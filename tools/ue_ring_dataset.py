"""Decide whether a captured point is a good activity spot.

Point it at one directory per candidate point. Every camera of that point is
fused into a single player-centred grid and judged once, because that is the
question being asked: is this spot open, not what does camera 7 think. Use
`--per-view` when a point needs explaining rather than deciding.

Engine-specific conversion lives here, not in the core package. One capture
folder is one camera on the ring; every frame in it shares the same pose.

Cost is linear at roughly 0.19 s per view, so a 68-camera point takes about
13 s and 230 MB. `occupancy._bresenham` is the hot spot if that ever matters.

Layout::

    <label>/render_metadata.json
    <label>/<CameraName>/NNNN.camera.json
    <label>/<CameraName>/NNNN.depth.png

Calibration verified against these captures:

- depth is linear Z-depth in metres, 16-bit, renormalized per frame with
  `depth_min`/`depth_max` in the PNG text chunks. Read as radial instead, the
  ground plane fit drops from 0.86 to 0.48 inliers.
- `up_vector = (0, cos pitch, -sin pitch)` lands 0.39 degrees from the RANSAC
  ground normal at pitch -15.
- the camera's world right vector is `(-sin yaw, cos yaw)`. Solved from the ring:
  this convention puts the character at one consistent world offset across all
  12 views (std 9 cm); the opposite sign scatters it (std 47 cm).
Two capture shapes exist, and they need different settings:

- No character in frame, one frame per camera. `render_metadata.location` is the
  spot being judged, so nothing is masked and the ground bias is zero. Terrain is
  often sloped, so fit the ground with `mode: ransac` (configs/ue_ring_env.yaml).
- A character in frame, many frames per camera. Then `location` is the actor and
  sits about 0.10 m above the floor (`--ground-bias-m 0.10`), the body is 0.66 to
  1.12 m away from it, and `locate_player_from_motion` finds them from the pixels
  that move between frames of a static camera.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import io
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ is None:  # run straight from a checkout, without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from free_space_analyzer import AnalyzerConfig, CameraInfo, FreeSpaceAnalyzer, load_config
from free_space_analyzer.depth import preprocess_depth
from free_space_analyzer.free_space import analyze_free_space
from free_space_analyzer.geometry import depth_to_point_cloud, points_to_ground_coordinates
from free_space_analyzer.ground import (
    GroundEstimationError,
    estimate_ground,
    known_height_plane,
)
from free_space_analyzer.models import GridState, GroundPlane, OccupancyGrid
from free_space_analyzer.occupancy import build_occupancy_grid
from free_space_analyzer.scoring import evaluate_requirements, score_space
from free_space_analyzer.visualization import render_occupancy

REPO_ROOT = Path(__file__).resolve().parents[1]

# How far the floor sits below `render_metadata.location`. Zero when the capture
# point is placed on the ground, which is the case whenever no character is in
# frame. Captures that put a character there report an actor pivot about 0.10 m
# above the floor, so those need `--ground-bias-m 0.10`.
DEFAULT_GROUND_BIAS_M = 0.0

# A depth pixel that shifts by more than this between two frames is the moving
# character rather than encoding noise.
MOTION_DELTA_M = 0.05
# Only the character's front surface is visible, so the median of those returns
# sits this much nearer the camera than the body really is. Measured against the
# 12-view ring mean: -0.122 m at point 1, -0.097 m at point 2, std 0.03-0.05.
SURFACE_BIAS_M = 0.11
# The character is somewhere near the capture point; anything moving further out
# is scenery, not the player.
PLAYER_SEARCH_RADIUS_M = 2.5
PLAYER_HEIGHT_BAND_M = (0.2, 2.3)
MIN_PLAYER_POINTS = 50


@dataclass(frozen=True)
class Capture:
    depth_m: np.ndarray
    camera: CameraInfo
    name: str
    frame: int
    player_located: bool = False


def decode_depth_png(path: Path) -> np.ndarray:
    """16-bit normalized engine depth to metres. Far-clamp pixels become NaN."""
    with Image.open(path) as image:
        image.load()
        raw = np.asarray(image).astype(np.float32)
        info = image.info
    if image.mode != "I;16":
        raise ValueError(f"{path}: expected a 16-bit depth PNG, got mode {image.mode}")
    missing = [key for key in ("depth_min", "depth_max") if key not in info]
    if missing:
        raise ValueError(f"{path}: depth PNG is missing metadata {missing}")
    low, high = float(info["depth_min"]), float(info["depth_max"])
    metres = low + (raw / 65535.0) * (high - low)
    clamp = float(info.get("max_depth_clamp", "inf"))
    # A pixel sitting on the far clamp is sky or a miss, not a surface at 50 m.
    metres[metres >= clamp - 1e-3] = np.nan
    return metres


def motion_mask(folder: Path, frame: int, other_frame: int) -> np.ndarray:
    """Pixels that move between two frames of a static camera: the character."""
    first = decode_depth_png(folder / f"{frame:04d}.depth.png")
    second = decode_depth_png(folder / f"{other_frame:04d}.depth.png")
    far = 1e4
    delta = np.abs(np.nan_to_num(first, nan=far) - np.nan_to_num(second, nan=far))
    return delta > MOTION_DELTA_M


def locate_player_from_motion(
    depth_m: np.ndarray,
    mask: np.ndarray,
    camera: CameraInfo,
    config: AnalyzerConfig,
) -> tuple[float, float] | None:
    """Ground (x, z) of the character, or None when too little of it moved."""
    masked = np.where(mask, preprocess_depth(depth_m, camera, config.depth), np.nan)
    points = depth_to_point_cloud(masked, camera, stride=config.depth.sample_stride)
    if len(points) < MIN_PLAYER_POINTS:
        return None
    plane = known_height_plane(points, camera, config.ground)
    x, z, height = points_to_ground_coordinates(points, plane)
    seed_x, seed_z = camera.player_offset_m
    low, high = PLAYER_HEIGHT_BAND_M
    keep = (
        (np.hypot(x - seed_x, z - seed_z) < PLAYER_SEARCH_RADIUS_M)
        & (height > low)
        & (height < high)
    )
    if int(keep.sum()) < MIN_PLAYER_POINTS:
        return None
    # +z is away from the camera, which is the direction the surface bias needs.
    return float(np.median(x[keep])), float(np.median(z[keep])) + SURFACE_BIAS_M


def camera_from_capture(
    camera_json: dict,
    subject_location_cm: list[float],
    subject_offset_cm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ground_bias_m: float = DEFAULT_GROUND_BIAS_M,
) -> CameraInfo:
    """Build CameraInfo from one engine pose plus where the character stands.

    `subject_offset_cm` moves the character off `render_metadata.location` in
    world centimetres, for the usual case where the actor is not spawned exactly
    on the capture point.
    """
    pitch_deg, yaw_deg, roll_deg = camera_json["ue_rotation_pyr"]
    if abs(roll_deg) > 1e-6:
        raise ValueError(f"roll {roll_deg} is not supported; up_vector assumes zero roll")

    camera_cm = np.asarray(camera_json["ue_location_cm"], dtype=np.float64)
    subject_cm = np.asarray(subject_location_cm, dtype=np.float64) + np.asarray(
        subject_offset_cm, dtype=np.float64
    )
    delta = subject_cm - camera_cm

    # Unreal is Z-up, so the ground plane is world XY.
    yaw = math.radians(yaw_deg)
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    right = np.array([-math.sin(yaw), math.cos(yaw)])
    forward_m = float(delta[:2] @ forward) / 100.0
    lateral_m = float(delta[:2] @ right) / 100.0
    if forward_m <= 0.0:
        raise ValueError("the subject is behind the camera; check yaw or subject_offset_cm")

    down_pitch = math.radians(-pitch_deg)
    return CameraInfo(
        width=int(camera_json["w"]),
        height=int(camera_json["h"]),
        fx=float(camera_json["fl_x"]),
        fy=float(camera_json["fl_y"]),
        cx=float(camera_json["cx"]),
        cy=float(camera_json["cy"]),
        camera_height_m=float(camera_cm[2] - subject_cm[2]) / 100.0 + ground_bias_m,
        up_vector=(0.0, math.cos(down_pitch), -math.sin(down_pitch)),
        player_offset_m=(lateral_m, forward_m),
    )


def load_capture(
    folder: Path,
    frame: int = 0,
    subject_offset_cm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    config: AnalyzerConfig | None = None,
    motion_frame: int | None = None,
    ground_bias_m: float = DEFAULT_GROUND_BIAS_M,
) -> Capture:
    """Load one frame. With `config`, the character is located and masked out.

    The mask is the honest way to drop the body: it removes exactly the pixels
    the character occupies instead of guessing a radius around them. The ground
    the character hides stays UNKNOWN, because it really was not observed.
    """
    metadata = json.loads((folder.parent / "render_metadata.json").read_text(encoding="utf-8"))
    camera_json = json.loads((folder / f"{frame:04d}.camera.json").read_text(encoding="utf-8"))
    depth = decode_depth_png(folder / f"{frame:04d}.depth.png")
    camera = camera_from_capture(
        camera_json, metadata["location"], subject_offset_cm, ground_bias_m
    )
    located = False
    # A single-frame capture has no motion to read, and nothing to read it for:
    # no character in frame means the capture point is the spot being judged.
    has_motion_frame = (
        motion_frame is not None
        and motion_frame != frame
        and (folder / f"{motion_frame:04d}.depth.png").exists()
    )
    if config is not None and has_motion_frame:
        mask = motion_mask(folder, frame, motion_frame)
        player = locate_player_from_motion(depth, mask, camera, config)
        if player is not None:
            camera = dataclasses.replace(camera, player_offset_m=player)
            depth = np.where(mask, np.nan, depth)
            located = True
    return Capture(
        depth_m=depth, camera=camera, name=folder.name, frame=frame, player_located=located
    )


def _world_basis(camera_json: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera ground position and its forward/right axes, all in world metres."""
    yaw = math.radians(camera_json["ue_rotation_pyr"][1])
    position = np.asarray(camera_json["ue_location_cm"], dtype=np.float64)[:2] / 100.0
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    right = np.array([-math.sin(yaw), math.cos(yaw)])
    return position, forward, right


def fuse_views(
    folders: list[Path],
    config: AnalyzerConfig,
    frame: int = 0,
    motion_frame: int | None = 12,
    ground_bias_m: float = DEFAULT_GROUND_BIAS_M,
) -> tuple[OccupancyGrid, int]:
    """Merge every view of one spot into a single player-centred grid.

    A lone view cannot see past the character it is judging, and it cannot see
    outside its own frustum. Views around a ring fill in each other's blind
    spots, so the fused grid is the honest picture of the spot.

    Evidence merges the same way it does inside one grid: OCCUPIED beats FREE
    beats UNKNOWN. That keeps an obstacle only one camera saw, and never lets a
    cell nobody observed pass as free.
    """
    # ponytail: every view's depth is held at once, about 3 MB each. Fine at 68
    # cameras; stream in two passes if a rig ever gets much larger.
    views = []
    for folder in folders:
        capture = load_capture(
            folder, frame, config=config, motion_frame=motion_frame, ground_bias_m=ground_bias_m
        )
        camera_json = json.loads((folder / f"{frame:04d}.camera.json").read_text(encoding="utf-8"))
        position, forward, right = _world_basis(camera_json)
        lateral, ahead = capture.camera.player_offset_m
        player = position + right * lateral + forward * ahead
        views.append((capture, position, forward, right, player))
    if not views:
        raise ValueError("no views to fuse")

    # Each view sees the character from its own side, so the ring mean cancels
    # the front-surface bias that survives in any single estimate.
    player_world = np.mean([view[4] for view in views], axis=0)
    flat = GroundPlane(normal=np.array([0.0, 1.0, 0.0]), d=0.0)
    half_depth = -math.ceil(config.occupancy.depth_m / config.occupancy.resolution_m) / 2.0
    z_min = half_depth * config.occupancy.resolution_m

    merged: OccupancyGrid | None = None
    observed = 0
    for capture, position, forward, right, _ in views:
        depth = preprocess_depth(capture.depth_m, capture.camera, config.depth)
        points = depth_to_point_cloud(depth, capture.camera, stride=config.depth.sample_stride)
        try:
            plane = estimate_ground(points, capture.camera, config.ground)
        except GroundEstimationError:
            # One view failing to find a floor is not a reason to lose the point;
            # the rig's own camera height still describes a usable plane.
            plane = known_height_plane(points, capture.camera, config.ground)
        x, z, height = points_to_ground_coordinates(points, plane)
        # This view's ground coordinates into the shared player-centred frame.
        world = position + np.outer(x, right) + np.outer(z, forward) - player_world
        grid = build_occupancy_grid(
            np.column_stack((world[:, 0], height, world[:, 1])),
            flat,
            config.occupancy,
            config.ground,
            player_offset_m=(0.0, 0.0),
            camera_offset_m=tuple(position - player_world),
            z_min_m=z_min,
        )
        observed += grid.observed_points
        if merged is None:
            merged = grid
            continue
        free = (grid.cells == GridState.FREE) & (merged.cells == GridState.UNKNOWN)
        merged.cells[free] = GridState.FREE
        merged.cells[grid.cells == GridState.OCCUPIED] = GridState.OCCUPIED
    assert merged is not None
    merged.observed_points = observed
    return merged, len(views)


def _thumbnail(path: Path, width: int = 300) -> str:
    """One image as an inline data URI, so the report stays a single file."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((width, width), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def write_report(cards: list[dict], path: Path) -> Path:
    """A single self-contained page: every point at a glance, best first.

    Written locally rather than published, because these are unreleased game
    frames. Open it in a browser; nothing is uploaded anywhere.
    """
    cards = sorted(cards, key=lambda card: -card["score"])
    blocks = []
    for card in cards:
        views = "".join(f'<img src="{src}">' for src in card["views"])
        reasons = "".join(f"<li>{reason}</li>" for reason in card["reasons"]) or "<li>-</li>"
        blocks.append(f"""
<section class="{'ok' if card['open'] else 'bad'}">
  <header>
    <span class="badge">{'OPEN' if card['open'] else 'NOT OPEN'}</span>
    <h2>{card['name']}</h2>
    <span class="score">{card['score']:.0f}</span>
  </header>
  <div class="body">
    <figure><img class="grid" src="{card['grid']}">
      <figcaption>fused occupancy</figcaption></figure>
    <div class="views">{views}</div>
    <dl>
      <dt>reachable</dt><dd>{card['reach']:.1f} m&sup2;</dd>
      <dt>rectangle</dt><dd>{card['rect']}</dd>
      <dt>nearest obstacle</dt><dd>{card['obstacle']}</dd>
      <dt>obstacle / unknown</dt><dd>{card['obstacle_ratio']:.0%} / {card['unknown']:.0%}</dd>
      <dt>ground fit</dt><dd>{card['ground']:.2f}</dd>
      <dt>fails</dt><dd><ul>{reasons}</ul></dd>
    </dl>
  </div>
</section>""")
    style = """
body{font:14px system-ui,sans-serif;margin:0;padding:24px;background:#f6f7f9;color:#1c1f23}
h1{font-size:20px;margin:0 0 4px}
p.sub{color:#666;margin:0 0 20px}
section{background:#fff;border-radius:10px;margin-bottom:16px;overflow:hidden;
  box-shadow:0 1px 3px rgba(0,0,0,.12);border-left:6px solid #d0d4da}
section.ok{border-left-color:#2e9e5b}
section.bad{border-left-color:#d4453f}
header{display:flex;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid #eceef1}
header h2{font-size:15px;margin:0;font-weight:600;flex:1;word-break:break-all}
.badge{font-size:11px;font-weight:700;letter-spacing:.04em;padding:3px 8px;border-radius:4px;
  background:#eceef1;color:#555}
.ok .badge{background:#e3f4ea;color:#1c6b3c}.bad .badge{background:#fbe6e5;color:#9e2b27}
.score{font-size:24px;font-weight:700;font-variant-numeric:tabular-nums}
.body{display:flex;gap:16px;padding:16px;flex-wrap:wrap;align-items:flex-start}
.grid{width:260px;border-radius:6px;background:#3f4650}
figure{margin:0}figcaption{font-size:11px;color:#888;text-align:center;margin-top:4px}
.views{display:grid;grid-template-columns:repeat(2,150px);gap:6px}
.views img{width:150px;border-radius:4px;display:block}
dl{display:grid;grid-template-columns:auto auto;gap:4px 14px;margin:0;font-size:13px}
dl{align-content:start}
dt{color:#777}dd{margin:0;font-variant-numeric:tabular-nums}
dd ul{margin:0;padding-left:16px}dd li{color:#9e2b27}
@media(max-width:640px){.body{flex-direction:column}.grid,.views{width:100%}}
"""
    opened = sum(1 for card in cards if card["open"])
    html = (
        "<!doctype html><meta charset='utf-8'><title>Capture points</title>"
        f"<style>{style}</style>"
        f"<h1>Capture points</h1><p class='sub'>{opened} of {len(cards)} open, best first. "
        "Grey is unobserved, white free, red blocked (safety margin included); "
        "green box is the largest usable rectangle, blue dot the evaluated point.</p>"
        + "".join(blocks)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def _ground_fit(folders: list[Path], config: AnalyzerConfig, args: argparse.Namespace) -> float:
    """Mean ground inlier ratio over the views, as a trust score for the geometry."""
    ratios = []
    for folder in folders:
        capture = load_capture(folder, args.frame, ground_bias_m=args.ground_bias_m)
        depth = preprocess_depth(capture.depth_m, capture.camera, config.depth)
        points = depth_to_point_cloud(depth, capture.camera, stride=config.depth.sample_stride)
        try:
            ratios.append(estimate_ground(points, capture.camera, config.ground).inlier_ratio)
        except GroundEstimationError:
            ratios.append(0.0)
    return float(np.mean(ratios)) if ratios else 0.0


def capture_folders(point_dir: Path) -> list[Path]:
    """The camera folders of one capture point."""
    return sorted(p for p in point_dir.iterdir() if p.is_dir() and any(p.glob("*.depth.png")))


def capture_points(path: Path) -> list[Path]:
    """Capture points at `path`, whether it is one point or a folder of them.

    Windows shells do not expand `images/*/`, so passing the parent has to work.
    """
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory")
    if (path / "render_metadata.json").exists():
        return [path]
    points = sorted(
        child
        for child in path.iterdir()
        if child.is_dir() and (child / "render_metadata.json").exists()
    )
    if not points:
        raise ValueError(
            f"{path} holds no capture points; expected render_metadata.json in it "
            "or in its subdirectories"
        )
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # argparse does not run `type` over defaults, so these are Paths already.
    parser.add_argument("labels", nargs="*", default=[Path("good"), Path("bad")], type=Path)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "ue_ring.yaml")
    parser.add_argument("--frame", type=int, default=0, help="frame index within each folder")
    parser.add_argument(
        "--motion-frame",
        type=int,
        default=12,
        help="second frame used to find the character; -1 keeps the metadata position",
    )
    parser.add_argument("--debug-dir", type=Path, help="write one occupancy PNG per camera")
    parser.add_argument(
        "--per-view",
        action="store_true",
        help="report each camera separately instead of fusing them (diagnostic)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="write a single self-contained HTML page showing every point at a glance",
    )
    parser.add_argument(
        "--ground-bias-m",
        type=float,
        default=DEFAULT_GROUND_BIAS_M,
        help="how far the floor sits below the capture point (0.10 with a character in frame)",
    )
    parser.add_argument(
        "--subject-offset-cm",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="where the character stands relative to render_metadata.location",
    )
    args = parser.parse_args()

    analyzer = FreeSpaceAnalyzer(load_config(args.config))
    exit_code = 0
    cards: list[dict] = []
    labels: list[Path] = []
    for given in args.labels:
        try:
            labels.extend(capture_points(given))
        except ValueError as error:
            print(f"error: {error}")
            exit_code = 1
    for label in labels:
        folders = capture_folders(label)
        if not folders:
            print(f"{label}: no camera folders found")
            exit_code = 1
            continue
        metadata = json.loads((label / "render_metadata.json").read_text(encoding="utf-8"))
        print(f"\n{label}  ({metadata['combo_id']}, point {metadata['point_id']})")
        if not args.per_view:
            grid, views = fuse_views(
                folders,
                analyzer.config,
                args.frame,
                None if args.motion_frame < 0 else args.motion_frame,
                args.ground_bias_m,
            )
            metrics = analyze_free_space(grid, analyzer.config.open_space)
            is_open, reasons = evaluate_requirements(
                metrics, analyzer.config.open_space, grid.resolution_m
            )
            score = score_space(metrics, analyzer.config.open_space, analyzer.config.scoring)
            print(
                f"  fused {views} views -> {'OPEN' if is_open else 'no':<4} "
                f"score={score:6.2f}  reach={metrics.player_reachable_area_m2:6.2f} m2  "
                f"unk={metrics.unknown_ratio:.2f}  clearance={metrics.max_clearance_m:.2f} m  "
                + ", ".join(r.replace("_", " ") for r in reasons)
            )
            if args.debug_dir:
                render_occupancy(
                    grid,
                    args.debug_dir / f"{label.name}-fused.png",
                    rectangle=metrics.largest_rectangle,
                )
            if args.report:
                rect = metrics.largest_rectangle
                with tempfile.TemporaryDirectory() as tmp:
                    png = render_occupancy(
                        grid, Path(tmp) / "g.png", rectangle=rect, scale=3
                    )
                    grid_uri = _thumbnail(png, 320)
                # Four cameras spread around the ring is enough to recognize a place.
                spread = [folders[round(i * len(folders) / 4)] for i in range(4)]
                cards.append(
                    {
                        "name": label.name,
                        "open": is_open,
                        "score": score,
                        "reach": metrics.player_reachable_area_m2,
                        "rect": f"{rect.width_m:.1f} x {rect.depth_m:.1f} m" if rect else "none",
                        "obstacle": (
                            f"{metrics.nearest_obstacle_m:.2f} m"
                            if metrics.nearest_obstacle_m is not None
                            else "none"
                        ),
                        "obstacle_ratio": metrics.obstacle_ratio,
                        "unknown": metrics.unknown_ratio,
                        "ground": _ground_fit(folders, analyzer.config, args),
                        "reasons": [r.replace("_", " ") for r in reasons],
                        "grid": grid_uri,
                        "views": [_thumbnail(f / f"{args.frame:04d}.rgb.png") for f in spread],
                    }
                )
            continue
        verdicts = []
        for folder in folders:
            capture = load_capture(
                folder,
                args.frame,
                tuple(args.subject_offset_cm),
                config=analyzer.config,
                motion_frame=None if args.motion_frame < 0 else args.motion_frame,
                ground_bias_m=args.ground_bias_m,
            )
            result = analyzer.analyze(capture.depth_m, capture.camera)
            verdicts.append(result.is_open_space)
            nearest = result.nearest_obstacle_m
            print(
                f"  {capture.name:<28} {'OPEN' if result.is_open_space else 'no':<4} "
                f"score={result.score:6.2f}  "
                f"reach={result.player_reachable_area_m2:6.2f} m2  "
                f"obstacle={nearest if nearest is not None else float('nan'):5.2f}  "
                f"unk={result.unknown_ratio:.2f}  ground={result.ground_inlier_ratio:.2f}  "
                f"{'' if capture.player_located else '[player NOT located] '}"
                + ", ".join(r.replace("_", " ") for r in result.failure_reasons[:2])
            )
            if args.debug_dir:
                render_occupancy(
                    analyzer.last_occupancy_grid,
                    args.debug_dir / label.name / f"{capture.name}.png",
                    rectangle=result.largest_free_rectangle,
                    player_xz=capture.camera.player_offset_m,
                )
        opened = sum(verdicts)
        agreement = "all views agree" if opened in (0, len(verdicts)) else "VIEWS DISAGREE"
        print(f"  -> {opened}/{len(verdicts)} views call it open, {agreement}")

    if args.report and cards:
        print(f"\nwrote {write_report(cards, args.report)}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
