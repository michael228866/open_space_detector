"""Adapter and batch runner for Unreal bullet-time ring captures.

Engine-specific conversion lives here, not in the core package. One capture
folder is one camera on the ring; every frame in it shares the same pose.

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
- `render_metadata.location` sits `GROUND_BIAS_M` above the floor, measured
  identically in both captured points.

`render_metadata.location` is NOT the character's position: the body sits 0.66 m
away at point 1 and 1.12 m at point 2, in different directions. The engine does
not export the actor location, so `locate_player_from_motion` recovers it from
the only cue left in the frame. The camera is static within a folder, so every
pixel that changes between frames belongs to the character.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from free_space_analyzer import AnalyzerConfig, CameraInfo, FreeSpaceAnalyzer, load_config
from free_space_analyzer.depth import preprocess_depth
from free_space_analyzer.free_space import analyze_free_space
from free_space_analyzer.geometry import depth_to_point_cloud, points_to_ground_coordinates
from free_space_analyzer.ground import known_height_plane
from free_space_analyzer.models import GridState, GroundPlane, OccupancyGrid
from free_space_analyzer.occupancy import build_occupancy_grid
from free_space_analyzer.scoring import evaluate_requirements, score_space
from free_space_analyzer.visualization import render_occupancy

REPO_ROOT = Path(__file__).resolve().parents[1]

# The floor sits this far below `render_metadata.location`. Measured as the
# median ground height under the assumed plane: -0.097 m at point 1 and
# -0.101 m at point 2, across all 12 views of each.
GROUND_BIAS_M = 0.10

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
        camera_height_m=float(camera_cm[2] - subject_cm[2]) / 100.0 + GROUND_BIAS_M,
        up_vector=(0.0, math.cos(down_pitch), -math.sin(down_pitch)),
        player_offset_m=(lateral_m, forward_m),
    )


def load_capture(
    folder: Path,
    frame: int = 0,
    subject_offset_cm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    config: AnalyzerConfig | None = None,
    motion_frame: int | None = None,
) -> Capture:
    """Load one frame. With `config`, the character is located and masked out.

    The mask is the honest way to drop the body: it removes exactly the pixels
    the character occupies instead of guessing a radius around them. The ground
    the character hides stays UNKNOWN, because it really was not observed.
    """
    metadata = json.loads((folder.parent / "render_metadata.json").read_text(encoding="utf-8"))
    camera_json = json.loads((folder / f"{frame:04d}.camera.json").read_text(encoding="utf-8"))
    depth = decode_depth_png(folder / f"{frame:04d}.depth.png")
    camera = camera_from_capture(camera_json, metadata["location"], subject_offset_cm)
    located = False
    if config is not None and motion_frame is not None and motion_frame != frame:
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
) -> tuple[OccupancyGrid, int]:
    """Merge every view of one spot into a single player-centred grid.

    A lone view cannot see past the character it is judging, and it cannot see
    outside its own frustum. Views around a ring fill in each other's blind
    spots, so the fused grid is the honest picture of the spot.

    Evidence merges the same way it does inside one grid: OCCUPIED beats FREE
    beats UNKNOWN. That keeps an obstacle only one camera saw, and never lets a
    cell nobody observed pass as free.
    """
    views = []
    for folder in folders:
        capture = load_capture(folder, frame, config=config, motion_frame=motion_frame)
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


def capture_folders(label_dir: Path) -> list[Path]:
    return sorted(p for p in label_dir.iterdir() if p.is_dir() and any(p.glob("*.depth.png")))


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
        "--fuse",
        action="store_true",
        help="merge every view of a spot into one player-centred grid",
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
    for label in args.labels:
        folders = capture_folders(label)
        if not folders:
            print(f"{label}: no capture folders found")
            exit_code = 1
            continue
        metadata = json.loads((label / "render_metadata.json").read_text(encoding="utf-8"))
        print(f"\n{label}  ({metadata['combo_id']}, point {metadata['point_id']})")
        if args.fuse:
            grid, views = fuse_views(
                folders,
                analyzer.config,
                args.frame,
                None if args.motion_frame < 0 else args.motion_frame,
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
            continue
        verdicts = []
        for folder in folders:
            capture = load_capture(
                folder,
                args.frame,
                tuple(args.subject_offset_cm),
                config=analyzer.config,
                motion_frame=None if args.motion_frame < 0 else args.motion_frame,
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

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
