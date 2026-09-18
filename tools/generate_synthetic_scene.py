from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _intersect_aabb(
    directions: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> np.ndarray:
    safe = np.where(np.abs(directions) < 1e-9, np.nan, directions)
    t1 = minimum / safe
    t2 = maximum / safe
    near = np.nanmax(np.minimum(t1, t2), axis=-1)
    far = np.nanmin(np.maximum(t1, t2), axis=-1)
    valid = (far >= np.maximum(near, 0.0)) & np.isfinite(near)
    return np.where(valid, np.maximum(near, 0.0), np.inf)


def synthetic_depth(
    width: int = 320,
    height: int = 240,
    horizontal_fov_deg: float = 90.0,
    camera_height_m: float = 1.7,
    with_obstacles: bool = True,
) -> np.ndarray:
    fx = width / (2 * np.tan(np.deg2rad(horizontal_fov_deg) / 2))
    fy = fx
    cx = (width - 1) / 2
    cy = (height - 1) / 2
    rows, cols = np.indices((height, width), dtype=np.float64)
    # X right, Y up, Z forward, with direction Z fixed to one.
    directions = np.stack(
        ((cols - cx) / fx, -(rows - cy) / fy, np.ones_like(rows)),
        axis=-1,
    )
    y_direction = directions[..., 1]
    ground_t = np.where(y_direction < -1e-9, -camera_height_m / y_direction, np.inf)
    wall_t = np.full((height, width), 9.0, dtype=np.float64)
    depth = np.minimum(ground_t, wall_t)

    if with_obstacles:
        boxes = [
            # Coordinates are relative to the camera. Ground is y=-camera_height.
            (np.array([-0.9, -camera_height_m, 3.0]), np.array([0.9, -0.5, 4.2])),
            (np.array([2.2, -camera_height_m, 5.2]), np.array([3.2, 0.1, 6.2])),
        ]
        for minimum, maximum in boxes:
            depth = np.minimum(depth, _intersect_aabb(directions, minimum, maximum))
    depth[~np.isfinite(depth)] = 0.0
    return depth.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic depth data")
    parser.add_argument("--output-dir", type=Path, default=Path("sample_data"))
    parser.add_argument("--clear", action="store_true", help="Generate a scene without boxes")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    depth = synthetic_depth(with_obstacles=not args.clear)
    np.save(args.output_dir / "depth.npy", depth)
    camera = {
        "width": int(depth.shape[1]),
        "height": int(depth.shape[0]),
        "horizontal_fov_deg": 90.0,
        "camera_height_m": 1.7,
        "depth_type": "z_depth",
    }
    (args.output_dir / "camera.json").write_text(
        json.dumps(camera, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

