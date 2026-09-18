from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .analyzer import FreeSpaceAnalyzer
from .models import CameraInfo, DepthType
from .visualization import render_occupancy


def camera_from_mapping(data: dict[str, Any], depth_shape: tuple[int, int]) -> CameraInfo:
    height, width = depth_shape
    declared_width = int(data.get("width", width))
    declared_height = int(data.get("height", height))
    if (declared_height, declared_width) != depth_shape:
        raise ValueError("Camera JSON resolution does not match the depth array")
    depth_type = DepthType(data.get("depth_type", DepthType.Z_DEPTH.value))
    if "camera_height_m" not in data:
        raise ValueError("Camera JSON is missing: camera_height_m")
    common = {
        "width": declared_width,
        "height": declared_height,
        "camera_height_m": float(data["camera_height_m"]),
        "depth_type": depth_type,
        "up_vector": tuple(data.get("up_vector", (0.0, 1.0, 0.0))),
    }
    if "horizontal_fov_deg" in data:
        return CameraInfo.from_horizontal_fov(
            horizontal_fov_deg=float(data["horizontal_fov_deg"]),
            **common,
        )
    required = ("fx", "fy", "cx", "cy")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"Camera JSON is missing: {', '.join(missing)}")
    return CameraInfo(
        fx=float(data["fx"]),
        fy=float(data["fy"]),
        cx=float(data["cx"]),
        cy=float(data["cy"]),
        **common,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze free space from a depth map")
    parser.add_argument("depth", type=Path, help="H x W .npy depth map")
    parser.add_argument("--camera", type=Path, required=True, help="Camera JSON file")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--debug-png", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        depth = np.load(args.depth, allow_pickle=False)
        with args.camera.open("r", encoding="utf-8") as handle:
            camera_data = json.load(handle)
        camera = camera_from_mapping(camera_data, depth.shape)
        analyzer = FreeSpaceAnalyzer.from_config(args.config)
        result = analyzer.analyze(depth, camera)
        result_json = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
        print(result_json)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(result_json + "\n", encoding="utf-8")
        if args.debug_png:
            if analyzer.last_occupancy_grid is None:
                raise RuntimeError("Analyzer did not produce an occupancy grid")
            render_occupancy(
                analyzer.last_occupancy_grid,
                args.debug_png,
                rectangle=result.largest_free_rectangle,
            )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

