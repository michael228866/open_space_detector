from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from free_space_analyzer import CameraInfo, FreeSpaceAnalyzer
from free_space_analyzer.visualization import render_occupancy


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    depth = np.load(root / "sample_data/depth.npy", allow_pickle=False)
    camera = CameraInfo.from_horizontal_fov(
        width=depth.shape[1],
        height=depth.shape[0],
        horizontal_fov_deg=90.0,
        camera_height_m=1.7,
    )
    analyzer = FreeSpaceAnalyzer.from_config(root / "configs/default.yaml")
    result = analyzer.analyze(depth, camera)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    if analyzer.last_occupancy_grid is not None:
        render_occupancy(
            analyzer.last_occupancy_grid,
            root / "output/occupancy.png",
            rectangle=result.largest_free_rectangle,
        )


if __name__ == "__main__":
    main()
