import tempfile
import unittest
from pathlib import Path

import numpy as np

from free_space_analyzer import CameraInfo, DepthFrame, FreeSpaceAnalyzer
from free_space_analyzer.config import (
    AnalyzerConfig,
    DepthConfig,
    GroundConfig,
    OccupancyConfig,
    OpenSpaceConfig,
    ScoringConfig,
)
from free_space_analyzer.visualization import render_occupancy


def floor_and_wall_depth(camera: CameraInfo, wall_z: float = 4.0) -> np.ndarray:
    rows, _ = np.indices((camera.height, camera.width), dtype=np.float64)
    y_up = -(rows - camera.cy) / camera.fy
    ground = np.where(y_up < -1e-8, -camera.camera_height_m / y_up, np.inf)
    return np.minimum(ground, wall_z).astype(np.float32)


class AnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = CameraInfo.from_horizontal_fov(160, 120, 90.0, 1.7)
        self.config = AnalyzerConfig(
            depth=DepthConfig(0.1, 10.0, 2),
            ground=GroundConfig(mode="known_height", distance_tolerance_m=0.15),
            occupancy=OccupancyConfig(
                width_m=4.0,
                depth_m=4.0,
                resolution_m=0.2,
                min_obstacle_height_m=0.15,
                max_obstacle_height_m=2.2,
                safety_margin_m=0.0,
                raycast_free_space=True,
            ),
            open_space=OpenSpaceConfig(
                min_free_area_m2=2.0,
                min_rectangle_width_m=1.0,
                min_rectangle_depth_m=1.0,
                min_player_clearance_m=0.5,
                max_obstacle_ratio=0.5,
                max_unknown_ratio=0.7,
                nearby_unknown_is_unsafe=False,
            ),
            scoring=ScoringConfig(),
        )

    def test_end_to_end_returns_explainable_result(self) -> None:
        analyzer = FreeSpaceAnalyzer(self.config)
        result = analyzer.analyze(floor_and_wall_depth(self.camera), self.camera)
        self.assertGreater(result.observed_points, 0)
        self.assertGreater(result.largest_free_area_m2, 1.0)
        self.assertIsNotNone(result.largest_free_rectangle)
        self.assertGreaterEqual(result.score, 0)
        self.assertLessEqual(result.score, 100)

    def test_near_wall_fails_clearance(self) -> None:
        analyzer = FreeSpaceAnalyzer(self.config)
        result = analyzer.analyze(floor_and_wall_depth(self.camera, wall_z=0.35), self.camera)
        self.assertFalse(result.is_open_space)
        self.assertIn("player_clearance_below_minimum", result.failure_reasons)

    def test_depth_frame_api(self) -> None:
        analyzer = FreeSpaceAnalyzer(self.config)
        frame = DepthFrame(floor_and_wall_depth(self.camera), self.camera, frame_id="test")
        result = analyzer.analyze_frame(frame)
        self.assertGreater(result.observed_points, 0)

    def test_debug_render(self) -> None:
        analyzer = FreeSpaceAnalyzer(self.config)
        result = analyzer.analyze(floor_and_wall_depth(self.camera), self.camera)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.png"
            render_occupancy(analyzer.last_occupancy_grid, path, result.largest_free_rectangle)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
