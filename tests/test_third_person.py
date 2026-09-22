"""A third-person rig puts the player in front of the camera, inside the frame."""

import unittest

import numpy as np

from free_space_analyzer import CameraInfo, FreeSpaceAnalyzer
from free_space_analyzer.config import (
    AnalyzerConfig,
    DepthConfig,
    GroundConfig,
    OccupancyConfig,
    OpenSpaceConfig,
)
from free_space_analyzer.free_space import analyze_free_space
from free_space_analyzer.models import GridState, GroundPlane, OccupancyGrid
from free_space_analyzer.occupancy import build_occupancy_grid

PLAYER = (0.0, 3.0)
PERMISSIVE = OpenSpaceConfig(
    min_free_area_m2=1.0,
    min_rectangle_width_m=1.0,
    min_rectangle_depth_m=1.0,
    min_player_clearance_m=0.5,
    max_obstacle_ratio=1.0,
    max_unknown_ratio=1.0,
    nearby_unknown_is_unsafe=False,
)


class PlayerOriginTests(unittest.TestCase):
    def test_reachability_follows_the_player_not_the_camera(self) -> None:
        # Wall at z = 1.5 m separates the camera pocket from the player.
        cells = np.full((20, 20), GridState.FREE, dtype=np.int8)
        cells[3, :] = GridState.OCCUPIED
        grid = OccupancyGrid(cells, 0.5, -5.0, 0.0)

        from_camera = analyze_free_space(grid, PERMISSIVE)
        from_player = analyze_free_space(grid, PERMISSIVE, PLAYER)

        self.assertAlmostEqual(from_camera.player_reachable_area_m2, 15.0)
        self.assertAlmostEqual(from_player.player_reachable_area_m2, 80.0)

    def test_clearance_is_measured_from_the_player(self) -> None:
        cells = np.full((20, 20), GridState.FREE, dtype=np.int8)
        cells[7, 10] = GridState.OCCUPIED  # ~0.5 m in front of the player
        grid = OccupancyGrid(cells, 0.5, -5.0, 0.0)

        self.assertGreater(analyze_free_space(grid, PERMISSIVE).nearest_obstacle_m, 3.0)
        self.assertLess(analyze_free_space(grid, PERMISSIVE, PLAYER).nearest_obstacle_m, 1.0)

    def test_first_person_stays_the_default(self) -> None:
        camera = CameraInfo(320, 240, 160.0, 160.0, 159.5, 119.5, 1.7)
        self.assertEqual(camera.player_offset_m, (0.0, 0.0))
        cells = np.full((10, 10), GridState.FREE, dtype=np.int8)
        grid = OccupancyGrid(cells, 0.5, -2.5, 0.0)
        self.assertEqual(
            analyze_free_space(grid, PERMISSIVE),
            analyze_free_space(grid, PERMISSIVE, (0.0, 0.0)),
        )

    def test_player_offset_must_be_a_finite_pair(self) -> None:
        for bad in ((0.0, float("nan")), (1.0, 2.0, 3.0)):
            with self.assertRaises(ValueError):
                CameraInfo(320, 240, 160.0, 160.0, 159.5, 119.5, 1.7, player_offset_m=bad)


class PlayerBodyTests(unittest.TestCase):
    """The character's own depth returns stand exactly where space is judged."""

    def scene(self) -> tuple[np.ndarray, GroundPlane]:
        plane = GroundPlane(np.array([0.0, 1.0, 0.0]), 1.7)
        floor = np.column_stack(
            (
                np.linspace(-2.0, 2.0, 200),
                np.full(200, -1.7),
                np.full(200, 2.0),
            )
        )
        # A torso-height blob standing at the player position.
        body = np.array([[0.0, -0.9, 3.0], [0.1, -0.7, 3.0], [-0.1, -1.1, 3.05]])
        return np.vstack((floor, body)), plane

    def occupancy(self, player_radius_m: float) -> OccupancyConfig:
        return OccupancyConfig(
            width_m=8.0,
            depth_m=8.0,
            resolution_m=0.25,
            safety_margin_m=0.0,
            player_radius_m=player_radius_m,
        )

    def test_body_counts_as_an_obstacle_without_the_exclusion(self) -> None:
        points, plane = self.scene()
        grid = build_occupancy_grid(points, plane, self.occupancy(0.0), GroundConfig(), PLAYER)
        self.assertEqual(grid.cells[grid.world_to_cell(*PLAYER)], GridState.OCCUPIED)

    def test_body_is_excluded_and_the_player_cell_is_walkable(self) -> None:
        points, plane = self.scene()
        grid = build_occupancy_grid(points, plane, self.occupancy(0.5), GroundConfig(), PLAYER)
        self.assertEqual(grid.cells[grid.world_to_cell(*PLAYER)], GridState.FREE)

    def test_exclusion_does_not_erase_obstacles_away_from_the_player(self) -> None:
        points, plane = self.scene()
        far = np.array([[0.0, -0.9, 5.0], [0.05, -0.8, 5.0]])
        grid = build_occupancy_grid(
            np.vstack((points, far)), plane, self.occupancy(0.5), GroundConfig(), PLAYER
        )
        self.assertEqual(grid.cells[grid.world_to_cell(0.0, 5.0)], GridState.OCCUPIED)

    def test_player_outside_the_grid_is_rejected(self) -> None:
        points, plane = self.scene()
        with self.assertRaises(RuntimeError) as error:
            build_occupancy_grid(
                points, plane, self.occupancy(0.5), GroundConfig(), (0.0, 12.0)
            )
        self.assertIn("player_offset_m", str(error.exception))


class ThirdPersonPipelineTests(unittest.TestCase):
    def test_analyze_runs_with_a_third_person_rig(self) -> None:
        camera = CameraInfo.from_horizontal_fov(
            160, 120, 90.0, camera_height_m=2.4, player_offset_m=(0.0, 2.0)
        )
        rows, _ = np.indices((camera.height, camera.width), dtype=np.float64)
        y_up = -(rows - camera.cy) / camera.fy
        ground = np.where(y_up < -1e-8, -camera.camera_height_m / y_up, np.inf)
        depth = np.minimum(ground, 8.0).astype(np.float32)

        config = AnalyzerConfig(
            depth=DepthConfig(0.1, 12.0, 2),
            ground=GroundConfig(mode="known_height", distance_tolerance_m=0.15),
            occupancy=OccupancyConfig(
                width_m=8.0,
                depth_m=8.0,
                resolution_m=0.2,
                safety_margin_m=0.0,
                player_radius_m=0.35,
            ),
            open_space=PERMISSIVE,
        )
        result = FreeSpaceAnalyzer(config).analyze(depth, camera)
        self.assertGreater(result.player_reachable_area_m2, 1.0)
        self.assertIsNotNone(result.largest_free_rectangle)


if __name__ == "__main__":
    unittest.main()
