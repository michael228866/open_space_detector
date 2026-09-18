import time
import unittest

import numpy as np

from free_space_analyzer.config import OpenSpaceConfig
from free_space_analyzer.free_space import analyze_free_space, largest_free_rectangle
from free_space_analyzer.models import GridState, OccupancyGrid
from free_space_analyzer.scoring import evaluate_requirements


def permissive_config(**overrides: object) -> OpenSpaceConfig:
    defaults = dict(
        min_free_area_m2=1.0,
        min_rectangle_width_m=1.0,
        min_rectangle_depth_m=1.0,
        min_player_clearance_m=0.5,
        max_obstacle_ratio=1.0,
        max_unknown_ratio=1.0,
        nearby_unknown_is_unsafe=False,
    )
    defaults.update(overrides)
    return OpenSpaceConfig(**defaults)  # type: ignore[arg-type]


class FreeSpaceTests(unittest.TestCase):
    def test_largest_rectangle_on_all_free_grid(self) -> None:
        grid = OccupancyGrid(
            cells=np.full((3, 4), GridState.FREE, dtype=np.int8),
            resolution_m=0.5,
            x_min_m=-1.0,
            z_min_m=0.0,
        )
        rectangle = largest_free_rectangle(grid)
        self.assertIsNotNone(rectangle)
        self.assertAlmostEqual(rectangle.width_m, 2.0)
        self.assertAlmostEqual(rectangle.depth_m, 1.5)
        self.assertAlmostEqual(rectangle.area_m2, 3.0)

    def test_unknown_is_not_free(self) -> None:
        cells = np.array(
            [
                [0, 0, -1, -1],
                [0, 0, -1, -1],
                [1, 0, -1, -1],
            ],
            dtype=np.int8,
        )
        grid = OccupancyGrid(cells, 1.0, -2.0, 0.0)
        metrics = analyze_free_space(grid, permissive_config())
        self.assertEqual(metrics.largest_free_area_m2, 5.0)
        self.assertAlmostEqual(metrics.unknown_ratio, 0.5)


class ReachabilityTests(unittest.TestCase):
    """A wall 1.5 m ahead must invalidate everything behind it."""

    def walled_grid(self) -> OccupancyGrid:
        cells = np.full((20, 20), GridState.FREE, dtype=np.int8)
        cells[3, :] = GridState.OCCUPIED  # full-width wall at z = 1.5 m
        return OccupancyGrid(cells, 0.5, -5.0, 0.0)

    def test_area_behind_a_wall_does_not_satisfy_requirements(self) -> None:
        grid = self.walled_grid()
        config = OpenSpaceConfig(nearby_unknown_is_unsafe=False)
        metrics = analyze_free_space(grid, config)

        self.assertAlmostEqual(metrics.player_reachable_area_m2, 15.0)
        self.assertAlmostEqual(metrics.largest_free_area_m2, 80.0)

        is_open, reasons = evaluate_requirements(metrics, config, grid.resolution_m)
        self.assertFalse(is_open)
        self.assertIn("largest_free_area_below_minimum", reasons)
        self.assertIn("required_free_rectangle_not_found", reasons)

    def test_rectangle_behind_a_wall_is_never_reported(self) -> None:
        metrics = analyze_free_space(self.walled_grid(), OpenSpaceConfig())
        rectangle = metrics.largest_rectangle
        self.assertIsNotNone(rectangle)
        # The wall starts at z = 1.5 m; nothing beyond it may appear.
        self.assertLessEqual(rectangle.z_max_m, 1.5)

    def test_same_room_passes_without_the_wall(self) -> None:
        cells = np.full((20, 20), GridState.FREE, dtype=np.int8)
        grid = OccupancyGrid(cells, 0.5, -5.0, 0.0)
        config = OpenSpaceConfig(nearby_unknown_is_unsafe=False)
        metrics = analyze_free_space(grid, config)
        is_open, reasons = evaluate_requirements(metrics, config, grid.resolution_m)
        self.assertTrue(is_open, reasons)

    def test_unreachable_free_space_is_excluded_from_clearance(self) -> None:
        grid = self.walled_grid()
        metrics = analyze_free_space(grid, permissive_config())
        # Reachable strip is 1.5 m deep, so clearance cannot exceed it.
        self.assertLessEqual(metrics.max_clearance_m, 1.5)


class ClearanceTests(unittest.TestCase):
    def test_axis_aligned_clearance_is_exact(self) -> None:
        cells = np.full((13, 13), GridState.FREE, dtype=np.int8)
        cells[6, :] = GridState.OCCUPIED
        grid = OccupancyGrid(cells, 1.0, -6.5, 0.0)
        metrics = analyze_free_space(grid, permissive_config())
        # Player stands at row 0, the wall is 6 cells ahead, and the free rows
        # behind the wall are unreachable.
        self.assertAlmostEqual(metrics.max_clearance_m, 6.0)

    def test_diagonal_clearance_is_within_chamfer_tolerance(self) -> None:
        cells = np.full((21, 21), GridState.FREE, dtype=np.int8)
        cells[10, 10] = GridState.OCCUPIED
        grid = OccupancyGrid(cells, 1.0, -10.5, 0.0)
        metrics = analyze_free_space(grid, permissive_config())
        expected = float(np.hypot(10.0, 10.0))
        self.assertLess(abs(metrics.max_clearance_m - expected) / expected, 0.08)

    def test_clearance_is_zero_without_reachable_space(self) -> None:
        cells = np.full((6, 6), GridState.UNKNOWN, dtype=np.int8)
        grid = OccupancyGrid(cells, 0.5, -1.5, 0.0)
        metrics = analyze_free_space(grid, permissive_config())
        self.assertEqual(metrics.max_clearance_m, 0.0)
        self.assertEqual(metrics.player_reachable_area_m2, 0.0)

    def test_open_grid_without_obstacles(self) -> None:
        cells = np.full((8, 8), GridState.FREE, dtype=np.int8)
        grid = OccupancyGrid(cells, 0.5, -2.0, 0.0)
        metrics = analyze_free_space(grid, permissive_config())
        self.assertAlmostEqual(metrics.max_clearance_m, float(np.hypot(4.0, 4.0)))

    def test_large_grid_stays_interactive(self) -> None:
        """The pairwise implementation needed ~49 s for this grid."""
        rng = np.random.default_rng(3)
        cells = np.full((400, 400), GridState.FREE, dtype=np.int8)
        blocked = rng.integers(0, 400, size=(300, 2))
        cells[blocked[:, 0], blocked[:, 1]] = GridState.OCCUPIED
        cells[0, 200] = GridState.FREE
        grid = OccupancyGrid(cells, 0.05, -10.0, 0.0)
        start = time.perf_counter()
        analyze_free_space(grid, permissive_config())
        self.assertLess(time.perf_counter() - start, 5.0)


if __name__ == "__main__":
    unittest.main()
