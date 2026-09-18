import unittest

import numpy as np

from free_space_analyzer.config import OpenSpaceConfig
from free_space_analyzer.free_space import analyze_free_space, largest_free_rectangle
from free_space_analyzer.models import GridState, OccupancyGrid


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
        metrics = analyze_free_space(
            grid,
            OpenSpaceConfig(
                min_free_area_m2=1,
                min_rectangle_width_m=1,
                min_rectangle_depth_m=1,
                min_player_clearance_m=0.5,
                max_obstacle_ratio=1,
                max_unknown_ratio=1,
                nearby_unknown_is_unsafe=False,
            ),
        )
        self.assertEqual(metrics.largest_free_area_m2, 5.0)
        self.assertAlmostEqual(metrics.unknown_ratio, 0.5)


if __name__ == "__main__":
    unittest.main()

