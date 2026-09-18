import unittest

import numpy as np

from free_space_analyzer.config import GroundConfig, OccupancyConfig
from free_space_analyzer.models import GridState, GroundPlane
from free_space_analyzer.occupancy import build_occupancy_grid


class OccupancyTests(unittest.TestCase):
    def test_floor_rays_and_obstacle_endpoint(self) -> None:
        plane = GroundPlane(np.array([0.0, 1.0, 0.0]), 1.7)
        points = np.array(
            [
                [0.0, -1.7, 3.2],
                [-1.2, -1.7, 3.2],
                [1.2, -1.7, 3.2],
                [0.0, -0.7, 2.2],
            ]
        )
        occupancy = OccupancyConfig(
            width_m=4.0,
            depth_m=4.0,
            resolution_m=0.5,
            min_obstacle_height_m=0.15,
            max_obstacle_height_m=2.2,
            safety_margin_m=0.0,
            raycast_free_space=True,
        )
        grid = build_occupancy_grid(points, plane, occupancy, GroundConfig())
        obstacle_cell = grid.world_to_cell(0.0, 2.2)
        self.assertIsNotNone(obstacle_cell)
        self.assertEqual(grid.cells[obstacle_cell], GridState.OCCUPIED)
        self.assertEqual(grid.cells[grid.world_to_cell(0.0, 1.0)], GridState.FREE)
        self.assertEqual(grid.cells[grid.world_to_cell(1.8, 0.2)], GridState.UNKNOWN)

    def test_occupied_wins_over_floor(self) -> None:
        plane = GroundPlane(np.array([0.0, 1.0, 0.0]), 1.7)
        points = np.array([[0.0, -1.7, 2.2], [0.0, -0.7, 2.2]])
        occupancy = OccupancyConfig(
            width_m=2.0,
            depth_m=3.0,
            resolution_m=0.5,
            safety_margin_m=0.0,
        )
        grid = build_occupancy_grid(points, plane, occupancy, GroundConfig())
        self.assertEqual(grid.cells[grid.world_to_cell(0.0, 2.2)], GridState.OCCUPIED)


if __name__ == "__main__":
    unittest.main()

