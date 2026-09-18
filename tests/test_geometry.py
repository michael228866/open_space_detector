import unittest

import numpy as np

from free_space_analyzer.geometry import depth_to_point_cloud, points_to_ground_coordinates
from free_space_analyzer.models import CameraInfo, GroundPlane


class GeometryTests(unittest.TestCase):
    def test_depth_projection_uses_y_up(self) -> None:
        camera = CameraInfo(3, 3, 1.0, 1.0, 1.0, 1.0, 1.7)
        depth = np.full((3, 3), np.nan)
        depth[0, 1] = 2.0
        depth[1, 1] = 2.0
        depth[1, 2] = 2.0
        points = depth_to_point_cloud(depth, camera)
        expected = np.array([[0.0, 2.0, 2.0], [0.0, 0.0, 2.0], [2.0, 0.0, 2.0]])
        np.testing.assert_allclose(points, expected)

    def test_ground_coordinates(self) -> None:
        plane = GroundPlane(np.array([0.0, 1.0, 0.0]), 1.7)
        points = np.array([[2.0, -1.7, 3.0], [0.0, -0.7, 4.0]])
        x, z, height = points_to_ground_coordinates(points, plane)
        np.testing.assert_allclose(x, [2.0, 0.0])
        np.testing.assert_allclose(z, [3.0, 4.0])
        np.testing.assert_allclose(height, [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()

