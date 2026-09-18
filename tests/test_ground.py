import unittest

import numpy as np

from free_space_analyzer.config import GroundConfig
from free_space_analyzer.ground import estimate_ground_ransac, known_height_plane
from free_space_analyzer.models import CameraInfo


class GroundTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = CameraInfo(320, 240, 160.0, 160.0, 159.5, 119.5, 1.7)

    def test_known_height_plane(self) -> None:
        plane = known_height_plane(self.camera)
        self.assertAlmostEqual(plane.d, 1.7)
        np.testing.assert_allclose(plane.normal, [0.0, 1.0, 0.0])

    def test_ransac_recovers_floor(self) -> None:
        rng = np.random.default_rng(42)
        floor = np.column_stack(
            (
                rng.uniform(-4, 4, 1500),
                rng.normal(-1.7, 0.008, 1500),
                rng.uniform(0.5, 9, 1500),
            )
        )
        clutter = np.column_stack(
            (
                rng.uniform(-2, 2, 300),
                rng.uniform(-1.3, 0.5, 300),
                rng.uniform(1, 6, 300),
            )
        )
        config = GroundConfig(
            mode="ransac",
            distance_tolerance_m=0.04,
            max_tilt_deg=15.0,
            ransac_iterations=150,
            min_inlier_ratio=0.5,
        )
        plane = estimate_ground_ransac(np.vstack((floor, clutter)), self.camera, config)
        self.assertGreater(plane.inlier_ratio, 0.7)
        self.assertGreater(float(np.dot(plane.normal, [0, 1, 0])), 0.99)
        self.assertAlmostEqual(plane.d, 1.7, delta=0.03)


if __name__ == "__main__":
    unittest.main()

