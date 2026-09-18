import unittest

import numpy as np

from free_space_analyzer.config import GroundConfig
from free_space_analyzer.ground import estimate_ground_ransac, known_height_plane
from free_space_analyzer.models import CameraInfo


def floor_points(height_m: float = 1.7, count: int = 1200, noise_m: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(7)
    return np.column_stack(
        (
            rng.uniform(-3.0, 3.0, count),
            rng.normal(-height_m, noise_m, count) if noise_m else np.full(count, -height_m),
            rng.uniform(0.5, 8.0, count),
        )
    )


class GroundTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = CameraInfo(320, 240, 160.0, 160.0, 159.5, 119.5, 1.7)
        self.config = GroundConfig(mode="known_height", distance_tolerance_m=0.12)

    def test_known_height_plane(self) -> None:
        plane = known_height_plane(floor_points(), self.camera, self.config)
        self.assertAlmostEqual(plane.d, 1.7)
        np.testing.assert_allclose(plane.normal, [0.0, 1.0, 0.0])
        self.assertEqual(plane.method, "known_height")

    def test_known_height_support_is_measured_not_assumed(self) -> None:
        plane = known_height_plane(floor_points(noise_m=0.02), self.camera, self.config)
        self.assertGreater(plane.inlier_ratio, 0.95)

    def test_known_height_support_collapses_for_wrong_camera_height(self) -> None:
        wrong = CameraInfo(320, 240, 160.0, 160.0, 159.5, 119.5, camera_height_m=1.0)
        plane = known_height_plane(floor_points(), wrong, self.config)
        # The floor sits 0.7 m away from the assumed plane, so nothing supports it.
        self.assertLess(plane.inlier_ratio, 0.05)

    def test_known_height_support_without_ground_candidates(self) -> None:
        above = np.array([[0.0, 0.5, 2.0], [0.2, 0.4, 2.5]])
        plane = known_height_plane(above, self.camera, self.config)
        self.assertEqual(plane.inlier_ratio, 0.0)

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
