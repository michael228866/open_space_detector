import unittest

import numpy as np

from free_space_analyzer.config import DepthConfig
from free_space_analyzer.depth import preprocess_depth, radial_to_z_depth
from free_space_analyzer.models import CameraInfo, DepthType


class DepthTests(unittest.TestCase):
    def test_invalid_values_become_nan(self) -> None:
        camera = CameraInfo(3, 1, 1.0, 1.0, 1.0, 0.0, 1.7)
        depth = np.array([[0.0, 2.0, 99.0]], dtype=np.float32)
        result = preprocess_depth(depth, camera, DepthConfig(0.1, 10.0, 1))
        self.assertTrue(np.isnan(result[0, 0]))
        self.assertEqual(float(result[0, 1]), 2.0)
        self.assertTrue(np.isnan(result[0, 2]))

    def test_radial_depth_is_converted_to_z(self) -> None:
        camera = CameraInfo(
            3, 1, 1.0, 1.0, 1.0, 0.0, 1.7, depth_type=DepthType.RADIAL
        )
        radial = np.full((1, 3), np.sqrt(2.0), dtype=np.float32)
        result = radial_to_z_depth(radial, camera)
        self.assertAlmostEqual(float(result[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(result[0, 1]), np.sqrt(2.0), places=5)

    def test_shape_mismatch_fails(self) -> None:
        camera = CameraInfo(3, 2, 1.0, 1.0, 1.0, 0.5, 1.7)
        with self.assertRaises(ValueError):
            preprocess_depth(np.ones((1, 3)), camera, DepthConfig())


if __name__ == "__main__":
    unittest.main()

