import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from free_space_analyzer.cli import camera_from_mapping, main
from free_space_analyzer.models import CameraInfo, DepthType

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "default.yaml"


def floor_depth(width: int = 160, height: int = 120, camera_height_m: float = 1.7) -> np.ndarray:
    camera = CameraInfo.from_horizontal_fov(width, height, 90.0, camera_height_m)
    rows, _ = np.indices((height, width), dtype=np.float64)
    y_up = -(rows - camera.cy) / camera.fy
    ground = np.where(y_up < -1e-8, -camera_height_m / y_up, np.inf)
    return np.minimum(ground, 6.0).astype(np.float32)


class CameraMappingTests(unittest.TestCase):
    def test_horizontal_fov_mapping(self) -> None:
        camera = camera_from_mapping(
            {"width": 4, "height": 2, "horizontal_fov_deg": 90.0, "camera_height_m": 1.7},
            (2, 4),
        )
        self.assertEqual(camera.depth_type, DepthType.Z_DEPTH)
        self.assertAlmostEqual(camera.fx, 2.0)

    def test_resolution_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            camera_from_mapping({"width": 8, "height": 8, "camera_height_m": 1.7}, (2, 4))

    def test_missing_intrinsics_are_reported(self) -> None:
        with self.assertRaises(ValueError) as error:
            camera_from_mapping({"width": 4, "height": 2, "camera_height_m": 1.7}, (2, 4))
        self.assertIn("fx", str(error.exception))


class CliSmokeTests(unittest.TestCase):
    def write_scene(self, directory: Path, **camera_overrides: object) -> tuple[Path, Path]:
        depth = floor_depth()
        depth_path = directory / "depth.npy"
        np.save(depth_path, depth)
        camera = {
            "width": int(depth.shape[1]),
            "height": int(depth.shape[0]),
            "horizontal_fov_deg": 90.0,
            "camera_height_m": 1.7,
            "depth_type": "z_depth",
        }
        camera.update(camera_overrides)
        camera_path = directory / "camera.json"
        camera_path.write_text(json.dumps(camera), encoding="utf-8")
        return depth_path, camera_path

    def test_cli_writes_json_and_debug_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            depth_path, camera_path = self.write_scene(directory)
            result_path = directory / "out" / "result.json"
            png_path = directory / "out" / "occupancy.png"
            code = main(
                [
                    str(depth_path),
                    "--camera",
                    str(camera_path),
                    "--config",
                    str(CONFIG),
                    "--output-json",
                    str(result_path),
                    "--debug-png",
                    str(png_path),
                ]
            )
            self.assertEqual(code, 0)
            self.assertGreater(png_path.stat().st_size, 0)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            for key in ("is_open_space", "score", "player_reachable_area_m2", "failure_reasons"):
                self.assertIn(key, result)

    def test_cli_reports_bad_camera_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            depth_path, camera_path = self.write_scene(directory)
            camera_path.write_text(json.dumps({"width": 160, "height": 120}), encoding="utf-8")
            code = main(
                [str(depth_path), "--camera", str(camera_path), "--config", str(CONFIG)]
            )
            self.assertEqual(code, 2)

    def test_cli_reports_missing_depth_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _, camera_path = self.write_scene(directory)
            code = main(
                [
                    str(directory / "missing.npy"),
                    "--camera",
                    str(camera_path),
                    "--config",
                    str(CONFIG),
                ]
            )
            self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
