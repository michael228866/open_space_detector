import unittest
from pathlib import Path

from free_space_analyzer.config import (
    AnalyzerConfig,
    DepthConfig,
    GroundConfig,
    OccupancyConfig,
    OpenSpaceConfig,
    ScoringConfig,
    config_from_mapping,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class ShippedConfigTests(unittest.TestCase):
    def test_default_yaml_loads_and_validates(self) -> None:
        config = load_config(REPO_ROOT / "configs" / "default.yaml")
        self.assertEqual(config.ground.mode, "known_height")
        self.assertTrue(
            config.open_space.nearby_unknown_is_unsafe,
            "UNKNOWN must stay unsafe by default",
        )

    def test_unknown_keys_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            config_from_mapping({"typo": {}})
        with self.assertRaises(ValueError):
            config_from_mapping({"occupancy": {"resolution": 0.1}})


class ValidationTests(unittest.TestCase):
    def assert_invalid(self, **sections: object) -> None:
        with self.assertRaises(ValueError):
            AnalyzerConfig(**sections).validate()  # type: ignore[arg-type]

    def test_valid_defaults(self) -> None:
        AnalyzerConfig().validate()

    def test_negative_safety_margin_is_rejected(self) -> None:
        self.assert_invalid(occupancy=OccupancyConfig(safety_margin_m=-0.1))

    def test_negative_player_clearance_is_rejected(self) -> None:
        self.assert_invalid(open_space=OpenSpaceConfig(min_player_clearance_m=-1.0))

    def test_negative_area_and_rectangle_thresholds_are_rejected(self) -> None:
        self.assert_invalid(open_space=OpenSpaceConfig(min_free_area_m2=-1.0))
        self.assert_invalid(open_space=OpenSpaceConfig(min_rectangle_width_m=-1.0))
        self.assert_invalid(open_space=OpenSpaceConfig(min_rectangle_depth_m=-1.0))

    def test_ratios_must_be_fractions(self) -> None:
        self.assert_invalid(open_space=OpenSpaceConfig(max_obstacle_ratio=1.5))
        self.assert_invalid(open_space=OpenSpaceConfig(max_unknown_ratio=-0.1))

    def test_depth_and_ground_bounds(self) -> None:
        self.assert_invalid(depth=DepthConfig(min_depth_m=5.0, max_depth_m=1.0))
        self.assert_invalid(depth=DepthConfig(sample_stride=0))
        self.assert_invalid(ground=GroundConfig(mode="magic"))
        self.assert_invalid(ground=GroundConfig(max_tilt_deg=90.0))

    def test_occupancy_bounds(self) -> None:
        self.assert_invalid(occupancy=OccupancyConfig(resolution_m=0.0))
        self.assert_invalid(
            occupancy=OccupancyConfig(min_obstacle_height_m=2.0, max_obstacle_height_m=1.0)
        )

    def test_scoring_weights_cannot_all_be_zero(self) -> None:
        self.assert_invalid(
            scoring=ScoringConfig(
                free_area_weight=0.0,
                rectangle_weight=0.0,
                clearance_weight=0.0,
                obstacle_weight=0.0,
                known_space_weight=0.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
