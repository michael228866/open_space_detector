import unittest

from free_space_analyzer.config import OpenSpaceConfig, ScoringConfig
from free_space_analyzer.free_space import FreeSpaceMetrics
from free_space_analyzer.models import Rectangle
from free_space_analyzer.scoring import evaluate_requirements, score_space


def rectangle(width_m: float, depth_m: float) -> Rectangle:
    return Rectangle(
        width_m=width_m,
        depth_m=depth_m,
        area_m2=width_m * depth_m,
        x_min_m=-width_m / 2,
        z_min_m=0.5,
        x_max_m=width_m / 2,
        z_max_m=0.5 + depth_m,
    )


def metrics(**overrides: object) -> FreeSpaceMetrics:
    values = dict(
        largest_free_area_m2=20.0,
        player_reachable_area_m2=20.0,
        largest_rectangle=rectangle(4.5, 4.5),
        nearest_obstacle_m=2.0,
        nearest_unsafe_m=2.0,
        max_clearance_m=2.0,
        obstacle_ratio=0.05,
        unknown_ratio=0.1,
        player_clearance=True,
    )
    values.update(overrides)
    return FreeSpaceMetrics(**values)  # type: ignore[arg-type]


CLEARANCE_ONLY = ScoringConfig(
    free_area_weight=0.0,
    rectangle_weight=0.0,
    clearance_weight=1.0,
    obstacle_weight=0.0,
    known_space_weight=0.0,
)


class ScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.thresholds = OpenSpaceConfig()

    def test_unknown_next_to_player_is_not_a_perfect_clearance_score(self) -> None:
        # No obstacle anywhere, but UNKNOWN 5 cm away counts as unsafe.
        unsafe = metrics(nearest_obstacle_m=None, nearest_unsafe_m=0.05, player_clearance=False)
        self.assertAlmostEqual(score_space(unsafe, self.thresholds, CLEARANCE_ONLY), 5.0)

    def test_clearance_score_is_full_only_when_nothing_is_unsafe(self) -> None:
        clear = metrics(nearest_obstacle_m=None, nearest_unsafe_m=None)
        self.assertAlmostEqual(score_space(clear, self.thresholds, CLEARANCE_ONLY), 100.0)

    def test_score_uses_reachable_area_not_global_free_area(self) -> None:
        weights = ScoringConfig(
            free_area_weight=1.0,
            rectangle_weight=0.0,
            clearance_weight=0.0,
            obstacle_weight=0.0,
            known_space_weight=0.0,
        )
        walled_in = metrics(largest_free_area_m2=80.0, player_reachable_area_m2=4.0)
        self.assertAlmostEqual(score_space(walled_in, self.thresholds, weights), 25.0)

    def test_score_stays_within_bounds(self) -> None:
        worst = metrics(
            largest_free_area_m2=0.0,
            player_reachable_area_m2=0.0,
            largest_rectangle=None,
            nearest_obstacle_m=0.0,
            nearest_unsafe_m=0.0,
            max_clearance_m=0.0,
            obstacle_ratio=1.0,
            unknown_ratio=1.0,
            player_clearance=False,
        )
        self.assertGreaterEqual(score_space(worst, self.thresholds, ScoringConfig()), 0.0)
        self.assertLessEqual(score_space(metrics(), self.thresholds, ScoringConfig()), 100.0)


class RequirementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.thresholds = OpenSpaceConfig()

    def test_all_requirements_met(self) -> None:
        is_open, reasons = evaluate_requirements(metrics(), self.thresholds)
        self.assertTrue(is_open, reasons)
        self.assertEqual(reasons, ())

    def test_unreachable_area_fails_even_when_the_room_is_large(self) -> None:
        walled_in = metrics(largest_free_area_m2=80.0, player_reachable_area_m2=4.0)
        is_open, reasons = evaluate_requirements(walled_in, self.thresholds)
        self.assertFalse(is_open)
        self.assertIn("largest_free_area_below_minimum", reasons)

    def test_rotated_rectangle_is_accepted(self) -> None:
        thresholds = OpenSpaceConfig(min_rectangle_width_m=5.0, min_rectangle_depth_m=3.0)
        rotated = metrics(largest_rectangle=rectangle(3.2, 5.4))
        is_open, reasons = evaluate_requirements(rotated, thresholds)
        self.assertTrue(is_open, reasons)

    def test_one_cell_of_quantization_is_tolerated(self) -> None:
        nearly = metrics(largest_rectangle=rectangle(3.9, 3.9))
        _, without = evaluate_requirements(nearly, self.thresholds, measurement_tolerance_m=0.0)
        self.assertIn("required_free_rectangle_not_found", without)
        _, with_tolerance = evaluate_requirements(nearly, self.thresholds, 0.1)
        self.assertNotIn("required_free_rectangle_not_found", with_tolerance)

    def test_missing_rectangle_fails(self) -> None:
        _, reasons = evaluate_requirements(metrics(largest_rectangle=None), self.thresholds)
        self.assertIn("required_free_rectangle_not_found", reasons)

    def test_each_hard_condition_reports_its_own_reason(self) -> None:
        _, reasons = evaluate_requirements(metrics(player_clearance=False), self.thresholds)
        self.assertIn("player_clearance_below_minimum", reasons)
        _, reasons = evaluate_requirements(metrics(obstacle_ratio=0.9), self.thresholds)
        self.assertIn("obstacle_ratio_above_maximum", reasons)
        _, reasons = evaluate_requirements(metrics(unknown_ratio=0.9), self.thresholds)
        self.assertIn("unknown_ratio_above_maximum", reasons)


if __name__ == "__main__":
    unittest.main()
