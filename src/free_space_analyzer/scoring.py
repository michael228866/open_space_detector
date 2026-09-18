from __future__ import annotations

from .config import OpenSpaceConfig, ScoringConfig
from .free_space import FreeSpaceMetrics


def _threshold_score(value: float, target: float) -> float:
    if target <= 0:
        return 100.0
    return max(0.0, min(100.0, value / target * 100.0))


def _inverse_threshold_score(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 100.0 if value <= 0 else 0.0
    if value <= maximum:
        return 100.0
    return max(0.0, 100.0 * (2.0 - value / maximum))


def score_space(
    metrics: FreeSpaceMetrics,
    thresholds: OpenSpaceConfig,
    weights: ScoringConfig,
) -> float:
    rectangle = metrics.largest_rectangle
    rectangle_score = 0.0
    if rectangle is not None:
        direct = min(
            _threshold_score(rectangle.width_m, thresholds.min_rectangle_width_m),
            _threshold_score(rectangle.depth_m, thresholds.min_rectangle_depth_m),
        )
        rotated = min(
            _threshold_score(rectangle.width_m, thresholds.min_rectangle_depth_m),
            _threshold_score(rectangle.depth_m, thresholds.min_rectangle_width_m),
        )
        rectangle_score = max(direct, rotated)
    clearance_value = metrics.nearest_obstacle_m
    if clearance_value is None:
        clearance_score = 100.0
    else:
        clearance_score = _threshold_score(
            clearance_value, thresholds.min_player_clearance_m
        )
    values = {
        "free_area_weight": _threshold_score(
            metrics.largest_free_area_m2, thresholds.min_free_area_m2
        ),
        "rectangle_weight": rectangle_score,
        "clearance_weight": clearance_score,
        "obstacle_weight": _inverse_threshold_score(
            metrics.obstacle_ratio, thresholds.max_obstacle_ratio
        ),
        "known_space_weight": _inverse_threshold_score(
            metrics.unknown_ratio, thresholds.max_unknown_ratio
        ),
    }
    total_weight = sum(vars(weights).values())
    weighted = sum(getattr(weights, name) * value for name, value in values.items())
    return round(weighted / total_weight, 2)


def evaluate_requirements(
    metrics: FreeSpaceMetrics,
    thresholds: OpenSpaceConfig,
    measurement_tolerance_m: float = 0.0,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if metrics.largest_free_area_m2 < thresholds.min_free_area_m2:
        reasons.append("largest_free_area_below_minimum")
    rectangle = metrics.largest_rectangle
    rectangle_pass = False
    if rectangle is not None:
        # A rasterized grid can conservatively lose one boundary cell even when
        # the physical span exactly matches the requested dimension.
        width = rectangle.width_m + measurement_tolerance_m
        depth = rectangle.depth_m + measurement_tolerance_m
        rectangle_pass = (
            width >= thresholds.min_rectangle_width_m
            and depth >= thresholds.min_rectangle_depth_m
        ) or (
            width >= thresholds.min_rectangle_depth_m
            and depth >= thresholds.min_rectangle_width_m
        )
    if not rectangle_pass:
        reasons.append("required_free_rectangle_not_found")
    if not metrics.player_clearance:
        reasons.append("player_clearance_below_minimum")
    if metrics.obstacle_ratio > thresholds.max_obstacle_ratio:
        reasons.append("obstacle_ratio_above_maximum")
    if metrics.unknown_ratio > thresholds.max_unknown_ratio:
        reasons.append("unknown_ratio_above_maximum")
    return not reasons, tuple(reasons)
