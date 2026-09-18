from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class DepthConfig:
    min_depth_m: float = 0.15
    max_depth_m: float = 15.0
    sample_stride: int = 2


@dataclass(frozen=True)
class GroundConfig:
    mode: str = "known_height"
    distance_tolerance_m: float = 0.12
    max_tilt_deg: float = 25.0
    ransac_iterations: int = 300
    min_inlier_ratio: float = 0.08


@dataclass(frozen=True)
class OccupancyConfig:
    width_m: float = 10.0
    depth_m: float = 10.0
    resolution_m: float = 0.10
    min_obstacle_height_m: float = 0.15
    max_obstacle_height_m: float = 2.20
    safety_margin_m: float = 0.25
    raycast_free_space: bool = True


@dataclass(frozen=True)
class OpenSpaceConfig:
    min_free_area_m2: float = 16.0
    min_rectangle_width_m: float = 4.0
    min_rectangle_depth_m: float = 4.0
    min_player_clearance_m: float = 1.0
    max_obstacle_ratio: float = 0.15
    max_unknown_ratio: float = 0.35
    nearby_unknown_is_unsafe: bool = True


@dataclass(frozen=True)
class ScoringConfig:
    free_area_weight: float = 0.30
    rectangle_weight: float = 0.25
    clearance_weight: float = 0.20
    obstacle_weight: float = 0.10
    known_space_weight: float = 0.15


@dataclass(frozen=True)
class AnalyzerConfig:
    depth: DepthConfig = DepthConfig()
    ground: GroundConfig = GroundConfig()
    occupancy: OccupancyConfig = OccupancyConfig()
    open_space: OpenSpaceConfig = OpenSpaceConfig()
    scoring: ScoringConfig = ScoringConfig()

    def validate(self) -> None:
        if not 0 < self.depth.min_depth_m < self.depth.max_depth_m:
            raise ValueError("depth min/max values are invalid")
        if self.depth.sample_stride < 1:
            raise ValueError("sample_stride must be >= 1")
        if self.ground.mode not in {"known_height", "ransac"}:
            raise ValueError("ground.mode must be known_height or ransac")
        if self.ground.distance_tolerance_m <= 0:
            raise ValueError("ground distance tolerance must be positive")
        if not 0 < self.ground.max_tilt_deg < 90:
            raise ValueError("ground max tilt must be between 0 and 90 degrees")
        if self.ground.ransac_iterations < 1:
            raise ValueError("ransac_iterations must be positive")
        if not 0 <= self.ground.min_inlier_ratio <= 1:
            raise ValueError("min_inlier_ratio must be between 0 and 1")
        if min(
            self.occupancy.width_m,
            self.occupancy.depth_m,
            self.occupancy.resolution_m,
        ) <= 0:
            raise ValueError("occupancy dimensions and resolution must be positive")
        if self.occupancy.min_obstacle_height_m < 0:
            raise ValueError("min obstacle height cannot be negative")
        if self.occupancy.max_obstacle_height_m <= self.occupancy.min_obstacle_height_m:
            raise ValueError("max obstacle height must exceed min obstacle height")
        if self.occupancy.safety_margin_m < 0:
            raise ValueError("safety_margin_m cannot be negative")
        for name in (
            "min_free_area_m2",
            "min_rectangle_width_m",
            "min_rectangle_depth_m",
            "min_player_clearance_m",
        ):
            if getattr(self.open_space, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        for name in ("max_obstacle_ratio", "max_unknown_ratio"):
            value = getattr(self.open_space, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        weights = list(vars(self.scoring).values())
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("scoring weights must be non-negative and not all zero")


def _construct_dataclass(cls: type[T], values: dict[str, Any] | None) -> T:
    values = values or {}
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def config_from_mapping(data: dict[str, Any]) -> AnalyzerConfig:
    allowed = {"depth", "ground", "occupancy", "open_space", "scoring"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    config = AnalyzerConfig(
        depth=_construct_dataclass(DepthConfig, data.get("depth")),
        ground=_construct_dataclass(GroundConfig, data.get("ground")),
        occupancy=_construct_dataclass(OccupancyConfig, data.get("occupancy")),
        open_space=_construct_dataclass(OpenSpaceConfig, data.get("open_space")),
        scoring=_construct_dataclass(ScoringConfig, data.get("scoring")),
    )
    config.validate()
    return config


def load_config(path: str | Path) -> AnalyzerConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping")
    return config_from_mapping(data)
