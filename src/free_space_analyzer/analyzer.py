from __future__ import annotations

from pathlib import Path

import numpy.typing as npt

from .config import AnalyzerConfig, load_config
from .depth import preprocess_depth
from .free_space import analyze_free_space
from .geometry import depth_to_point_cloud
from .ground import estimate_ground
from .models import CameraInfo, DepthFrame, OccupancyGrid, SpaceAnalysisResult
from .occupancy import build_occupancy_grid
from .scoring import evaluate_requirements, score_space


class FreeSpaceAnalyzer:
    def __init__(self, config: AnalyzerConfig | None = None) -> None:
        self.config = config or AnalyzerConfig()
        self.config.validate()
        self.last_occupancy_grid: OccupancyGrid | None = None

    @classmethod
    def from_config(cls, path: str | Path) -> FreeSpaceAnalyzer:
        return cls(load_config(path))

    def analyze(
        self,
        depth: npt.ArrayLike,
        camera: CameraInfo,
    ) -> SpaceAnalysisResult:
        processed = preprocess_depth(depth, camera, self.config.depth)
        points = depth_to_point_cloud(
            processed,
            camera,
            stride=self.config.depth.sample_stride,
        )
        plane = estimate_ground(points, camera, self.config.ground)
        grid = build_occupancy_grid(
            points,
            plane,
            self.config.occupancy,
            self.config.ground,
        )
        self.last_occupancy_grid = grid
        metrics = analyze_free_space(grid, self.config.open_space)
        is_open, reasons = evaluate_requirements(
            metrics,
            self.config.open_space,
            measurement_tolerance_m=grid.resolution_m,
        )
        score = score_space(metrics, self.config.open_space, self.config.scoring)
        return SpaceAnalysisResult(
            is_open_space=is_open,
            score=score,
            largest_free_area_m2=round(metrics.largest_free_area_m2, 3),
            player_reachable_area_m2=round(metrics.player_reachable_area_m2, 3),
            largest_free_rectangle=metrics.largest_rectangle,
            nearest_obstacle_m=(
                round(metrics.nearest_obstacle_m, 3)
                if metrics.nearest_obstacle_m is not None
                else None
            ),
            max_clearance_m=round(metrics.max_clearance_m, 3),
            obstacle_ratio=round(metrics.obstacle_ratio, 5),
            unknown_ratio=round(metrics.unknown_ratio, 5),
            player_clearance=metrics.player_clearance,
            ground_inlier_ratio=round(plane.inlier_ratio, 5),
            ground_method=plane.method,
            observed_points=grid.observed_points,
            failure_reasons=reasons,
        )

    def analyze_frame(self, frame: DepthFrame) -> SpaceAnalysisResult:
        return self.analyze(frame.data, frame.camera)
