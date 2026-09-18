"""Public API for free-space-analyzer."""

from .analyzer import FreeSpaceAnalyzer
from .config import AnalyzerConfig, load_config
from .ground import GroundEstimationError
from .models import (
    CameraInfo,
    DepthFrame,
    DepthType,
    GridState,
    GroundPlane,
    OccupancyGrid,
    Rectangle,
    SpaceAnalysisResult,
)

__all__ = [
    "AnalyzerConfig",
    "CameraInfo",
    "DepthFrame",
    "DepthType",
    "FreeSpaceAnalyzer",
    "GroundEstimationError",
    "GroundPlane",
    "GridState",
    "OccupancyGrid",
    "Rectangle",
    "SpaceAnalysisResult",
    "load_config",
]
