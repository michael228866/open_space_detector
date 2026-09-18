"""Public API for free-space-analyzer."""

from .analyzer import FreeSpaceAnalyzer
from .config import AnalyzerConfig, load_config
from .models import (
    CameraInfo,
    DepthFrame,
    DepthType,
    GroundPlane,
    GridState,
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
    "GroundPlane",
    "GridState",
    "OccupancyGrid",
    "Rectangle",
    "SpaceAnalysisResult",
    "load_config",
]
