from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating[Any]]
IntArray = npt.NDArray[np.integer[Any]]


class DepthType(str, Enum):
    """Meaning of each valid depth pixel."""

    Z_DEPTH = "z_depth"
    RADIAL = "radial"


class GridState(IntEnum):
    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 1


@dataclass(frozen=True)
class CameraInfo:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    camera_height_m: float
    depth_type: DepthType = DepthType.Z_DEPTH
    up_vector: tuple[float, float, float] = (0.0, 1.0, 0.0)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera resolution must be positive")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Camera focal lengths must be positive")
        if self.camera_height_m <= 0:
            raise ValueError("camera_height_m must be positive")
        up = np.asarray(self.up_vector, dtype=np.float64)
        if up.shape != (3,) or not np.all(np.isfinite(up)) or np.linalg.norm(up) < 1e-9:
            raise ValueError("up_vector must be a finite non-zero 3-vector")

    @classmethod
    def from_horizontal_fov(
        cls,
        width: int,
        height: int,
        horizontal_fov_deg: float,
        camera_height_m: float,
        depth_type: DepthType = DepthType.Z_DEPTH,
        up_vector: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> CameraInfo:
        if not 1.0 < horizontal_fov_deg < 179.0:
            raise ValueError("horizontal_fov_deg must be between 1 and 179 degrees")
        fx = width / (2.0 * np.tan(np.deg2rad(horizontal_fov_deg) / 2.0))
        fy = fx
        return cls(
            width=width,
            height=height,
            fx=float(fx),
            fy=float(fy),
            cx=(width - 1) / 2.0,
            cy=(height - 1) / 2.0,
            camera_height_m=camera_height_m,
            depth_type=depth_type,
            up_vector=up_vector,
        )

    def normalized_up(self) -> npt.NDArray[np.float64]:
        up = np.asarray(self.up_vector, dtype=np.float64)
        return up / np.linalg.norm(up)


@dataclass(frozen=True)
class DepthFrame:
    data: npt.NDArray[np.float32]
    camera: CameraInfo
    timestamp_s: float | None = None
    frame_id: str | None = None

    def __post_init__(self) -> None:
        array = np.asarray(self.data, dtype=np.float32)
        expected = (self.camera.height, self.camera.width)
        if array.shape != expected:
            raise ValueError(f"Depth shape {array.shape} does not match camera resolution {expected}")
        object.__setattr__(self, "data", array)


@dataclass(frozen=True)
class GroundPlane:
    """Plane `normal dot point + d = 0`; normal points away from the floor."""

    normal: npt.NDArray[np.float64]
    d: float
    inlier_ratio: float = 1.0
    method: str = "known_height"

    def __post_init__(self) -> None:
        normal = np.asarray(self.normal, dtype=np.float64)
        norm = np.linalg.norm(normal)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)) or norm < 1e-9:
            raise ValueError("Ground plane normal must be a finite non-zero 3-vector")
        object.__setattr__(self, "normal", normal / norm)
        object.__setattr__(self, "d", float(self.d) / norm)

    def signed_height(self, points: FloatArray) -> npt.NDArray[np.float64]:
        return np.asarray(points, dtype=np.float64) @ self.normal + self.d


@dataclass(frozen=True)
class Rectangle:
    width_m: float
    depth_m: float
    area_m2: float
    x_min_m: float
    z_min_m: float
    x_max_m: float
    z_max_m: float


@dataclass
class OccupancyGrid:
    cells: npt.NDArray[np.int8]
    resolution_m: float
    x_min_m: float
    z_min_m: float
    observed_points: int = 0

    def __post_init__(self) -> None:
        if self.cells.ndim != 2:
            raise ValueError("Occupancy grid must be a 2D array")
        if self.resolution_m <= 0:
            raise ValueError("resolution_m must be positive")
        allowed = np.isin(self.cells, [state.value for state in GridState])
        if not bool(np.all(allowed)):
            raise ValueError("Occupancy grid contains an invalid state")
        if self.cells.dtype != np.int8:
            self.cells = self.cells.astype(np.int8, copy=False)

    @property
    def height(self) -> int:
        return int(self.cells.shape[0])

    @property
    def width(self) -> int:
        return int(self.cells.shape[1])

    @property
    def width_m(self) -> float:
        return self.width * self.resolution_m

    @property
    def depth_m(self) -> float:
        return self.height * self.resolution_m

    def world_to_cell(self, x_m: float, z_m: float) -> tuple[int, int] | None:
        col = int(np.floor((x_m - self.x_min_m) / self.resolution_m))
        row = int(np.floor((z_m - self.z_min_m) / self.resolution_m))
        if 0 <= row < self.height and 0 <= col < self.width:
            return row, col
        return None

    def cell_center(self, row: int, col: int) -> tuple[float, float]:
        x = self.x_min_m + (col + 0.5) * self.resolution_m
        z = self.z_min_m + (row + 0.5) * self.resolution_m
        return x, z


@dataclass(frozen=True)
class SpaceAnalysisResult:
    is_open_space: bool
    score: float
    largest_free_area_m2: float
    player_reachable_area_m2: float
    largest_free_rectangle: Rectangle | None
    nearest_obstacle_m: float | None
    max_clearance_m: float
    obstacle_ratio: float
    unknown_ratio: float
    player_clearance: bool
    ground_inlier_ratio: float
    ground_method: str
    observed_points: int
    failure_reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["failure_reasons"] = list(self.failure_reasons)
        return data
