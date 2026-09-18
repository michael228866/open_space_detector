from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .config import OpenSpaceConfig
from .models import GridState, OccupancyGrid, Rectangle


@dataclass(frozen=True)
class FreeSpaceMetrics:
    largest_free_area_m2: float
    player_reachable_area_m2: float
    largest_rectangle: Rectangle | None
    nearest_obstacle_m: float | None
    max_clearance_m: float
    obstacle_ratio: float
    unknown_ratio: float
    player_clearance: bool


def _component_sizes(free: npt.NDArray[np.bool_]) -> tuple[list[int], npt.NDArray[np.int32]]:
    height, width = free.shape
    labels = np.full((height, width), -1, dtype=np.int32)
    sizes: list[int] = []
    label = 0
    for row in range(height):
        for col in range(width):
            if not free[row, col] or labels[row, col] >= 0:
                continue
            queue: deque[tuple[int, int]] = deque([(row, col)])
            labels[row, col] = label
            size = 0
            while queue:
                current_row, current_col = queue.popleft()
                size += 1
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = current_row + dr, current_col + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and free[nr, nc]
                        and labels[nr, nc] < 0
                    ):
                        labels[nr, nc] = label
                        queue.append((nr, nc))
            sizes.append(size)
            label += 1
    return sizes, labels


def largest_free_rectangle(grid: OccupancyGrid) -> Rectangle | None:
    """Largest axis-aligned all-FREE rectangle using a histogram stack."""
    free = grid.cells == GridState.FREE
    heights = np.zeros(grid.width, dtype=np.int64)
    best_area_cells = 0
    best_bounds: tuple[int, int, int, int] | None = None
    for row in range(grid.height):
        heights = np.where(free[row], heights + 1, 0)
        stack: list[int] = []
        for col in range(grid.width + 1):
            current = int(heights[col]) if col < grid.width else 0
            while stack and int(heights[stack[-1]]) > current:
                index = stack.pop()
                rectangle_height = int(heights[index])
                left = stack[-1] + 1 if stack else 0
                rectangle_width = col - left
                area = rectangle_height * rectangle_width
                if area > best_area_cells:
                    best_area_cells = area
                    top = row - rectangle_height + 1
                    best_bounds = (top, left, row + 1, col)
            stack.append(col)
    if best_bounds is None:
        return None
    top, left, bottom, right = best_bounds
    width_m = (right - left) * grid.resolution_m
    depth_m = (bottom - top) * grid.resolution_m
    return Rectangle(
        width_m=width_m,
        depth_m=depth_m,
        area_m2=width_m * depth_m,
        x_min_m=grid.x_min_m + left * grid.resolution_m,
        z_min_m=grid.z_min_m + top * grid.resolution_m,
        x_max_m=grid.x_min_m + right * grid.resolution_m,
        z_max_m=grid.z_min_m + bottom * grid.resolution_m,
    )


def _distance_to_mask(
    grid: OccupancyGrid,
    mask: npt.NDArray[np.bool_],
    x_m: float,
    z_m: float,
) -> float | None:
    locations = np.argwhere(mask)
    if len(locations) == 0:
        return None
    x = grid.x_min_m + (locations[:, 1] + 0.5) * grid.resolution_m
    z = grid.z_min_m + (locations[:, 0] + 0.5) * grid.resolution_m
    return float(np.sqrt((x - x_m) ** 2 + (z - z_m) ** 2).min())


def _max_clearance(grid: OccupancyGrid) -> float:
    free_locations = np.argwhere(grid.cells == GridState.FREE)
    blocked_locations = np.argwhere(grid.cells != GridState.FREE)
    if len(free_locations) == 0:
        return 0.0
    if len(blocked_locations) == 0:
        return float(np.hypot(grid.width_m, grid.depth_m))
    # Grid sizes are normally around 100x100. Chunking avoids a large NxM allocation.
    maximum = 0.0
    for start in range(0, len(free_locations), 512):
        chunk = free_locations[start : start + 512]
        delta = chunk[:, None, :] - blocked_locations[None, :, :]
        squared = np.sum(delta * delta, axis=2)
        maximum = max(maximum, float(np.sqrt(np.min(squared, axis=1)).max()))
    return maximum * grid.resolution_m


def analyze_free_space(grid: OccupancyGrid, config: OpenSpaceConfig) -> FreeSpaceMetrics:
    free = grid.cells == GridState.FREE
    occupied = grid.cells == GridState.OCCUPIED
    unknown = grid.cells == GridState.UNKNOWN
    total_cells = grid.cells.size
    cell_area = grid.resolution_m**2

    sizes, labels = _component_sizes(free)
    largest_area = (max(sizes) if sizes else 0) * cell_area
    origin = grid.world_to_cell(0.0, 0.0)
    reachable_area = 0.0
    if origin is not None:
        row, col = origin
        origin_label = int(labels[row, col])
        if origin_label >= 0:
            reachable_area = sizes[origin_label] * cell_area

    nearest_obstacle = _distance_to_mask(grid, occupied, 0.0, 0.0)
    clearance_mask = occupied.copy()
    if config.nearby_unknown_is_unsafe:
        clearance_mask |= unknown
    nearest_unsafe = _distance_to_mask(grid, clearance_mask, 0.0, 0.0)
    player_clearance = nearest_unsafe is None or nearest_unsafe >= config.min_player_clearance_m

    return FreeSpaceMetrics(
        largest_free_area_m2=float(largest_area),
        player_reachable_area_m2=float(reachable_area),
        largest_rectangle=largest_free_rectangle(grid),
        nearest_obstacle_m=nearest_obstacle,
        max_clearance_m=_max_clearance(grid),
        obstacle_ratio=float(np.count_nonzero(occupied) / total_cells),
        unknown_ratio=float(np.count_nonzero(unknown) / total_cells),
        player_clearance=player_clearance,
    )
