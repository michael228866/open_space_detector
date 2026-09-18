from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from .config import GroundConfig, OccupancyConfig
from .geometry import points_to_ground_coordinates
from .models import GridState, GroundPlane, OccupancyGrid


def _bresenham(row0: int, col0: int, row1: int, col1: int) -> list[tuple[int, int]]:
    """Integer line traversal, including both endpoints."""
    points: list[tuple[int, int]] = []
    dx = abs(col1 - col0)
    sx = 1 if col0 < col1 else -1
    dy = -abs(row1 - row0)
    sy = 1 if row0 < row1 else -1
    error = dx + dy
    row, col = row0, col0
    while True:
        points.append((row, col))
        if row == row1 and col == col1:
            return points
        twice = 2 * error
        if twice >= dy:
            error += dy
            col += sx
        if twice <= dx:
            error += dx
            row += sy


def _unique_cells(rows: npt.NDArray[np.int64], cols: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    if len(rows) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(np.column_stack((rows, cols)), axis=0)


def _dilate_occupied(cells: npt.NDArray[np.int8], radius_cells: int) -> None:
    if radius_cells <= 0:
        return
    occupied = np.argwhere(cells == GridState.OCCUPIED)
    if len(occupied) == 0:
        return
    source = cells.copy()
    offsets: list[tuple[int, int]] = []
    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            if dr * dr + dc * dc <= radius_cells * radius_cells:
                offsets.append((dr, dc))
    height, width = cells.shape
    for dr, dc in offsets:
        target_rows = occupied[:, 0] + dr
        target_cols = occupied[:, 1] + dc
        valid = (
            (target_rows >= 0)
            & (target_rows < height)
            & (target_cols >= 0)
            & (target_cols < width)
        )
        source[target_rows[valid], target_cols[valid]] = GridState.OCCUPIED
    cells[:] = source


def build_occupancy_grid(
    points: npt.ArrayLike,
    plane: GroundPlane,
    occupancy_config: OccupancyConfig,
    ground_config: GroundConfig,
) -> OccupancyGrid:
    """Create a forward-facing ground grid from a point cloud.

    Floor returns mark their ray and endpoint FREE. Obstacle returns mark the
    ray FREE and endpoint OCCUPIED. Cells with no evidence remain UNKNOWN.
    """
    rows = int(math.ceil(occupancy_config.depth_m / occupancy_config.resolution_m))
    cols = int(math.ceil(occupancy_config.width_m / occupancy_config.resolution_m))
    x_min = -cols * occupancy_config.resolution_m / 2.0
    cells = np.full((rows, cols), GridState.UNKNOWN, dtype=np.int8)
    grid = OccupancyGrid(
        cells=cells,
        resolution_m=occupancy_config.resolution_m,
        x_min_m=x_min,
        z_min_m=0.0,
        observed_points=0,
    )

    array = np.asarray(points, dtype=np.float64)
    if len(array) == 0:
        return grid
    x, z, height = points_to_ground_coordinates(array, plane)
    inside = (
        (x >= grid.x_min_m)
        & (x < grid.x_min_m + grid.width_m)
        & (z >= grid.z_min_m)
        & (z < grid.z_min_m + grid.depth_m)
    )
    floor = inside & (np.abs(height) <= ground_config.distance_tolerance_m)
    obstacle = (
        inside
        & (height >= occupancy_config.min_obstacle_height_m)
        & (height <= occupancy_config.max_obstacle_height_m)
    )
    useful = floor | obstacle
    grid.observed_points = int(np.count_nonzero(useful))
    if grid.observed_points == 0:
        return grid

    cell_cols = np.floor((x - grid.x_min_m) / grid.resolution_m).astype(np.int64)
    cell_rows = np.floor((z - grid.z_min_m) / grid.resolution_m).astype(np.int64)
    floor_cells = _unique_cells(cell_rows[floor], cell_cols[floor])
    obstacle_cells = _unique_cells(cell_rows[obstacle], cell_cols[obstacle])

    origin = grid.world_to_cell(0.0, 0.0)
    if origin is None:
        raise RuntimeError("Player origin is outside the occupancy grid")
    origin_row, origin_col = origin

    if occupancy_config.raycast_free_space:
        for endpoint_row, endpoint_col in floor_cells:
            line = _bresenham(origin_row, origin_col, int(endpoint_row), int(endpoint_col))
            rr, cc = np.asarray(line, dtype=np.int64).T
            cells[rr, cc] = GridState.FREE
        for endpoint_row, endpoint_col in obstacle_cells:
            line = _bresenham(origin_row, origin_col, int(endpoint_row), int(endpoint_col))
            if len(line) > 1:
                rr, cc = np.asarray(line[:-1], dtype=np.int64).T
                not_occupied = cells[rr, cc] != GridState.OCCUPIED
                cells[rr[not_occupied], cc[not_occupied]] = GridState.FREE
    else:
        if len(floor_cells):
            cells[floor_cells[:, 0], floor_cells[:, 1]] = GridState.FREE

    if len(obstacle_cells):
        cells[obstacle_cells[:, 0], obstacle_cells[:, 1]] = GridState.OCCUPIED
    cells[origin_row, origin_col] = GridState.FREE

    radius_cells = int(math.ceil(occupancy_config.safety_margin_m / grid.resolution_m))
    _dilate_occupied(cells, radius_cells)
    return grid

