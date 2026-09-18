from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .models import GridState, OccupancyGrid, Rectangle


def render_occupancy(
    grid: OccupancyGrid,
    output_path: str | Path,
    rectangle: Rectangle | None = None,
    scale: int = 6,
) -> Path:
    """Write a top-down PNG: grey unknown, white free, red occupied."""
    if scale < 1:
        raise ValueError("scale must be >= 1")
    colors = np.zeros((grid.height, grid.width, 3), dtype=np.uint8)
    colors[grid.cells == GridState.UNKNOWN] = (75, 85, 99)
    colors[grid.cells == GridState.FREE] = (235, 240, 245)
    colors[grid.cells == GridState.OCCUPIED] = (220, 60, 65)
    # Far space is shown at the top of the image.
    colors = np.flipud(colors)
    image = Image.fromarray(colors).resize(
        (grid.width * scale, grid.height * scale),
        resample=Image.Resampling.NEAREST,
    )
    draw = ImageDraw.Draw(image)

    def pixel_for_world(x_m: float, z_m: float) -> tuple[int, int]:
        col = (x_m - grid.x_min_m) / grid.resolution_m
        row = (z_m - grid.z_min_m) / grid.resolution_m
        return int(round(col * scale)), int(round((grid.height - row) * scale))

    if rectangle is not None:
        left, bottom = pixel_for_world(rectangle.x_min_m, rectangle.z_min_m)
        right, top = pixel_for_world(rectangle.x_max_m, rectangle.z_max_m)
        draw.rectangle((left, top, right, bottom), outline=(40, 180, 90), width=max(2, scale // 2))

    player_x, player_y = pixel_for_world(0.0, 0.0)
    radius = max(3, scale)
    draw.ellipse(
        (player_x - radius, player_y - radius, player_x + radius, player_y + radius),
        fill=(40, 110, 240),
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path

