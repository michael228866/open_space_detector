from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .config import DepthConfig
from .models import CameraInfo, DepthType


def validate_depth_shape(depth: npt.ArrayLike, camera: CameraInfo) -> npt.NDArray[np.float32]:
    array = np.asarray(depth, dtype=np.float32)
    expected = (camera.height, camera.width)
    if array.shape != expected:
        raise ValueError(f"Depth shape {array.shape} does not match camera resolution {expected}")
    return array


def radial_to_z_depth(
    radial_depth: npt.ArrayLike,
    camera: CameraInfo,
) -> npt.NDArray[np.float32]:
    """Convert distance along the viewing ray to distance along camera Z."""
    radial = validate_depth_shape(radial_depth, camera)
    rows, cols = np.indices(radial.shape, dtype=np.float32)
    x = (cols - camera.cx) / camera.fx
    y = (rows - camera.cy) / camera.fy
    ray_norm = np.sqrt(x * x + y * y + 1.0)
    return (radial / ray_norm).astype(np.float32, copy=False)


def preprocess_depth(
    depth: npt.ArrayLike,
    camera: CameraInfo,
    config: DepthConfig,
) -> npt.NDArray[np.float32]:
    """Return linear Z-depth in metres; invalid pixels are NaN."""
    array = validate_depth_shape(depth, camera).copy()
    if camera.depth_type is DepthType.RADIAL:
        array = radial_to_z_depth(array, camera)
    invalid = (
        ~np.isfinite(array)
        | (array < config.min_depth_m)
        | (array > config.max_depth_m)
    )
    array[invalid] = np.nan
    return array

