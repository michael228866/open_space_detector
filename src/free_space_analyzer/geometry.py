from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .models import CameraInfo, GroundPlane


def depth_to_point_cloud(
    z_depth: npt.ArrayLike,
    camera: CameraInfo,
    stride: int = 1,
) -> npt.NDArray[np.float64]:
    """Project Z-depth into camera space: X right, Y up, Z forward."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    depth = np.asarray(z_depth, dtype=np.float64)
    expected = (camera.height, camera.width)
    if depth.shape != expected:
        raise ValueError(f"Depth shape {depth.shape} does not match {expected}")

    rows = np.arange(0, camera.height, stride, dtype=np.float64)
    cols = np.arange(0, camera.width, stride, dtype=np.float64)
    uu, vv = np.meshgrid(cols, rows)
    sampled = depth[::stride, ::stride]
    valid = np.isfinite(sampled) & (sampled > 0)
    z = sampled[valid]
    x = (uu[valid] - camera.cx) * z / camera.fx
    y = -(vv[valid] - camera.cy) * z / camera.fy
    return np.column_stack((x, y, z))


def ground_basis(
    plane: GroundPlane,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return right, forward, and the point directly below the camera."""
    up = plane.normal
    camera_right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right = camera_right - np.dot(camera_right, up) * up
    if np.linalg.norm(right) < 1e-6:
        camera_forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(up, camera_forward)
    right /= np.linalg.norm(right)
    forward = np.cross(right, up)
    forward /= np.linalg.norm(forward)
    if np.dot(forward, np.array([0.0, 0.0, 1.0])) < 0:
        forward *= -1.0
        right *= -1.0
    camera_floor_origin = -plane.d * up
    return right, forward, camera_floor_origin


def points_to_ground_coordinates(
    points: npt.ArrayLike,
    plane: GroundPlane,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return x, z and signed height relative to the fitted ground plane."""
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    right, forward, origin = ground_basis(plane)
    height = plane.signed_height(array)
    projected = array - height[:, None] * plane.normal
    relative = projected - origin
    x = relative @ right
    z = relative @ forward
    return x, z, height

