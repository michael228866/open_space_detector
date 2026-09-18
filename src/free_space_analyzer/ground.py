from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .config import GroundConfig
from .models import CameraInfo, GroundPlane


class GroundEstimationError(RuntimeError):
    pass


def known_height_plane(camera: CameraInfo) -> GroundPlane:
    up = camera.normalized_up()
    return GroundPlane(normal=up, d=camera.camera_height_m, method="known_height")


def estimate_ground_ransac(
    points: npt.ArrayLike,
    camera: CameraInfo,
    config: GroundConfig,
    rng: np.random.Generator | None = None,
) -> GroundPlane:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) < 3:
        raise GroundEstimationError("At least three 3D points are required")

    up = camera.normalized_up()
    # Keep points plausibly below the camera. This removes most walls/ceilings.
    below = array[(array @ up) < -0.2]
    if len(below) < 3:
        raise GroundEstimationError("Not enough points below the camera to estimate ground")
    if len(below) > 50_000:
        step = max(1, len(below) // 50_000)
        below = below[::step]

    generator = rng or np.random.default_rng(0)
    min_up_dot = float(np.cos(np.deg2rad(config.max_tilt_deg)))
    best_count = 0
    best_normal: npt.NDArray[np.float64] | None = None
    best_d = 0.0

    for _ in range(config.ransac_iterations):
        sample = below[generator.choice(len(below), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal /= norm
        if np.dot(normal, up) < 0:
            normal *= -1.0
        if np.dot(normal, up) < min_up_dot:
            continue
        d = -float(np.dot(normal, sample[0]))
        camera_distance = abs(d)
        if abs(camera_distance - camera.camera_height_m) > max(0.75, camera.camera_height_m * 0.5):
            continue
        distances = np.abs(below @ normal + d)
        count = int(np.count_nonzero(distances <= config.distance_tolerance_m))
        if count > best_count:
            best_count = count
            best_normal = normal.copy()
            best_d = d

    if best_normal is None:
        raise GroundEstimationError("RANSAC did not find a plausible ground plane")
    inlier_ratio = best_count / len(below)
    if inlier_ratio < config.min_inlier_ratio:
        raise GroundEstimationError(
            f"Ground inlier ratio {inlier_ratio:.3f} is below {config.min_inlier_ratio:.3f}"
        )

    distances = np.abs(below @ best_normal + best_d)
    inliers = below[distances <= config.distance_tolerance_m]
    centroid = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    refined_normal = vh[-1]
    if np.dot(refined_normal, up) < 0:
        refined_normal *= -1.0
    if np.dot(refined_normal, up) < min_up_dot:
        refined_normal = best_normal
    refined_d = -float(np.dot(refined_normal, centroid))
    refined_ratio = float(
        np.mean(np.abs(below @ refined_normal + refined_d) <= config.distance_tolerance_m)
    )
    return GroundPlane(
        normal=refined_normal,
        d=refined_d,
        inlier_ratio=refined_ratio,
        method="ransac",
    )


def estimate_ground(
    points: npt.ArrayLike,
    camera: CameraInfo,
    config: GroundConfig,
) -> GroundPlane:
    if config.mode == "known_height":
        return known_height_plane(camera)
    if config.mode == "ransac":
        return estimate_ground_ransac(points, camera, config)
    raise ValueError(f"Unsupported ground mode: {config.mode}")
