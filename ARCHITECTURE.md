# Free Space Analyzer Architecture

## Goal

Determine whether a depth-observed indoor scene contains a sufficiently large,
continuous and safe activity area. The result is geometry-based and explainable;
it does not classify a screenshot as "open" with a black-box model.

## Authoritative conventions

- Input depth is `H x W`, floating point, measured in metres after adaptation.
- `z_depth` means distance along the optical Z axis.
- `radial` means Euclidean distance along the pixel ray and is converted first.
- Camera coordinates are `X right`, `Y up`, `Z forward`.
- The player/camera projects to ground coordinate `(x=0, z=0)`.
- Ground coordinates are `X right`, `Z forward`; the grid covers only `z >= 0`.
- Occupancy states are immutable in meaning:
  - `UNKNOWN = -1`: no trustworthy visibility evidence.
  - `FREE = 0`: a ray passed through or ended on ground.
  - `OCCUPIED = 1`: a return lies in the configured obstacle-height band.
- UNKNOWN must never be silently treated as FREE.
- Public distances use metres and areas use square metres.

## Pipeline

```text
Depth + CameraInfo
        |
        v
validate, sanitize, radial-to-Z conversion
        |
        v
3D point cloud (camera coordinates)
        |
        v
known-height plane or RANSAC ground plane
        |
        v
ground-relative x/z/height
        |
        v
ray-cast occupancy grid + obstacle safety margin
        |
        v
connected free area + maximum rectangle + clearance
        |
        v
hard requirements + 0..100 diagnostic score
```

## Module ownership

| Module | Responsibility |
| --- | --- |
| `models.py` | Stable public data contracts and state enums |
| `config.py` | Strict YAML-to-dataclass configuration and validation |
| `depth.py` | Input validation, invalid-value filtering and depth semantics |
| `geometry.py` | Pinhole projection and ground coordinate transforms |
| `ground.py` | Known-height and RANSAC ground estimation |
| `occupancy.py` | Ray casting, state conflict policy and obstacle inflation |
| `free_space.py` | Connectivity, largest rectangle and clearance metrics |
| `scoring.py` | Explainable score and pass/fail requirements |
| `analyzer.py` | Stable high-level pipeline |
| `visualization.py` | Debug-only bird's-eye-view output |
| `cli.py` | `.npy`/JSON file adapter and command-line interface |

## Ground estimation

`known_height` is the default because an engine normally knows camera height and
orientation. Its plane is deterministic and should be preferred in production.

`ransac` is provided for captures where the camera-to-ground relationship is not
trusted. Candidates are restricted to points below the camera, plane normals must
stay within `max_tilt_deg` of `CameraInfo.up_vector`, and the plane distance must be
plausible relative to `camera_height_m`.

For a pitched or rolled camera, the integration layer must provide the world-up
direction expressed in camera coordinates as `CameraInfo.up_vector`. If the engine
can transform depth points to a level player frame, use `(0, 1, 0)` there.

## Occupancy evidence policy

Floor and obstacle endpoints are quantized to grid cells before ray casting.
Floor rays mark every traversed cell free. Obstacle rays mark cells before the
endpoint free and the endpoint occupied. Occupied evidence wins over free evidence.
Finally, occupied cells are dilated by `safety_margin_m`.

Points below the ground tolerance, points above the obstacle band and out-of-grid
points do not contribute evidence. Unsupported cells remain unknown.

## Decision semantics

`is_open_space` requires every configured hard condition:

1. largest connected free area;
2. an axis-aligned free rectangle, allowing width/depth rotation;
3. player clearance;
4. maximum obstacle ratio;
5. maximum unknown ratio.

`score` is diagnostic, not a replacement for hard safety conditions. A high score
can coexist with `is_open_space=false`; `failure_reasons` explains why.

## Engine integration boundary

The engine adapter owns:

- reading the depth buffer;
- converting normalized/nonlinear depth to metres;
- declaring Z-depth versus radial depth;
- supplying intrinsics or FOV;
- supplying camera height and up direction;
- deciding how transparent, masked and non-collidable materials enter depth.

Do not put Unreal- or Unity-specific conversion formulas inside the core package.
Add adapters beside the application integration and regression-test them using
captured depth frames.

## Known V1 limits

- One forward grid is analyzed; multi-view temporal fusion is not implemented.
- The maximum free rectangle is aligned with the grid axes.
- Stairs, holes, glass and mirrors require engine metadata or an RGB/depth policy.
- Dynamic versus static obstacles are not semantically distinguished.
- `last_occupancy_grid` is intended for single-threaded debug use. A service should
  return or store grids per request rather than sharing one analyzer instance.

