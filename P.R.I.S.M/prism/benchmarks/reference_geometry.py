"""Reference-input geometry checks for the TUM RGB-D sequence.

Every function here consumes benchmark depth, calibration and (where needed)
ground-truth poses.  These are consistency and interface checks on reference
sensor inputs; they do not validate monocular depth inference, metric scale
from a single camera, or traversability.

Coordinates: camera X right, Y down, Z forward (OpenCV).  Ground-truth
``T_w_c`` maps camera-frame points to the benchmark world frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from prism.benchmarks.tum_rgbd_adapter import (
    DEPTH_SCALE,
    FREIBURG1_RGB_INTRINSICS,
    FREIBURG1_ROS_DEFAULT_INTRINSICS,
    depth_to_metres,
    intrinsics_matrix,
)
from prism.perception import geometry as geo


@dataclass(frozen=True, slots=True)
class ReferenceFrame:
    timestamp_s: float
    rgb_bgr: np.ndarray
    depth_u16: np.ndarray
    pose_T_w_c: np.ndarray
    rgb_path: str
    depth_path: str


@dataclass(frozen=True, slots=True)
class DepthStats:
    width: int
    height: int
    valid_pixels: int
    valid_fraction: float
    depth_min_m: float
    depth_p05_m: float
    depth_median_m: float
    depth_p90_m: float
    depth_max_m: float


@dataclass(frozen=True, slots=True)
class CrossViewDepthResult:
    pair_count: int
    compared_pixels: int
    median_abs_error_m: float
    mean_abs_error_m: float
    p90_abs_error_m: float
    within_5cm_fraction: float
    within_10cm_fraction: float
    within_20cm_fraction: float
    mean_baseline_m: float


@dataclass(frozen=True, slots=True)
class CrossViewPhotometricResult:
    pair_count: int
    compared_pixels: int
    median_abs_gray: float
    mean_abs_gray: float
    p90_abs_gray: float
    within_20_gray_fraction: float
    mean_baseline_m: float
    note: str


@dataclass(frozen=True, slots=True)
class SupportPlaneFit:
    ok: bool
    normal_camera: tuple[float, float, float]
    height_m: float
    tilt_deg: float
    inlier_fraction: float
    median_abs_residual_m: float
    candidate_points: int
    note: str


@dataclass(frozen=True, slots=True)
class IntrinsicsSensitivity:
    compared_pixels: int
    median_distance_m: float
    mean_distance_m: float
    p90_distance_m: float
    max_distance_m: float
    note: str


@dataclass(frozen=True, slots=True)
class VehicleGeometry:
    valid_pixels: int
    height_above_support_min_m: float
    height_above_support_median_m: float
    height_above_support_p90_m: float
    slope_median_deg: float
    roughness_median_m: float
    note: str


def depth_frame_stats(depth_u16: np.ndarray) -> DepthStats:
    depth = np.asarray(depth_u16)
    if depth.ndim != 2:
        raise ValueError("depth image must be two-dimensional")
    z = depth_to_metres(depth)
    valid = z[np.isfinite(z) & (z > 0.0)]
    height, width = depth.shape
    if valid.size == 0:
        return DepthStats(width, height, 0, 0.0, *(float("nan"),) * 5)
    return DepthStats(
        width=width,
        height=height,
        valid_pixels=int(valid.size),
        valid_fraction=float(valid.size / depth.size),
        depth_min_m=float(valid.min()),
        depth_p05_m=float(np.percentile(valid, 5)),
        depth_median_m=float(np.median(valid)),
        depth_p90_m=float(np.percentile(valid, 90)),
        depth_max_m=float(valid.max()),
    )


def _sampled_valid_pixels(
    depth_u16: np.ndarray, *, stride: int, max_depth_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = depth_to_metres(depth_u16)
    depth = depth[::stride, ::stride]
    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth_m)
    rows, cols = np.nonzero(valid)
    return rows.astype(np.float64), cols.astype(np.float64), depth[rows, cols]


def _backproject_pixels(
    rows: np.ndarray, cols: np.ndarray, z: np.ndarray, *, intrinsics: dict, stride: int
) -> np.ndarray:
    u = cols * stride
    v = rows * stride
    x = (u - intrinsics["cx"]) * z / intrinsics["fx"]
    y = (v - intrinsics["cy"]) * z / intrinsics["fy"]
    return np.stack([x, y, z], axis=1)


def _cap_samples(
    *arrays: np.ndarray, max_samples: int
) -> tuple[np.ndarray, ...]:
    count = arrays[0].shape[0] if arrays else 0
    if count <= max_samples:
        return arrays
    selection = np.linspace(0, count - 1, max_samples).round().astype(int)
    return tuple(array[selection] for array in arrays)


def cross_view_depth_consistency(
    frames: Sequence[ReferenceFrame],
    *,
    intrinsics: dict = FREIBURG1_ROS_DEFAULT_INTRINSICS,
    max_depth_m: float = 5.0,
    stride: int = 8,
    max_samples_per_pair: int = 20000,
) -> CrossViewDepthResult:
    """Re-project depth from frame A into frame B using ground-truth poses.

    A depth sample from A is transformed to the world with ``T_w_c(A)``, into
    camera B with ``inv(T_w_c(B))``, projected with the same intrinsics, and
    compared with B's registered depth.  Errors therefore include depth noise,
    pose interpolation error, and registration error, but use the benchmark's
    own depth and poses on both sides; this is not independent absolute-depth
    accuracy.
    """
    if len(frames) < 2:
        return CrossViewDepthResult(0, 0, *(float("nan"),) * 7)
    if stride < 1:
        raise ValueError("stride must be at least 1")
    K = intrinsics_matrix(intrinsics)
    errors: list[np.ndarray] = []
    baselines: list[float] = []
    compared = 0
    for frame_a, frame_b in zip(frames[:-1], frames[1:]):
        rows, cols, z = _sampled_valid_pixels(
            frame_a.depth_u16, stride=stride, max_depth_m=max_depth_m)
        if rows.size == 0:
            continue
        rows, cols, z = _cap_samples(rows, cols, z, max_samples=max_samples_per_pair)
        points_a = _backproject_pixels(rows, cols, z, intrinsics=intrinsics, stride=stride)
        T_a = np.asarray(frame_a.pose_T_w_c, np.float64)
        T_b = np.asarray(frame_b.pose_T_w_c, np.float64)
        points_w = points_a @ T_a[:3, :3].T + T_a[:3, 3]
        points_b = (points_w - T_b[:3, 3]) @ T_b[:3, :3]
        in_front = points_b[:, 2] > 1e-6
        if not in_front.any():
            continue
        points_b = points_b[in_front]
        uv = points_b @ K.T
        u = uv[:, 0] / uv[:, 2]
        v = uv[:, 1] / uv[:, 2]
        height, width = frame_b.depth_u16.shape
        inside = (u >= 0.0) & (u <= width - 1) & (v >= 0.0) & (v <= height - 1)
        if not inside.any():
            continue
        ui = np.round(u[inside]).astype(np.int64)
        vi = np.round(v[inside]).astype(np.int64)
        measured = frame_b.depth_u16[vi, ui].astype(np.float64) / DEPTH_SCALE
        projected = points_b[inside, 2]
        valid = (measured > 0.0) & (measured <= max_depth_m)
        if not valid.any():
            continue
        pair_errors = np.abs(projected[valid] - measured[valid])
        errors.append(pair_errors)
        compared += int(pair_errors.size)
        baselines.append(float(np.linalg.norm(T_b[:3, 3] - T_a[:3, 3])))
    if not errors:
        return CrossViewDepthResult(0, 0, *(float("nan"),) * 7)
    all_errors = np.concatenate(errors)
    return CrossViewDepthResult(
        pair_count=len(errors),
        compared_pixels=compared,
        median_abs_error_m=float(np.median(all_errors)),
        mean_abs_error_m=float(np.mean(all_errors)),
        p90_abs_error_m=float(np.percentile(all_errors, 90)),
        within_5cm_fraction=float(np.mean(all_errors <= 0.05)),
        within_10cm_fraction=float(np.mean(all_errors <= 0.10)),
        within_20cm_fraction=float(np.mean(all_errors <= 0.20)),
        mean_baseline_m=float(np.mean(baselines)) if baselines else float("nan"),
    )


def cross_view_photometric_consistency(
    frames: Sequence[ReferenceFrame],
    *,
    max_depth_m: float = 5.0,
    stride: int = 8,
    max_samples_per_pair: int = 15000,
) -> CrossViewPhotometricResult:
    """Project depth A into RGB B and compare grayscale intensities.

    This is an approximate registration/pose/calibration consistency check.  It
    ignores occlusion boundaries, moving objects and sampling interpolation, so
    it is reported as a diagnostic rather than a pass/fail calibration result.
    """
    if len(frames) < 2:
        return CrossViewPhotometricResult(
            0, 0, *(float("nan"),) * 5,
            note="need at least two reference frames")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    K_depth = intrinsics_matrix(FREIBURG1_ROS_DEFAULT_INTRINSICS)
    K_rgb = intrinsics_matrix(FREIBURG1_RGB_INTRINSICS)
    distortion = np.asarray(FREIBURG1_RGB_INTRINSICS["distortion"], np.float64).reshape(1, -1)
    differences: list[np.ndarray] = []
    baselines: list[float] = []
    compared = 0
    for frame_a, frame_b in zip(frames[:-1], frames[1:]):
        rows, cols, z = _sampled_valid_pixels(
            frame_a.depth_u16, stride=stride, max_depth_m=max_depth_m)
        if rows.size == 0:
            continue
        rows, cols, z = _cap_samples(rows, cols, z, max_samples=max_samples_per_pair)
        points_a = _backproject_pixels(
            rows, cols, z, intrinsics=FREIBURG1_ROS_DEFAULT_INTRINSICS, stride=stride)
        T_a = np.asarray(frame_a.pose_T_w_c, np.float64)
        T_b = np.asarray(frame_b.pose_T_w_c, np.float64)
        points_w = points_a @ T_a[:3, :3].T + T_a[:3, 3]
        points_b = (points_w - T_b[:3, 3]) @ T_b[:3, :3]
        in_front = points_b[:, 2] > 1e-6
        if not in_front.any():
            continue
        projected, _ = cv2.projectPoints(
            points_b[in_front].astype(np.float64), np.zeros(3), np.zeros(3),
            K_rgb, distortion)
        uv = projected.reshape(-1, 2)
        u = uv[:, 0]
        v = uv[:, 1]
        height, width = frame_b.rgb_bgr.shape[:2]
        inside = (u >= 0.0) & (u <= width - 1) & (v >= 0.0) & (v <= height - 1)
        if not inside.any():
            continue
        ui_b = np.round(u[inside]).astype(np.int64)
        vi_b = np.round(v[inside]).astype(np.int64)
        # rows/cols index the strided grid; full-resolution RGB needs the stride back
        ui_a = (cols[in_front][inside] * stride).astype(np.int64)
        vi_a = (rows[in_front][inside] * stride).astype(np.int64)
        gray_a = cv2.cvtColor(frame_a.rgb_bgr, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b.rgb_bgr, cv2.COLOR_BGR2GRAY)
        pair_diff = np.abs(
            gray_a[vi_a, ui_a].astype(np.float64) - gray_b[vi_b, ui_b].astype(np.float64))
        differences.append(pair_diff)
        compared += int(pair_diff.size)
        baselines.append(float(np.linalg.norm(T_b[:3, 3] - T_a[:3, 3])))
    if not differences:
        return CrossViewPhotometricResult(
            0, 0, *(float("nan"),) * 5,
            note="no valid projections")
    all_diff = np.concatenate(differences)
    return CrossViewPhotometricResult(
        pair_count=len(differences),
        compared_pixels=compared,
        median_abs_gray=float(np.median(all_diff)),
        mean_abs_gray=float(np.mean(all_diff)),
        p90_abs_gray=float(np.percentile(all_diff, 90)),
        within_20_gray_fraction=float(np.mean(all_diff <= 20.0)),
        mean_baseline_m=float(np.mean(baselines)) if baselines else float("nan"),
        note=(
            "diagnostic only: approximate registered-depth/RGB/pose consistency; "
            "lens distortion is applied on the RGB projection but back-projection "
            "uses the TUM-recommended zero-distortion ROS default intrinsics"),
    )


def fit_support_plane(
    points_cam: np.ndarray,
    valid: np.ndarray,
    *,
    image_shape: tuple[int, int] | None = None,
    lower_band_fraction: float = 0.55,
    max_depth_m: float = 4.0,
    max_points: int = 20000,
    iterations: int = 5,
) -> SupportPlaneFit:
    """Robust dominant-plane fit in the lower image band.

    This is a measured support surface, not a traversability label.  In the
    freiburg1_xyz desk scene it may be a desk or floor surface; the fit does not
    establish that the plane is drivable.
    """
    points = np.asarray(points_cam, np.float64)
    mask = np.asarray(valid, bool)
    if points.shape[:2] != mask.shape:
        raise ValueError("points and valid mask must share the image shape")
    height = mask.shape[0]
    if image_shape is not None and tuple(image_shape) != mask.shape:
        raise ValueError("image_shape does not match the valid mask")
    band = np.zeros_like(mask)
    band[int(height * (1.0 - lower_band_fraction)):, :] = True
    z = points[..., 2]
    eligible = mask & band & np.isfinite(points).all(axis=2) & (z > 0.2) & (z <= max_depth_m)
    candidates = points[eligible]
    if candidates.shape[0] < 200:
        return SupportPlaneFit(
            False, (float("nan"),) * 3, float("nan"), float("nan"), 0.0,
            float("nan"), int(candidates.shape[0]),
            "insufficient valid lower-band points for a support-plane fit")

    if candidates.shape[0] > max_points:
        selection = np.linspace(0, candidates.shape[0] - 1, max_points).round().astype(int)
        candidates = candidates[selection]

    center = candidates.mean(axis=0)
    centered = candidates - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    if normal[1] < 0:
        normal = -normal

    weights = np.ones(candidates.shape[0], np.float64)
    for _ in range(iterations):
        weighted_center = (candidates * weights[:, None]).sum(axis=0) / weights.sum()
        weighted = (candidates - weighted_center) * np.sqrt(weights)[:, None]
        _, _, vt = np.linalg.svd(weighted, full_matrices=False)
        normal = vt[-1]
        if normal[1] < 0:
            normal = -normal
        distances = candidates @ normal - float(weighted_center @ normal)
        sigma = 1.4826 * float(np.median(np.abs(distances - np.median(distances)))) + 1e-6
        weights = 1.0 / np.sqrt(1.0 + (distances / (2.5 * sigma)) ** 2)

    height_m = float(np.median(candidates @ normal))
    residual = candidates @ normal - height_m
    inlier_threshold = max(0.03, 3.0 * 1.4826 * float(
        np.median(np.abs(residual - np.median(residual)))))
    inliers = np.abs(residual) <= inlier_threshold
    inlier_fraction = float(inliers.mean()) if inliers.size else 0.0
    tilt = float(np.degrees(np.arccos(np.clip(normal[1], -1.0, 1.0))))
    return SupportPlaneFit(
        ok=bool(inlier_fraction >= 0.30 and tilt <= 60.0),
        normal_camera=(float(normal[0]), float(normal[1]), float(normal[2])),
        height_m=height_m,
        tilt_deg=tilt,
        inlier_fraction=inlier_fraction,
        median_abs_residual_m=float(np.median(np.abs(residual))),
        candidate_points=int(candidates.shape[0]),
        note=(
            "dominant support plane from lower-band reference depth; "
            "not a traversability or drivable-surface label"),
    )


def intrinsics_sensitivity(
    depth_u16: np.ndarray,
    *,
    stride: int = 8,
    max_depth_m: float = 4.0,
    primary: dict = FREIBURG1_ROS_DEFAULT_INTRINSICS,
    alternate: dict = FREIBURG1_RGB_INTRINSICS,
) -> IntrinsicsSensitivity:
    """3-D disagreement between two back-projection intrinsic sets."""
    if stride < 1:
        raise ValueError("stride must be at least 1")
    rows, cols, z = _sampled_valid_pixels(
        depth_u16, stride=stride, max_depth_m=max_depth_m)
    if rows.size == 0:
        return IntrinsicsSensitivity(
            0, *(float("nan"),) * 4,
            note="no valid depth pixels in range")
    points_primary = _backproject_pixels(rows, cols, z, intrinsics=primary, stride=stride)
    points_alternate = _backproject_pixels(rows, cols, z, intrinsics=alternate, stride=stride)
    distances = np.linalg.norm(points_primary - points_alternate, axis=1)
    return IntrinsicsSensitivity(
        compared_pixels=int(distances.size),
        median_distance_m=float(np.median(distances)),
        mean_distance_m=float(np.mean(distances)),
        p90_distance_m=float(np.percentile(distances, 90)),
        max_distance_m=float(np.max(distances)),
        note=(
            "ROS-default depth-projection intrinsics vs Freiburg 1 RGB intrinsics; "
            "the official page recommends the ROS default for pre-registered depth "
            "and notes undistorting registered depth is not trivial"),
    )


def points_to_world(
    points_cam: np.ndarray, valid: np.ndarray, pose_T_w_c: np.ndarray
) -> np.ndarray:
    """Transform valid camera-frame points to the benchmark world frame (Nx3)."""
    points = np.asarray(points_cam, np.float64)
    mask = np.asarray(valid, bool)
    if points.shape[:2] != mask.shape:
        raise ValueError("points and valid mask must share shape")
    T = np.asarray(pose_T_w_c, np.float64)
    if T.shape != (4, 4):
        raise ValueError("pose_T_w_c must be 4x4")
    selected = points[mask]
    selected = selected[np.isfinite(selected).all(axis=1)]
    return selected @ T[:3, :3].T + T[:3, 3]


def point_cloud_summary(points_world: np.ndarray) -> dict:
    points = np.asarray(points_world, np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or points.shape[0] == 0:
        return {
            "points": 0,
            "bounds_min_m": None,
            "bounds_max_m": None,
            "extent_m": None,
            "centroid_m": None,
        }
    return {
        "points": int(points.shape[0]),
        "bounds_min_m": [float(v) for v in points.min(axis=0)],
        "bounds_max_m": [float(v) for v in points.max(axis=0)],
        "extent_m": [float(v) for v in (points.max(axis=0) - points.min(axis=0))],
        "centroid_m": [float(v) for v in points.mean(axis=0)],
    }


def camera_to_vehicle(
    points_cam: np.ndarray,
    valid: np.ndarray,
    plane: SupportPlaneFit,
) -> tuple[np.ndarray, np.ndarray, VehicleGeometry]:
    """Transform camera points into the existing vehicle convention.

    The vehicle origin is placed on the measured support plane below the camera
    (the plane's measured height), not on the mapping stage's assumed 0.12 m
    camera height.  Height/slope/roughness are then computed with the existing
    geometry helpers and reported relative to that measured support surface.
    """
    points = np.asarray(points_cam, np.float32)
    mask = np.asarray(valid, bool) & np.isfinite(points).all(axis=-1)
    if not plane.ok:
        nan = float("nan")
        return (np.full_like(points, np.nan), np.zeros_like(mask, bool),
                VehicleGeometry(0, nan, nan, nan, nan, nan,
                                "support-plane fit unavailable; no vehicle transform claimed"))
    fit = geo.GroundFit(
        a=1.0, b=0.0,
        normal=np.asarray(plane.normal_camera, np.float32),
        height=float(plane.height_m),
        ok=True,
    )
    points_veh = geo.to_vehicle(points, fit)
    points_veh[~mask] = np.nan
    height, slope, rough = geo.height_slope_roughness(points_veh, mask)
    valid_height = height[np.isfinite(height)]
    valid_slope = slope[np.isfinite(slope)]
    valid_rough = rough[np.isfinite(rough)]
    return (
        points_veh,
        mask,
        VehicleGeometry(
            valid_pixels=int(mask.sum()),
            height_above_support_min_m=float(valid_height.min()) if valid_height.size else float("nan"),
            height_above_support_median_m=float(np.median(valid_height)) if valid_height.size else float("nan"),
            height_above_support_p90_m=float(np.percentile(valid_height, 90)) if valid_height.size else float("nan"),
            slope_median_deg=float(np.median(valid_slope)) if valid_slope.size else float("nan"),
            roughness_median_m=float(np.median(valid_rough)) if valid_rough.size else float("nan"),
            note=(
                "vehicle frame derived from a measured support plane; height is "
                "relative to that plane and is not a traversability label"),
        ),
    )
