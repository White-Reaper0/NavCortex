"""Bounded sequential visual-odometry audit replay.

WHAT THIS IS
------------
This module runs the classical ORB front end from
:mod:`prism.perception.odometry` one frame at a time over a recorded clip and
records what the geometry actually did.  It deliberately does **not** integrate
a metric pose or claim one:

* detection, matching, essential/fundamental-matrix RANSAC and the Sampson /
  parallax gates are the ones already implemented in ``odometry.py``;
* no depth cache is read, so no metric baseline is invented;
* translation from a monocular essential matrix has an arbitrary scale, so it
  is reported as a unit direction only (``|t| == 1``);
* no camera-to-vehicle extrinsic exists, so the reported frame is the camera
  frame, not a vehicle trajectory.

CALIBRATION
-----------
There is no calibration asset in this repository.  ``CFG.cam.K`` is a
documented 92-degree-HFOV assumption, not a calibration result.  Therefore the
default replay is **tracking-only**:

* feature detection, mutual ratio matching and fundamental-matrix inliers are
  measured;
* no relative pose is estimated, because doing so requires intrinsics that have
  not been calibrated.

A caller may pass a validated :class:`CameraCalibration` explicitly, including
one loaded from a versioned artifact produced by :mod:`prism.calibration`.  If
its status is ``CALIBRATED`` the harness estimates relative motion and marks the
records as calibrated.  If the caller deliberately opts into the documented
config assumption (``ASSUMED``) the motion is still estimated, but every such
record is marked ``UNVALIDATED_MOTION`` rather than ``OK``.

This is visual-only.  It is not visual-inertial odometry and no IMU is used.

MEMORY / REPRODUCIBILITY
------------------------
Frames are decoded one at a time and discarded after their metadata is
recorded.  Only the previous frame's keypoint coordinates and descriptors are
retained.  Timestamps are capture-order timestamps derived from the decoded
frame index and the container FPS; they are not claimed to be camera hardware
timestamps.  Records contain no image pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Optional
import json
import time

import cv2
import numpy as np

from .config import CFG, PROC_H, PROC_W
from . import io_utils
from .calibration import (
    CALIBRATION_SCHEMA,
    CalibrationArtifact,
    CalibrationError,
    load_calibration_artifact,
)
from .perception.odometry import (
    OdomParams,
    _grid_detect,
    _mutual_ratio_match,
    _sampson,
    _yaw_from_R,
)
from .video_observation_replay import CaptureFactory


CALIBRATION_REQUIREMENT = (
    "Required for calibrated relative motion: a calibrated 3x3 pinhole camera "
    "matrix at 640x360 (fx, fy, cx, cy), the lens distortion coefficients, and a "
    "camera-to-vehicle rotation/translation. No calibration asset exists in this "
    "repository; CFG.cam.K is a documented 92-degree-HFOV assumption, not a "
    "calibration."
)

ARBITRARY_SCALE_NOTE = (
    "arbitrary scale: monocular two-view geometry fixes only the translation "
    "direction (|t| = 1). No metric distance, speed or metric trajectory is "
    "reported."
)

COORDINATE_FRAME_NOTE = (
    "camera frame (OpenCV: X right, Y down, Z forward). No camera-to-vehicle "
    "extrinsic is available, so this is not a vehicle-frame trajectory."
)


class CalibrationStatus(str, Enum):
    CALIBRATED = "calibrated"
    ASSUMED = "assumed"
    MISSING = "missing"
    INVALID = "invalid"


class MotionStatus(str, Enum):
    """Outcome of one inter-frame tracking/geometry attempt."""

    FIRST_FRAME = "first_frame"
    OK = "ok"
    UNVALIDATED_MOTION = "unvalidated_motion"
    TRACKING_ONLY = "tracking_only"
    INSUFFICIENT_FEATURES = "insufficient_features"
    INSUFFICIENT_MATCHES = "insufficient_matches"
    INSUFFICIENT_INLIERS = "insufficient_inliers"
    GEOMETRY_FAILED = "geometry_failed"
    DEGENERATE = "degenerate"
    IMPLAUSIBLE_YAW = "implausible_yaw"
    INVALID_CALIBRATION = "invalid_calibration"
    FRAME_ERROR = "frame_error"
    DECODE_FAILURE = "decode_failure"


TRACKING_SUCCESS_STATUSES = frozenset({
    MotionStatus.OK,
    MotionStatus.UNVALIDATED_MOTION,
    MotionStatus.TRACKING_ONLY,
})

TRACKING_LOSS_STATUSES = frozenset({
    MotionStatus.INSUFFICIENT_FEATURES,
    MotionStatus.INSUFFICIENT_MATCHES,
    MotionStatus.INSUFFICIENT_INLIERS,
    MotionStatus.GEOMETRY_FAILED,
    MotionStatus.DEGENERATE,
    MotionStatus.IMPLAUSIBLE_YAW,
    MotionStatus.INVALID_CALIBRATION,
    MotionStatus.FRAME_ERROR,
})


def _validated_intrinsics(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (3, 3):
        raise ValueError("camera matrix must have shape (3, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError("camera matrix contains non-finite values")
    if array[0, 0] <= 0.0 or array[1, 1] <= 0.0:
        raise ValueError("camera focal lengths fx and fy must be positive")
    return array.copy()


def _validated_distortion(distortion: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if distortion is None:
        return None
    array = np.asarray(distortion, dtype=np.float64).ravel()
    if array.size < 4:
        raise ValueError("distortion coefficients must contain at least 4 values")
    if not np.all(np.isfinite(array)):
        raise ValueError("distortion coefficients contain non-finite values")
    return array.copy()


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """Validated intrinsics plus their provenance, or an explicit absence."""

    matrix: Optional[np.ndarray]
    status: CalibrationStatus
    provenance: str
    requirement: str = CALIBRATION_REQUIREMENT
    distortion: Optional[np.ndarray] = None
    image_size: Optional[tuple[int, int]] = None
    calibration_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status in (CalibrationStatus.CALIBRATED, CalibrationStatus.ASSUMED):
            if self.matrix is None:
                raise ValueError(f"{self.status.value} calibration needs a camera matrix")
            object.__setattr__(self, "matrix", _validated_intrinsics(self.matrix))
        elif self.matrix is not None:
            object.__setattr__(self, "matrix", _validated_intrinsics(self.matrix))
        object.__setattr__(self, "distortion", _validated_distortion(self.distortion))
        if self.image_size is not None:
            if len(self.image_size) != 2:
                raise ValueError("image_size must contain width and height")
            width, height = self.image_size
            if isinstance(width, bool) or isinstance(height, bool) \
                    or not isinstance(width, int) or not isinstance(height, int) \
                    or width <= 0 or height <= 0:
                raise ValueError("image_size width and height must be positive integers")
            object.__setattr__(self, "image_size", (width, height))
        if self.calibration_id is not None \
                and (not isinstance(self.calibration_id, str)
                     or not self.calibration_id):
            raise ValueError("calibration_id must be a non-empty string")

    @classmethod
    def from_intrinsics(
        cls,
        matrix: np.ndarray,
        provenance: str,
        *,
        distortion: Optional[np.ndarray] = None,
        image_size: Optional[tuple[int, int]] = None,
        calibration_id: Optional[str] = None,
    ) -> "CameraCalibration":
        return cls(matrix, CalibrationStatus.CALIBRATED, provenance,
                   distortion=distortion, image_size=image_size,
                   calibration_id=calibration_id)

    @classmethod
    def from_artifact(
        cls,
        artifact: CalibrationArtifact | str | Path,
    ) -> "CameraCalibration":
        if not isinstance(artifact, CalibrationArtifact):
            artifact = load_calibration_artifact(artifact)
        provenance = (
            f"validated OpenCV checkerboard calibration "
            f"{artifact.calibration_id}: "
            f"{artifact.pattern.columns}x{artifact.pattern.rows} inner corners, "
            f"{artifact.pattern.square_size:g} {artifact.pattern.units} squares, "
            f"{artifact.detected_views} detected views, "
            f"RMS {artifact.rms_px:.4f} px"
        )
        return cls(
            artifact.camera_matrix,
            CalibrationStatus.CALIBRATED,
            provenance,
            distortion=artifact.distortion_coefficients,
            image_size=artifact.image_size,
            calibration_id=artifact.calibration_id,
        )

    @classmethod
    def assumed_from_config(cls) -> "CameraCalibration":
        provenance = (
            f"documented assumption in prism.config.CameraConfig: "
            f"{CFG.cam.hfov_deg:.1f} deg horizontal FOV at {PROC_W}x{PROC_H}, "
            "not a calibration result"
        )
        return cls(CFG.cam.K, CalibrationStatus.ASSUMED, provenance,
                   image_size=(PROC_W, PROC_H))

    @classmethod
    def missing(cls) -> "CameraCalibration":
        return cls(None, CalibrationStatus.MISSING,
                   "no calibration asset is present in the repository")

    @classmethod
    def invalid(cls, reason: str) -> "CameraCalibration":
        return cls(None, CalibrationStatus.INVALID, reason)

    def for_frame_size(self, width: int, height: int) -> "CameraCalibration":
        """Scale intrinsics to the image size actually fed to the front end."""
        if isinstance(width, bool) or isinstance(height, bool) \
                or not isinstance(width, int) or not isinstance(height, int) \
                or width <= 0 or height <= 0:
            raise ValueError("frame width and height must be positive integers")
        if self.matrix is None:
            return self
        target = (width, height)
        if self.image_size is None:
            return replace(self, image_size=target)
        if self.image_size == target:
            return self

        source_width, source_height = self.image_size
        scale_x = width / source_width
        scale_y = height / source_height
        if abs(scale_x - scale_y) > 0.01 * max(scale_x, scale_y):
            return CameraCalibration.invalid(
                "calibration image aspect ratio "
                f"{source_width}:{source_height} does not match the processing "
                f"frame aspect ratio {width}:{height}; the calibration was not "
                "rescaled anisotropically")
        matrix = self.matrix.copy()
        matrix[0, :] *= scale_x
        matrix[1, :] *= scale_y
        provenance = (
            f"{self.provenance}; intrinsics scaled from "
            f"{source_width}x{source_height} to {width}x{height}"
        )
        return replace(
            self, matrix=matrix, image_size=target, provenance=provenance)

    @property
    def usable_intrinsics(self) -> bool:
        return self.matrix is not None and self.status in (
            CalibrationStatus.CALIBRATED, CalibrationStatus.ASSUMED)

    @property
    def calibrated(self) -> bool:
        return self.status is CalibrationStatus.CALIBRATED

    def as_dict(self) -> dict:
        return {
            "status": self.status.value,
            "calibrated": self.calibrated,
            "provenance": self.provenance,
            "requirement": self.requirement,
            "matrix": None if self.matrix is None else self.matrix.tolist(),
            "distortion_coefficients": None if self.distortion is None
            else self.distortion.tolist(),
            "image_size": None if self.image_size is None
            else list(self.image_size),
            "calibration_id": self.calibration_id,
        }


@dataclass(frozen=True, slots=True)
class RelativeMotionEstimate:
    """One model-free two-view geometry result."""

    status: MotionStatus
    n_matches: int
    n_inliers: int
    inlier_ratio: float
    reason: str
    rotation: Optional[np.ndarray] = None
    translation_unit: Optional[np.ndarray] = None
    yaw_deg: Optional[float] = None
    rotation_deg: Optional[float] = None
    parallax_deg: Optional[float] = None
    sampson_px: Optional[float] = None
    translation_scale: str = ARBITRARY_SCALE_NOTE

    @property
    def motion_valid(self) -> bool:
        return self.status is MotionStatus.OK and self.translation_unit is not None

    @property
    def translation_direction(self) -> Optional[tuple[float, float, float]]:
        if self.translation_unit is None:
            return None
        unit = np.asarray(self.translation_unit, np.float64).ravel()
        return tuple(float(v) for v in unit)


def _finite_correspondences(p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(p0, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(p1, dtype=np.float64).reshape(-1, 2)
    if len(a) != len(b):
        raise ValueError("correspondence arrays must have the same length")
    finite = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    return a[finite], b[finite]


def _failure(status: MotionStatus, reason: str, n_matches: int = 0,
             n_inliers: int = 0, sampson_px: Optional[float] = None) -> RelativeMotionEstimate:
    ratio = float(n_inliers / n_matches) if n_matches else 0.0
    return RelativeMotionEstimate(status, int(n_matches), int(n_inliers), ratio,
                                  reason, sampson_px=sampson_px)


def estimate_relative_motion(
    p0: np.ndarray,
    p1: np.ndarray,
    K: np.ndarray,
    params: Optional[OdomParams] = None,
    *,
    dt_s: Optional[float] = None,
    distortion: Optional[np.ndarray] = None,
) -> RelativeMotionEstimate:
    """Estimate relative rotation and unit translation from correspondences.

    This is the calibrated path.  ``recoverPose`` returns ``|t| == 1``; the
    caller must not interpret it as metres.  A missing/invalid camera matrix
    returns ``INVALID_CALIBRATION`` instead of falling back to an assumed one.
    Distortion coefficients, when supplied, are applied before essential-matrix
    estimation and pose recovery.
    """
    p = params or OdomParams()
    try:
        matrix = _validated_intrinsics(K)
        coefficients = _validated_distortion(distortion)
    except ValueError as error:
        return _failure(MotionStatus.INVALID_CALIBRATION, str(error))

    a, b = _finite_correspondences(p0, p1)
    if coefficients is not None:
        try:
            a = cv2.undistortPoints(
                a.reshape(-1, 1, 2), matrix, coefficients.reshape(1, -1),
                P=matrix).reshape(-1, 2)
            b = cv2.undistortPoints(
                b.reshape(-1, 1, 2), matrix, coefficients.reshape(1, -1),
                P=matrix).reshape(-1, 2)
        except cv2.error as error:
            return _failure(MotionStatus.INVALID_CALIBRATION,
                            f"could not undistort correspondences: {error}")
    n = len(a)
    if n < p.min_matches:
        return _failure(MotionStatus.INSUFFICIENT_MATCHES,
                        f"only {n} finite correspondences (< {p.min_matches})", n)

    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    E, mask = cv2.findEssentialMat(
        a, b, matrix, method=method, prob=p.ransac_conf,
        threshold=p.ransac_thresh_px,
    )
    if E is None or np.asarray(E).shape != (3, 3) or mask is None:
        return _failure(MotionStatus.GEOMETRY_FAILED,
                        "essential matrix not estimated", n)
    inlier_mask = np.asarray(mask).ravel().astype(bool)
    if inlier_mask.size != n:
        return _failure(MotionStatus.GEOMETRY_FAILED,
                        "essential-matrix inlier mask has the wrong length", n)

    n_inliers = int(inlier_mask.sum())
    if n_inliers < p.min_inliers:
        return _failure(MotionStatus.INSUFFICIENT_INLIERS,
                        f"only {n_inliers} essential-matrix inliers (< {p.min_inliers})",
                        n, n_inliers)

    _, R, t, _ = cv2.recoverPose(
        E, a, b, matrix, mask=inlier_mask.astype(np.uint8) * 255)
    if R is None or t is None:
        return _failure(MotionStatus.GEOMETRY_FAILED, "pose recovery failed",
                        n, n_inliers)

    i0, i1 = a[inlier_mask], b[inlier_mask]
    Kinv = np.linalg.inv(matrix)
    F = Kinv.T @ E @ Kinv
    sampson = _sampson(F, i0, i1)
    sampson_med = float(np.median(sampson))

    f0 = np.hstack([i0, np.ones((len(i0), 1))]) @ Kinv.T
    f1 = np.hstack([i1, np.ones((len(i1), 1))]) @ Kinv.T
    f0 /= np.maximum(np.linalg.norm(f0, axis=1, keepdims=True), 1e-12)
    f1r = f1 @ R
    f1r /= np.maximum(np.linalg.norm(f1r, axis=1, keepdims=True), 1e-12)
    cosang = np.clip(np.einsum("ij,ij->i", f0, f1r), -1.0, 1.0)
    parallax_deg = float(np.degrees(np.median(np.arccos(cosang))))

    if parallax_deg < p.min_parallax_deg:
        return _failure(MotionStatus.DEGENERATE,
                        f"degenerate: parallax {parallax_deg:.3f} deg "
                        f"(< {p.min_parallax_deg:.3f} deg, near-pure rotation)",
                        n, n_inliers, sampson_med)

    yaw = float(_yaw_from_R(np.asarray(R, np.float64)))
    if dt_s is not None and dt_s > 0.0:
        if abs(np.degrees(yaw) / dt_s) > p.max_yaw_rate_dps:
            return _failure(MotionStatus.IMPLAUSIBLE_YAW,
                            f"implausible yaw rate {np.degrees(yaw) / dt_s:.0f} deg/s",
                            n, n_inliers, sampson_med)

    unit = np.asarray(t, np.float64).ravel()
    norm = float(np.linalg.norm(unit))
    if norm <= 1e-12:
        return _failure(MotionStatus.GEOMETRY_FAILED, "zero translation direction",
                        n, n_inliers, sampson_med)
    unit = unit / norm
    rotation_deg = float(np.degrees(np.arccos(
        np.clip((float(np.trace(R)) - 1.0) / 2.0, -1.0, 1.0))))
    return RelativeMotionEstimate(
        MotionStatus.OK, n, n_inliers, float(n_inliers / n), "ok",
        rotation=np.asarray(R, np.float64), translation_unit=unit,
        yaw_deg=float(np.degrees(yaw)), rotation_deg=rotation_deg,
        parallax_deg=parallax_deg, sampson_px=sampson_med,
    )


def estimate_fundamental_inliers(
    p0: np.ndarray,
    p1: np.ndarray,
    params: Optional[OdomParams] = None,
) -> RelativeMotionEstimate:
    """Tracking-only geometry: fundamental-matrix inliers, no pose.

    A fundamental matrix is independent of camera intrinsics, so this is the
    strongest geometric check that remains valid when no calibration exists.
    """
    p = params or OdomParams()
    a, b = _finite_correspondences(p0, p1)
    n = len(a)
    if n < p.min_matches:
        return _failure(MotionStatus.INSUFFICIENT_MATCHES,
                        f"only {n} finite correspondences (< {p.min_matches})", n)

    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    F, mask = cv2.findFundamentalMat(
        a, b, method, p.ransac_thresh_px, p.ransac_conf)
    if F is None or mask is None:
        return _failure(MotionStatus.GEOMETRY_FAILED,
                        "fundamental matrix not estimated", n)
    F = np.asarray(F, np.float64)
    if F.size < 9:
        return _failure(MotionStatus.GEOMETRY_FAILED,
                        "fundamental matrix has the wrong shape", n)
    F = F[:3, :3]
    inlier_mask = np.asarray(mask).ravel().astype(bool)
    if inlier_mask.size != n:
        return _failure(MotionStatus.GEOMETRY_FAILED,
                        "fundamental-matrix inlier mask has the wrong length", n)

    n_inliers = int(inlier_mask.sum())
    if n_inliers < p.min_inliers:
        return _failure(MotionStatus.INSUFFICIENT_INLIERS,
                        f"only {n_inliers} fundamental-matrix inliers (< {p.min_inliers})",
                        n, n_inliers)

    sampson = _sampson(F, a[inlier_mask], b[inlier_mask])
    sampson_med = float(np.median(sampson))
    return RelativeMotionEstimate(
        MotionStatus.TRACKING_ONLY, n, n_inliers, float(n_inliers / n),
        "calibration unavailable: fundamental-matrix inliers only; "
        "relative pose not estimated",
        sampson_px=sampson_med,
    )


@dataclass(frozen=True, slots=True)
class FrameOdometryRecord:
    """Serializable per-frame diagnostics; deliberately excludes image pixels."""

    frame_index: int
    capture_order_ns: int
    n_features: int
    n_matches: int
    n_inliers: int
    inlier_ratio: float
    status: MotionStatus
    tracking_ok: bool
    latency_ms: float
    reason: str
    yaw_deg: Optional[float] = None
    rotation_deg: Optional[float] = None
    translation_unit: Optional[tuple[float, float, float]] = None
    rotation: Optional[tuple[float, ...]] = None
    parallax_deg: Optional[float] = None
    sampson_px: Optional[float] = None
    translation_scale: str = ARBITRARY_SCALE_NOTE

    @property
    def is_loss(self) -> bool:
        return self.status in TRACKING_LOSS_STATUSES

    def as_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "capture_order_ns": self.capture_order_ns,
            "n_features": self.n_features,
            "n_matches": self.n_matches,
            "n_inliers": self.n_inliers,
            "inlier_ratio": self.inlier_ratio,
            "status": self.status.value,
            "tracking_ok": self.tracking_ok,
            "latency_ms": self.latency_ms,
            "reason": self.reason,
            "yaw_deg": self.yaw_deg,
            "rotation_deg": self.rotation_deg,
            "translation_unit": None if self.translation_unit is None
            else list(self.translation_unit),
            "rotation": None if self.rotation is None else list(self.rotation),
            "parallax_deg": self.parallax_deg,
            "sampson_px": self.sampson_px,
            "translation_scale": self.translation_scale,
        }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


@dataclass(frozen=True, slots=True)
class VideoOdometryReplayReport:
    """Bounded report for one sequential clip replay."""

    video_path: str
    opened: bool
    fps: float
    width: int
    height: int
    declared_frame_count: int
    decoded_frames: int
    processed_frames: int
    calibration: CameraCalibration
    tracking_only: bool
    records: tuple[FrameOdometryRecord, ...]
    errors: tuple[str, ...]
    coordinate_frame: str = COORDINATE_FRAME_NOTE
    translation_scale: str = ARBITRARY_SCALE_NOTE

    @property
    def status_counts(self) -> dict:
        counts = {status.value: 0 for status in MotionStatus}
        for record in self.records:
            counts[record.status.value] += 1
        return counts

    @property
    def tracking_attempts(self) -> int:
        return sum(1 for record in self.records
                   if record.status not in (MotionStatus.FIRST_FRAME,
                                            MotionStatus.DECODE_FAILURE))

    @property
    def successful_tracking_frames(self) -> int:
        return sum(1 for record in self.records
                   if record.status in TRACKING_SUCCESS_STATUSES)

    @property
    def tracking_losses(self) -> int:
        return sum(1 for record in self.records if record.is_loss)

    @property
    def tracking_success_rate(self) -> float:
        attempts = self.tracking_attempts
        return float(self.successful_tracking_frames / attempts) if attempts else 0.0

    @property
    def relative_motion_frames(self) -> int:
        return sum(1 for record in self.records if record.status is MotionStatus.OK)

    @property
    def unvalidated_motion_frames(self) -> int:
        return sum(1 for record in self.records
                   if record.status is MotionStatus.UNVALIDATED_MOTION)

    @property
    def tracking_only_frames(self) -> int:
        return sum(1 for record in self.records
                   if record.status is MotionStatus.TRACKING_ONLY)

    @property
    def loss_events(self) -> tuple[FrameOdometryRecord, ...]:
        return tuple(record for record in self.records if record.is_loss)

    @property
    def mean_features(self) -> float:
        values = [record.n_features for record in self.records
                  if record.status is not MotionStatus.DECODE_FAILURE]
        return _mean([float(v) for v in values])

    @property
    def mean_matches(self) -> float:
        values = [record.n_matches for record in self.records
                  if record.status not in (MotionStatus.FIRST_FRAME,
                                           MotionStatus.DECODE_FAILURE)]
        return _mean([float(v) for v in values])

    @property
    def mean_inliers(self) -> float:
        values = [record.n_inliers for record in self.records
                  if record.status not in (MotionStatus.FIRST_FRAME,
                                           MotionStatus.DECODE_FAILURE)]
        return _mean([float(v) for v in values])

    @property
    def mean_inlier_ratio(self) -> float:
        values = [record.inlier_ratio for record in self.records
                  if record.status not in (MotionStatus.FIRST_FRAME,
                                           MotionStatus.DECODE_FAILURE)
                  and record.n_matches > 0]
        return _mean(values)

    @property
    def latency_stats(self) -> dict:
        values = [record.latency_ms for record in self.records
                  if record.status is not MotionStatus.DECODE_FAILURE]
        if not values:
            return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0,
                    "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
        arr = np.asarray(values, np.float64)
        return {
            "count": int(arr.size),
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(arr.max()),
        }

    @property
    def trajectory_produced(self) -> bool:
        return False

    @property
    def metric_scale_meaningful(self) -> bool:
        return False

    def summary(self, *, include_latency: bool = True) -> dict:
        out = {
            "video_path": self.video_path,
            "opened": self.opened,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "declared_frame_count": self.declared_frame_count,
            "decoded_frames": self.decoded_frames,
            "processed_frames": self.processed_frames,
            "calibration": self.calibration.as_dict(),
            "tracking_only": self.tracking_only,
            "tracking_attempts": self.tracking_attempts,
            "successful_tracking_frames": self.successful_tracking_frames,
            "tracking_losses": self.tracking_losses,
            "tracking_success_rate": self.tracking_success_rate,
            "relative_motion_frames": self.relative_motion_frames,
            "unvalidated_motion_frames": self.unvalidated_motion_frames,
            "tracking_only_frames": self.tracking_only_frames,
            "mean_features": self.mean_features,
            "mean_matches": self.mean_matches,
            "mean_inliers": self.mean_inliers,
            "mean_inlier_ratio": self.mean_inlier_ratio,
            "status_counts": self.status_counts,
            "loss_events": [
                {
                    "frame_index": record.frame_index,
                    "status": record.status.value,
                    "reason": record.reason,
                    "n_matches": record.n_matches,
                    "n_inliers": record.n_inliers,
                }
                for record in self.loss_events
            ],
            "trajectory_produced": self.trajectory_produced,
            "metric_scale_meaningful": self.metric_scale_meaningful,
            "coordinate_frame": self.coordinate_frame,
            "translation_scale": self.translation_scale,
            "errors": list(self.errors),
        }
        if include_latency:
            out["latency_ms"] = self.latency_stats
        return out


class _OrbFrontend:
    """Detector/matcher configured exactly like ``OdometryStage``."""

    def __init__(self, params: OdomParams) -> None:
        self.p = params
        self.orb = cv2.ORB_create(
            nfeatures=params.n_features * 3, scaleFactor=params.scale_factor,
            nlevels=params.n_levels, edgeThreshold=19, patchSize=31,
            fastThreshold=params.fast_thresh,
            scoreType=cv2.ORB_HARRIS_SCORE)
        self.orb_relaxed = cv2.ORB_create(
            nfeatures=params.n_features, scaleFactor=params.scale_factor,
            nlevels=params.n_levels, edgeThreshold=19, patchSize=31,
            fastThreshold=params.fast_thresh_relaxed,
            scoreType=cv2.ORB_HARRIS_SCORE)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.barren = np.zeros(params.grid_cols * params.grid_rows, np.uint8)
        mask = io_utils.ego_mask(PROC_H, PROC_W).astype(np.uint8) * 255
        self.mask = mask

    def detect(self, bgr: np.ndarray, frame_index: int):
        if bgr.shape[1] != PROC_W or bgr.shape[0] != PROC_H:
            bgr = cv2.resize(bgr, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)
        gray = self.clahe.apply(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        keypoints = _grid_detect(self.orb, self.orb_relaxed, gray, self.mask,
                                 self.p, self.barren, frame_index)
        keypoints, descriptors = self.orb.compute(gray, keypoints)
        if keypoints:
            xy = np.asarray([kp.pt for kp in keypoints], np.float32).reshape(-1, 2)
        else:
            xy = np.zeros((0, 2), np.float32)
        return xy, descriptors


def _capture_order_timestamp(capture_start_ns: int, frame_index: int, fps: float) -> int:
    # Capture-order replay time derived from decode order, not a hardware timestamp.
    return capture_start_ns + round(frame_index * 1_000_000_000 / fps)


class VideoOdometryReplayHarness:
    """Decode one frame at a time and audit the classical VO front end."""

    def __init__(
        self,
        capture_factory: CaptureFactory = cv2.VideoCapture,
        params: Optional[OdomParams] = None,
        calibration: Optional[CameraCalibration] = None,
    ) -> None:
        self._capture_factory = capture_factory
        self._params = params or OdomParams()
        self._calibration = calibration if calibration is not None \
            else CameraCalibration.missing()

    def replay(
        self,
        video_path: str | Path,
        *,
        max_frames: Optional[int] = None,
        capture_start_ns: int = 0,
        require_calibration: bool = False,
    ) -> VideoOdometryReplayReport:
        if isinstance(capture_start_ns, bool) or not isinstance(capture_start_ns, int) \
                or capture_start_ns < 0:
            raise ValueError("capture_start_ns must be a non-negative integer")
        if max_frames is not None and (isinstance(max_frames, bool)
                                       or not isinstance(max_frames, int)
                                       or max_frames < 0):
            raise ValueError("max_frames must be a non-negative integer or None")

        path = str(video_path)
        calibration = self._calibration
        if calibration.usable_intrinsics:
            calibration = calibration.for_frame_size(PROC_W, PROC_H)
        if calibration.status is CalibrationStatus.INVALID:
            return VideoOdometryReplayReport(
                path, False, 0.0, 0, 0, 0, 0, 0, calibration, True, (), 
                (f"invalid calibration: {calibration.provenance}",))
        if require_calibration and not calibration.calibrated:
            return VideoOdometryReplayReport(
                path, False, 0.0, 0, 0, 0, 0, 0, calibration, True, (),
                ("calibration required but unavailable: "
                 + calibration.requirement,))

        cap = self._capture_factory(path)
        if not cap.isOpened():
            cap.release()
            return VideoOdometryReplayReport(
                path, False, 0.0, 0, 0, 0, 0, 0, calibration, True, (),
                ("video could not be opened",))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        declared = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if not np.isfinite(fps) or fps <= 0:
            cap.release()
            return VideoOdometryReplayReport(
                path, True, fps, width, height, declared, 0, 0, calibration, True, (),
                ("video reports no positive finite FPS",))

        frontend = _OrbFrontend(self._params)
        records: list[FrameOdometryRecord] = []
        errors: list[str] = []
        prev_kp: Optional[np.ndarray] = None
        prev_des: Optional[np.ndarray] = None
        tracking_only = not calibration.usable_intrinsics
        frame_index = 0
        try:
            while max_frames is None or frame_index < max_frames:
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    if declared > frame_index:
                        record = FrameOdometryRecord(
                            frame_index,
                            _capture_order_timestamp(capture_start_ns, frame_index, fps),
                            0, 0, 0, 0.0, MotionStatus.DECODE_FAILURE, False, 0.0,
                            "decode failed before declared end of video")
                        records.append(record)
                        errors.append(f"decode failed at frame {frame_index}")
                    break

                timestamp = _capture_order_timestamp(capture_start_ns, frame_index, fps)
                started = time.perf_counter()
                try:
                    if bgr.ndim != 3 or bgr.shape[2] != 3:
                        raise ValueError("decoded frame is not a 3-channel image")
                    kp_xy, descriptors = frontend.detect(bgr, frame_index)
                    n_features = int(len(kp_xy))
                    n_matches = 0
                    n_inliers = 0
                    yaw_deg = rotation_deg = parallax_deg = sampson_px = None
                    translation_unit = None
                    rotation_matrix = None

                    if prev_kp is None:
                        status = MotionStatus.FIRST_FRAME
                        reason = (f"first frame ({n_features} features extracted; "
                                  "no previous view)")
                    elif n_features < self._params.min_matches:
                        status = MotionStatus.INSUFFICIENT_FEATURES
                        reason = (f"only {n_features} features "
                                  f"(< {self._params.min_matches})")
                    else:
                        pairs = _mutual_ratio_match(
                            frontend.matcher, prev_des, descriptors, self._params.ratio)
                        n_matches = int(len(pairs))
                        if n_matches < self._params.min_matches:
                            status = MotionStatus.INSUFFICIENT_MATCHES
                            reason = (f"only {n_matches} matches "
                                      f"(< {self._params.min_matches})")
                        else:
                            p0 = prev_kp[pairs[:, 0]].astype(np.float64)
                            p1 = kp_xy[pairs[:, 1]].astype(np.float64)
                            if calibration.usable_intrinsics:
                                estimate = estimate_relative_motion(
                                    p0, p1, calibration.matrix, self._params,
                                    dt_s=1.0 / fps,
                                    distortion=calibration.distortion)
                                if estimate.status is MotionStatus.OK:
                                    status = (MotionStatus.OK if calibration.calibrated
                                              else MotionStatus.UNVALIDATED_MOTION)
                                else:
                                    status = estimate.status
                            else:
                                estimate = estimate_fundamental_inliers(
                                    p0, p1, self._params)
                                status = estimate.status
                            n_inliers = int(estimate.n_inliers)
                            reason = estimate.reason
                            yaw_deg = estimate.yaw_deg
                            rotation_deg = estimate.rotation_deg
                            parallax_deg = estimate.parallax_deg
                            sampson_px = estimate.sampson_px
                            translation_unit = estimate.translation_direction
                            if estimate.rotation is not None:
                                rotation_matrix = tuple(
                                    float(v) for v in np.asarray(
                                        estimate.rotation, np.float64).reshape(-1))

                    latency_ms = (time.perf_counter() - started) * 1000.0
                    records.append(FrameOdometryRecord(
                        frame_index, timestamp, n_features, n_matches, n_inliers,
                        float(n_inliers / n_matches) if n_matches else 0.0,
                        status, status not in TRACKING_LOSS_STATUSES, latency_ms,
                        reason, yaw_deg, rotation_deg, translation_unit,
                        rotation_matrix, parallax_deg, sampson_px))
                    prev_kp = kp_xy
                    prev_des = descriptors
                except Exception as error:  # keep the replay bounded and auditable
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    records.append(FrameOdometryRecord(
                        frame_index, timestamp, 0, 0, 0, 0.0,
                        MotionStatus.FRAME_ERROR, False, latency_ms,
                        f"{type(error).__name__}: {error}"))
                    errors.append(f"frame {frame_index}: {type(error).__name__}: {error}")
                frame_index += 1
        finally:
            cap.release()

        decoded = sum(1 for record in records
                      if record.status is not MotionStatus.DECODE_FAILURE)
        processed = sum(1 for record in records
                        if record.status is not MotionStatus.DECODE_FAILURE)
        return VideoOdometryReplayReport(
            path, True, fps, width, height, declared, decoded, processed,
            calibration, tracking_only, tuple(records), tuple(errors))


def _load_calibration(path: str) -> CameraCalibration:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return CameraCalibration.invalid(
            f"could not read calibration file {source}: {error}")
    if not isinstance(data, dict):
        return CameraCalibration.invalid("calibration file must contain a JSON object")
    if data.get("schema") == CALIBRATION_SCHEMA:
        try:
            return CameraCalibration.from_artifact(source)
        except CalibrationError as error:
            return CameraCalibration.invalid(str(error))

    matrix = data.get("K", data.get("matrix", data.get("camera_matrix")))
    provenance = data.get("provenance", f"user-supplied calibration file: {source}")
    if not isinstance(provenance, str):
        provenance = f"user-supplied calibration file: {source}"
    distortion = data.get("distortion_coefficients", data.get("distortion"))
    image_size = data.get("image_size")
    if image_size is None and "image_width" in data and "image_height" in data:
        image_size = [data.get("image_width"), data.get("image_height")]
    if isinstance(image_size, list):
        image_size = tuple(image_size)
    try:
        return CameraCalibration.from_intrinsics(
            matrix, provenance, distortion=distortion, image_size=image_size,
            calibration_id=data.get("calibration_id"))
    except (TypeError, ValueError) as error:
        return CameraCalibration.invalid(str(error))


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Bounded sequential ORB visual-odometry audit replay.")
    parser.add_argument("--clip", default="clip_01",
                        help="clip id under clips/ (used when --video is omitted)")
    parser.add_argument("--video", default=None, help="explicit video path")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--start-ns", type=int, default=0,
                        help="capture-order origin; not a hardware timestamp")
    parser.add_argument("--calibration", default=None,
                        help="validated prism.calibration artifact, or a legacy "
                             "JSON file with a 3x3 K (or matrix) and provenance")
    parser.add_argument("--assumed-intrinsics", action="store_true",
                        help="use CFG.cam.K as an explicitly unvalidated assumption")
    parser.add_argument("--json", default=None,
                        help="write the deterministic summary to this JSON path")
    args = parser.parse_args(argv)

    if args.calibration:
        calibration = _load_calibration(args.calibration)
    elif args.assumed_intrinsics:
        calibration = CameraCalibration.assumed_from_config()
    else:
        calibration = CameraCalibration.missing()

    video = Path(args.video) if args.video else io_utils.clip_path(args.clip)
    report = VideoOdometryReplayHarness(calibration=calibration).replay(
        video, max_frames=args.max_frames, capture_start_ns=args.start_ns)

    summary = report.summary(include_latency=True)
    print(f"video: {video}")
    print(f"calibration: {calibration.status.value} - {calibration.provenance}")
    if not calibration.calibrated:
        print(f"requirement: {calibration.requirement}")
    print(f"decoded {report.decoded_frames}/{report.declared_frame_count} frames, "
          f"processed {report.processed_frames}, fps {report.fps:g}")
    print(f"tracking: {report.successful_tracking_frames}/"
          f"{report.tracking_attempts} attempts "
          f"({100.0 * report.tracking_success_rate:.1f}%), "
          f"losses {report.tracking_losses}")
    print(f"relative motion frames: calibrated {report.relative_motion_frames}, "
          f"unvalidated {report.unvalidated_motion_frames}, "
          f"tracking-only {report.tracking_only_frames}")
    print(f"features mean {report.mean_features:.1f} | matches mean "
          f"{report.mean_matches:.1f} | inliers mean {report.mean_inliers:.1f} | "
          f"inlier ratio {report.mean_inlier_ratio:.3f}")
    lat = report.latency_stats
    print(f"latency ms: mean {lat['mean_ms']:.2f} p50 {lat['p50_ms']:.2f} "
          f"p95 {lat['p95_ms']:.2f} max {lat['max_ms']:.2f}")
    print(f"trajectory produced: {report.trajectory_produced}; "
          f"metric scale meaningful: {report.metric_scale_meaningful}")
    if report.errors:
        for error in report.errors:
            print(f"error: {error}")
    if report.loss_events:
        first = report.loss_events[0]
        print(f"first loss event: frame {first.frame_index} "
              f"{first.status.value}: {first.reason}")
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
