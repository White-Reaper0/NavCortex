"""Checkerboard camera calibration with OpenCV.

The repository does not ship a calibration asset.  This module produces one
from user-supplied checkerboard images and writes a versioned, validated JSON
artifact that records the intrinsics, distortion coefficients, image geometry,
reprojection statistics and provenance.

The calibration does not establish a camera-to-vehicle extrinsic.  It also does
not give monocular two-view translation a metric scale.  Those remain separate
requirements.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
import hashlib
import json
import math

import cv2
import numpy as np


CALIBRATION_SCHEMA = "prism.opencv_checkerboard_calibration"
CALIBRATION_ARTIFACT_VERSION = 1
DISTORTION_MODEL = "opencv_radial_tangential_5"
DEFAULT_MIN_VIEWS = 8
DEFAULT_MAX_RMS_PX = 1.0
IMAGE_EXTENSIONS = frozenset({
    ".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp",
})

CALIBRATION_NOTE = (
    "Intrinsics and lens distortion are estimated from checkerboard views. "
    "This artifact does not contain a camera-to-vehicle extrinsic and does not "
    "make monocular two-view translation metric."
)


class CalibrationError(ValueError):
    """A calibration input, calibration result or artifact failed validation."""


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CalibrationError(f"{name} must be a positive integer")
    return value


def _finite_float(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise CalibrationError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise CalibrationError(f"{name} must be a finite number")
    return number


@dataclass(frozen=True, slots=True)
class CalibrationPattern:
    """Checkerboard inner-corner geometry and physical square size."""

    columns: int
    rows: int
    square_size: float
    units: str = "mm"

    def __post_init__(self) -> None:
        if isinstance(self.columns, bool) or not isinstance(self.columns, int) \
                or self.columns < 3:
            raise CalibrationError("pattern columns must be an integer >= 3")
        if isinstance(self.rows, bool) or not isinstance(self.rows, int) \
                or self.rows < 3:
            raise CalibrationError("pattern rows must be an integer >= 3")
        size = _finite_float(self.square_size, "square_size")
        if size <= 0.0:
            raise CalibrationError("square_size must be positive")
        units = str(self.units).strip()
        if not units:
            raise CalibrationError("pattern units must not be empty")
        object.__setattr__(self, "square_size", size)
        object.__setattr__(self, "units", units)

    @property
    def corner_count(self) -> int:
        return self.columns * self.rows

    def object_points(self) -> np.ndarray:
        points = np.zeros((self.corner_count, 3), np.float32)
        points[:, :2] = np.mgrid[0:self.columns, 0:self.rows].T.reshape(-1, 2)
        points[:, :2] *= np.float32(self.square_size)
        return points

    def as_dict(self) -> dict:
        return {
            "columns": self.columns,
            "rows": self.rows,
            "square_size": self.square_size,
            "units": self.units,
        }


@dataclass(frozen=True, slots=True)
class CalibrationView:
    """One attempted checkerboard image and its detected-corner error."""

    path: str
    sha256: str
    width: int
    height: int
    detected: bool
    corner_count: int
    mean_reprojection_error_px: Optional[float] = None
    failure_reason: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "detected": self.detected,
            "corner_count": self.corner_count,
            "mean_reprojection_error_px": self.mean_reprojection_error_px,
            "failure_reason": self.failure_reason,
        }


def _calibration_id(
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    pattern: CalibrationPattern,
    image_width: int,
    image_height: int,
) -> str:
    payload = {
        "camera_matrix": np.round(np.asarray(camera_matrix, np.float64), 12).tolist(),
        "distortion_coefficients": np.round(
            np.asarray(distortion, np.float64), 12).ravel().tolist(),
        "distortion_model": DISTORTION_MODEL,
        "image_size": [int(image_width), int(image_height)],
        "pattern": pattern.as_dict(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class CalibrationArtifact:
    """A versioned OpenCV calibration result and its audit trail."""

    artifact_version: int
    schema: str
    calibration_id: str
    created_utc: str
    opencv_version: str
    pattern: CalibrationPattern
    image_width: int
    image_height: int
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    rms_px: float
    mean_error_px: float
    median_error_px: float
    p95_error_px: float
    max_error_px: float
    views: tuple[CalibrationView, ...]
    valid: bool
    validation: dict
    provenance: dict
    notes: str = CALIBRATION_NOTE

    @property
    def image_size(self) -> tuple[int, int]:
        return self.image_width, self.image_height

    @property
    def detected_views(self) -> int:
        return sum(1 for view in self.views if view.detected)

    def as_dict(self) -> dict:
        return {
            "schema": self.schema,
            "artifact_version": self.artifact_version,
            "calibration_id": self.calibration_id,
            "created_utc": self.created_utc,
            "opencv_version": self.opencv_version,
            "distortion_model": DISTORTION_MODEL,
            "pattern": self.pattern.as_dict(),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "camera_matrix": np.asarray(self.camera_matrix, np.float64).tolist(),
            "distortion_coefficients": np.asarray(
                self.distortion_coefficients, np.float64).ravel().tolist(),
            "rms_px": self.rms_px,
            "mean_reprojection_error_px": self.mean_error_px,
            "median_reprojection_error_px": self.median_error_px,
            "p95_reprojection_error_px": self.p95_error_px,
            "max_reprojection_error_px": self.max_error_px,
            "detected_views": self.detected_views,
            "attempted_views": len(self.views),
            "views": [view.as_dict() for view in self.views],
            "valid": self.valid,
            "validation": self.validation,
            "provenance": self.provenance,
            "notes": self.notes,
        }

    def save(self, path: str | Path) -> None:
        save_calibration_artifact(self, path)


def _validated_camera_matrix(matrix: object) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (3, 3):
        raise CalibrationError("camera_matrix must have shape (3, 3)")
    if not np.all(np.isfinite(array)):
        raise CalibrationError("camera_matrix contains non-finite values")
    if array[0, 0] <= 0.0 or array[1, 1] <= 0.0:
        raise CalibrationError("camera_matrix fx and fy must be positive")
    return array.copy()


def _validated_distortion(distortion: object) -> np.ndarray:
    array = np.asarray(distortion, dtype=np.float64).ravel()
    if array.size < 4:
        raise CalibrationError("distortion_coefficients must contain at least 4 values")
    if not np.all(np.isfinite(array)):
        raise CalibrationError("distortion_coefficients contains non-finite values")
    return array.copy()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_gray(path: Path) -> np.ndarray:
    try:
        raw = np.fromfile(path, dtype=np.uint8)
    except OSError as error:
        raise CalibrationError(f"could not read calibration image {path}: {error}") from error
    image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise CalibrationError(f"could not decode calibration image {path}")
    return image


def _detect_corners(image: np.ndarray, pattern: CalibrationPattern) -> Optional[np.ndarray]:
    size = (pattern.columns, pattern.rows)
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        flags |= getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0)
        flags |= getattr(cv2, "CALIB_CB_ACCURACY", 0)
        found, corners = cv2.findChessboardCornersSB(image, size, flags=flags)
        if found and corners is not None:
            points = np.asarray(corners, np.float32).reshape(-1, 2)
            if len(points) == pattern.corner_count:
                return points

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(image, size, flags=flags)
    if not found or corners is None:
        return None
    points = np.asarray(corners, np.float32).reshape(-1, 2)
    if len(points) != pattern.corner_count:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    refined = cv2.cornerSubPix(image, points.reshape(-1, 1, 2), (11, 11), (-1, -1),
                               criteria)
    return np.asarray(refined, np.float32).reshape(-1, 2)


def collect_image_paths(inputs: Iterable[str | Path] | str | Path) -> list[Path]:
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(
                child for child in path.iterdir()
                if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS))
        elif path.is_file():
            paths.append(path)
        else:
            raise CalibrationError(f"calibration image path does not exist: {path}")
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _validation_checks(
    pattern: CalibrationPattern,
    detected_views: int,
    image_width: int,
    image_height: int,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rms_px: float,
    statistics: dict,
    min_views: int,
    max_rms_px: float,
) -> dict:
    checks = [
        {
            "name": "enough_detected_views",
            "passed": detected_views >= min_views,
            "value": detected_views,
            "threshold": min_views,
        },
        {
            "name": "rms_within_limit",
            "passed": rms_px <= max_rms_px,
            "value": rms_px,
            "threshold": max_rms_px,
        },
        {
            "name": "positive_image_geometry",
            "passed": image_width > 0 and image_height > 0,
            "value": [image_width, image_height],
            "threshold": "positive width and height",
        },
        {
            "name": "positive_focal_lengths",
            "passed": bool(camera_matrix[0, 0] > 0.0 and camera_matrix[1, 1] > 0.0),
            "value": [float(camera_matrix[0, 0]), float(camera_matrix[1, 1])],
            "threshold": "> 0",
        },
        {
            "name": "finite_calibration_arrays",
            "passed": bool(np.all(np.isfinite(camera_matrix))
                           and np.all(np.isfinite(distortion))),
            "value": True,
            "threshold": "all finite",
        },
        {
            "name": "finite_reprojection_statistics",
            "passed": all(math.isfinite(float(value)) for value in statistics.values()),
            "value": {key: float(value) for key, value in statistics.items()},
            "threshold": "all finite",
        },
        {
            "name": "valid_pattern",
            "passed": pattern.columns >= 3 and pattern.rows >= 3
            and pattern.square_size > 0.0,
            "value": pattern.as_dict(),
            "threshold": "columns >= 3, rows >= 3, square_size > 0",
        },
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "min_views": min_views,
        "max_rms_px": max_rms_px,
        "checks": checks,
    }


def calibrate_checkerboard(
    image_paths: Iterable[str | Path],
    pattern: CalibrationPattern,
    *,
    min_views: int = DEFAULT_MIN_VIEWS,
    max_rms_px: float = DEFAULT_MAX_RMS_PX,
    save_path: Optional[str | Path] = None,
) -> CalibrationArtifact:
    """Detect checkerboards, calibrate, validate and optionally save an artifact."""
    if isinstance(min_views, bool) or not isinstance(min_views, int) or min_views < 3:
        raise CalibrationError("min_views must be an integer >= 3")
    max_rms = _finite_float(max_rms_px, "max_rms_px")
    if max_rms <= 0.0:
        raise CalibrationError("max_rms_px must be positive")

    paths = collect_image_paths(image_paths)
    if len(paths) < 3:
        raise CalibrationError(
            f"at least 3 checkerboard images are required; found {len(paths)}")

    object_template = pattern.object_points()
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    views: list[CalibrationView] = []
    image_size: Optional[tuple[int, int]] = None

    for path in paths:
        image = _read_gray(path)
        height, width = image.shape[:2]
        digest = _sha256_file(path)
        corners = _detect_corners(image, pattern)
        if corners is None:
            views.append(CalibrationView(
                str(path), digest, width, height, False, 0, None,
                "checkerboard not detected"))
            continue
        if image_size is None:
            image_size = (width, height)
        elif image_size != (width, height):
            raise CalibrationError(
                "all detected checkerboard images must have the same size; "
                f"found {image_size[0]}x{image_size[1]} and {width}x{height}")
        object_points.append(object_template.copy())
        image_points.append(corners.reshape(-1, 1, 2))
        views.append(CalibrationView(
            str(path), digest, width, height, True, len(corners)))

    detected = len(image_points)
    if detected < 3:
        raise CalibrationError(
            f"checkerboard detected in only {detected} image(s); at least 3 are required")
    assert image_size is not None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    try:
        rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, image_size, None, None,
            flags=0, criteria=criteria)
    except cv2.error as error:
        raise CalibrationError(f"OpenCV calibrateCamera failed: {error}") from error

    matrix = _validated_camera_matrix(camera_matrix)
    coefficients = _validated_distortion(distortion)
    rms_px = _finite_float(rms, "rms_px")
    if rms_px < 0.0:
        raise CalibrationError("rms_px must not be negative")

    per_view_errors: list[float] = []
    all_errors: list[np.ndarray] = []
    for objp, imgp, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, matrix, coefficients)
        errors = np.linalg.norm(
            np.asarray(projected, np.float64).reshape(-1, 2)
            - np.asarray(imgp, np.float64).reshape(-1, 2), axis=1)
        per_view_errors.append(float(errors.mean()))
        all_errors.append(errors)

    if len(per_view_errors) != detected:
        raise CalibrationError("calibration returned an unexpected number of views")
    error_values = np.concatenate(all_errors)
    statistics = {
        "mean": float(error_values.mean()),
        "median": float(np.median(error_values)),
        "p95": float(np.percentile(error_values, 95)),
        "max": float(error_values.max()),
    }

    detected_index = 0
    enriched_views: list[CalibrationView] = []
    for view in views:
        if view.detected:
            enriched_views.append(replace(
                view, mean_reprojection_error_px=per_view_errors[detected_index]))
            detected_index += 1
        else:
            enriched_views.append(view)

    validation = _validation_checks(
        pattern, detected, image_size[0], image_size[1], matrix, coefficients,
        rms_px, statistics, min_views, max_rms)
    calibration_id = _calibration_id(
        matrix, coefficients, pattern, image_size[0], image_size[1])
    provenance = {
        "method": "cv2.calibrateCamera",
        "source": "user-supplied checkerboard images",
        "image_count": len(paths),
        "detected_views": detected,
        "min_views": min_views,
        "max_rms_px": max_rms,
        "square_size": pattern.square_size,
        "units": pattern.units,
        "distortion_model": DISTORTION_MODEL,
    }
    artifact = CalibrationArtifact(
        artifact_version=CALIBRATION_ARTIFACT_VERSION,
        schema=CALIBRATION_SCHEMA,
        calibration_id=calibration_id,
        created_utc=datetime.now(timezone.utc).isoformat(),
        opencv_version=cv2.__version__,
        pattern=pattern,
        image_width=image_size[0],
        image_height=image_size[1],
        camera_matrix=matrix,
        distortion_coefficients=coefficients,
        rms_px=rms_px,
        mean_error_px=statistics["mean"],
        median_error_px=statistics["median"],
        p95_error_px=statistics["p95"],
        max_error_px=statistics["max"],
        views=tuple(enriched_views),
        valid=bool(validation["passed"]),
        validation=validation,
        provenance=provenance,
    )
    if save_path is not None:
        artifact.save(save_path)
    return artifact


def save_calibration_artifact(
    artifact: CalibrationArtifact,
    path: str | Path,
) -> None:
    if not artifact.valid:
        failed = [
            check["name"] for check in artifact.validation.get("checks", ())
            if not check.get("passed")
        ]
        raise CalibrationError(
            "refusing to save an invalid calibration artifact: "
            + (", ".join(failed) if failed else "validation failed"))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(artifact.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def calibration_artifact_from_dict(data: object) -> CalibrationArtifact:
    if not isinstance(data, dict):
        raise CalibrationError("calibration artifact must be a JSON object")
    if data.get("schema") != CALIBRATION_SCHEMA:
        raise CalibrationError("unrecognized calibration artifact schema")
    if data.get("artifact_version") != CALIBRATION_ARTIFACT_VERSION:
        raise CalibrationError(
            f"unsupported calibration artifact version: "
            f"{data.get('artifact_version')!r}")
    if data.get("distortion_model") != DISTORTION_MODEL:
        raise CalibrationError("unrecognized distortion model")

    pattern_data = data.get("pattern")
    if not isinstance(pattern_data, dict):
        raise CalibrationError("calibration artifact pattern must be an object")
    pattern = CalibrationPattern(
        columns=pattern_data.get("columns"),
        rows=pattern_data.get("rows"),
        square_size=pattern_data.get("square_size"),
        units=pattern_data.get("units", ""),
    )
    width = _positive_int(data.get("image_width"), "image_width")
    height = _positive_int(data.get("image_height"), "image_height")
    matrix = _validated_camera_matrix(data.get("camera_matrix"))
    distortion = _validated_distortion(data.get("distortion_coefficients"))
    rms = _finite_float(data.get("rms_px"), "rms_px")
    mean = _finite_float(data.get("mean_reprojection_error_px"),
                         "mean_reprojection_error_px")
    median = _finite_float(data.get("median_reprojection_error_px"),
                           "median_reprojection_error_px")
    p95 = _finite_float(data.get("p95_reprojection_error_px"),
                        "p95_reprojection_error_px")
    maximum = _finite_float(data.get("max_reprojection_error_px"),
                            "max_reprojection_error_px")
    if min(rms, mean, median, p95, maximum) < 0.0:
        raise CalibrationError("reprojection statistics must not be negative")
    if not (mean <= rms <= maximum and median <= p95 <= maximum):
        raise CalibrationError("reprojection statistics are inconsistent")

    views_data = data.get("views")
    if not isinstance(views_data, list) or not views_data:
        raise CalibrationError("calibration artifact must contain views")
    views: list[CalibrationView] = []
    for item in views_data:
        if not isinstance(item, dict):
            raise CalibrationError("calibration view must be an object")
        detected = item.get("detected")
        if not isinstance(detected, bool):
            raise CalibrationError("calibration view detected flag must be boolean")
        corner_count = item.get("corner_count")
        if isinstance(corner_count, bool) or not isinstance(corner_count, int) \
                or corner_count < 0:
            raise CalibrationError("calibration view corner_count is invalid")
        error = item.get("mean_reprojection_error_px")
        if error is not None:
            error = _finite_float(error, "mean_reprojection_error_px")
            if error < 0.0:
                raise CalibrationError("view reprojection error must not be negative")
        sha = item.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise CalibrationError("calibration view sha256 is invalid")
        view_width = _positive_int(item.get("width"), "view width")
        view_height = _positive_int(item.get("height"), "view height")
        path = item.get("path")
        if not isinstance(path, str) or not path:
            raise CalibrationError("calibration view path is invalid")
        reason = item.get("failure_reason")
        if reason is not None and not isinstance(reason, str):
            raise CalibrationError("calibration view failure_reason is invalid")
        views.append(CalibrationView(
            path, sha, view_width, view_height, detected, corner_count,
            error, reason))

    validation = data.get("validation")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise CalibrationError("calibration artifact is not valid")
    if data.get("valid") is not True:
        raise CalibrationError("calibration artifact is marked invalid")
    checks = validation.get("checks")
    if not isinstance(checks, list) or not checks \
            or not all(isinstance(check, dict) and check.get("passed") is True
                       for check in checks):
        raise CalibrationError("calibration artifact validation checks did not pass")
    min_views = _positive_int(validation.get("min_views"), "validation min_views")
    if min_views < 3:
        raise CalibrationError("validation min_views must be >= 3")
    max_rms = _finite_float(validation.get("max_rms_px"), "validation max_rms_px")
    if max_rms <= 0.0:
        raise CalibrationError("validation max_rms_px must be positive")
    if rms > max_rms:
        raise CalibrationError("calibration RMS exceeds the artifact limit")

    detected_count = sum(1 for view in views if view.detected)
    if detected_count < min_views:
        raise CalibrationError("calibration artifact has too few detected views")
    if any(view.detected and view.mean_reprojection_error_px is None
           for view in views):
        raise CalibrationError("detected view is missing its reprojection error")
    if detected_count != int(data.get("detected_views", detected_count)):
        raise CalibrationError("detected view count is inconsistent")

    expected_id = _calibration_id(matrix, distortion, pattern, width, height)
    if data.get("calibration_id") != expected_id:
        raise CalibrationError("calibration_id does not match the calibrated data")
    created = data.get("created_utc")
    if not isinstance(created, str) or not created:
        raise CalibrationError("created_utc is invalid")
    opencv_version = data.get("opencv_version")
    if not isinstance(opencv_version, str) or not opencv_version:
        raise CalibrationError("opencv_version is invalid")
    provenance = data.get("provenance")
    if not isinstance(provenance, dict):
        raise CalibrationError("provenance must be an object")
    notes = data.get("notes", CALIBRATION_NOTE)
    if not isinstance(notes, str):
        raise CalibrationError("notes must be a string")

    return CalibrationArtifact(
        artifact_version=CALIBRATION_ARTIFACT_VERSION,
        schema=CALIBRATION_SCHEMA,
        calibration_id=expected_id,
        created_utc=created,
        opencv_version=opencv_version,
        pattern=pattern,
        image_width=width,
        image_height=height,
        camera_matrix=matrix,
        distortion_coefficients=distortion,
        rms_px=rms,
        mean_error_px=mean,
        median_error_px=median,
        p95_error_px=p95,
        max_error_px=maximum,
        views=tuple(views),
        valid=True,
        validation=validation,
        provenance=provenance,
        notes=notes,
    )


def load_calibration_artifact(path: str | Path) -> CalibrationArtifact:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError(
            f"could not read calibration artifact {source}: {error}") from error
    return calibration_artifact_from_dict(data)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Calibrate a pinhole camera from checkerboard images.")
    parser.add_argument("images", nargs="+",
                        help="checkerboard image files or directories")
    parser.add_argument("--columns", type=int, required=True,
                        help="number of inner corners along a row")
    parser.add_argument("--rows", type=int, required=True,
                        help="number of inner corners along a column")
    parser.add_argument("--square-size", type=float, required=True,
                        help="physical square size (used for object points)")
    parser.add_argument("--units", default="mm", help="square-size units")
    parser.add_argument("--min-views", type=int, default=DEFAULT_MIN_VIEWS)
    parser.add_argument("--max-rms-px", type=float, default=DEFAULT_MAX_RMS_PX)
    parser.add_argument("--output", required=True,
                        help="validated JSON calibration artifact to write")
    args = parser.parse_args(argv)

    pattern = CalibrationPattern(
        args.columns, args.rows, args.square_size, args.units)
    try:
        artifact = calibrate_checkerboard(
            args.images, pattern, min_views=args.min_views,
            max_rms_px=args.max_rms_px, save_path=args.output)
    except CalibrationError as error:
        print(f"calibration failed: {error}")
        return 1

    print(f"calibration: {artifact.calibration_id}")
    print(f"pattern: {pattern.columns}x{pattern.rows} inner corners, "
          f"{pattern.square_size:g} {pattern.units} squares")
    print(f"views: {artifact.detected_views}/{len(artifact.views)} detected")
    print(f"image: {artifact.image_width}x{artifact.image_height}")
    print(f"fx {artifact.camera_matrix[0, 0]:.3f} fy "
          f"{artifact.camera_matrix[1, 1]:.3f} cx "
          f"{artifact.camera_matrix[0, 2]:.3f} cy "
          f"{artifact.camera_matrix[1, 2]:.3f}")
    print(f"distortion: {np.asarray(artifact.distortion_coefficients).tolist()}")
    print(f"reprojection px: rms {artifact.rms_px:.4f} mean "
          f"{artifact.mean_error_px:.4f} median {artifact.median_error_px:.4f} "
          f"p95 {artifact.p95_error_px:.4f} max {artifact.max_error_px:.4f}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
