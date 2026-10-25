"""Model-free tests for the OpenCV checkerboard calibration workflow."""
from pathlib import Path
import json

import cv2
import numpy as np
import pytest

from prism.calibration import (
    CALIBRATION_SCHEMA,
    CalibrationError,
    CalibrationPattern,
    calibrate_checkerboard,
    load_calibration_artifact,
)
from prism.visual_odometry_replay import (
    CalibrationStatus,
    CameraCalibration,
    MotionStatus,
    _load_calibration,
    estimate_relative_motion,
)


K_DRAW = np.array([[620.0, 0.0, 320.0],
                   [0.0, 620.0, 240.0],
                   [0.0, 0.0, 1.0]], np.float64)
K_MOTION = np.array([[500.0, 0.0, 320.0],
                     [0.0, 500.0, 180.0],
                     [0.0, 0.0, 1.0]], np.float64)


def _checkerboard_texture(columns=9, rows=6, scale=100, margin=100):
    squares_x, squares_y = columns + 1, rows + 1
    width = squares_x * scale + 2 * margin
    height = squares_y * scale + 2 * margin
    texture = np.full((height, width), 255, np.uint8)
    for row in range(squares_y):
        for column in range(squares_x):
            if (row + column) % 2 == 0:
                x0 = margin + column * scale
                y0 = margin + row * scale
                texture[y0:y0 + scale, x0:x0 + scale] = 0
    return texture, margin, scale


def _write_checkerboard_images(directory: Path) -> None:
    texture, margin, scale = _checkerboard_texture()
    translate = np.array([[1.0 / scale, 0.0, -margin / scale],
                          [0.0, 1.0 / scale, -margin / scale],
                          [0.0, 0.0, 1.0]], np.float64)
    board_center = np.array([5.0, 3.5, 0.0])
    poses = [
        (-0.22, 0.08, -0.08, -2.0, -1.0, 20.0),
        (0.18, -0.22, 0.10, 1.8, 0.2, 19.5),
        (-0.12, 0.20, 0.16, -1.6, 1.2, 21.0),
        (0.24, 0.12, -0.18, 1.2, -1.4, 20.5),
        (0.04, -0.18, 0.24, -2.2, 0.4, 19.0),
        (-0.20, -0.08, 0.12, 2.0, 1.0, 21.5),
        (0.14, 0.24, -0.20, -0.8, -1.6, 20.2),
        (-0.16, 0.04, 0.20, 1.6, 1.4, 19.8),
        (0.20, -0.10, -0.14, -1.8, -0.2, 20.8),
        (-0.06, 0.16, -0.22, 0.6, 0.6, 19.2),
        (0.10, -0.20, 0.04, 1.8, -0.6, 21.2),
        (-0.18, 0.12, 0.08, -1.4, 1.6, 20.4),
    ]
    directory.mkdir(parents=True, exist_ok=True)
    for index, (rx, ry, rz, tx, ty, tz) in enumerate(poses):
        rotation = cv2.Rodrigues(np.array([rx, ry, rz]))[0]
        translation = np.array([tx, ty, tz]) - rotation @ board_center
        plane_homography = K_DRAW @ np.column_stack(
            [rotation[:, 0], rotation[:, 1], translation])
        homography = plane_homography @ translate
        image = cv2.warpPerspective(
            texture, homography, (640, 480), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        cv2.imwrite(str(directory / f"view_{index:02d}.png"), image)


def _synthetic_correspondences(n=200, seed=7):
    rng = np.random.default_rng(seed)
    z = rng.uniform(2.0, 8.0, n)
    x = rng.uniform(-0.50, 0.50, n) * z
    y = rng.uniform(-0.30, 0.30, n) * z
    points = np.stack([x, y, z], axis=1)
    rotation = cv2.Rodrigues(np.array([0.010, -0.020, 0.005]))[0]
    translation = np.array([0.08, -0.02, 0.30], np.float64)

    def project(rotation_matrix, translation_vector):
        camera = points @ rotation_matrix.T + translation_vector.reshape(1, 3)
        uv = camera @ K_MOTION.T
        return uv[:, :2] / uv[:, 2:3], camera[:, 2]

    p0, z0 = project(np.eye(3), np.zeros(3))
    p1, z1 = project(rotation, translation)
    keep = (z0 > 0) & (z1 > 0)
    return p0[keep], p1[keep]


@pytest.fixture(scope="module")
def calibrated_artifact(tmp_path_factory):
    directory = tmp_path_factory.mktemp("checkerboard")
    _write_checkerboard_images(directory)
    pattern = CalibrationPattern(9, 6, 25.0, "mm")
    output = directory / "calibration.json"
    artifact = calibrate_checkerboard(
        directory, pattern, min_views=8, max_rms_px=1.0, save_path=output)
    return directory, output, artifact


def test_checkerboard_calibration_produces_a_valid_versioned_artifact(
        calibrated_artifact):
    directory, output, artifact = calibrated_artifact

    assert output.is_file()
    assert artifact.schema == CALIBRATION_SCHEMA
    assert artifact.artifact_version == 1
    assert artifact.valid is True
    assert artifact.detected_views == 12
    assert artifact.image_size == (640, 480)
    assert artifact.pattern.columns == 9 and artifact.pattern.rows == 6
    assert artifact.pattern.units == "mm"
    assert 600.0 < artifact.camera_matrix[0, 0] < 640.0
    assert 600.0 < artifact.camera_matrix[1, 1] < 640.0
    assert abs(artifact.camera_matrix[0, 2] - 320.0) < 5.0
    assert abs(artifact.camera_matrix[1, 2] - 240.0) < 5.0
    assert artifact.rms_px < 0.5
    assert artifact.max_error_px >= artifact.p95_error_px >= artifact.median_error_px
    assert artifact.mean_error_px <= artifact.rms_px
    assert all(view.detected for view in artifact.views)
    assert all(view.mean_reprojection_error_px is not None
               for view in artifact.views)
    assert all(len(view.sha256) == 64 for view in artifact.views)

    loaded = load_calibration_artifact(output)
    assert loaded.calibration_id == artifact.calibration_id
    assert np.allclose(loaded.camera_matrix, artifact.camera_matrix)
    assert np.allclose(loaded.distortion_coefficients,
                       artifact.distortion_coefficients)
    assert len(list(directory.glob("view_*.png"))) == 12


def test_artifact_loads_into_camera_calibration_and_scales_intrinsics(
        calibrated_artifact):
    _, output, artifact = calibrated_artifact

    calibration = CameraCalibration.from_artifact(output)

    assert calibration.status is CalibrationStatus.CALIBRATED
    assert calibration.calibrated is True
    assert calibration.image_size == (640, 480)
    assert calibration.distortion is not None
    assert calibration.calibration_id == artifact.calibration_id
    assert "RMS" in calibration.provenance

    scaled = calibration.for_frame_size(320, 240)
    assert scaled.status is CalibrationStatus.CALIBRATED
    assert scaled.image_size == (320, 240)
    assert np.isclose(scaled.matrix[0, 0], artifact.camera_matrix[0, 0] / 2.0)
    assert np.isclose(scaled.matrix[1, 1], artifact.camera_matrix[1, 1] / 2.0)
    assert np.isclose(scaled.matrix[0, 2], artifact.camera_matrix[0, 2] / 2.0)
    assert np.isclose(scaled.matrix[1, 2], artifact.camera_matrix[1, 2] / 2.0)
    assert "scaled from 640x480 to 320x240" in scaled.provenance

    incompatible = calibration.for_frame_size(640, 360)
    assert incompatible.status is CalibrationStatus.INVALID
    assert "aspect ratio" in incompatible.provenance

    loaded_by_cli = _load_calibration(str(output))
    assert loaded_by_cli.status is CalibrationStatus.CALIBRATED
    assert loaded_by_cli.image_size == (640, 480)
    assert loaded_by_cli.calibration_id == artifact.calibration_id


def test_tampered_artifact_is_rejected(calibrated_artifact):
    _, output, _ = calibrated_artifact
    data = json.loads(output.read_text(encoding="utf-8"))
    data["camera_matrix"][0][0] += 1.0
    tampered = output.with_name("tampered.json")
    tampered.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(CalibrationError, match="calibration_id"):
        load_calibration_artifact(tampered)


def test_invalid_calibration_is_not_written(calibrated_artifact):
    directory, _, _ = calibrated_artifact
    output = directory / "too_strict.json"
    pattern = CalibrationPattern(9, 6, 25.0, "mm")

    with pytest.raises(CalibrationError, match="refusing to save"):
        calibrate_checkerboard(
            directory, pattern, max_rms_px=1e-6, save_path=output)

    assert not output.exists()


def test_distortion_coefficients_are_validated_and_applied():
    p0, p1 = _synthetic_correspondences()

    estimate = estimate_relative_motion(
        p0, p1, K_MOTION, distortion=np.zeros(5))
    assert estimate.status is MotionStatus.OK
    assert estimate.translation_unit is not None

    invalid = estimate_relative_motion(
        p0, p1, K_MOTION, distortion=np.array([np.nan, 0.0, 0.0, 0.0, 0.0]))
    assert invalid.status is MotionStatus.INVALID_CALIBRATION
    assert invalid.translation_unit is None
