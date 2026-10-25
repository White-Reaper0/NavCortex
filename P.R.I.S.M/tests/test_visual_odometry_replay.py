"""Model-free tests for the bounded sequential visual-odometry audit replay."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from prism.visual_odometry_replay import (
    CameraCalibration,
    CalibrationStatus,
    MotionStatus,
    VideoOdometryReplayHarness,
    estimate_relative_motion,
)


ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "clips" / "clip_01.mp4"
K = np.array([[500.0, 0.0, 320.0],
              [0.0, 500.0, 180.0],
              [0.0, 0.0, 1.0]], np.float64)


def _project(points, R, t):
    camera = points @ R.T + np.asarray(t, np.float64).reshape(1, 3)
    uv = camera @ K.T
    return uv[:, :2] / uv[:, 2:3], camera[:, 2]


def _synthetic_correspondences(n=200, seed=7):
    rng = np.random.default_rng(seed)
    z = rng.uniform(2.0, 8.0, n)
    x = rng.uniform(-0.50, 0.50, n) * z
    y = rng.uniform(-0.30, 0.30, n) * z
    points = np.stack([x, y, z], axis=1)
    R = cv2.Rodrigues(np.array([0.010, -0.020, 0.005]))[0]
    t = np.array([0.08, -0.02, 0.30], np.float64)
    p0, z0 = _project(points, np.eye(3), np.zeros(3))
    p1, z1 = _project(points, R, t)
    keep = (z0 > 0) & (z1 > 0)
    return p0[keep], p1[keep], R, t


def test_valid_motion_estimates_unit_translation_and_rotation():
    p0, p1, R_true, t_true = _synthetic_correspondences()

    estimate = estimate_relative_motion(p0, p1, K)

    assert estimate.status is MotionStatus.OK
    assert estimate.motion_valid and estimate.translation_unit is not None
    assert estimate.n_inliers >= 180
    assert np.isclose(np.linalg.norm(estimate.translation_unit), 1.0)
    direction = t_true / np.linalg.norm(t_true)
    assert float(estimate.translation_unit @ direction) > 0.95
    assert np.linalg.norm(estimate.rotation - R_true) < 0.05
    assert "arbitrary scale" in estimate.translation_scale


def test_insufficient_correspondences_are_reported_not_estimated():
    p0, p1, _, _ = _synthetic_correspondences(n=10)

    estimate = estimate_relative_motion(p0, p1, K)

    assert estimate.status is MotionStatus.INSUFFICIENT_MATCHES
    assert estimate.translation_unit is None
    assert not estimate.motion_valid


def test_geometric_outliers_are_rejected_by_robust_inlier_count():
    p0, p1, _, _ = _synthetic_correspondences(n=160)
    rng = np.random.default_rng(11)
    outlier_p0 = rng.uniform([0.0, 0.0], [640.0, 360.0], size=(80, 2))
    outlier_p1 = rng.uniform([0.0, 0.0], [640.0, 360.0], size=(80, 2))

    estimate = estimate_relative_motion(
        np.vstack([p0, outlier_p0]), np.vstack([p1, outlier_p1]), K)

    assert estimate.status is MotionStatus.OK
    assert estimate.n_matches == 240
    assert 140 <= estimate.n_inliers < estimate.n_matches
    assert 0.5 < estimate.inlier_ratio < 1.0


def test_invalid_calibration_is_rejected_at_construction_and_in_replay():
    with pytest.raises(ValueError):
        CameraCalibration.from_intrinsics(np.zeros((3, 3)), "no focal lengths")

    bad = CameraCalibration.invalid("calibration file has no fx")
    report = VideoOdometryReplayHarness(calibration=bad).replay("synthetic.mp4")

    assert report.opened is False
    assert report.decoded_frames == 0 and report.records == ()
    assert "invalid calibration" in report.errors[0]
    assert report.summary()["calibration"]["status"] == "invalid"


class _FakeCapture:
    def __init__(self, frames, fps=30.0):
        self._frames = iter(frames)
        self._fps = fps
        self._count = len(frames)
        self.released = False

    def isOpened(self):
        return True

    def get(self, property_id):
        values = {
            cv2.CAP_PROP_FPS: self._fps,
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 360.0,
            cv2.CAP_PROP_FRAME_COUNT: float(self._count),
        }
        return values[property_id]

    def read(self):
        try:
            return True, next(self._frames)
        except StopIteration:
            return False, None

    def release(self):
        self.released = True


def _blank_frame():
    return np.zeros((360, 640, 3), np.uint8)


def _textured_frame(seed=1):
    return np.random.default_rng(seed).integers(0, 255, (360, 640, 3), np.uint8)


def test_feature_starvation_reports_explicit_tracking_loss_events():
    frames = [_textured_frame(), _blank_frame(), _blank_frame()]
    fake = _FakeCapture(frames)
    harness = VideoOdometryReplayHarness(capture_factory=lambda _path: fake)

    report = harness.replay("synthetic.mp4")

    assert fake.released
    assert report.decoded_frames == 3
    assert report.records[0].status is MotionStatus.FIRST_FRAME
    assert all(record.status is MotionStatus.INSUFFICIENT_FEATURES
               for record in report.records[1:])
    assert report.tracking_losses == 2
    assert len(report.loss_events) == 2
    assert report.tracking_only is True
    assert report.relative_motion_frames == 0
    assert report.trajectory_produced is False


def test_status_reporting_is_deterministic_and_timestamps_are_decode_order():
    frames = [_blank_frame() for _ in range(4)]

    def run():
        fake = _FakeCapture(frames)
        return fake, VideoOdometryReplayHarness(
            capture_factory=lambda _path: fake).replay(
                "synthetic.mp4", capture_start_ns=10_000)

    fake_a, first = run()
    fake_b, second = run()

    assert fake_a.released and fake_b.released
    assert first.summary(include_latency=False) == second.summary(include_latency=False)
    assert [record.status for record in first.records] == [
        MotionStatus.FIRST_FRAME,
        MotionStatus.INSUFFICIENT_FEATURES,
        MotionStatus.INSUFFICIENT_FEATURES,
        MotionStatus.INSUFFICIENT_FEATURES,
    ]
    assert [record.capture_order_ns for record in first.records] == [
        10_000, 33_343_333, 66_676_667, 100_010_000,
    ]
    assert first.summary()["trajectory_produced"] is False
    assert first.summary()["translation_scale"].startswith("arbitrary scale")


def test_missing_calibration_is_reported_and_real_clip_decodes_sequentially():
    assert CLIP.is_file()
    report = VideoOdometryReplayHarness().replay(CLIP, max_frames=2)

    assert report.calibration.status is CalibrationStatus.MISSING
    assert "camera matrix" in report.calibration.requirement
    assert report.decoded_frames == 2
    assert [record.frame_index for record in report.records] == [0, 1]
    assert report.records[0].capture_order_ns == 0
    assert report.records[1].capture_order_ns == 33_333_333
    assert report.relative_motion_frames == 0
    assert report.metric_scale_meaningful is False
