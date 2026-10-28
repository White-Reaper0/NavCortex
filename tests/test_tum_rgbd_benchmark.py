"""Model-free tests for the TUM RGB-D benchmark adapter and metrics."""
from pathlib import Path
import tarfile

import numpy as np
import pytest

from prism.benchmarks.reference_geometry import (
    ReferenceFrame,
    camera_to_vehicle,
    cross_view_depth_consistency,
    cross_view_photometric_consistency,
    depth_frame_stats,
    fit_support_plane,
    intrinsics_sensitivity,
)
from prism.benchmarks.trajectory_evaluation import (
    evaluate_trajectories,
    quaternion_to_rotation_xyzw,
    umeyama_sim3,
)
from prism.benchmarks.tum_rgbd_adapter import (
    FREIBURG1_RGB_INTRINSICS,
    FREIBURG1_ROS_DEFAULT_INTRINSICS,
    TimestampedPath,
    TumPose,
    associate_groundtruth_poses,
    associate_timestamp_lists,
    backproject_dense,
    depth_to_metres,
    depth_to_points,
    download_archive,
    extract_archive,
    freiburg1_calibration,
    freiburg1_rgb_processing_calibration,
    groundtruth_positions_at_timestamps,
    load_sequence_layout,
    sequence_integrity,
)
from prism.visual_odometry_replay import CalibrationStatus


def test_associate_timestamp_lists_is_one_to_one():
    rgb = (
        TimestampedPath(0.000, "a.png"),
        TimestampedPath(0.010, "b.png"),
        TimestampedPath(0.020, "c.png"),
    )
    depth = (TimestampedPath(0.005, "d0.png"),)
    pairs = associate_timestamp_lists(rgb, depth, max_difference_s=0.02)
    assert len(pairs) == 1
    assert pairs[0][0].relative_path == "a.png"
    assert pairs[0][1].relative_path == "d0.png"


def test_associate_timestamp_lists_chooses_globally_smallest_differences():
    first = (TimestampedPath(0.000, "a.png"), TimestampedPath(0.030, "b.png"))
    second = (TimestampedPath(0.001, "d0.png"), TimestampedPath(0.029, "d1.png"))
    pairs = associate_timestamp_lists(first, second, max_difference_s=0.03)
    assert [pair[0].relative_path for pair in pairs] == ["a.png", "b.png"]
    assert [pair[1].relative_path for pair in pairs] == ["d0.png", "d1.png"]


def test_processing_calibration_matches_harness_resize():
    calibration = freiburg1_rgb_processing_calibration()
    assert calibration.status is CalibrationStatus.CALIBRATED
    assert calibration.image_size == (640, 360)
    K = calibration.matrix
    assert K[0, 0] == pytest.approx(FREIBURG1_RGB_INTRINSICS["fx"])
    assert K[0, 2] == pytest.approx(FREIBURG1_RGB_INTRINSICS["cx"])
    assert K[1, 1] == pytest.approx(FREIBURG1_RGB_INTRINSICS["fy"] * 360.0 / 480.0)
    assert K[1, 2] == pytest.approx(FREIBURG1_RGB_INTRINSICS["cy"] * 360.0 / 480.0)
    assert calibration.distortion.tolist() == list(FREIBURG1_RGB_INTRINSICS["distortion"])


def test_native_calibrations_are_reported_with_official_values():
    rgb = freiburg1_calibration(ros_default=False)
    ros = freiburg1_calibration()
    assert rgb.image_size == (640, 480)
    assert rgb.matrix[0, 0] == pytest.approx(517.3)
    assert rgb.matrix[1, 1] == pytest.approx(516.5)
    assert ros.matrix[0, 0] == pytest.approx(525.0)
    assert ros.matrix[1, 2] == pytest.approx(239.5)


def test_groundtruth_pose_association_and_nan_fallback():
    gt = (
        TumPose(1.0, (0.0, 1.0, 2.0), (0.0, 0.0, 0.0, 1.0)),
        TumPose(1.1, (1.0, 1.0, 2.0), (0.0, 0.0, 0.0, 1.0)),
    )
    poses = associate_groundtruth_poses(gt, (1.005, 1.5), max_difference_s=0.02)
    assert poses[0] is not None and poses[0].position == (0.0, 1.0, 2.0)
    assert poses[1] is None
    positions = groundtruth_positions_at_timestamps(gt, (1.005, 1.5), max_difference_s=0.02)
    assert np.isnan(positions[1][0])


def test_quaternion_to_rotation_identity_and_90_degrees():
    assert np.allclose(quaternion_to_rotation_xyzw(0, 0, 0, 1), np.eye(3))
    half = np.sqrt(0.5)
    R = quaternion_to_rotation_xyzw(0, 0, half, half)
    assert np.allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])


def test_depth_conversion_and_backprojection():
    depth = np.array([[0, 5000], [2500, 10000]], dtype=np.uint16)
    z = depth_to_metres(depth)
    assert np.isnan(z[0, 0])
    assert z[0, 1] == pytest.approx(1.0)
    assert z[1, 0] == pytest.approx(0.5)

    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 0.0, "cy": 0.0}
    points = depth_to_points(depth, intrinsics=intrinsics, stride=1)
    assert points.shape == (3, 3)
    assert np.allclose(
        sorted(map(tuple, points)),
        sorted([(0.0, 0.005, 0.5), (0.02, 0.02, 2.0), (0.01, 0.0, 1.0)]),
        atol=1e-12,
    )

    dense, valid = backproject_dense(depth, intrinsics=intrinsics)
    assert not valid[0, 0]
    assert np.isnan(dense[0, 0]).all()
    assert dense[1, 1].tolist() == pytest.approx([0.02, 0.02, 2.0])
    assert depth_frame_stats(depth).valid_pixels == 3


def test_intrinsics_sensitivity_is_zero_for_identical_intrinsics():
    depth = np.full((24, 32), 2000, dtype=np.uint16)
    result = intrinsics_sensitivity(
        depth, stride=2, primary=FREIBURG1_ROS_DEFAULT_INTRINSICS,
        alternate=FREIBURG1_ROS_DEFAULT_INTRINSICS)
    assert result.compared_pixels > 0
    assert result.median_distance_m == 0.0
    assert result.max_distance_m == 0.0


def test_umeyama_sim3_recovers_known_scale_and_transform():
    src = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    scale_true = 2.0
    rot_true = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    trans_true = np.array([0.5, -0.25, 0.1])
    dst = (scale_true * (src @ rot_true.T)) + trans_true
    scale, rot, trans = umeyama_sim3(src, dst)
    est = (scale * (src @ rot.T)) + trans
    assert np.allclose(est, dst, atol=1e-6)
    assert abs(scale - scale_true) < 1e-6


def _non_collinear_path(count: int = 10) -> np.ndarray:
    t = np.linspace(0.0, 1.0, count)
    return np.stack([t, 0.3 * np.sin(2.0 * np.pi * t), 0.1 * t ** 2], axis=1)


def test_evaluate_trajectories_identical_paths_with_rotations():
    path = _non_collinear_path()
    rotations = np.repeat(np.eye(3)[None, :, :], len(path), axis=0)
    metrics = evaluate_trajectories(
        path, path, with_sim3_scale=True,
        estimated_rotations=rotations, reference_rotations=rotations)
    assert metrics.ate_rmse_m == pytest.approx(0.0, abs=1e-12)
    assert metrics.ate_rot_rmse_deg == pytest.approx(0.0, abs=1e-9)
    assert metrics.rpe_rot_rmse_deg == pytest.approx(0.0, abs=1e-9)


def test_evaluate_trajectories_recovers_sim3_scale():
    reference = _non_collinear_path()
    estimated = 0.25 * reference
    metrics = evaluate_trajectories(estimated, reference, with_sim3_scale=True)
    assert metrics.scale == pytest.approx(4.0, rel=1e-6)
    assert metrics.ate_rmse_m == pytest.approx(0.0, abs=1e-9)


def test_evaluate_trajectories_measures_rotation_error():
    reference = _non_collinear_path()
    estimated_rotations = np.repeat(np.eye(3)[None, :, :], len(reference), axis=0)
    reference_rotations = estimated_rotations.copy()
    for index in range(len(reference)):
        angle = np.deg2rad(2.0 * index)
        reference_rotations[index] = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
    metrics = evaluate_trajectories(
        reference, reference, with_sim3_scale=True,
        estimated_rotations=estimated_rotations,
        reference_rotations=reference_rotations)
    assert metrics.rpe_rot_rmse_deg == pytest.approx(2.0, abs=1e-6)
    assert metrics.ate_rot_rmse_deg > 0.0


def test_evaluate_trajectories_removes_only_global_orientation_offset():
    path = _non_collinear_path()
    angle = np.deg2rad(73.0)
    axis = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    global_rotation = (
        np.eye(3) * np.cos(angle)
        + np.sin(angle) * cross
        + (1.0 - np.cos(angle)) * np.outer(axis, axis))
    estimated_rotations = np.repeat(np.eye(3)[None], len(path), axis=0)
    reference_rotations = np.repeat(global_rotation[None], len(path), axis=0)
    metrics = evaluate_trajectories(
        path, path, with_sim3_scale=True,
        estimated_rotations=estimated_rotations,
        reference_rotations=reference_rotations)
    assert metrics.ate_rot_rmse_deg == pytest.approx(0.0, abs=1e-6)
    assert metrics.ate_rot_alignment_deg == pytest.approx(73.0, abs=1e-6)
    assert metrics.alignment_rotation_deg == pytest.approx(0.0, abs=1e-6)


def test_ate_rotation_increases_with_frame_count_due_to_drift():
    """ATE rotation accumulates drift over more frames; RPE does not."""
    path = _non_collinear_path()
    n_short = 6
    n_long = len(path)

    drift_per_frame_deg = 0.5
    estimated_rotations_short = np.array([
        _rotation_matrix_z(np.deg2rad(i * drift_per_frame_deg))
        for i in range(n_short)
    ])
    reference_rotations_short = np.repeat(np.eye(3)[None], n_short, axis=0)

    estimated_rotations_long = np.array([
        _rotation_matrix_z(np.deg2rad(i * drift_per_frame_deg))
        for i in range(n_long)
    ])
    reference_rotations_long = np.repeat(np.eye(3)[None], n_long, axis=0)

    metrics_short = evaluate_trajectories(
        path[:n_short], path[:n_short], with_sim3_scale=True,
        estimated_rotations=estimated_rotations_short,
        reference_rotations=reference_rotations_short)
    metrics_long = evaluate_trajectories(
        path, path, with_sim3_scale=True,
        estimated_rotations=estimated_rotations_long,
        reference_rotations=reference_rotations_long)

    assert metrics_long.ate_rot_rmse_deg > metrics_short.ate_rot_rmse_deg
    assert metrics_short.rpe_rot_rmse_deg == pytest.approx(
        metrics_long.rpe_rot_rmse_deg, rel=0.1)


def test_rpe_rotation_is_alignment_free():
    """RPE rotation measures local consistency without global alignment."""
    path = _non_collinear_path()
    angle = np.deg2rad(45.0)
    global_rotation = _rotation_matrix_z(angle)

    estimated_rotations = np.repeat(np.eye(3)[None], len(path), axis=0)
    reference_rotations = np.repeat(global_rotation[None], len(path), axis=0)

    metrics = evaluate_trajectories(
        path, path, with_sim3_scale=True,
        estimated_rotations=estimated_rotations,
        reference_rotations=reference_rotations)

    assert metrics.rpe_rot_rmse_deg == pytest.approx(0.0, abs=1e-6)
    assert metrics.ate_rot_alignment_deg == pytest.approx(45.0, abs=1e-6)


def test_orientation_metrics_handle_missing_rotations():
    """Orientation metrics are NaN when rotations are not provided."""
    path = _non_collinear_path()
    metrics = evaluate_trajectories(path, path, with_sim3_scale=True)
    assert np.isnan(metrics.ate_rot_rmse_deg)
    assert np.isnan(metrics.rpe_rot_rmse_deg)
    assert np.isnan(metrics.ate_rot_alignment_deg)


def _rotation_matrix_z(angle_rad: float) -> np.ndarray:
    """Rotation matrix about Z-axis."""
    return np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0.0],
        [np.sin(angle_rad), np.cos(angle_rad), 0.0],
        [0.0, 0.0, 1.0],
    ])


def test_evaluate_trajectories_degenerate_input_returns_nan():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    metrics = evaluate_trajectories(points, points)
    assert metrics.degenerate
    assert np.isnan(metrics.ate_rmse_m)
    assert metrics.association_count == 2


def test_support_plane_fit_and_vehicle_transform():
    height, width = 120, 160
    v, u = np.mgrid[0:height, 0:width]
    depth = 0.5 + 2.0 * (v / height)
    z = depth
    x = (u - width / 2.0) * z / 100.0
    y = np.ones_like(z)
    points = np.stack([x, y, z], axis=-1).astype(np.float64)
    valid = np.ones((height, width), bool)
    plane = fit_support_plane(points, valid)
    assert plane.ok
    assert plane.normal_camera[1] > 0.99
    assert plane.height_m == pytest.approx(1.0, abs=1e-6)
    assert plane.inlier_fraction > 0.99

    points_veh, mask, geometry = camera_to_vehicle(points, valid, plane)
    assert mask.all()
    assert geometry.valid_pixels == height * width
    assert geometry.height_above_support_median_m == pytest.approx(0.0, abs=1e-5)
    assert np.nanmedian(points_veh[..., 2]) == pytest.approx(0.0, abs=1e-5)


def _synthetic_cross_view_frames() -> list[ReferenceFrame]:
    height, width = 60, 80
    depth = np.full((height, width), 5000, dtype=np.uint16)
    rgb = np.full((height, width, 3), 100, dtype=np.uint8)
    pose_a = np.eye(4)
    pose_b = np.eye(4)
    pose_b[0, 3] = 0.1
    return [
        ReferenceFrame(0.0, rgb.copy(), depth.copy(), pose_a, "a.png", "a_depth.png"),
        ReferenceFrame(0.1, rgb.copy(), depth.copy(), pose_b, "b.png", "b_depth.png"),
    ]


def test_cross_view_depth_consistency_on_fronto_parallel_plane():
    frames = _synthetic_cross_view_frames()
    result = cross_view_depth_consistency(frames, stride=1, max_depth_m=5.0)
    assert result.pair_count == 1
    assert result.compared_pixels > 0
    assert result.median_abs_error_m == pytest.approx(0.0, abs=1e-12)
    assert result.within_10cm_fraction == 1.0


def test_cross_view_photometric_consistency_on_constant_images():
    frames = _synthetic_cross_view_frames()
    result = cross_view_photometric_consistency(frames, stride=1, max_depth_m=5.0)
    assert result.compared_pixels > 0
    assert result.median_abs_gray == pytest.approx(0.0, abs=1e-9)


def test_cross_view_photometric_uses_full_resolution_source_pixels():
    height, width = 60, 80
    _, u = np.mgrid[0:height, 0:width]
    rgb = np.repeat(u.astype(np.uint8)[:, :, None], 3, axis=2)
    depth = np.full((height, width), 5000, np.uint16)
    pose = np.eye(4)
    frames = [
        ReferenceFrame(0.0, rgb.copy(), depth.copy(), pose, "a.png", "a_depth.png"),
        ReferenceFrame(0.1, rgb.copy(), depth.copy(), pose, "b.png", "b_depth.png"),
    ]
    result = cross_view_photometric_consistency(frames, stride=2, max_depth_m=5.0)
    # The ROS/RGB intrinsics mismatch contributes a small ramp error; using the
    # strided column index as a pixel coordinate inflates it to ~17 gray levels.
    assert result.median_abs_gray < 8.0


def test_extract_archive_rejects_path_traversal(tmp_path: Path):
    archive_path = tmp_path / "bad.tgz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("../escape.txt")
        payload = b"not allowed"
        info.size = len(payload)
        import io
        archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="escapes destination|unsafe absolute"):
        extract_archive(archive_path, tmp_path / "extract")


def test_load_sequence_layout_and_integrity_from_minimal_fixture(tmp_path: Path):
    root = tmp_path / "rgbd_dataset_freiburg1_xyz"
    root.mkdir()
    (root / "rgb.txt").write_text("1305031453.359684 rgb/000001.png\n", encoding="utf-8")
    (root / "depth.txt").write_text("1305031453.374112 depth/000001.png\n", encoding="utf-8")
    (root / "groundtruth.txt").write_text(
        "1305031453.359684 0 0 0 0 0 0 1\n", encoding="utf-8")
    layout = load_sequence_layout(root)
    assert layout.sequence_name == root.name
    assert len(layout.rgb_depth_associations) == 1
    integrity = sequence_integrity(layout)
    assert integrity["reused_depth_entries"] == 0
    assert integrity["sampled_paths_missing"]


def test_download_archive_provenance_is_sequence_specific(tmp_path: Path):
    archive = tmp_path / "rgbd_dataset_freiburg1_rpy.tgz"
    archive.write_bytes(b"x" * 16)
    provenance = download_archive(
        tmp_path, archive_name=archive.name,
        urls=("https://example.invalid/freiburg1_rpy.tgz",), expected_bytes=16)
    assert provenance.download_status == "already_present"
    assert provenance.archive_name == archive.name
    assert provenance.source_urls == ("https://example.invalid/freiburg1_rpy.tgz",)
    assert provenance.official_archive_bytes == 16
    assert provenance.archive_size_matches_official is True


def test_extract_archive_selects_requested_sequence_not_xyz(tmp_path: Path):
    import io
    name = "rgbd_dataset_freiburg1_rpy"
    archive_path = tmp_path / f"{name}.tgz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for filename in ("rgb.txt", "depth.txt", "groundtruth.txt"):
            payload = b""
            info = tarfile.TarInfo(f"{name}/{filename}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    decoy = tmp_path / "extract" / "rgbd_dataset_freiburg1_xyz"
    decoy.mkdir(parents=True)
    for filename in ("rgb.txt", "depth.txt", "groundtruth.txt"):
        (decoy / filename).write_text("", encoding="utf-8")
    root = extract_archive(archive_path, tmp_path / "extract", sequence_name=name)
    assert root.name == name


def test_multi_sequence_summary_treats_passed_status_as_success():
    from types import SimpleNamespace
    from tools.run_prism_dataset_validation import _build_multi_sequence_summary

    reports = [{
        "command": {"sequence": sequence},
        "validation_summary": {
            "software_checks": "passed",
            "geometric_accuracy_checks": "passed",
            "rgb_only_tracking_success_rate": 0.9,
            "rgb_only_ate_rmse_m": 0.03,
            "rgb_only_sim3_scale": 0.02,
        },
        "failures": [],
    } for sequence in ("xyz", "rpy", "360")]
    summary = _build_multi_sequence_summary(
        reports, SimpleNamespace(dataset_dir="dataset", output="report.json"))
    assert summary["validation_summary"]["sequences_validated"] == 3
    assert summary["validation_summary"]["all_software_checks_passed"] is True
    assert summary["validation_summary"]["all_geometric_checks_passed"] is True
    assert [item["sequence"] for item in summary["per_sequence"]] == ["xyz", "rpy", "360"]
