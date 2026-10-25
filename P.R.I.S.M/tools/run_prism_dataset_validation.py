"""Ground-truth validation harness for TUM RGB-D (offline, bounded).

This command is separate from the ``clips/clip_01.mp4`` workflow.  It runs:

1. RGB-only classical visual odometry with the published Freiburg 1 RGB
   intrinsics scaled to the harness's 640x360 processing size.
2. RGB-D *reference-input* geometry checks using benchmark depth, calibration
   and ground-truth poses (not monocular depth inference).
3. A bounded mapping-interface test through the existing observation and
   ``MappingStage`` contracts.
4. Belief Ledger / Doubt Engine conservative-behaviour checks.
5. Terrain-boundary status with honest "unvalidated" reporting.

The JSON report separates software-path checks from geometric accuracy checks
and states explicitly when a quantity is unmeasured.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from prism.belief_ledger import BeliefLedger, EvidenceRelation, HypothesisStatus
from prism.benchmarks.reference_geometry import (
    ReferenceFrame,
    camera_to_vehicle,
    cross_view_depth_consistency,
    cross_view_photometric_consistency,
    depth_frame_stats,
    fit_support_plane,
    intrinsics_sensitivity,
    point_cloud_summary,
    points_to_world,
)
from prism.benchmarks.trajectory_evaluation import (
    evaluate_trajectories,
    integrate_vo_trajectory,
    quaternion_to_rotation_xyzw,
)
from prism.benchmarks.tum_rgbd_adapter import (
    FREIBURG1_ROS_DEFAULT_INTRINSICS,
    TumPose,
    associate_groundtruth_poses,
    backproject_dense,
    depth_to_metres,
    ensure_freiburg1_xyz,
    ensure_freiburg1_rpy,
    ensure_freiburg1_360,
    freiburg1_rgb_processing_calibration,
    load_depth_png,
    sequence_integrity,
    tum_capture_factory,
)
from prism.config import (
    CFG,
    OBSTACLE,
    PROC_H,
    PROC_W,
    RISKY,
    SAFE,
    UNKNOWN,
)
from prism.doubt_engine import DoubtEngine
from prism.integrations.mapping_diagnostic import mapping_diagnostic_status
from prism.integrations.terrain import terrain_status
from prism.integrations.terrain_grid import terrain_grid_from_bev
from prism.io_utils import ego_mask
from prism.observations import CoordinateFrame, EvidencePayload, Observation, Validity
from prism.perception import geometry as geo
from prism.perception.mapping import MappingStage, cell_stats
from prism.runtime_adapter import RuntimeAdapter
from prism.visual_odometry_replay import (
    CalibrationStatus,
    MotionStatus,
    VideoOdometryReplayHarness,
)


DEFAULT_DATASET_DIR = PROJECT_ROOT / "output" / "datasets" / "tum_rgbd"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "tum_rgbd_validation_report.json"
CAPTURE_START_NS = 1_000_000_000
USED_MOTION_STATUSES = frozenset({MotionStatus.OK, MotionStatus.UNVALIDATED_MOTION})

SOFTWARE_VS_GEOMETRY_NOTE = (
    "Software checks show that files, calibration, coordinate transforms, "
    "observation contracts, mapping interfaces, and the ledger/doubt path ran. "
    "Geometric accuracy checks compare measured reference data against benchmark "
    "ground truth under declared thresholds; they are not a navigation "
    "certification and do not establish GPS-denied navigation readiness. That "
    "group also records preconditions (minimum associated poses, tracking "
    "continuity), which are not accuracy results by themselves."
)


def _json(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return [_json(item) for item in value.tolist()]
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: _json(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _stage(
    timings: list[dict],
    failures: list[str],
    name: str,
    action: Callable[[], Any],
) -> Any:
    started = perf_counter()
    try:
        result = action()
        timings.append({
            "name": name,
            "duration_s": perf_counter() - started,
            "status": "ok",
            "detail": None,
        })
        return result
    except Exception as error:  # each stage is isolated so later evidence still runs
        detail = f"{type(error).__name__}: {error}"
        timings.append({
            "name": name,
            "duration_s": perf_counter() - started,
            "status": "failed",
            "detail": detail,
        })
        failures.append(f"{name}: {detail}")
        return None


def _pose_matrix(pose: TumPose) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_to_rotation_xyzw(*pose.quaternion_xyzw)
    matrix[:3, 3] = pose.position
    return matrix


def _timestamp_ns(base_ns: int, timestamp_s: float, origin_s: float) -> int:
    return int(base_ns + round((timestamp_s - origin_s) * 1_000_000_000.0))


def _tracking_summary(report) -> dict:
    summary = report.summary(include_latency=True)
    return {
        "opened": report.opened,
        "fps": report.fps,
        "width": report.width,
        "height": report.height,
        "decoded_frames": report.decoded_frames,
        "processed_frames": report.processed_frames,
        "calibration": report.calibration.as_dict(),
        "tracking_attempts": report.tracking_attempts,
        "successful_tracking_frames": report.successful_tracking_frames,
        "tracking_success_rate": report.tracking_success_rate,
        "tracking_losses": report.tracking_losses,
        "relative_motion_frames": report.relative_motion_frames,
        "unvalidated_motion_frames": report.unvalidated_motion_frames,
        "tracking_only_frames": report.tracking_only_frames,
        "status_counts": summary["status_counts"],
        "mean_features": report.mean_features,
        "mean_matches": report.mean_matches,
        "mean_inliers": report.mean_inliers,
        "mean_inlier_ratio": report.mean_inlier_ratio,
        "latency_ms": summary["latency_ms"],
        "loss_events": summary["loss_events"],
        "errors": list(report.errors),
        "coordinate_frame": report.coordinate_frame,
        "translation_scale": report.translation_scale,
        "trajectory_produced_by_harness": report.trajectory_produced,
        "metric_scale_meaningful": report.metric_scale_meaningful,
    }


def _run_rgb_only(
    layout,
    *,
    frame_limit: int,
    fps: float,
) -> dict:
    calibration = freiburg1_rgb_processing_calibration()
    if calibration.status is not CalibrationStatus.CALIBRATED:
        raise RuntimeError(f"RGB calibration is not usable: {calibration.provenance}")
    capture = tum_capture_factory(layout.sequence_root, fps=fps)
    harness = VideoOdometryReplayHarness(
        capture_factory=lambda _: capture,
        calibration=calibration,
    )
    vo_report = harness.replay(
        str(layout.sequence_root),
        max_frames=frame_limit,
        capture_start_ns=CAPTURE_START_NS,
    )
    if not vo_report.opened:
        raise RuntimeError("visual-odometry replay could not open the TUM RGB sequence")
    if vo_report.calibration.status is CalibrationStatus.INVALID:
        raise RuntimeError(f"harness rejected calibration: {vo_report.calibration.provenance}")

    positions, rotations = integrate_vo_trajectory(vo_report.records)
    record_timestamps = [
        layout.rgb_entries[record.frame_index].timestamp_s
        if 0 <= record.frame_index < len(layout.rgb_entries) else float("nan")
        for record in vo_report.records
    ]
    gt_poses = associate_groundtruth_poses(layout.groundtruth, record_timestamps)
    reference_positions = np.asarray([
        pose.position if pose is not None
        else (float("nan"), float("nan"), float("nan"))
        for pose in gt_poses
    ], dtype=np.float64)
    reference_rotations = np.asarray([
        quaternion_to_rotation_xyzw(*pose.quaternion_xyzw) if pose is not None
        else np.eye(3, dtype=np.float64)
        for pose in gt_poses
    ], dtype=np.float64)

    metrics = evaluate_trajectories(
        positions,
        reference_positions,
        with_sim3_scale=True,
        estimated_rotations=rotations,
        reference_rotations=reference_rotations,
    )
    used_steps = sum(
        1 for record in vo_report.records[1:]
        if record.status in USED_MOTION_STATUSES
        and record.rotation is not None
        and record.translation_unit is not None
    )
    held_steps = max(0, len(vo_report.records) - 1 - used_steps)
    gt_associated = sum(1 for pose in gt_poses if pose is not None)
    tracking = _tracking_summary(vo_report)

    finite_metrics = bool(
        np.isfinite(metrics.ate_rmse_m)
        and np.isfinite(metrics.rpe_trans_rmse_m)
        and np.isfinite(metrics.ate_rot_rmse_deg)
        and np.isfinite(metrics.rpe_rot_rmse_deg)
    )
    thresholds = {
        "min_associated_poses": min(20, frame_limit),
        "min_tracking_success_rate": 0.80,
        "max_ate_rmse_m": 0.10,
        "max_ate_rmse_fraction_of_reference_path": 0.10,
        "max_rpe_translation_rmse_m": 0.05,
        "max_ate_rotation_rmse_deg": 10.0,
    }
    checks = {
        "finite_metrics": finite_metrics,
        "enough_associated_poses": metrics.association_count >= thresholds["min_associated_poses"],
        "tracking_success": tracking["tracking_success_rate"] >= thresholds["min_tracking_success_rate"],
        "ate_rmse": bool(np.isfinite(metrics.ate_rmse_m)
                         and metrics.ate_rmse_m <= thresholds["max_ate_rmse_m"]),
        "ate_path_fraction": bool(
            np.isfinite(metrics.path_length_ref_m)
            and metrics.path_length_ref_m > 0
            and metrics.ate_rmse_m <= thresholds["max_ate_rmse_fraction_of_reference_path"]
            * metrics.path_length_ref_m),
        "rpe_translation": bool(
            np.isfinite(metrics.rpe_trans_rmse_m)
            and metrics.rpe_trans_rmse_m <= thresholds["max_rpe_translation_rmse_m"]),
        "ate_rotation": bool(
            np.isfinite(metrics.ate_rot_rmse_deg)
            and metrics.ate_rot_rmse_deg <= thresholds["max_ate_rotation_rmse_deg"]),
    }
    drift_per_m = (metrics.ate_rmse_m / metrics.path_length_ref_m
                   if np.isfinite(metrics.ate_rmse_m)
                   and np.isfinite(metrics.path_length_ref_m)
                   and metrics.path_length_ref_m > 0 else float("nan"))
    return {
        "label": "RGB-only classical monocular ORB visual odometry",
        "reference_inputs": "TUM RGB images + published Freiburg 1 RGB intrinsics",
        "groundtruth_used_for": "post-hoc evaluation only (not fed to VO)",
        "calibration": calibration.as_dict(),
        "frames_requested": frame_limit,
        "frames_decoded": vo_report.decoded_frames,
        "frames_processed": vo_report.processed_frames,
        "tracking": tracking,
        "trajectory": {
            "frames": int(positions.shape[0]),
            "associated_groundtruth_poses": gt_associated,
            "finite_metric_pairs": metrics.association_count,
            "used_relative_motion_steps": used_steps,
            "held_steps_on_loss_or_invalid_motion": held_steps,
            "metric_scale_meaningful": False,
            "scale_source": "post-hoc Sim(3) alignment only",
            "rotation_error_alignment": (
                "best-fit single SO(3) rotation; independent of the Sim(3) position "
                "alignment, whose rotation is unreliable on short trajectories"),
            "monocular_scale_note": metrics.monocular_scale_note,
        },
        "trajectory_error": _json(metrics),
        "drift": {
            "ate_rmse_m": metrics.ate_rmse_m,
            "final_position_error_m": metrics.final_position_error_m,
            "ate_rmse_per_reference_metre": drift_per_m,
            "path_length_ratio_after_sim3": metrics.path_length_ratio,
            "note": (
                "Drift is measured after post-hoc Sim(3) alignment. Tracking-loss "
                "frames hold the last pose and are included in the integration."),
        },
        "thresholds": thresholds,
        "checks": checks,
        "bounded_geometric_check_passed": bool(all(checks.values())),
        "note": (
            "Monocular scale is unobservable. The reported scale factor is an "
            "evaluation alignment, not an online metric result; reported metres "
            "for the estimate are therefore conditional on that alignment."),
    }


def _select_associations(layout, *, max_reference_frames: int, stride: int):
    associations = list(layout.rgb_depth_associations)
    if stride > 1:
        associations = associations[::stride]
    return associations[:max_reference_frames]


def _load_reference_frames(layout, selected) -> tuple[list[ReferenceFrame], list[str]]:
    frames: list[ReferenceFrame] = []
    skipped: list[str] = []
    for rgb_entry, depth_entry in selected:
        pose = associate_groundtruth_poses(
            layout.groundtruth, (depth_entry.timestamp_s,))[0]
        if pose is None:
            skipped.append(f"no ground-truth pose near {depth_entry.timestamp_s:.6f}")
            continue
        rgb_path = layout.sequence_root / rgb_entry.relative_path
        depth_path = layout.sequence_root / depth_entry.relative_path
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            skipped.append(f"could not read RGB image {rgb_entry.relative_path}")
            continue
        try:
            depth = load_depth_png(depth_path)
        except (OSError, ValueError) as error:
            skipped.append(str(error))
            continue
        if bgr.shape[:2] != depth.shape:
            skipped.append(
                f"RGB/depth shape mismatch at {depth_entry.timestamp_s:.6f}: "
                f"{bgr.shape[:2]} vs {depth.shape}")
            continue
        frames.append(ReferenceFrame(
            timestamp_s=depth_entry.timestamp_s,
            rgb_bgr=bgr,
            depth_u16=depth,
            pose_T_w_c=_pose_matrix(pose),
            rgb_path=rgb_entry.relative_path,
            depth_path=depth_entry.relative_path,
        ))
    return frames, skipped


def _run_reference_geometry(frames: list[ReferenceFrame], *, depth_stride: int) -> dict:
    if not frames:
        return {
            "status": "unmeasured",
            "reason": "no loadable reference frame with an associated ground-truth pose",
        }
    first = frames[0]
    stats = [depth_frame_stats(frame.depth_u16) for frame in frames]
    valid_fractions = [item.valid_fraction for item in stats]
    medians = [item.depth_median_m for item in stats if np.isfinite(item.depth_median_m)]

    cross_depth = cross_view_depth_consistency(
        frames, max_depth_m=5.0, stride=depth_stride)
    cross_photo = cross_view_photometric_consistency(
        frames, max_depth_m=5.0, stride=depth_stride)
    sensitivity = intrinsics_sensitivity(
        first.depth_u16, stride=depth_stride, max_depth_m=4.0)

    points_cam, valid = backproject_dense(
        first.depth_u16, intrinsics=FREIBURG1_ROS_DEFAULT_INTRINSICS, max_depth_m=5.0)
    plane = fit_support_plane(points_cam, valid)
    world_points = points_to_world(points_cam, valid, first.pose_T_w_c)
    cloud = point_cloud_summary(world_points)
    _, _, vehicle_geometry = camera_to_vehicle(points_cam, valid, plane)

    thresholds = {
        "max_cross_view_median_error_m": 0.05,
        "min_cross_view_within_10cm_fraction": 0.80,
        "min_support_plane_inlier_fraction": 0.30,
        "max_support_plane_tilt_deg": 60.0,
    }
    checks = {
        "cross_view_depth_available": bool(
            cross_depth.pair_count > 0 and cross_depth.compared_pixels > 0),
        "cross_view_median_error": bool(
            np.isfinite(cross_depth.median_abs_error_m)
            and cross_depth.median_abs_error_m <= thresholds["max_cross_view_median_error_m"]),
        "cross_view_within_10cm": bool(
            np.isfinite(cross_depth.within_10cm_fraction)
            and cross_depth.within_10cm_fraction
            >= thresholds["min_cross_view_within_10cm_fraction"]),
        "support_plane_fit": bool(plane.ok),
        "support_plane_inliers": plane.inlier_fraction
        >= thresholds["min_support_plane_inlier_fraction"],
        "support_plane_tilt": bool(
            np.isfinite(plane.tilt_deg)
            and plane.tilt_deg <= thresholds["max_support_plane_tilt_deg"]),
        "point_cloud_created": cloud["points"] > 0,
        "vehicle_transform_created": vehicle_geometry.valid_pixels > 0,
    }
    return {
        "status": "measured",
        "label": "reference sensor inputs (benchmark depth + calibration + GT poses)",
        "reference_input_note": (
            "Uses benchmark-supplied registered depth and ground-truth poses. "
            "This is not proof that P.R.I.S.M. can infer metric depth from its "
            "monocular detector pipeline."),
        "frame_count": len(frames),
        "frames_skipped": [],
        "frame_timestamps_s": [frame.timestamp_s for frame in frames],
        "depth_stats": [_json(item) for item in stats],
        "depth_summary": {
            "mean_valid_fraction": float(np.mean(valid_fractions)) if valid_fractions else float("nan"),
            "min_valid_fraction": float(np.min(valid_fractions)) if valid_fractions else float("nan"),
            "max_valid_fraction": float(np.max(valid_fractions)) if valid_fractions else float("nan"),
            "median_of_frame_medians_m": float(np.median(medians)) if medians else float("nan"),
        },
        "cross_view_depth_consistency": _json(cross_depth),
        "cross_view_photometric_consistency": _json(cross_photo),
        "intrinsics_sensitivity": _json(sensitivity),
        "point_cloud_world": cloud,
        "support_plane": _json(plane),
        "vehicle_geometry": _json(vehicle_geometry),
        "thresholds": thresholds,
        "checks": checks,
        "bounded_geometric_check_passed": bool(all(checks.values())),
        "note": (
            "Cross-view consistency re-projects frame A depth into frame B using "
            "ground-truth poses. Because benchmark depth is used on both sides, it "
            "tests depth/pose/calibration plumbing, not independent absolute-depth "
            "or monocular-depth accuracy. The support plane is not a traversability "
            "label."),
    }


def _expected_bev_cells(points_veh: np.ndarray, valid: np.ndarray, stride: int) -> int:
    full_valid = valid & ego_mask(*points_veh.shape[:2])
    sampled_points = points_veh[::stride, ::stride]
    sampled_valid = full_valid[::stride, ::stride]
    # bev_indices floors every coordinate before applying its validity mask, and the
    # float->int cast emits a RuntimeWarning for NaN. Substituting zero for invalid
    # samples changes no kept cell because sampled_valid already excludes them.
    safe_points = np.where(np.isfinite(sampled_points), sampled_points, 0.0)
    row, col, keep = geo.bev_indices(safe_points, sampled_valid)
    z = sampled_points[..., 2]
    keep &= np.isfinite(z) & (z > CFG.bev.z_min) & (z < CFG.bev.z_max)
    if not keep.any():
        return 0
    cells = row[keep].astype(np.int64) * CFG.bev.W + col[keep].astype(np.int64)
    return int(np.unique(cells).size)


def _run_mapping_interface(
    frame: ReferenceFrame,
    *,
    mapping_stride: int,
) -> dict:
    points_cam, valid = backproject_dense(
        frame.depth_u16, intrinsics=FREIBURG1_ROS_DEFAULT_INTRINSICS, max_depth_m=5.0)
    plane = fit_support_plane(points_cam, valid)
    if not plane.ok:
        raise RuntimeError("support-plane fit unavailable; refusing a vehicle-frame map test")
    points_veh, vehicle_valid, vehicle_geometry = camera_to_vehicle(points_cam, valid, plane)
    expected_cells = _expected_bev_cells(points_veh, vehicle_valid, mapping_stride)

    observation = Observation(
        observation_id="tum-rgbd-reference-mapping-sample",
        source="tum-rgbd-reference-input",
        captured_at_ns=_timestamp_ns(CAPTURE_START_NS, frame.timestamp_s, frame.timestamp_s),
        coordinate_frame=CoordinateFrame.VEHICLE,
        validity=Validity.VALID,
        provenance={
            "mode": "rgbd_reference_input",
            "sequence_frame_timestamp_s": frame.timestamp_s,
            "depth_source": frame.depth_path,
            "support_plane": _json(plane),
            "note": (
                "benchmark depth transformed to the vehicle frame with the "
                "measured support plane; reference-input wiring test"),
        },
        evidence=EvidencePayload(points=points_veh),
    )
    adapter_result = RuntimeAdapter().to_packet(
        observation, packet_index=0, clip_id="tum-rgbd-reference-input")
    mapping = MappingStage(device="cpu", pixel_stride=mapping_stride)
    packet = mapping(adapter_result.packet)
    bev = packet.bev
    stats = cell_stats(bev)
    observed = np.asarray(bev.observed, bool)
    unobserved = ~observed
    unobserved_all_unknown = bool(
        unobserved.sum() == 0 or (bev.trav[unobserved] == UNKNOWN).all())
    unobserved_height_nan = bool(
        unobserved.sum() == 0 or np.isnan(bev.height[unobserved]).all())
    observed_cells = int(observed.sum())
    total_cells = int(observed.size)
    layered_grid = terrain_grid_from_bev(
        bev,
        metric_status="REFERENCE_INPUT_UNVERIFIED",
        origin_description="vehicle frame, TUM RGB-D reference-input map (interface test)",
    )
    layered_classes = layered_grid.layer("trav_class")
    layered_unknown_preserved = bool(
        unobserved.sum() == 0 or (layered_classes[unobserved] == UNKNOWN).all())
    coverage = (float(observed_cells / expected_cells) if expected_cells > 0
                else float("nan"))
    observed_fraction = observed_cells / total_cells if total_cells else 0.0
    checks = {
        "adapter_applied_points": adapter_result.audit.applied_fields == ("points",),
        "map_observed_cells": observed_cells > 0,
        "unobserved_cells_unknown": unobserved_all_unknown,
        "unobserved_height_unknown": unobserved_height_nan,
        "geometric_traversability_fallback": mapping.trav_source == "geometric",
        "unknown_areas_retained": stats["frac_unknown"] > 0.0,
        "expected_cells_reached": coverage >= 0.50 if expected_cells > 0 else False,
        "layered_grid_unknown_preserved": layered_unknown_preserved,
    }
    return {
        "status": "measured",
        "label": "bounded MappingStage interface test with reference vehicle-frame points",
        "reference_input_note": (
            "The points are benchmark-derived and converted with a measured support "
            "plane. This validates the existing map-update interface; it does not "
            "validate monocular depth, terrain traversability, or the active "
            "navigator map."),
        "adapter_applied_fields": list(adapter_result.audit.applied_fields),
        "mapping_pixel_stride": mapping_stride,
        "map_grid": {"height": int(bev.height.shape[0]), "width": int(bev.height.shape[1]),
                     "resolution_m": CFG.bev.res_m,
                     "z_range_m": [CFG.bev.z_min, CFG.bev.z_max]},
        "trav_source": mapping.trav_source,
        "observed_cells": observed_cells,
        "total_cells": total_cells,
        "expected_cells_from_points": expected_cells,
        "observed_coverage_of_expected": coverage,
        "observed_fraction": observed_fraction,
        "cell_stats": stats,
        "trav_counts": {
            "safe": int((bev.trav == SAFE).sum()),
            "risky": int((bev.trav == RISKY).sum()),
            "obstacle": int((bev.trav == OBSTACLE).sum()),
            "unknown": int((bev.trav == UNKNOWN).sum()),
        },
        "finite_height_cells": int(np.isfinite(bev.height).sum()),
        "unobserved_cells_all_unknown": unobserved_all_unknown,
        "unobserved_height_all_nan": unobserved_height_nan,
        "vehicle_geometry": _json(vehicle_geometry),
        "mapping_last_stats": _json(mapping.last_stats),
        "checks": checks,
        "software_check_passed": bool(all(checks.values())),
        "layered_terrain_grid": layered_grid.summary(),
        "traversability_accuracy": "unmeasured",
        "traversability_unmeasured_reason": (
            "indoor desk sequence has no traversable/obstacle labels or vehicle-frame "
            "traversability ground truth; the prototype grid is an interface test, "
            "not the active navigator map"),
        "note": (
            "MappingStage applies its existing ego-mask preprocessing (a clip-specific "
            "asset) to the reference frame; this only affects the interface test."),
    }


def _run_belief_doubt(frame: ReferenceFrame) -> dict:
    depth_m = depth_to_metres(frame.depth_u16).astype(np.float32)
    depth_valid = np.isfinite(depth_m)
    capture_ns = _timestamp_ns(CAPTURE_START_NS, frame.timestamp_s, frame.timestamp_s)
    expires_ns = capture_ns + 1_000_000_000

    depth_observation = Observation(
        observation_id="tum-rgbd-reference-depth-evidence",
        source="tum-rgbd-reference-input",
        captured_at_ns=capture_ns,
        coordinate_frame=CoordinateFrame.CAMERA,
        validity=Validity.VALID,
        provenance={
            "mode": "rgbd_reference_input",
            "depth_source": frame.depth_path,
            "units": "metres along the optical axis",
            "note": "benchmark-supplied depth; no confidence was reported by the dataset",
        },
        evidence=EvidencePayload(metric_depth_m=depth_m, depth_valid=depth_valid),
    )
    terrain_observation = Observation(
        observation_id="tum-rgbd-terrain-evidence-unavailable",
        source="tum-rgbd-validation",
        captured_at_ns=capture_ns,
        coordinate_frame=CoordinateFrame.LOCAL_MAP,
        validity=Validity.UNAVAILABLE,
        provenance={
            "reason": (
                "indoor desk sequence provides no validated metric terrain grid or "
                "traversability labels"),
        },
        evidence=EvidencePayload(),
    )

    ledger = BeliefLedger()
    ledger.record(
        "reference-depth-availability", depth_observation,
        EvidenceRelation.SUPPORTS, expires_at_ns=expires_ns)
    ledger.record(
        "terrain-traversability-validated", terrain_observation,
        EvidenceRelation.SUPPORTS, expires_at_ns=expires_ns)

    before_ns = capture_ns + 100_000_000
    before = DoubtEngine().assess(
        ledger, at_ns=before_ns,
        hypothesis_ids=("reference-depth-availability", "terrain-traversability-validated"))
    status_before = {
        hypothesis: ledger.hypothesis_status(hypothesis, at_ns=before_ns).value
        for hypothesis in ledger.hypotheses()
    }
    ledger.advance_to(expires_ns)
    after_ns = expires_ns + 1
    after = DoubtEngine().assess(
        ledger, at_ns=after_ns,
        hypothesis_ids=("reference-depth-availability", "terrain-traversability-validated"))
    status_after = {
        hypothesis: ledger.hypothesis_status(hypothesis, at_ns=after_ns).value
        for hypothesis in ledger.hypotheses()
    }
    before_kinds = [item.kind.value for item in before.recommendations]
    after_kinds = [item.kind.value for item in after.recommendations]
    checks = {
        "reference_depth_evidence_current": status_before.get(
            "reference-depth-availability") == HypothesisStatus.SUPPORTED.value,
        "terrain_remains_unknown": status_before.get(
            "terrain-traversability-validated") == HypothesisStatus.UNKNOWN.value,
        "terrain_requests_observation": "request_one_observation" in before_kinds,
        "expired_depth_requests_observation": "request_one_observation" in after_kinds,
        "expired_depth_cautious_stop": "cautious_stop" in after_kinds,
        "no_clearance_field": not any(
            key in _json(before) for key in ("clearance", "permission_to_move")),
    }
    return {
        "label": "existing Belief Ledger / Doubt Engine contract test",
        "reference_depth_evidence_units": "metric_depth_m (benchmark supplied)",
        "hypothesis_status_at_capture": status_before,
        "hypothesis_status_after_expiry": status_after,
        "assessment_at_capture": _json(before),
        "assessment_after_expiry": _json(after),
        "recommendation_kinds_at_capture": before_kinds,
        "recommendation_kinds_after_expiry": after_kinds,
        "checks": checks,
        "software_check_passed": bool(all(checks.values())),
        "conservative_behavior_note": (
            "A supported depth hypothesis is not a clearance; the unavailable terrain "
            "hypothesis and the expired depth evidence both produce advisory "
            "REQUEST_ONE_OBSERVATION/CAUTIOUS_STOP recommendations, and no recommendation "
            "is executed."),
    }


def run_validation(
    *,
    dataset_dir: Path,
    sequence_name: str = "xyz",
    max_frames: int,
    max_reference_frames: int,
    reference_stride: int,
    depth_stride: int,
    mapping_stride: int,
    output_path: Path,
    allow_download: bool,
) -> dict:
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if max_reference_frames < 2:
        raise ValueError("max_reference_frames must be at least 2 for cross-view checks")
    if reference_stride <= 0 or depth_stride <= 0 or mapping_stride <= 0:
        raise ValueError("strides must be positive")

    ensure_fn = {
        "xyz": ensure_freiburg1_xyz,
        "rpy": ensure_freiburg1_rpy,
        "360": ensure_freiburg1_360,
    }.get(sequence_name.lower())
    if ensure_fn is None:
        raise ValueError(f"Unknown sequence: {sequence_name}. Must be one of: xyz, rpy, 360")

    timings: list[dict] = []
    failures: list[str] = []
    started_total = perf_counter()
    report: dict[str, Any] = {
        "format": "prism-tum-rgbd-validation/v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": {
            "dataset_dir": str(dataset_dir),
            "sequence": f"freiburg1_{sequence_name}",
            "output": str(output_path),
            "max_frames": max_frames,
            "max_reference_frames": max_reference_frames,
            "reference_stride": reference_stride,
            "depth_stride": depth_stride,
            "mapping_stride": mapping_stride,
            "allow_download": allow_download,
        },
        "limits": [
            f"Indoor sequence freiburg1_{sequence_name}; results do not establish GPS-denied navigation readiness.",
            "RGB-only mode is monocular ORB VO with post-hoc Sim(3) alignment; metric scale is unobservable.",
            "RGB-D reference mode uses benchmark-supplied depth and calibration; it does not validate monocular depth inference.",
            "Terrain traversability accuracy is unmeasured on this sequence (no labels).",
            "Prediction accuracy is unmeasured: there are no labelled future events.",
            "No neural model, ROS, OpenVINS, SLAM backend, or planner was added or tuned.",
        ],
        "dataset": None,
        "input_availability": {},
        "checks": {
            "software_checks": {"status": "not_run", "items": {}},
            "geometric_accuracy_checks": {
                "status": "not_run",
                "items": {},
                "thresholds": {},
                "note": SOFTWARE_VS_GEOMETRY_NOTE,
            },
        },
        "modes": {},
        "evidence_separation": {},
        "mapping_terrain": {},
        "belief_doubt": {},
        "timing_stages": [],
        "failures": failures,
        "validation_summary": {},
    }

    dataset_started = perf_counter()
    layout, provenance = ensure_fn(dataset_dir, allow_download=allow_download)
    report["dataset"] = _json(provenance)
    if layout is None:
        timings.append({
            "name": "dataset_availability",
            "duration_s": perf_counter() - dataset_started,
            "status": "blocked",
            "detail": provenance.download_error,
        })
        failures.append(provenance.download_error or "TUM sequence unavailable")
        report["checks"]["software_checks"].update({
            "status": "blocked",
            "items": {"dataset_available": False},
            "detail": provenance.download_error,
        })
        report["checks"]["geometric_accuracy_checks"].update({
            "status": "blocked",
            "detail": "dataset unavailable",
        })
        report["validation_summary"] = {
            "software_checks": "blocked",
            "geometric_accuracy_checks": "blocked_missing_dataset",
            "prediction_accuracy": "unmeasured",
            "gps_denied_navigation_ready": False,
        }
        report["timing_stages"] = _json(timings)
        report["runtime_s"] = perf_counter() - started_total
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_json(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        return report

    integrity = sequence_integrity(layout)
    timings.append({
        "name": "dataset_availability",
        "duration_s": perf_counter() - dataset_started,
        "status": "ok",
        "detail": provenance.download_status,
    })
    frame_limit = min(max_frames, len(layout.rgb_entries))
    selected_associations = _select_associations(
        layout, max_reference_frames=max_reference_frames, stride=reference_stride)
    report["input_availability"] = {
        **integrity,
        "frames_requested": frame_limit,
        "reference_frames_requested": len(selected_associations),
        "calibration_primary_for_depth": _json(FREIBURG1_ROS_DEFAULT_INTRINSICS),
        "processing_resolution": [PROC_W, PROC_H],
    }
    source_repaired = {
        "sampled_paths_missing": len(integrity["sampled_paths_missing"]) == 0,
        "association_is_one_to_one": integrity["reused_depth_entries"] == 0,
        "rgb_timestamps_unique": integrity["rgb_timestamps_unique"],
        "depth_timestamps_unique": integrity["depth_timestamps_unique"],
    }
    fps = integrity["estimated_rgb_fps"] or 30.0

    rgb_only = _stage(
        timings, failures, "rgb_only_vo",
        lambda: _run_rgb_only(layout, frame_limit=frame_limit, fps=fps))
    report["modes"]["rgb_only"] = rgb_only if rgb_only is not None else {
        "status": "failed",
        "reason": "see timing_stages/failures",
    }

    loaded = _stage(
        timings, failures, "rgbd_reference_frame_load",
        lambda: _load_reference_frames(layout, selected_associations))
    frames, skipped = loaded if loaded is not None else ([], ["frame load stage failed"])
    reference = _stage(
        timings, failures, "rgbd_reference_geometry",
        lambda: _run_reference_geometry(frames, depth_stride=depth_stride))
    if reference is not None and skipped:
        reference["frames_skipped"] = skipped
    report["modes"]["rgbd_reference"] = reference if reference is not None else {
        "status": "failed",
        "reason": "see timing_stages/failures",
    }

    mapping_report = None
    if frames:
        mapping_report = _stage(
            timings, failures, "mapping_interface",
            lambda: _run_mapping_interface(frames[0], mapping_stride=mapping_stride))

    terrain = terrain_status(geometry_report={"physical_scale": "METRIC_SCALE_ALIGNED"})
    vo_summary = {}
    if rgb_only is not None:
        vo_summary = {
            "tracking_only": False,
            "trajectory_produced": False,
        }
    diagnostic = mapping_diagnostic_status(
        geometry_report={"physical_scale": "METRIC_SCALE_ALIGNED"},
        visual_motion_summary=vo_summary or {
            "tracking_only": True, "trajectory_produced": False},
    )
    report["mapping_terrain"] = {
        "terrain_status": _json(terrain),
        "mapping_diagnostic": {
            **_json(diagnostic),
            "scope": (
                "legacy integrated-replay diagnostic evaluated from the RGB-only "
                "visual-motion summary; it does not describe the separate "
                "reference-input mapping interface test"),
        },
        "terrain_traversability_validation": {
            "status": "unmeasured",
            "reason": (
                "indoor desk sequence lacks labelled traversable/obstacle regions and "
                "UGV/vehicle-frame terrain ground truth"
                if terrain.get("status") == "UNVALIDATED" else
                terrain.get("reason")),
            "grid_produced_for_interface_test": bool(mapping_report is not None),
            "mapping_interface_software_check": (
                mapping_report.get("software_check_passed")
                if isinstance(mapping_report, dict) else False),
            "unobserved_cells_preserved_unknown": (
                mapping_report.get("unobserved_cells_all_unknown")
                if isinstance(mapping_report, dict) else None),
            "trav_source": mapping_report.get("trav_source") if isinstance(
                mapping_report, dict) else None,
            "active_map_updated": False,
            "note": (
                "The prototype BEV grid is an interface test, not the active "
                "navigator map and not a traversability accuracy result."),
        },
        "prediction_accuracy": {
            "status": "unmeasured",
            "reason": "no labelled or tracked future events suitable for prediction scoring",
        },
    }
    if mapping_report is not None:
        report["mapping_terrain"]["mapping_interface"] = mapping_report

    belief = None
    if frames:
        belief = _stage(timings, failures, "belief_ledger_doubt",
                        lambda: _run_belief_doubt(frames[0]))
    report["belief_doubt"] = belief if belief is not None else {
        "status": "failed",
        "reason": "see timing_stages/failures",
    }

    software_items = {
        "dataset_available": True,
        "sampled_files_present": source_repaired["sampled_paths_missing"],
        "one_to_one_rgb_depth_association": source_repaired["association_is_one_to_one"],
        "rgb_only_path_ran": bool(
            isinstance(rgb_only, dict)
            and rgb_only.get("tracking", {}).get("decoded_frames", 0) > 0),
        "rgbd_reference_path_ran": bool(
            isinstance(reference, dict) and reference.get("status") == "measured"),
        "mapping_interface_ran": bool(
            isinstance(mapping_report, dict)
            and mapping_report.get("software_check_passed", False)),
        "belief_doubt_path_ran": bool(
            isinstance(belief, dict) and belief.get("software_check_passed", False)),
        "calibration_status_calibrated": bool(
            isinstance(rgb_only, dict)
            and rgb_only.get("calibration", {}).get("status") == "calibrated"),
    }
    software_passed = all(software_items.values())

    geometric_items = {}
    if isinstance(rgb_only, dict):
        for key, value in rgb_only.get("checks", {}).items():
            geometric_items[f"rgb_only_{key}"] = bool(value)
    if isinstance(reference, dict):
        for key, value in reference.get("checks", {}).items():
            geometric_items[f"rgbd_reference_{key}"] = bool(value)
    if not geometric_items:
        geometric_status = "unmeasured"
    elif not all(geometric_items.values()):
        geometric_status = "failed"
    elif isinstance(rgb_only, dict) and isinstance(reference, dict) \
            and reference.get("status") == "measured":
        geometric_status = "passed"
    else:
        geometric_status = "partial"

    report["checks"] = {
        "software_checks": {
            "status": "passed" if software_passed else "failed",
            "items": software_items,
            "note": (
                "These checks show that the software path executed; they are not "
                "geometric accuracy results."),
        },
        "geometric_accuracy_checks": {
            "status": geometric_status,
            "items": geometric_items,
            "thresholds": {
                "rgb_only": rgb_only.get("thresholds") if isinstance(rgb_only, dict) else {},
                "rgbd_reference": (
                    reference.get("thresholds") if isinstance(reference, dict) else {}),
            },
            "note": SOFTWARE_VS_GEOMETRY_NOTE,
            "measured_scope": [
                "monocular RGB-only trajectory after post-hoc Sim(3) alignment",
                "cross-view reference-depth consistency using ground-truth poses",
                "reference point-cloud/vehicle-transform interface",
                "support-plane fit on reference depth (not traversability)",
            ],
            "unmeasured_scope": [
                "online monocular metric scale and metric odometry",
                "monocular depth inference accuracy",
                "terrain traversability accuracy",
                "prediction accuracy versus future events",
                "GPS-denied navigation readiness",
            ],
        },
    }

    report["evidence_separation"] = {
        "ground_truth": {
            "source": "TUM rgbd_dataset_freiburg1_xyz groundtruth.txt (motion capture)",
            "used_for": [
                "post-hoc RGB-only trajectory evaluation",
                "reference-frame pose transforms for cross-view geometry",
                "placement of the reference point cloud in world coordinates",
            ],
            "not_used_for": [
                "visual-odometry motion estimation",
                "monocular depth inference",
                "traversability or prediction labels",
            ],
        },
        "measured_depth": {
            "rgbd_reference_mode": (
                "benchmark-registered 16-bit depth PNGs, scale 5000, missing pixels "
                "left invalid"),
            "rgb_only_mode": (
                "not used; no monocular depth inference was run, so no inferred "
                "depth is reported as measured"),
        },
        "detection_scores": {
            "status": "none_produced_or_consumed",
            "reason": (
                "the RGB-only path is classical ORB visual odometry and reports "
                "feature/match/inlier counts, not object-detection scores"),
        },
        "predictions": {
            "status": "not_scored",
            "reason": (
                "no labelled or tracked future events exist in this sequence; "
                "traversability and prediction accuracy are unmeasured"),
        },
    }
    report["timing_stages"] = _json(timings)
    report["runtime_s"] = perf_counter() - started_total
    report["validation_summary"] = {
        "software_checks": report["checks"]["software_checks"]["status"],
        "geometric_accuracy_checks": geometric_status,
        "rgb_only_tracking_success_rate": (
            rgb_only.get("tracking", {}).get("tracking_success_rate")
            if isinstance(rgb_only, dict) else None),
        "rgb_only_ate_rmse_m": (
            rgb_only.get("trajectory_error", {}).get("ate_rmse_m")
            if isinstance(rgb_only, dict) else None),
        "rgb_only_sim3_scale": (
            rgb_only.get("trajectory_error", {}).get("scale")
            if isinstance(rgb_only, dict) else None),
        "cross_view_depth_median_abs_error_m": (
            reference.get("cross_view_depth_consistency", {}).get("median_abs_error_m")
            if isinstance(reference, dict) else None),
        "mapping_observed_cells": (
            mapping_report.get("observed_cells") if isinstance(mapping_report, dict) else None),
        "prediction_accuracy": "unmeasured",
        "gps_denied_navigation_ready": False,
        "final_live_camera_test": "out_of_scope",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return report


def _build_multi_sequence_summary(reports: list[dict], args: argparse.Namespace) -> dict:
    """Aggregate multiple per-sequence reports into a summary."""
    sequence_names = [r.get("command", {}).get("sequence", "unknown") for r in reports]
    all_failures = []
    for r in reports:
        all_failures.extend(r.get("failures", []))

    per_sequence_summaries = []
    for r in reports:
        summary = r.get("validation_summary", {})
        seq_name = r.get("command", {}).get("sequence", "unknown")
        per_sequence_summaries.append({
            "sequence": seq_name,
            "software_checks": summary.get("software_checks"),
            "geometric_accuracy_checks": summary.get("geometric_accuracy_checks"),
            "rgb_only_tracking_success_rate": summary.get("rgb_only_tracking_success_rate"),
            "rgb_only_ate_rmse_m": summary.get("rgb_only_ate_rmse_m"),
            "rgb_only_sim3_scale": summary.get("rgb_only_sim3_scale"),
        })

    return {
        "format": "prism-tum-rgbd-validation/v2-multi",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": {
            "dataset_dir": str(args.dataset_dir),
            "sequences": sequence_names,
            "output": str(args.output),
        },
        "per_sequence": per_sequence_summaries,
        "failures": all_failures,
        "validation_summary": {
            "sequences_validated": len(reports),
            "all_software_checks_passed": all(
                s.get("software_checks") == "passed" for s in per_sequence_summaries),
            "all_geometric_checks_passed": all(
                s.get("geometric_accuracy_checks") == "passed" for s in per_sequence_summaries),
            "prediction_accuracy": "unmeasured",
            "gps_denied_navigation_ready": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sequences", type=str, default="xyz",
                        help="comma-separated sequence names: xyz,rpy,360 (default: xyz)")
    parser.add_argument("--max-frames", type=int, default=80,
                        help="bounded RGB frames for visual odometry (default: 80)")
    parser.add_argument("--max-reference-frames", type=int, default=12,
                        help="bounded RGB-D reference frames (default: 12)")
    parser.add_argument("--reference-stride", type=int, default=8,
                        help="stride over the RGB-depth associations for reference frames")
    parser.add_argument("--depth-stride", type=int, default=8,
                        help="pixel stride for depth consistency/back-projection checks")
    parser.add_argument("--mapping-stride", type=int, default=4,
                        help="pixel stride for the bounded MappingStage interface test")
    parser.add_argument("--no-download", action="store_true",
                        help="do not attempt to download the TUM archive")
    args = parser.parse_args()
    if args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")
    if args.max_reference_frames < 2:
        raise SystemExit("--max-reference-frames must be at least 2")
    if args.reference_stride <= 0 or args.depth_stride <= 0 or args.mapping_stride <= 0:
        raise SystemExit("strides must be positive")

    sequence_names = [s.strip().lower() for s in args.sequences.split(",") if s.strip()]
    if not sequence_names:
        raise SystemExit("--sequences must specify at least one sequence")

    all_reports = []
    for seq_name in sequence_names:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Processing sequence: freiburg1_{seq_name}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        seq_output = args.output.parent / f"tum_rgbd_validation_{seq_name}.json"
        report = run_validation(
            dataset_dir=args.dataset_dir,
            sequence_name=seq_name,
            max_frames=args.max_frames,
            max_reference_frames=args.max_reference_frames,
            reference_stride=args.reference_stride,
            depth_stride=args.depth_stride,
            mapping_stride=args.mapping_stride,
            output_path=seq_output,
            allow_download=not args.no_download,
        )
        all_reports.append(report)

    if len(all_reports) == 1:
        summary_report = all_reports[0]
    else:
        summary_report = _build_multi_sequence_summary(all_reports, args)

    summary = summary_report.get("validation_summary", {})
    print(json.dumps({
        "output": str(args.output),
        "sequences": sequence_names,
        "summary": _json(summary),
        "failures": _json(summary_report.get("failures", [])),
    }, sort_keys=True, allow_nan=False))
    return 0 if summary_report.get("input_availability") else 1


if __name__ == "__main__":
    raise SystemExit(main())
