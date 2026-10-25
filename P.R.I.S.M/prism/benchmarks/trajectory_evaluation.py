"""Trajectory alignment and error metrics for benchmark validation.

The monocular visual-odometry path only supplies translation directions, so the
only defensible trajectory comparison is a post-hoc Sim(3) alignment.  The scale
recovered by that alignment is reported explicitly; it is not an online metric
odometry estimate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class AlignedTrajectoryMetrics:
    association_count: int
    scale: float
    ate_rmse_m: float
    ate_mean_m: float
    ate_median_m: float
    ate_std_m: float
    rpe_trans_rmse_m: float
    rpe_rot_rmse_deg: float
    monocular_scale_note: str
    alignment: str
    ate_rot_rmse_deg: float = float("nan")
    rpe_rot_mean_deg: float = float("nan")
    path_length_est_m: float = float("nan")
    path_length_ref_m: float = float("nan")
    path_length_ratio: float = float("nan")
    final_position_error_m: float = float("nan")
    alignment_rotation_deg: float = float("nan")       # rotation of the Sim(3) position alignment
    ate_rot_alignment_deg: float = float("nan")        # rotation of the orientation-only alignment
    input_count: int = 0
    degenerate: bool = False
    rpe_delta: int = 1


def quaternion_to_rotation_xyzw(
    qx: float, qy: float, qz: float, qw: float
) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("quaternion must be finite and non-zero")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Backward-compatible private alias."""
    return quaternion_to_rotation_xyzw(qx, qy, qz, qw)


def rotation_to_quat_xyzw(R: np.ndarray) -> tuple[float, float, float, float]:
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError("rotation matrix must be 3x3")
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    return float(qx), float(qy), float(qz), float(qw)


def rotation_angle_deg(R: np.ndarray) -> float:
    R = np.asarray(R, dtype=np.float64)
    cosine = np.clip((float(np.trace(R)) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def umeyama_sim3(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares Sim(3) mapping ``source`` (Nx3) onto ``target`` (Nx3)."""
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    if src.ndim != 2 or dst.ndim != 2 or src.shape[1:] != (3,) or dst.shape[1:] != (3,):
        raise ValueError("source and target must both be Nx3 arrays")
    if src.shape[0] != dst.shape[0]:
        raise ValueError("source and target must contain the same number of poses")
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("source and target must be finite")
    count = src.shape[0]
    if count < 3:
        raise ValueError("need at least three associated poses for Sim(3) alignment")
    src_t = src.T
    dst_t = dst.T
    mean_src = src_t.mean(axis=1)
    mean_dst = dst_t.mean(axis=1)
    delta_src = src_t - mean_src[:, None]
    delta_dst = dst_t - mean_dst[:, None]
    sigma_src = float((delta_src ** 2).sum() / count)
    if sigma_src <= 1e-18:
        raise ValueError("degenerate source trajectory for scale estimation")
    cov = (delta_dst @ delta_src.T) / count
    u, singular, vt = np.linalg.svd(cov)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = u @ vt
    scale = float(np.sum(singular) / sigma_src)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("non-positive or non-finite alignment scale")
    trans = mean_dst - scale * (rot @ mean_src)
    return scale, rot, trans


def align_orientations(estimated_rotations: np.ndarray,
                       reference_rotations: np.ndarray) -> np.ndarray:
    """Single rotation minimising ``sum_i ||R R_est_i - R_ref_i||_F`` over SO(3).

    Camera-to-world rotations from monocular VO live in an arbitrary world frame
    whose orientation differs from the reference world by an unknown constant.
    This removes *only* that constant; the residual is the absolute orientation
    error.  It is deliberately independent of the Sim(3) position alignment,
    whose rotation is poorly constrained by short, nearly straight trajectories.

    Returns
    -------
    rot : np.ndarray
        The 3x3 rotation matrix that best aligns estimated orientations to
        reference orientations. The angle of this rotation (via
        ``rotation_angle_deg``) is reported as ``ate_rot_alignment_deg`` and
        represents the arbitrary world-frame offset, not an error.

    Notes
    -----
    This is an orthogonal Procrustes problem on rotation matrices. The solution
    maximises ``tr(R_align @ M)`` where ``M = sum_i R_est_i @ R_ref_i^T``,
    yielding ``R_align = U @ V^T`` from the SVD of ``M = U @ S @ V^T``.
    """
    est = np.asarray(estimated_rotations, dtype=np.float64)
    ref = np.asarray(reference_rotations, dtype=np.float64)
    if est.shape != ref.shape or est.ndim != 3 or est.shape[1:] != (3, 3):
        raise ValueError("estimated and reference rotations must both be Nx3x3")
    if est.shape[0] == 0:
        raise ValueError("orientation alignment requires at least one pose")
    cross = np.einsum("nij,nkj->ik", ref, est)
    u, singular, vt = np.linalg.svd(cross)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = u @ vt
    return rot


def integrate_vo_trajectory(
    records: Sequence,
    *,
    use_statuses: Optional[set] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate relative OpenCV ``recoverPose`` outputs into camera-to-world poses.

    ``recoverPose`` returns the mapping from the previous camera frame to the
    current one (``X_current = R X_previous + t``), so the camera pose update is
    the inverse of that transform.  Failed frames hold the previous pose and are
    visible as zero-motion steps in the diagnostics.
    """
    from prism.visual_odometry_replay import MotionStatus

    if use_statuses is None:
        use_statuses = {MotionStatus.OK, MotionStatus.UNVALIDATED_MOTION}
    if not records:
        return np.empty((0, 3), np.float64), np.empty((0, 3, 3), np.float64)

    positions = [np.zeros(3, dtype=np.float64)]
    rotations = [np.eye(3, dtype=np.float64)]
    for record in records[1:]:
        rotation = getattr(record, "rotation", None)
        translation = record.translation_unit
        if record.status not in use_statuses or rotation is None or translation is None:
            positions.append(positions[-1].copy())
            rotations.append(rotations[-1].copy())
            continue
        R_rel = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        t_rel = np.asarray(translation, dtype=np.float64).reshape(3)
        if not np.isfinite(R_rel).all() or not np.isfinite(t_rel).all():
            positions.append(positions[-1].copy())
            rotations.append(rotations[-1].copy())
            continue
        T_inv = np.eye(4, dtype=np.float64)
        T_inv[:3, :3] = R_rel.T
        T_inv[:3, 3] = -R_rel.T @ t_rel
        T_w_c_prev = np.eye(4, dtype=np.float64)
        T_w_c_prev[:3, :3] = rotations[-1]
        T_w_c_prev[:3, 3] = positions[-1]
        T_w_c = T_w_c_prev @ T_inv
        positions.append(T_w_c[:3, 3].copy())
        rotations.append(T_w_c[:3, :3].copy())
    return np.asarray(positions, dtype=np.float64), np.asarray(rotations, dtype=np.float64)


def _empty_metrics(
    *,
    association_count: int,
    input_count: int,
    note: str,
    alignment: str = "none",
    degenerate: bool = True,
    rpe_delta: int = 1,
) -> AlignedTrajectoryMetrics:
    nan = float("nan")
    return AlignedTrajectoryMetrics(
        association_count=int(association_count),
        scale=nan,
        ate_rmse_m=nan,
        ate_mean_m=nan,
        ate_median_m=nan,
        ate_std_m=nan,
        rpe_trans_rmse_m=nan,
        rpe_rot_rmse_deg=nan,
        monocular_scale_note=note,
        alignment=alignment,
        input_count=int(input_count),
        degenerate=degenerate,
        rpe_delta=int(rpe_delta),
    )


def evaluate_trajectories(
    estimated_positions: np.ndarray,
    reference_positions: np.ndarray,
    *,
    with_sim3_scale: bool = True,
    estimated_rotations: Optional[np.ndarray] = None,
    reference_rotations: Optional[np.ndarray] = None,
    rpe_delta: int = 1,
) -> AlignedTrajectoryMetrics:
    """Evaluate an estimated camera trajectory against reference camera poses.

    Positions are camera centres in the same arbitrary world frame; a Sim(3)
    alignment is required because monocular VO has no metric scale.  Rotation
    metrics require full camera-to-world rotations for both trajectories.
    Degenerate input returns NaN metrics with a reason instead of raising.
    """
    if rpe_delta < 1:
        raise ValueError("rpe_delta must be at least 1")
    est = np.asarray(estimated_positions, dtype=np.float64)
    ref = np.asarray(reference_positions, dtype=np.float64)
    if est.ndim != 2 or est.shape[1:] != (3,) or ref.ndim != 2 or ref.shape[1:] != (3,):
        raise ValueError("positions must be Nx3 arrays")
    if est.shape[0] != ref.shape[0]:
        raise ValueError("estimated and reference trajectories must have equal length")

    input_count = int(est.shape[0])
    mask = np.isfinite(est).all(axis=1) & np.isfinite(ref).all(axis=1)
    est = est[mask]
    ref = ref[mask]
    association_count = int(est.shape[0])

    est_rot = None
    ref_rot = None
    if estimated_rotations is not None or reference_rotations is not None:
        if estimated_rotations is None or reference_rotations is None:
            raise ValueError("both estimated and reference rotations are required together")
        est_rot = np.asarray(estimated_rotations, dtype=np.float64)
        ref_rot = np.asarray(reference_rotations, dtype=np.float64)
        if est_rot.shape != (input_count, 3, 3) or ref_rot.shape != (input_count, 3, 3):
            raise ValueError("rotations must be Nx3x3 arrays matching the trajectories")
        if not np.isfinite(est_rot).all() or not np.isfinite(ref_rot).all():
            raise ValueError("rotation arrays must be finite")
        est_rot = est_rot[mask]
        ref_rot = ref_rot[mask]

    if association_count < 3:
        return _empty_metrics(
            association_count=association_count, input_count=input_count,
            note="insufficient associated finite poses for alignment and error metrics",
            rpe_delta=rpe_delta,
        )

    ref_path = float(np.linalg.norm(np.diff(ref, axis=0), axis=1).sum()) \
        if association_count > 1 else 0.0
    if ref_path <= 1e-12:
        return _empty_metrics(
            association_count=association_count, input_count=input_count,
            note="reference trajectory has no spatial extent; ATE is undefined",
            rpe_delta=rpe_delta,
        )

    try:
        if with_sim3_scale:
            scale, rot, trans = umeyama_sim3(est, ref)
            est_aligned = (scale * (est @ rot.T)) + trans
            alignment = "sim3"
        else:
            scale = 1.0
            rot = np.eye(3)
            trans = np.zeros(3)
            est_aligned = est.copy()
            alignment = "none"
    except (ValueError, np.linalg.LinAlgError) as error:
        return _empty_metrics(
            association_count=association_count, input_count=input_count,
            note=f"trajectory alignment failed: {error}",
            alignment="failed",
            rpe_delta=rpe_delta,
        )

    errors = np.linalg.norm(est_aligned - ref, axis=1)
    ate_rmse = float(np.sqrt(np.mean(errors ** 2)))
    ate_mean = float(np.mean(errors))
    ate_median = float(np.median(errors))
    ate_std = float(np.std(errors))
    final_error = float(errors[-1])
    aligned_path = float(np.linalg.norm(np.diff(est_aligned, axis=0), axis=1).sum()) \
        if association_count > 1 else 0.0

    rpe_trans_errors: list[float] = []
    for i in range(0, association_count - rpe_delta):
        j = i + rpe_delta
        delta_est = est_aligned[j] - est_aligned[i]
        delta_ref = ref[j] - ref[i]
        rpe_trans_errors.append(float(np.linalg.norm(delta_est - delta_ref)))
    rpe_trans = (float(np.sqrt(np.mean(np.square(rpe_trans_errors))))
                 if rpe_trans_errors else float("nan"))

    ate_rot = float("nan")
    rpe_rot_rmse = float("nan")
    rpe_rot_mean = float("nan")
    alignment_rot = float("nan")
    ate_rot_alignment = float("nan")
    if est_rot is not None and ref_rot is not None:
        # Absolute orientation error (ATE rotation) removes only the unknown constant
        # world-frame rotation via a single best-fit SO(3) alignment. The residual is
        # the absolute orientation error in degrees. This is distinct from RPE rotation,
        # which measures frame-to-frame relative rotation error without alignment.
        #
        # Metric definitions:
        # - ate_rot_rmse_deg: RMS of per-frame orientation errors after removing the
        #   best-fit global rotation (ate_rot_alignment_deg). This is an absolute error.
        # - rpe_rot_rmse_deg: RMS of relative rotation errors between consecutive frames
        #   (delta=rpe_delta). This is alignment-free and measures local consistency.
        # - ate_rot_alignment_deg: The angle of the best-fit rotation that aligns
        #   estimated orientations to reference orientations. This is not an error; it's
        #   the arbitrary world-frame offset removed before computing ATE rotation.
        # - alignment_rotation_deg: The rotation angle from the Sim(3) position alignment.
        #   This is separate from orientation alignment and is reported for diagnostics.
        #
        # Frame-count effects: ATE rotation typically increases with more frames because
        # drift accumulates over time. RPE rotation is local and less sensitive to frame
        # count. A short trajectory (e.g., 6 frames) may show lower ATE rotation than a
        # longer one (e.g., 80 frames) even with similar per-frame quality.
        #
        # The Sim(3) position alignment is *not* reused for orientation alignment: on
        # short trajectories its rotation is dominated by translation noise and does not
        # represent the camera-frame offset.
        try:
            rot_align = align_orientations(est_rot, ref_rot)
            aligned_est_rot = np.einsum("ij,njk->nik", rot_align, est_rot)
            rot_errors = np.asarray([
                rotation_angle_deg(aligned_est_rot[i].T @ ref_rot[i])
                for i in range(association_count)
            ])
            ate_rot = float(np.sqrt(np.mean(rot_errors ** 2)))
            ate_rot_alignment = rotation_angle_deg(rot_align)
        except (ValueError, np.linalg.LinAlgError):
            ate_rot = float("nan")
        alignment_rot = rotation_angle_deg(rot)
        rel_errors: list[float] = []
        for i in range(0, association_count - rpe_delta):
            j = i + rpe_delta
            rel_est = est_rot[i].T @ est_rot[j]
            rel_ref = ref_rot[i].T @ ref_rot[j]
            rel_errors.append(rotation_angle_deg(rel_est.T @ rel_ref))
        if rel_errors:
            rpe_rot_rmse = float(np.sqrt(np.mean(np.square(rel_errors))))
            rpe_rot_mean = float(np.mean(rel_errors))

    note = (
        "Sim(3) alignment is a post-hoc evaluation alignment; monocular "
        "two-view geometry fixes translation direction only and does not "
        "observe metric scale. The reported scale is not online odometry."
        if with_sim3_scale else
        "no alignment applied; estimated and reference trajectories are in different frames"
    )
    return AlignedTrajectoryMetrics(
        association_count=association_count,
        scale=float(scale),
        ate_rmse_m=ate_rmse,
        ate_mean_m=ate_mean,
        ate_median_m=ate_median,
        ate_std_m=ate_std,
        rpe_trans_rmse_m=rpe_trans,
        rpe_rot_rmse_deg=rpe_rot_rmse,
        monocular_scale_note=note,
        alignment=alignment,
        ate_rot_rmse_deg=ate_rot,
        rpe_rot_mean_deg=rpe_rot_mean,
        path_length_est_m=aligned_path,
        path_length_ref_m=ref_path,
        path_length_ratio=float(aligned_path / ref_path) if ref_path > 0 else float("nan"),
        final_position_error_m=final_error,
        alignment_rotation_deg=alignment_rot,
        ate_rot_alignment_deg=ate_rot_alignment,
        input_count=input_count,
        degenerate=False,
        rpe_delta=rpe_delta,
    )
