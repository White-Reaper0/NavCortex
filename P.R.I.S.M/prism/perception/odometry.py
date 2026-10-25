"""ORB-SLAM3-style monocular visual odometry front end (Python / OpenCV).

WHAT THIS IS -- read this before quoting any number out of it
-------------------------------------------------------------
This module is **not** ORB-SLAM3. ORB-SLAM3 is a C++ system with a full local
bundle-adjustment back end, covisibility graph, keyframe culling, loop closing and
an inertial initialiser. Building it is out of scope for this repo. What lives here
is a faithful re-implementation of its *monocular tracking front end* in Python and
OpenCV:

    grid-bucketed ORB -> BFMatcher(Hamming) + Lowe ratio + mutual cross-check
    -> essential-matrix RANSAC with the configured intrinsics -> recoverPose
    -> cheirality / parallax model check -> scale recovery -> 2-D pose integration

Everything ORB-SLAM3 gets from its back end (drift-free maps, global BA, IMU scale)
is absent. Label it "ORB-SLAM3-style monocular front end", never "ORB-SLAM3".

SCALE -- the honest bit
-----------------------
A monocular essential matrix determines translation only up to an unknown positive
scalar (``|t| == 1`` out of ``cv2.recoverPose``). There is no IMU in this footage and
no stereo baseline, so the scale must come from somewhere else. prism takes it from
the depth stage:

  1. triangulate the RANSAC inlier correspondences with the baseline fixed at 1
     -> per-point depth ``Z_tri`` in "baseline units",
  2. read the metric depth map produced by the depth stage at the *same* pixels
     -> ``Z_met`` in metres,
  3. robustly fit ``Z_met ~= s * Z_tri`` (median ratio seed + Cauchy IRLS) --
     ``_robust_scale_ratio``,
  4. refine that with a 1-D robust least squares on the *reprojection* constraint,
     which is far better conditioned at this parallax -- ``_reproj_scale``. Both
     steps read their metric depths from the same depth map; the second only
     changes how the residual is weighted.

``s`` is then the inter-frame baseline in metres. Only depths inside
``[scale_z_min, scale_z_max]`` vote, and each residual is normalised by point depth
-- without both, the fit is dominated by far points whose monocular depth is
unreliable (measured: clip_01 came out 3.4x too fast). A short running median over
the accepted baselines runs before the EMA, because the raw per-frame distribution
is strongly right-skewed.

So this is **depth-anchored monocular VO**. It is not stereo VO and not
visual-inertial VO. And the metric depth is itself anchored on the *assumed* camera
height ``CFG.cam.height_above_ground_m`` (see ``perception/geometry.py``), so every
metre reported here is proportional to that assumption. If the fit has too few
surviving points, or its relative standard error exceeds ``max_scale_se``, the stage
falls back to a smoothed previous scale; if no scale has ever been recovered (no
depth cache at all) it falls back to a documented constant (``fallback_step_m``).
``scale_source`` always says which of the three was used.

Cross-check: on clip_01/02/04 the recovered baseline was compared per frame against
an independent estimate that uses only the fitted ground plane and the assumed
camera height (no network depth at all). Paired median ratio 0.63 - 0.93, i.e. the
depth-anchored speed agrees with the plane-only speed to within roughly 10-35 %.

Track health -- three states, all visible in the renderer
---------------------------------------------------------
``OK``     two-view geometry accepted, pose integrated from it.
``COAST``  geometry solved but its translation *direction* was rejected by the
           constant-velocity motion model. This is the analogue of ORB-SLAM's
           ``TrackWithMotionModel`` fallback. It is needed here because at 30 fps the
           inter-frame baseline of this RC car is only a few centimetres, the median
           parallax is ~0.3-0.5 deg, and at that parallax ``recoverPose``'s cheirality
           vote flips the sign of ``t`` on roughly 15 % of frames. Measured on these
           clips, see the report. ``tracking_ok`` stays True; ``track_quality`` drops.
``LOST``   too few matches / too few RANSAC inliers / degenerate geometry
           (median parallax below ``min_parallax_deg`` -- near-pure rotation).
           ``tracking_ok`` goes False, the pose is **held**, not extrapolated.
           The trajectory visibly stalls, which is what a supervisor needs to see.

A forward-motion prior is used to seed the motion model (all five clips are
forward-driving; the car never reverses). That is an assumption about this footage,
not a general property of the algorithm, and it is stated on the video panel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time
import warnings

import numpy as np
import cv2

from ..config import CFG, CLIP_FPS, PROC_W, PROC_H
from ..types import FramePacket, OdometryResult, Pose
from .. import io_utils


# scale_source codes -- kept as small ints so they survive an npz round trip
SCALE_DEPTH = 0        # depth-anchored robust fit succeeded this frame
SCALE_SMOOTHED = 1     # fit failed, reused the smoothed previous scale
SCALE_CONST = 2        # no depth available at all, constant-step assumption
SCALE_NONE = 3         # tracking lost, no motion integrated
SCALE_SOURCE_NAMES = ["depth-anchored", "smoothed-prev", "const-fallback", "no-track"]

# track state codes
TRACK_OK, TRACK_COAST, TRACK_LOST = 0, 1, 2
TRACK_STATE_NAMES = ["OK", "COAST", "LOST"]


@dataclass
class OdomParams:
    n_features: int = 1500
    n_levels: int = 8
    scale_factor: float = 1.2
    grid_cols: int = 5
    grid_rows: int = 4
    fast_thresh: int = 12
    fast_thresh_relaxed: int = 4
    ratio: float = 0.78
    ransac_thresh_px: float = 1.5
    ransac_conf: float = 0.9999
    min_matches: int = 28
    min_inliers: int = 22
    min_parallax_deg: float = 0.08
    max_yaw_rate_dps: float = 150.0    # implausible for this platform -> reject
    min_scale_pts: int = 25
    scale_z_min: float = 0.20      # depth band trusted for anchoring the baseline
    scale_z_max: float = 8.0
    max_scale_se: float = 0.45     # accept the depth fit only if its rel. SE is below this
    max_step_m: float = 0.30           # hard sanity clamp on per-frame baseline
    dir_accept_deg: float = 55.0       # motion-model gate on translation direction
    alpha_dir: float = 0.25            # EMA on the unit translation direction
    alpha_speed: float = 0.35          # EMA on speed
    alpha_yaw: float = 0.45            # EMA on yaw rate
    scale_med_win: int = 7             # median filter on the raw baseline before the EMA
    alpha_scale: float = 0.25          # EMA on the recovered baseline
    fallback_step_m: float = 0.035     # ~1.05 m/s at 30 fps, used only without depth


# --------------------------------------------------------------------- helpers

def _grid_detect(orb: cv2.ORB, orb_relaxed: cv2.ORB, gray: np.ndarray,
                 mask: np.ndarray, p: OdomParams,
                 barren: Optional[np.ndarray] = None, frame_i: int = 0) -> list:
    """Detect ORB keypoints per grid cell so they spread over the whole frame.

    A single whole-image ORB detect piles features onto whatever corner of the
    image happens to be textured; on this footage that is the gravel directly in
    front of the car. Bucketing by cell is exactly what ORB-SLAM does and it keeps
    the essential matrix well conditioned.
    """
    h, w = gray.shape[:2]
    gc, gr = p.grid_cols, p.grid_rows
    per_cell = int(np.ceil(p.n_features / (gc * gr)))
    ys = np.linspace(0, h, gr + 1).astype(int)
    xs = np.linspace(0, w, gc + 1).astype(int)

    # One whole-image detect (one image pyramid instead of gc*gr of them), then bucket.
    all_kps = orb.detect(gray, mask)
    buckets: list[list] = [[] for _ in range(gr * gc)]
    for k in all_kps:
        cx = min(int(k.pt[0] * gc / w), gc - 1)
        cy = min(int(k.pt[1] * gr / h), gr - 1)
        buckets[cy * gc + cx].append(k)

    out: list = []
    pad = 16                                        # ORB needs context around a corner
    for r in range(gr):
        for c in range(gc):
            b = buckets[r * gc + c]
            y0, y1 = ys[r], ys[r + 1]
            x0, x1 = xs[c], xs[c + 1]
            ci = r * gc + c
            # Cells that stay empty even with a relaxed threshold (sky, blown-out
            # tarmac) are remembered and only re-probed every 15 frames -- the retry
            # loop was otherwise half the stage's runtime.
            skip = barren is not None and barren[ci] > 0 and (frame_i % 15)
            if len(b) < max(6, per_cell // 3) and not skip:
                # starved cell (typical in the low-light clips): re-detect locally
                # with a relaxed FAST threshold instead of leaving a hole in the grid.
                ey0, ey1 = max(0, y0 - pad), min(h, y1 + pad)
                ex0, ex1 = max(0, x0 - pad), min(w, x1 + pad)
                msk = mask[ey0:ey1, ex0:ex1]
                if msk.max():
                    loc = orb_relaxed.detect(gray[ey0:ey1, ex0:ex1], msk)
                    for k in loc:
                        k.pt = (float(k.pt[0] + ex0), float(k.pt[1] + ey0))
                    got = [k for k in loc if x0 <= k.pt[0] < x1 and y0 <= k.pt[1] < y1]
                    if barren is not None:
                        barren[ci] = 0 if len(got) >= max(6, per_cell // 3) else 1
                    b = got or b
                elif barren is not None:
                    barren[ci] = 1
            if not b:
                continue
            b.sort(key=lambda k: -k.response)
            out.extend(b[:per_cell])
    return out


def _mutual_ratio_match(bf: cv2.BFMatcher, d0: np.ndarray, d1: np.ndarray,
                        ratio: float) -> np.ndarray:
    """Lowe ratio test in both directions plus a mutual (cross-check) constraint."""
    if d0 is None or d1 is None or len(d0) < 2 or len(d1) < 2:
        return np.zeros((0, 2), np.int32)
    fwd = bf.knnMatch(d0, d1, k=2)
    bwd = bf.knnMatch(d1, d0, k=2)

    best_bwd = np.full(len(d1), -1, np.int32)
    for mm in bwd:
        if len(mm) < 2:
            if len(mm) == 1:
                best_bwd[mm[0].queryIdx] = mm[0].trainIdx
            continue
        a, b = mm[0], mm[1]
        if a.distance < ratio * b.distance:
            best_bwd[a.queryIdx] = a.trainIdx

    pairs = []
    for mm in fwd:
        if len(mm) < 2:
            continue
        a, b = mm[0], mm[1]
        if a.distance >= ratio * b.distance:
            continue
        if best_bwd[a.trainIdx] == a.queryIdx:       # cross-check
            pairs.append((a.queryIdx, a.trainIdx))
    return np.asarray(pairs, np.int32).reshape(-1, 2)


def _sampson(F: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Sampson epipolar distance in pixels."""
    x0 = np.hstack([p0, np.ones((len(p0), 1))])
    x1 = np.hstack([p1, np.ones((len(p1), 1))])
    Fx0 = x0 @ F.T
    Ftx1 = x1 @ F
    num = np.einsum("ij,ij->i", x1, Fx0) ** 2
    den = Fx0[:, 0] ** 2 + Fx0[:, 1] ** 2 + Ftx1[:, 0] ** 2 + Ftx1[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12))


def _robust_scale_ratio(z_tri: np.ndarray, z_met: np.ndarray,
                        z_min: float = 0.20, z_max: float = 8.0,
                        iters: int = 6) -> tuple[float, float, int]:
    """Median-ratio + Cauchy-IRLS fit of ``z_met ~= s * z_tri``.

    Returns ``(s, relative_standard_error, n_used)``. The SE is the classic
    ``1.253 * MAD / sqrt(n)`` standard error of a median, expressed relative to the
    estimate: at the sub-degree parallax of this footage the per-point triangulated
    depth is very noisy, so the *spread* is large but the *median* is still well
    determined once a few hundred points vote. The SE is what the acceptance test
    uses; a per-point inlier fraction would reject almost every frame.

    Points outside ``[z_min, z_max]`` metres are dropped: monocular depth past a few
    metres is not trustworthy enough to anchor a centimetre-scale baseline.
    """
    good = (np.isfinite(z_tri) & np.isfinite(z_met) & (z_tri > 1e-4)
            & (z_met > z_min) & (z_met < z_max))
    zt, zm = z_tri[good], z_met[good]
    n = int(zt.size)
    if n < 8:
        return 0.0, 9.9, n
    ratio = zm / zt
    med = float(np.median(ratio))
    if not np.isfinite(med) or med <= 0:
        return 0.0, 9.9, n
    s = med
    w = np.ones_like(zt)
    for _ in range(iters):                           # Cauchy IRLS on z_met = s * z_tri
        den = float(np.sum(w * zt * zt))
        if den <= 1e-12:
            break
        s = float(np.sum(w * zt * zm)) / den
        r = zm - s * zt
        sig = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        w = 1.0 / (1.0 + (r / (2.5 * sig)) ** 2)
    if not np.isfinite(s) or s <= 0:
        return 0.0, 9.9, n
    mad = 1.4826 * float(np.median(np.abs(ratio - med)))
    se_rel = 1.253 * mad / np.sqrt(n) / max(med, 1e-9)
    return float(s), float(se_rel), n


def _reproj_scale(K: np.ndarray, Kinv: np.ndarray, R: np.ndarray, t: np.ndarray,
                  i0: np.ndarray, i1: np.ndarray, z_met: np.ndarray, b0: float,
                  z_min: float = 0.20, z_max: float = 8.0,
                  iters: int = 5) -> tuple[float, float, int]:
    """Robust 1-D least squares for the metric baseline ``b``, seeded by ``b0``.

    With the metric depth ``Z0`` of an inlier known from the depth stage, its 3-D
    point in camera k-1 is ``X0 = Z0 * m0``. In camera k it is ``X1 = R X0 + b t``
    with ``|t| = 1``, so the only unknown left is the scalar ``b``. Writing the
    projection constraint for the observed pixel ``(u1, v1)`` with
    ``p = (u1-cx)/fx``, ``q = (v1-cy)/fy`` and ``A = R X0`` gives two *linear*
    equations per point::

        (A_x - p A_z) + b (t_x - p t_z) = 0
        (A_y - q A_z) + b (t_y - q t_z) = 0

    a 1-D weighted least squares ``b = -sum(w c d) / sum(w d^2)`` refined by Cauchy
    IRLS. Far better conditioned than dividing by a triangulated depth, and it is
    still "fit the geometry against the metric depth map": the depth map supplies
    every ``Z0``.

    Two details matter and both were measured on this footage:

    * **Row normalisation.** That linear residual is the reprojection error
      *multiplied by the point depth*, so an unnormalised fit is dominated by the
      farthest points. Each row is divided by ``A_z``, making the cost a uniform
      normalised-image-plane reprojection error. Without this, clip_01 came out
      3.4x too fast because the 5-25 m depth bin outvoted everything else.
    * **Depth band.** Same ``[z_min, z_max]`` restriction as the ratio fit.
    """
    good = np.isfinite(z_met) & (z_met > z_min) & (z_met < z_max)
    if int(good.sum()) < 8:
        return 0.0, 9.9, int(good.sum())
    a0, a1, zm = i0[good], i1[good], z_met[good]
    m0 = np.hstack([a0, np.ones((len(a0), 1))]) @ Kinv.T           # (N,3), m_z == 1
    A = (m0 * zm[:, None]) @ R.T                                   # (N,3)
    tv = np.asarray(t, np.float64).ravel()
    p = (a1[:, 0] - K[0, 2]) / K[0, 0]
    q = (a1[:, 1] - K[1, 2]) / K[1, 1]
    nz = np.maximum(np.abs(A[:, 2]), 1e-3)                         # row normaliser
    c = np.concatenate([(A[:, 0] - p * A[:, 2]) / nz, (A[:, 1] - q * A[:, 2]) / nz])
    d = np.concatenate([(tv[0] - p * tv[2]) / nz, (tv[1] - q * tv[2]) / nz])
    w = np.ones_like(c)
    b = float(b0) if b0 > 0 else 0.0
    for _ in range(iters):
        den = float(np.sum(w * d * d))
        if den <= 1e-16:
            return 0.0, 9.9, int(len(zm))
        b = -float(np.sum(w * c * d)) / den
        r = c + b * d
        sig = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
        w = 1.0 / (1.0 + (r / (2.5 * sig)) ** 2)
    if not np.isfinite(b):
        return 0.0, 9.9, int(len(zm))
    r = c + b * d
    dof = max(len(c) - 1, 1)
    var_b = float(np.sum(w * r * r)) / dof / max(float(np.sum(w * d * d)), 1e-16)
    se_rel = float(np.sqrt(max(var_b, 0.0)) / max(abs(b), 1e-9))
    return float(b), se_rel, int(len(zm))


def _yaw_from_R(R: np.ndarray) -> float:
    """Vehicle yaw increment (CCW positive, radians) from a camera-frame rotation.

    ``R`` maps points from camera k-1 into camera k. The camera's own rotation is
    ``R.T``; its yaw component about the camera Y axis (which points *down*) is
    ``atan2(Rc[0,2], Rc[2,2])``. A right-handed rotation about a down-pointing axis
    turns the vehicle to the right, i.e. clockwise, i.e. negative vehicle yaw.
    """
    Rc = R.T
    return float(-np.arctan2(Rc[0, 2], Rc[2, 2]))


# --------------------------------------------------------------------- stage

class OdometryStage:
    """ORB-SLAM3-style monocular front end with depth-anchored metric scale.

    Fills ``packet.odom`` with a complete :class:`~prism.types.OdometryResult`.
    """

    def __init__(self, device: str = "cuda", params: Optional[OdomParams] = None,
                 use_depth_cache: bool = True, verbose: bool = False, **kw):
        # device is accepted for interface symmetry; ORB/RANSAC are CPU-only.
        self.device = device
        self.p = params or OdomParams()
        for k, v in kw.items():
            if hasattr(self.p, k):
                setattr(self.p, k, v)
        self.use_depth_cache = use_depth_cache
        self.verbose = verbose

        p = self.p
        # over-detect globally so every grid cell has candidates to bucket from
        self._orb = cv2.ORB_create(nfeatures=p.n_features * 3, scaleFactor=p.scale_factor,
                                   nlevels=p.n_levels, edgeThreshold=19, patchSize=31,
                                   fastThreshold=p.fast_thresh,
                                   scoreType=cv2.ORB_HARRIS_SCORE)
        self._orb_relaxed = cv2.ORB_create(nfeatures=p.n_features, scaleFactor=p.scale_factor,
                                           nlevels=p.n_levels, edgeThreshold=19, patchSize=31,
                                           fastThreshold=p.fast_thresh_relaxed,
                                           scoreType=cv2.ORB_HARRIS_SCORE)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        # CLAHE keeps the low-light clips (04/05) from starving the FAST detector
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        # MAGSAC++ is markedly more stable than plain RANSAC on this small-baseline
        # forward motion (measured: yaw-rate std 10.4 -> 0.5 deg/frame on clip_01).
        self._ransac_method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        self._K = np.asarray(CFG.cam.K, np.float64)
        self._Kinv = np.linalg.inv(self._K)
        self._mask_cache: dict = {}
        self._depth_clip: Optional[str] = None
        self._depth_arr: Optional[np.ndarray] = None
        self._depth_valid: Optional[np.ndarray] = None
        self._warned_no_depth = False
        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self) -> None:
        """Clear all temporal state. Call between clips."""
        self.pose = Pose()
        self.traj: list[tuple[float, float]] = []   # one entry per processed frame
        self._prev_kp: Optional[np.ndarray] = None
        self._prev_des: Optional[np.ndarray] = None
        self._prev_depth: Optional[np.ndarray] = None
        self._prev_t: Optional[float] = None
        self._scale_ema: float = 0.0
        self._scale_hist: list[float] = []
        self._speed_ema: float = 0.0
        self._yawrate_ema: float = 0.0
        # forward-motion prior in the body frame (right, forward); see module docstring
        self._dir_ema: np.ndarray = np.array([0.0, 1.0])
        self._barren = np.zeros(self.p.grid_cols * self.p.grid_rows, np.uint8)
        self._n_lost = 0
        self._n_coast = 0
        self._n_frames = 0
        self.last_scale_source = SCALE_NONE
        self.last_scale = 0.0
        self.last_sampson = 0.0
        self.last_parallax_deg = 0.0
        self.last_cheirality = 0.0
        self.last_scale_se = 9.9
        self.last_scale_n = 0
        self.last_scale_tri = 0.0
        self.last_state = TRACK_LOST
        self.last_reason = "init"

    # ---------------------------------------------------------------- depth
    def _ego_mask_u8(self, h: int, w: int) -> np.ndarray:
        key = (h, w)
        if key not in self._mask_cache:
            self._mask_cache[key] = (io_utils.ego_mask(h, w).astype(np.uint8) * 255)
        return self._mask_cache[key]

    def _ensure_depth_cache(self, clip_id: str) -> None:
        if not self.use_depth_cache or self._depth_clip == clip_id:
            return
        self._depth_clip = clip_id
        self._depth_arr = None
        self._depth_valid = None
        try:
            d = io_utils.load_stage(clip_id, "depth")
        except FileNotFoundError:
            if not self._warned_no_depth:
                warnings.warn(
                    f"[odometry] no depth cache for {clip_id}: falling back to a CONSTANT "
                    f"per-frame baseline of {self.p.fallback_step_m:.3f} m. Re-run this "
                    f"stage after the depth stage exists to get depth-anchored scale.",
                    RuntimeWarning, stacklevel=2)
                self._warned_no_depth = True
            return
        for k in ("depth", "depth_m", "metric_depth"):
            if k in d:
                self._depth_arr = d[k]
                break
        for k in ("valid", "depth_valid"):
            if k in d:
                self._depth_valid = d[k]
                break
        if self.verbose and self._depth_arr is not None:
            print(f"[odometry] depth cache {clip_id}: {self._depth_arr.shape} "
                  f"{self._depth_arr.dtype}")

    def _depth_for(self, packet: FramePacket) -> Optional[np.ndarray]:
        """Metric depth map (H,W) float32 with NaN where invalid, or None."""
        if packet.depth is not None and getattr(packet.depth, "depth_m", None) is not None:
            d = np.asarray(packet.depth.depth_m, np.float32).copy()
            v = getattr(packet.depth, "valid", None)
            if v is not None:
                d[~np.asarray(v, bool)] = np.nan
            return d
        self._ensure_depth_cache(packet.clip_id)
        if self._depth_arr is None or packet.idx >= len(self._depth_arr):
            return None
        d = np.asarray(self._depth_arr[packet.idx], np.float32).copy()
        if self._depth_valid is not None and packet.idx < len(self._depth_valid):
            d[~np.asarray(self._depth_valid[packet.idx], bool)] = np.nan
        return d

    # ---------------------------------------------------------------- main
    def __call__(self, packet: FramePacket) -> FramePacket:
        t_start = time.perf_counter()
        rgb = packet.rgb
        if rgb is None:
            raise ValueError("OdometryStage needs packet.rgb")
        if rgb.shape[1] != PROC_W or rgb.shape[0] != PROC_H:
            rgb = cv2.resize(rgb, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)
        gray = self._clahe.apply(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY))
        h, w = gray.shape
        mask = self._ego_mask_u8(h, w)

        kps = _grid_detect(self._orb, self._orb_relaxed, gray, mask, self.p,
                           self._barren, packet.idx)
        kps, des = self._orb.compute(gray, kps)
        kp_xy = (np.array([k.pt for k in kps], np.float32).reshape(-1, 2)
                 if kps else np.zeros((0, 2), np.float32))

        depth_now = self._depth_for(packet)
        dt = 1.0 / CLIP_FPS
        if self._prev_t is not None and packet.t > self._prev_t:
            dt = float(min(max(packet.t - self._prev_t, 1e-3), 0.5))

        res = OdometryResult(keypoints=kp_xy)
        res.n_matches = 0
        res.n_inliers = 0

        first = self._prev_des is None
        if first:
            res.tracking_ok = True
            res.track_quality = float(np.clip(len(kp_xy) / 600.0, 0.0, 1.0))
            res.pose = Pose(self.pose.x, self.pose.y, self.pose.yaw)
            res.flow = np.zeros((0, 4), np.float32)
            self.last_reason = "first frame (no previous view)"
            self.last_scale_source = SCALE_NONE
            self.last_state = TRACK_OK
        else:
            self._track(res, kp_xy, des, depth_now, dt)

        # ------------------------------------------------------- bookkeeping
        self._prev_kp = kp_xy
        self._prev_des = des
        self._prev_depth = depth_now
        self._prev_t = packet.t
        self._n_frames += 1
        if self.last_state == TRACK_LOST:
            self._n_lost += 1
        elif self.last_state == TRACK_COAST:
            self._n_coast += 1

        res.pose = Pose(float(self.pose.x), float(self.pose.y), float(self.pose.yaw))
        self.traj.append((self.pose.x, self.pose.y))
        res.trajectory = np.asarray(self.traj, np.float32)
        # extra diagnostics (not part of the frozen dataclass, attached for renderers)
        res.scale_source = int(self.last_scale_source)
        res.scale_m = float(self.last_scale)
        res.scale_source_name = SCALE_SOURCE_NAMES[int(self.last_scale_source)]
        res.sampson_px = float(self.last_sampson)
        res.parallax_deg = float(self.last_parallax_deg)
        res.scale_se = float(self.last_scale_se)
        res.scale_n = int(self.last_scale_n)
        res.reason = self.last_reason
        res.state = int(self.last_state)
        res.state_name = TRACK_STATE_NAMES[int(self.last_state)]
        res.lost_frac = self._n_lost / max(self._n_frames, 1)
        res.coast_frac = self._n_coast / max(self._n_frames, 1)

        packet.odom = res
        packet.timings_ms["odom"] = (time.perf_counter() - t_start) * 1000.0
        return packet

    # ---------------------------------------------------------------- tracking
    def _track(self, res: OdometryResult, kp_xy: np.ndarray, des: Optional[np.ndarray],
               depth_now: Optional[np.ndarray], dt: float) -> None:
        p = self.p
        pairs = _mutual_ratio_match(self._bf, self._prev_des, des, p.ratio)
        res.n_matches = int(len(pairs))
        res.flow = np.zeros((0, 4), np.float32)

        if len(pairs) < p.min_matches:
            self._declare_lost(res, f"only {len(pairs)} matches (< {p.min_matches})")
            return

        p0 = self._prev_kp[pairs[:, 0]].astype(np.float64)
        p1 = kp_xy[pairs[:, 1]].astype(np.float64)

        E, emask = cv2.findEssentialMat(p0, p1, self._K, method=self._ransac_method,
                                        prob=p.ransac_conf, threshold=p.ransac_thresh_px)
        if E is None or E.shape != (3, 3) or emask is None:
            self._declare_lost(res, "essential matrix not estimated")
            return
        emask = emask.ravel().astype(bool)
        n_e = int(emask.sum())
        res.n_inliers = n_e
        if n_e < p.min_inliers:
            self._declare_lost(res, f"only {n_e} E-inliers (< {p.min_inliers})")
            return

        # recoverPose selects one of the four (R, t) decompositions by cheirality vote.
        # Its cheirality *count* is kept only as a conditioning diagnostic: at the
        # sub-degree parallax of this footage it is small even for correct solutions.
        n_chi, R, t, _ = cv2.recoverPose(E, p0, p1, self._K,
                                         mask=emask.astype(np.uint8) * 255)
        i0, i1 = p0[emask], p1[emask]
        res.flow = np.concatenate([i0, i1], axis=1).astype(np.float32)

        # ---- geometry quality: Sampson residual + parallax
        F = self._Kinv.T @ E @ self._Kinv
        samp = _sampson(F, i0, i1)
        self.last_sampson = float(np.median(samp))

        f0 = np.hstack([i0, np.ones((len(i0), 1))]) @ self._Kinv.T
        f1 = np.hstack([i1, np.ones((len(i1), 1))]) @ self._Kinv.T
        f0 /= np.linalg.norm(f0, axis=1, keepdims=True)
        f1r = f1 @ R                                   # == (R.T @ f1_i) stacked row-wise
        f1r /= np.linalg.norm(f1r, axis=1, keepdims=True)
        cosang = np.clip(np.einsum("ij,ij->i", f0, f1r), -1.0, 1.0)
        self.last_parallax_deg = float(np.degrees(np.median(np.arccos(cosang))))
        self.last_cheirality = float(n_chi) / max(n_e, 1)

        if self.last_parallax_deg < p.min_parallax_deg:
            self._declare_lost(res, f"degenerate: parallax {self.last_parallax_deg:.3f} deg "
                                    f"(near-pure rotation)")
            return

        d_yaw = _yaw_from_R(R)
        if abs(np.degrees(d_yaw) / dt) > p.max_yaw_rate_dps:
            self._declare_lost(res, f"implausible yaw rate "
                                    f"{np.degrees(d_yaw)/dt:.0f} deg/s")
            return

        # ---- translation direction vs the constant-velocity motion model
        C = (-R.T @ t).ravel()                        # camera-1 origin in camera-0 frame
        step = np.array([C[0], C[2]], np.float64)     # body frame (right, forward)
        nrm = np.linalg.norm(step)
        step = step / nrm if nrm > 1e-9 else self._dir_ema.copy()
        dir_prev = self._dir_ema / (np.linalg.norm(self._dir_ema) + 1e-12)
        if float(step @ dir_prev) < -0.35:
            # documented sign repair: at ~0.3 deg parallax the cheirality vote flips
            # the sign of t on a sizeable minority of frames. Flipping back is the
            # correct decomposition, not a fudge -- E is sign-ambiguous by construction.
            step = -step
            t = -t
        cos_dir = float(step @ dir_prev)
        coasting = cos_dir < float(np.cos(np.deg2rad(p.dir_accept_deg)))

        # ---- depth-anchored scale
        scale, scale_src = self._recover_scale(R, t, i0, i1)
        self.last_scale = float(scale)
        self.last_scale_source = scale_src

        if coasting:
            # keep the previous heading/speed for this frame; the two-view direction is
            # not trustworthy. This mirrors ORB-SLAM's TrackWithMotionModel fallback.
            step = dir_prev
            self.last_state = TRACK_COAST
            self.last_reason = (f"motion-model coast: two-view direction "
                                f"{np.degrees(np.arccos(np.clip(cos_dir,-1,1))):.0f} deg "
                                f"off the constant-velocity prediction")
        else:
            self._dir_ema = ((1 - p.alpha_dir) * self._dir_ema + p.alpha_dir * step)
            self.last_state = TRACK_OK
            self.last_reason = "ok"

        # ---- integrate (light temporal smoothing; the source is handheld-ish RC footage)
        speed_raw = float(scale / dt)
        yawrate_raw = float(d_yaw / dt)
        self._speed_ema = p.alpha_speed * speed_raw + (1 - p.alpha_speed) * self._speed_ema
        self._yawrate_ema = p.alpha_yaw * yawrate_raw + (1 - p.alpha_yaw) * self._yawrate_ema

        d_yaw_s = self._yawrate_ema * dt
        d_len_s = self._speed_ema * dt
        d_right, d_fwd = step * d_len_s

        yaw = self.pose.yaw
        fwd_w = np.array([-np.sin(yaw), np.cos(yaw)])
        right_w = np.array([np.cos(yaw), np.sin(yaw)])
        dw = right_w * d_right + fwd_w * d_fwd
        self.pose.x += float(dw[0])
        self.pose.y += float(dw[1])
        self.pose.yaw = float((yaw + d_yaw_s + np.pi) % (2 * np.pi) - np.pi)

        res.d_trans = float(d_len_s)
        res.d_yaw = float(d_yaw_s)
        res.speed_mps = float(self._speed_ema)
        res.tracking_ok = True
        res.track_quality = self._quality(res)
        if coasting:
            res.track_quality = float(min(res.track_quality, 0.45))

    # ---------------------------------------------------------------- scale
    def _recover_scale(self, R: np.ndarray, t: np.ndarray,
                       i0: np.ndarray, i1: np.ndarray) -> tuple[float, int]:
        """Return (baseline_metres, scale_source_code)."""
        p = self.p
        dmap = self._prev_depth
        self.last_scale_se = 9.9
        self.last_scale_n = 0
        self.last_scale_tri = 0.0
        if dmap is not None:
            u = np.clip(np.round(i0[:, 0]).astype(int), 0, dmap.shape[1] - 1)
            v = np.clip(np.round(i0[:, 1]).astype(int), 0, dmap.shape[0] - 1)
            z_met = dmap[v, u].astype(np.float64)

            # (1) triangulate with the baseline fixed at 1, ratio-fit against metric depth
            P0 = self._K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P1 = self._K @ np.hstack([R, t.reshape(3, 1)])
            X = cv2.triangulatePoints(P0, P1, i0.T.copy(), i1.T.copy())
            with np.errstate(divide="ignore", invalid="ignore"):
                z_tri = np.asarray(X[2] / X[3], np.float64)
            s_tri, se_tri, n_tri = _robust_scale_ratio(z_tri, z_met,
                                                        p.scale_z_min, p.scale_z_max)
            self.last_scale_tri = float(s_tri)

            # (2) refine as a 1-D reprojection least squares on the same depths
            s, se, n_used = _reproj_scale(self._K, self._Kinv, R, t, i0, i1, z_met,
                                          s_tri, p.scale_z_min, p.scale_z_max)
            if not (s > 0 and se < p.max_scale_se and n_used >= p.min_scale_pts):
                s, se, n_used = s_tri, se_tri, n_tri     # fall back to the ratio fit
            self.last_scale_se = float(se)
            self.last_scale_n = int(n_used)

            if s > 1e-4 and n_used >= p.min_scale_pts and se < p.max_scale_se:
                s = float(np.clip(s, 0.0, p.max_step_m))
                # A short running median before the EMA. The per-frame baseline
                # distribution is strongly right-skewed (occasional large spurious
                # values when the parallax collapses); an EMA alone tracks the *mean*
                # of that and came out ~1.7x the plane-based cross-check. The median
                # filter removes the tail, after which the two agree to ~10%.
                self._scale_hist.append(s)
                if len(self._scale_hist) > p.scale_med_win:
                    self._scale_hist.pop(0)
                s = float(np.median(self._scale_hist))
                self._scale_ema = (p.alpha_scale * s + (1 - p.alpha_scale) * self._scale_ema
                                   if self._scale_ema > 0 else s)
                return s, SCALE_DEPTH

        if self._scale_ema > 0:
            return float(self._scale_ema), SCALE_SMOOTHED
        return float(p.fallback_step_m), SCALE_CONST

    # ---------------------------------------------------------------- health
    def _quality(self, res: OdometryResult) -> float:
        inl_ratio = res.n_inliers / max(res.n_matches, 1)
        q_inl = np.clip(inl_ratio / 0.60, 0.0, 1.0)
        q_cnt = np.clip(res.n_matches / 300.0, 0.0, 1.0)
        q_res = np.clip(1.0 - self.last_sampson / 2.0, 0.0, 1.0)
        return float(np.clip(0.45 * q_inl + 0.30 * q_cnt + 0.25 * q_res, 0.0, 1.0))

    def _declare_lost(self, res: OdometryResult, reason: str) -> None:
        """Hold the pose, zero the motion, flag it loudly. No silent drift."""
        res.tracking_ok = False
        res.d_trans = 0.0
        res.d_yaw = 0.0
        self._speed_ema *= 0.6
        self._yawrate_ema *= 0.5
        res.speed_mps = float(self._speed_ema)
        res.track_quality = float(np.clip(0.35 * (res.n_inliers / max(self.p.min_inliers, 1)),
                                          0.0, 0.34))
        self.last_scale_source = SCALE_NONE
        self.last_scale = 0.0
        self.last_state = TRACK_LOST
        self.last_reason = reason


# --------------------------------------------------------------------- self-test

def _selftest(clip_ids=("clip_01", "clip_04"), n_frames: int = 120) -> None:
    import sys
    print("ORB-SLAM3-style monocular front end (Python/OpenCV) -- self test")
    print(f"  K (at {PROC_W}x{PROC_H}) fx={CFG.cam.fx:.1f} cx={CFG.cam.cx:.1f}")
    stage = OdometryStage(verbose=True)
    for cid in clip_ids:
        stage.reset()
        t0 = time.perf_counter()
        inl, mat, q, srcs, speeds, states, par = [], [], [], [], [], [], []
        for i, frame in io_utils.read_frames(cid, max_frames=n_frames):
            pk = FramePacket(clip_id=cid, idx=i, t=i / CLIP_FPS, rgb=frame)
            pk = stage(pk)
            o = pk.odom
            if i == 0:
                continue
            inl.append(o.n_inliers); mat.append(o.n_matches)
            q.append(o.track_quality); states.append(o.state)
            srcs.append(o.scale_source); speeds.append(o.speed_mps)
            par.append(o.parallax_deg)
        el = time.perf_counter() - t0
        inl = np.array(inl); mat = np.array(mat); srcs = np.array(srcs)
        states = np.array(states)
        print(f"\n{cid}: {len(inl)+1} frames, {el*1000/max(len(inl),1):.1f} ms/frame")
        print(f"  matches  mean {mat.mean():6.1f}  p10 {np.percentile(mat,10):6.1f}")
        print(f"  inliers  mean {inl.mean():6.1f}  p10 {np.percentile(inl,10):6.1f}"
              f"  inlier-ratio {np.mean(inl/np.maximum(mat,1)):.3f}")
        print(f"  state: " + ", ".join(
            f"{TRACK_STATE_NAMES[k]}={100*float((states==k).mean()):.1f}%" for k in range(3)))
        print(f"  quality mean {np.mean(q):.3f}   parallax med {np.median(par):.3f} deg")
        print(f"  scale src: " + ", ".join(
            f"{SCALE_SOURCE_NAMES[k]}={int((srcs==k).sum())}" for k in range(4)))
        print(f"  speed m/s mean {np.mean(speeds):.3f} max {np.max(speeds):.3f}")
        tr = np.array(stage.traj)
        print(f"  final pose x={stage.pose.x:+.2f} y={stage.pose.y:+.2f} "
              f"yaw={np.degrees(stage.pose.yaw):+.1f} deg  path_len="
              f"{np.linalg.norm(np.diff(tr,axis=0),axis=1).sum():.2f} m")
        print(f"  last reason: {stage.last_reason}")
    sys.stdout.flush()


if __name__ == "__main__":
    _selftest()
