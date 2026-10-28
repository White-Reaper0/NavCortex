r"""Geometric + semantic PSEUDO-LABEL generator for the traversability / uncertainty heads.

    python -m prism.training.make_trav_labels --per-clip 40 --extra 48

There is no off-road traversability ground truth in this environment - no RELLIS-3D, no
GOOSE, no ORFD, and nobody hand-annotated these clips.  So the supervision is
**derived from the physics of the vehicle envelope**, which is exactly the argument the
project makes: if terrain height exceeds the estimated UGV clearance the region becomes
non-traversable.  Nothing here is a human label.  Every number the trained model is
later scored against is *agreement with these rules*, and must be reported that way.

What goes in
------------
* ``load_stage(clip_id, "depth")`` -> ``depth`` (metric m), ``valid`` (bool), ``q`` (relative inv depth)
* ``load_stage(clip_id, "seg")``   -> ``label`` (prism-7 uint8), ``prob_max``, ``entropy``
                                      (+ optionally ``teacher_label`` / ``teacher_entropy``)
* ``load_stage(clip_id, "odom")``  -> vehicle poses, used only for the depth-uncertainty target
All three are produced by other agents.  If one is missing this script says exactly
which clip and which stage, and can poll for it (``--wait-min``).  It is idempotent:
re-running overwrites ``work/dataset/trav/`` cleanly.

The four-class rule set  (every threshold is a function of ``CFG.ugv`` / ``CFG.bev``)
------------------------------------------------------------------------------------
``h``   = patch-median height above **local** ground (metric BEV reference, see
          ``models.traversability.local_terrain_fields``), metres
``zp``  = height above the single fitted ground *plane*, metres
``s``   = ground slope in degrees   ``r`` = local height std (roughness), metres
STEP  = ugv.max_step_m  = 0.030 m      SLOPE = ugv.max_slope_deg = 22 deg
Z_MIN = bev.z_min       = -0.45 m      HALF_W = ugv.width_m / 2  = 0.110 m

OBSTACLE (2)   any of
  h > STEP, **sustained over half the vehicle footprint** - the exceedance is projected
      into the metric BEV grid, connected components are taken there, and only
      components >= 0.5 * 0.22 m * 0.34 m = 0.037 m^2 (11 cells) survive.  A per-pixel
      3 cm test on monocular depth fires on gravel texture; a kerb survives, speckle
      does not, and the patch size means the same *physical* extent is required at
      0.5 m and at 6 m.
  zp > 8*STEP = 0.24 m            -> a structure, far above any plane-fit drift
  zp < Z_MIN  = -0.45 m           -> negative obstacle / drop-off below the map floor
  s  > 1.4*SLOPE = 30.8 deg       -> beyond any recoverable climb for this chassis
  terrain in {obstacle, dynamic} **and geometry does not veto it** (see below)
RISKY (1)   (only where not OBSTACLE) any of
  h  > 0.5*STEP = 0.015 m         -> a step the chassis can take, but not at speed
  zp < -2*STEP  = -0.060 m        -> a dip, not yet a drop-off
  s  > 0.6*SLOPE = 13.2 deg
  r  > 0.5*STEP = 0.015 m         -> broken ground; wheel contact is not guaranteed
  terrain in {rough_veg, water}
  terrain == grass and the geometry is not clean (grass hides what is underneath)
  terrain in {obstacle, dynamic} but the geometry vetoed it (the two cues disagree)
  within HALF_W + corridor_margin = 0.160 m of an OBSTACLE cell, dilated in **BEV
      metric space**: a 0.16 m halo is a few pixels at 4 m and a third of the image at
      0.6 m, and only a metric dilation gets that right
SAFE (0)
  terrain in {trail, grass}, |h| <= 0.015 m, s <= 13.2 deg, r <= 0.015 m
UNKNOWN (3)
  invalid depth, sky, segmentation prob_max < 0.35 (= safety.conf_unknown) or entropy > 0.80, depth beyond
  D_TRUST = 8.0 m (monocular scale anchored on an assumed 0.12 m camera height is not
  worth trusting further out), a failed ground fit, **or the traced RC-chassis
  silhouette** (``io_utils.ego_mask``) - white bodywork reads as smooth low ground to
  any geometric rule, so it must never be labelled SAFE.
  Exception: a confident obstacle/dynamic terrain pixel stays OBSTACLE even where the
  geometry is unusable.  Failing safe beats failing silent.
255 = ignore: only the thin mixed-pixel band around the chassis silhouette.

The geometry veto
-----------------
The ADE20K-derived terrain teacher calls large patches of this gravel path "wall" on a
ground-level fisheye view.  A pixel sitting within 0.015 m of local ground with low
slope and low roughness is far more likely a segmentation error than a wall lying flat
on the trail, so semantics alone cannot promote it to OBSTACLE - it is demoted to RISKY
(the cues disagree, so it is not SAFE either).  This is the "geometric" half of
"geometric + semantic" doing real work, and it is what keeps OBSTACLE near 22% rather
than 50% of the daylight frames.

Continuous risk target in [0,1]
-------------------------------
A noisy-OR over smoothsteps of the same quantities, so the network can *regress* risk
instead of only classifying it:

    risk = 1 - (1-r_h)(1-r_neg)(1-r_s)(1-r_r)(1-0.80*r_prox)(1-0.65*r_terr)
    r_h    = smoothstep(h,   0.3*STEP, 1.2*STEP)   r_s = smoothstep(s, 0.5*SLOPE, 1.4*SLOPE)
    r_neg  = smoothstep(-zp, 1.0*STEP, 6.0*STEP)   r_r = smoothstep(r, 0.3*STEP, 1.5*STEP)
    r_prox = exp(-d_obstacle_metres / 0.25)        r_terr = 1 - CFG.TERRAIN_DRIVE_PRIOR[class]
    risk = 1 where OBSTACLE, floored at 0.50 where UNKNOWN (unknown ground is not free)

"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..config import (CFG, CLIP_IDS, DATA_DIR, N_TERRAIN, N_TRAV, PROC_H, PROC_W,
                      TERRAIN_DRIVE_PRIOR, VIDEO_IN, SAFE, RISKY, OBSTACLE, UNKNOWN)
from ..io_utils import (ego_mask, has_stage, load_stage, read_frames, save_stage,
                        clip_path, device as pick_device)
from ..models.traversability import (IN_CH, NET_H, NET_W, build_feature_stack,
                                     geometry_from_depth, resize_stack)
from ..models.uncertainty import (depth_error_target, seg_error_target,
                                  warp_prev_into_current)
from ..perception import geometry as G

OUT_DIR = DATA_DIR / "trav"
EXTRA_CLIP = "extra_input"          # pseudo-clip holding frames sampled from video/input.mp4

# ------------------------------------------------------------------ thresholds
U, B, S = CFG.ugv, CFG.bev, CFG.safety
STEP = U.max_step_m                      # 0.030 m
CLEAR = U.clearance_m                    # 0.045 m (reported to the viewer; STEP binds first)
HALF_W = U.width_m / 2.0                 # 0.110 m
SLOPE_MAX = U.max_slope_deg              # 22 deg
Z_MIN = B.z_min                          # -0.45 m

T_OBST_H = STEP                          # 0.030  above LOCAL ground
T_RISK_H = 0.5 * STEP                    # 0.015
T_STRUCT = 8.0 * STEP                    # 0.240  above the fitted PLANE -> a structure
T_NEG_RISK = -2.0 * STEP                 # -0.060
T_OBST_SLOPE = 1.4 * SLOPE_MAX           # 30.8
T_RISK_SLOPE = 0.6 * SLOPE_MAX           # 13.2
T_RISK_ROUGH = 0.5 * STEP                # 0.015
DILATE_M = HALF_W + S.corridor_margin_m  # 0.160
D_TRUST = 8.0                            # m; beyond this monocular scale is extrapolation
SEG_CONF_MIN = S.conf_unknown            # 0.35, the supervisor's own UNKNOWN gate
SEG_ENT_MAX = 0.80
RISK_UNKNOWN_FLOOR = 0.50

THRESHOLDS = dict(
    max_step_m=STEP, clearance_m=CLEAR, half_width_m=HALF_W, max_slope_deg=SLOPE_MAX,
    z_min_m=Z_MIN, obst_height_m=T_OBST_H, risky_height_m=T_RISK_H,
    structure_height_above_plane_m=T_STRUCT,
    min_obstacle_patch_m2=0.5 * U.width_m * U.length_m,
    neg_risky_height_m=T_NEG_RISK, obst_slope_deg=T_OBST_SLOPE, risky_slope_deg=T_RISK_SLOPE,
    risky_rough_m=T_RISK_ROUGH, obstacle_dilate_m=DILATE_M, depth_trust_m=D_TRUST,
    seg_conf_min=SEG_CONF_MIN, seg_entropy_max=SEG_ENT_MAX,
    risk_unknown_floor=RISK_UNKNOWN_FLOOR,
    note="all thresholds derive from CFG.ugv / CFG.bev; metric scale is anchored on the "
         "ASSUMED camera height CFG.cam.height_above_ground_m = "
         f"{CFG.cam.height_above_ground_m} m, not on a calibration",
)


def smoothstep(x: np.ndarray, e0: float, e1: float) -> np.ndarray:
    t = np.clip((np.nan_to_num(x, nan=e0) - e0) / max(e1 - e0, 1e-9), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


# ------------------------------------------------------------------ BEV metric dilation

def _sustained_over_footprint(points_veh: np.ndarray, geo_valid: np.ndarray,
                              raw: np.ndarray) -> np.ndarray:
    """Keep only height exceedances that persist over a patch the size of the vehicle.

    A per-pixel "height > 3 cm" test on monocular depth fires on gravel texture and on
    every depth-edge speckle.  What actually stops the vehicle is a *sustained* rise, so
    the raw exceedance is projected into the metric BEV grid, connected components are
    taken there, and only components of at least half the vehicle footprint
    (0.5 * 0.22 m * 0.34 m = 0.037 m^2 = 11 cells at 0.06 m) survive.  Doing it in BEV
    rather than in pixels means the same physical patch size is required at 0.5 m and at
    6 m, which an image-space morphology cannot give.
    """
    min_cells = max(4, int(np.ceil(0.5 * U.width_m * U.length_m / (B.res_m ** 2))))
    row, col, keep = G.bev_indices(points_veh, geo_valid)
    sel = keep & raw
    if not sel.any():
        return np.zeros_like(raw)
    grid = np.zeros((B.H, B.W), np.uint8)
    grid[row[sel], col[sel]] = 1
    n, cc, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    good = np.zeros(n, bool)
    good[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_cells
    out = np.zeros_like(raw)
    out[sel] = good[cc[row[sel], col[sel]]]
    return out


def _bev_obstacle_fields(points_veh: np.ndarray, geo_valid: np.ndarray,
                         obst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dilate obstacles by the vehicle half-width **in metric BEV space** and measure
    the metric distance from every pixel to the nearest obstacle cell.

    Returns ``(near_obstacle_mask, distance_to_obstacle_metres)`` back in image space.
    """
    h, w = geo_valid.shape
    near = np.zeros((h, w), bool)
    dist = np.full((h, w), 9.9, np.float32)

    row, col, keep = G.bev_indices(points_veh, geo_valid)
    if not keep.any():
        return near, dist

    grid = np.zeros((B.H, B.W), np.uint8)
    om = keep & obst
    if om.any():
        grid[row[om], col[om]] = 1
    if grid.any():
        k = int(np.ceil(DILATE_M / B.res_m))                # 0.16 / 0.06 -> 3 cells
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        dil = cv2.dilate(grid, ker)
        # metric distance transform on the un-dilated occupancy
        dt = cv2.distanceTransform((1 - grid).astype(np.uint8), cv2.DIST_L2, 3) * B.res_m
    else:
        dil = grid
        dt = np.full((B.H, B.W), 9.9, np.float32)

    near[keep] = dil[row[keep], col[keep]] > 0
    dist[keep] = np.minimum(dt[row[keep], col[keep]], 9.9)
    return near, dist


# ------------------------------------------------------------------ the rules

def pseudo_label_frame(depth_m: np.ndarray, valid: np.ndarray,
                       height: np.ndarray, slope: np.ndarray, rough: np.ndarray,
                       points_veh: np.ndarray,
                       seg_label: np.ndarray, prob_max: np.ndarray, entropy: np.ndarray,
                       fit_ok: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply the documented rule set.  Returns (label uint8 [0..3, 255], risk float32, aux)."""
    h_img, w_img = depth_m.shape
    em = ego_mask(h_img, w_img)

    # `height` is height above LOCAL ground (models.traversability.local_ground_height);
    # z_plane is height above the single fitted plane, which is what the drop-off and
    # "definitely a structure" tests want.
    z_plane = points_veh[..., 2].astype(np.float32)
    hgt = np.asarray(height, np.float32)
    h_med = cv2.medianBlur(np.nan_to_num(hgt, nan=0.0), 5)
    h_med[~np.asarray(valid, bool)] = np.nan
    slp = np.nan_to_num(slope, nan=0.0).astype(np.float32)
    rgh = np.nan_to_num(rough, nan=0.0).astype(np.float32)
    lab = np.asarray(seg_label, np.int32)
    pmx = np.nan_to_num(prob_max, nan=0.0).astype(np.float32)
    ent = np.nan_to_num(entropy, nan=1.0).astype(np.float32)
    d = np.nan_to_num(depth_m, nan=1e3).astype(np.float32)

    geo_ok = np.asarray(valid, bool) & em & np.isfinite(h_med)

    # ---------------- OBSTACLE ----------------
    # 1) a step above local ground, sustained over half the vehicle footprint in BEV
    step_raw = (geo_ok & (h_med > T_OBST_H))
    ker3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    step_raw = cv2.morphologyEx(step_raw.astype(np.uint8), cv2.MORPH_OPEN, ker3).astype(bool)
    step_patch = _sustained_over_footprint(points_veh, geo_ok, step_raw)
    # 2) anything standing T_STRUCT above the fitted plane is a structure, not terrain
    struct = geo_ok & (z_plane > T_STRUCT)

    # 3) semantics say obstacle/dynamic - but geometry gets a veto.  The ADE20K-derived
    #    teacher calls large patches of gravel path "wall" on this ground-level fisheye
    #    view; a pixel sitting within half a max-step of local ground, with low slope and
    #    low roughness, is far more likely a segmentation error than a wall lying flat on
    #    the trail.  Those are demoted to RISKY (the two cues disagree, so not SAFE) -
    #    this is the geometric half of "geometric + semantic" actually doing work.
    flat_evidence = (geo_ok & (np.abs(h_med) <= T_RISK_H)
                     & (slp <= T_RISK_SLOPE) & (rgh <= T_RISK_ROUGH))
    sem_obst = np.isin(lab, (4, 6)) & ~flat_evidence

    obst = (step_patch
            | struct
            | (geo_ok & (z_plane < Z_MIN))
            | sem_obst
            | (geo_ok & (slp > T_OBST_SLOPE)))
    obst &= em

    # ---------------- proximity in metric BEV ----------------
    near_obst, dist_obst = _bev_obstacle_fields(points_veh, geo_ok, obst)

    # ---------------- SAFE geometry gate ----------------
    safe_geo = (geo_ok & (np.abs(h_med) <= T_RISK_H)
                & (slp <= T_RISK_SLOPE) & (rgh <= T_RISK_ROUGH))

    # ---------------- RISKY ----------------
    risky = (~obst) & (
        (geo_ok & (h_med > T_RISK_H))
        | (geo_ok & (z_plane < T_NEG_RISK))
        | (geo_ok & (slp > T_RISK_SLOPE))
        | (geo_ok & (rgh > T_RISK_ROUGH))
        | np.isin(lab, (3, 5))
        | ((lab == 2) & ~safe_geo)
        | (np.isin(lab, (4, 6)) & flat_evidence)   # semantics/geometry disagree
        | near_obst
    )
    risky &= em

    # ---------------- SAFE ----------------
    safe = (~obst) & (~risky) & np.isin(lab, (1, 2)) & safe_geo

    # ---------------- UNKNOWN ----------------
    unknown = (~np.asarray(valid, bool)) | (lab == 0) | (pmx < SEG_CONF_MIN) \
        | (ent > SEG_ENT_MAX) | (d > D_TRUST) | (not fit_ok)
    hard_obst = np.isin(lab, (4, 6)) & (pmx >= 0.50) & ~flat_evidence
    unknown = unknown & ~hard_obst

    out = np.full((h_img, w_img), UNKNOWN, np.uint8)
    out[safe] = SAFE
    out[risky] = RISKY
    out[obst] = OBSTACLE
    out[unknown] = UNKNOWN
    out[obst & hard_obst] = OBSTACLE
    # The traced RC-chassis silhouette is UNKNOWN, never SAFE: white bodywork reads as
    # smooth low ground to any geometric rule, which is precisely the wrong lesson.
    # A thin band around the silhouette holds mixed pixels and is ignored (255).
    out[~em] = UNKNOWN
    band = cv2.dilate((~em).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(bool) & em
    out[band] = 255

    # ---------------- continuous risk ----------------
    r_h = smoothstep(np.where(geo_ok, h_med, 0.0), 0.3 * STEP, 1.2 * STEP)
    r_neg = smoothstep(np.where(geo_ok, -z_plane, 0.0), 1.0 * STEP, 6.0 * STEP)
    r_s = smoothstep(np.where(geo_ok, slp, 0.0), 0.5 * SLOPE_MAX, 1.4 * SLOPE_MAX)
    r_r = smoothstep(np.where(geo_ok, rgh, 0.0), 0.3 * STEP, 1.5 * STEP)
    r_prox = np.exp(-dist_obst / 0.25).astype(np.float32)
    r_terr = (1.0 - TERRAIN_DRIVE_PRIOR[np.clip(lab, 0, N_TERRAIN - 1)]).astype(np.float32)

    keep_prob = ((1 - r_h) * (1 - r_neg) * (1 - r_s) * (1 - r_r)
                 * (1 - 0.80 * r_prox) * (1 - 0.65 * r_terr))
    risk = np.clip(1.0 - keep_prob, 0.0, 1.0).astype(np.float32)
    risk[obst] = 1.0
    risk[unknown & ~obst] = np.maximum(risk[unknown & ~obst], RISK_UNKNOWN_FLOOR)
    risk = cv2.blur(risk, (5, 5)).astype(np.float32)
    risk[~em] = RISK_UNKNOWN_FLOOR

    scene = out != 255
    aux = dict(near_obst=near_obst, dist_obst=dist_obst, h_med=h_med,
               frac=np.array([(out[scene] == c).mean() for c in range(N_TRAV)], np.float32))
    return out, risk, aux


# ------------------------------------------------------------------ odometry / relative pose

def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def load_relative_motion(clip_id: str, n_frames: int):
    """(dx, dy, dyaw) per frame in the PREVIOUS vehicle frame; index 0 is zero motion.

    Reads whatever the odometry agent wrote.  Understood key sets, in order:
      ``pose`` / ``poses`` / ``xyyaw``  as (N,3) absolute (x, y, yaw)
      ``x``,``y``,``yaw``               as three (N,) arrays
      ``d_trans``,``d_yaw``             per-frame deltas (assumes motion along heading)
    Returns ``(dx, dy, dyaw, source)`` or ``(None, None, None, "missing")``.
    """
    if not has_stage(clip_id, "odom"):
        return None, None, None, "missing"
    z = load_stage(clip_id, "odom")
    pose = None
    for k in ("pose", "poses", "xyyaw", "pose_xyyaw"):
        if k in z and np.asarray(z[k]).ndim == 2 and np.asarray(z[k]).shape[1] >= 3:
            pose = np.asarray(z[k], np.float64)[:, :3]
            break
    if pose is None and all(k in z for k in ("x", "y", "yaw")):
        pose = np.stack([np.asarray(z["x"], np.float64),
                         np.asarray(z["y"], np.float64),
                         np.asarray(z["yaw"], np.float64)], 1)
    n = n_frames
    dx = np.zeros(n, np.float32)
    dy = np.zeros(n, np.float32)
    dyaw = np.zeros(n, np.float32)
    if pose is not None:
        m = min(n, len(pose))
        for i in range(1, m):
            th = pose[i - 1, 2]
            c, s = np.cos(-th), np.sin(-th)
            ddx = pose[i, 0] - pose[i - 1, 0]
            ddy = pose[i, 1] - pose[i - 1, 1]
            dx[i] = c * ddx - s * ddy
            dy[i] = s * ddx + c * ddy
            dyaw[i] = _wrap(pose[i, 2] - pose[i - 1, 2])
        return dx, dy, dyaw, "odom:absolute_pose"
    if "d_trans" in z and "d_yaw" in z:
        dt = np.asarray(z["d_trans"], np.float32).ravel()
        dw = np.asarray(z["d_yaw"], np.float32).ravel()
        m = min(n, len(dt), len(dw))
        dy[:m] = dt[:m]                    # motion along the previous heading
        dyaw[:m] = dw[:m]
        return dx, dy, dyaw, "odom:d_trans/d_yaw"
    return None, None, None, f"odom present but unrecognised keys: {sorted(z)}"


_ORB = None


def _orb_yaw(gray_prev, gray_cur, fit) -> tuple[float, bool]:
    """Yaw change between two frames from ORB + the essential matrix.

    Only the *rotation* is taken from here.  The two-view translation is recovered only
    up to scale, and every metric scale estimate available offline (matching depth maps,
    or plane-projected features) is swamped by the frame-to-frame drift of the monocular
    depth itself at this vehicle scale (~0.075 m of motion against a ~12% depth
    inconsistency at a 1 m median range).  Rotation, by contrast, is well conditioned.
    """
    global _ORB
    if _ORB is None:
        _ORB = cv2.ORB_create(1500, scaleFactor=1.2, nlevels=6)
    K = CFG.cam.K
    m8 = (ego_mask(*gray_prev.shape).astype(np.uint8) * 255)
    k1, d1 = _ORB.detectAndCompute(gray_prev, m8)
    k2, d2 = _ORB.detectAndCompute(gray_cur, m8)
    if d1 is None or d2 is None or len(k1) < 30 or len(k2) < 30:
        return 0.0, False
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = bf.knnMatch(d1, d2, k=2)
    good = [a for a, b in (q for q in raw if len(q) == 2) if a.distance < 0.78 * b.distance]
    if len(good) < 25:
        return 0.0, False
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    E, inl = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=1.2)
    if E is None or E.shape != (3, 3) or inl is None or int(inl.sum()) < 15:
        return 0.0, False
    _, R, _t, mask = cv2.recoverPose(E, p1, p2, K, mask=inl.copy())
    if int((mask.ravel() > 0).sum()) < 12:
        return 0.0, False
    Rb = G.vehicle_basis(fit.normal)
    M = Rb @ np.asarray(R, np.float32) @ Rb.T
    dyaw = float(np.arctan2(-M[1, 0], M[0, 0]))
    return (dyaw, True) if abs(dyaw) < np.deg2rad(12.0) else (0.0, False)


def relative_motion_fallback(gray_prev: np.ndarray, gray_cur: np.ndarray,
                             depth_prev: np.ndarray, depth_cur: np.ndarray,
                             valid_prev: np.ndarray, valid_cur: np.ndarray,
                             fit: "G.GroundFit",
                             speeds=np.arange(0.0, 0.155, 0.02)):
    """Relative pose when the VO cache is absent: ORB yaw + a *direct* forward-speed search.

    The forward translation is chosen as the value in ``speeds`` (0 to 0.15 m per frame,
    i.e. up to 4.5 m/s, comfortably above ``ugv.max_speed_mps``) that minimises the
    geometric reprojection residual - a direct/photometric speed estimate rather than a
    feature-triangulated one, because two-view triangulation cannot beat the depth
    map's own drift here.

    **Honesty note that must survive into the report:** because the translation is chosen
    to minimise the very residual that becomes the uncertainty target, targets built this
    way are a *conservative lower bound* on the true depth error.  Whenever
    ``load_stage(clip_id, "odom")`` exists its pose is used instead and this function is
    not called.  Returns ``(dx, dy, dyaw, ok)`` in the previous vehicle frame.
    """
    dyaw, ok = _orb_yaw(gray_prev, gray_cur, fit)
    dc = np.nan_to_num(depth_cur, nan=0.0).astype(np.float32)
    base = np.asarray(valid_cur, bool) & (dc > 0.05) & ego_mask(*dc.shape)
    best, best_r = 0.0, np.inf
    for s in speeds:
        wd, _wg, cov = warp_prev_into_current(depth_prev, valid_prev, gray_prev,
                                              fit, fit, 0.0, float(s), dyaw)
        m = base & cov & np.isfinite(wd)
        if m.sum() < 2000:
            continue
        r = float(np.mean(np.abs(dc[m] - wd[m]) / np.maximum(dc[m], 0.20)))
        if r < best_r:
            best_r, best = r, float(s)
    return 0.0, best, dyaw, bool(ok or best > 0)

# ------------------------------------------------------------------ SegFormer teacher / extras

# ADE20K -> prism-7, matched on the model's own id2label strings (ordered, first hit
# wins) so it survives any id reshuffle.  Anything unmatched falls to `obstacle`, which
# is the conservative choice for a navigation stack.
_ADE_RULES = [
    (("sky",), 0),
    (("person", "animal", "car", "truck", "bus", "van", "bicycle", "minibike",
      "motorbike", "boat", "train", "airplane"), 6),
    (("waterfall", "water", "sea", "river", "lake", "swimming pool"), 5),
    (("grass", "field"), 2),
    (("tree", "plant", "bush", "flower", "palm", "hedge", "vegetation"), 3),
    (("road", "sidewalk", "pavement", "earth", "ground", "path", "dirt track", "sand",
      "floor", "runway"), 1),
]
_TOKEN_RE = None


def _ade_to_prism77(id2label: dict) -> np.ndarray:
    """Whole-word matching: substring matching turns 'skyscraper' into sky, 'seat' into
    water and 'streetlight' into a tree.  Multi-word keys still match as phrases."""
    import re
    n = max(int(k) for k in id2label) + 1
    lut = np.full(n, 4, np.int64)                # default: obstacle (conservative)
    for i in range(n):
        name = str(id2label.get(i, id2label.get(str(i), ""))).lower()
        toks = set(re.split(r"[^a-z]+", name)) - {""}
        for keys, cls in _ADE_RULES:
            if any((k in toks) if " " not in k else (k in name) for k in keys):
                lut[i] = cls
                break
    return lut


class SegFormerTeacher:
    """SegFormer-B0 ADE20K teacher, mapped into prism-7 by summing class probabilities.

    Used here for two offline jobs only: (a) the student/teacher disagreement target for
    the uncertainty head, (b) terrain labels for extra frames sampled from
    ``video/input.mp4`` that have no seg cache.  It never runs at inference.
    """

    REPO = "nvidia/segformer-b0-finetuned-ade-512-512"

    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        from ..models.pretrained_paths import (SEGMENTATION_ENV_VAR,
                                                SEGMENTATION_MODEL_ID, local_model_dir)
        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        local = local_model_dir(SEGMENTATION_MODEL_ID, env_var=SEGMENTATION_ENV_VAR)
        source = str(local) if local is not None else self.REPO
        self.model_source = source
        # local_files_only: this offline label job must never start a download of its own
        self.proc = SegformerImageProcessor.from_pretrained(source, local_files_only=True)
        self.net = SegformerForSemanticSegmentation.from_pretrained(
            source, local_files_only=True).to(self.device).eval()
        self.lut = _ade_to_prism77(self.net.config.id2label)

    def __call__(self, rgb_bgr: np.ndarray):
        import torch
        import torch.nn.functional as F
        h, w = rgb_bgr.shape[:2]
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        inp = self.proc(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            lg = self.net(**inp).logits
            lg = F.interpolate(lg, size=(h, w), mode="bilinear", align_corners=False)
            p = torch.softmax(lg.float(), 1)[0]                     # (150,H,W)
            lut = torch.as_tensor(self.lut, device=p.device)
            p7 = torch.zeros((N_TERRAIN, h, w), device=p.device, dtype=p.dtype)
            p7.index_add_(0, lut, p)
            p7 = p7 / p7.sum(0, keepdim=True).clamp_min(1e-6)
            pm, lab = p7.max(0)
            ent = -(p7 * p7.clamp_min(1e-8).log()).sum(0) / float(np.log(N_TERRAIN))
        return (lab.byte().cpu().numpy(), pm.float().cpu().numpy(),
                ent.clamp(0, 1).float().cpu().numpy())


class DepthAnythingRunner:
    """Depth-Anything-V2-Small, used only to give extra input.mp4 frames a `q` map."""

    REPO = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        from ..models.pretrained_paths import DEPTH_ENV_VAR, DEPTH_MODEL_ID, local_model_dir
        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        local = local_model_dir(DEPTH_MODEL_ID, env_var=DEPTH_ENV_VAR)
        source = str(local) if local is not None else self.REPO
        self.model_source = source
        # local_files_only: same rule as the segmentation teacher - no surprise download
        self.proc = AutoImageProcessor.from_pretrained(source, local_files_only=True)
        self.net = AutoModelForDepthEstimation.from_pretrained(
            source, local_files_only=True).to(self.device).eval()

    def __call__(self, rgb_bgr: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        h, w = rgb_bgr.shape[:2]
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        inp = self.proc(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            q = self.net(**inp).predicted_depth            # relative INVERSE depth
            q = F.interpolate(q[:, None].float(), size=(h, w), mode="bicubic",
                              align_corners=False)[0, 0]
        return q.cpu().numpy().astype(np.float32)


def provisional_seg(clip_id: str, device: str = "cuda", force: bool = False) -> dict:
    """Fallback terrain labels for a clip whose ``seg`` cache does not exist yet.

    Runs the SegFormer-B0 *teacher* over the clip and caches it as
    ``seg_provisional`` - a **different** stage name, so it can never be mistaken for,
    or clobber, the segmentation agent's real PIDNet-S student output.  Anything built
    on it is tagged ``seg_source = "provisional_segformer_teacher"`` and must be rebuilt
    once the real cache lands.
    """
    if not force and has_stage(clip_id, "seg_provisional"):
        return load_stage(clip_id, "seg_provisional")
    tea = SegFormerTeacher(device)
    labs, pms, ents = [], [], []
    t0 = time.time()
    for i, f in read_frames(clip_id):
        l, p, e = tea(f)
        labs.append(l)
        pms.append(p.astype(np.float16))
        ents.append(e.astype(np.float16))
    # No teacher_* keys on purpose: here the "student" IS the teacher, so a
    # student/teacher disagreement target would be identically zero.  Leaving them out
    # makes seg_error_target fall back to the honest student-confidence proxy.
    out = dict(label=np.stack(labs), prob_max=np.stack(pms), entropy=np.stack(ents))
    save_stage(clip_id, "seg_provisional", **out)
    print(f"[seg-fallback] {clip_id}: {len(labs)} frames from the SegFormer teacher "
          f"in {time.time()-t0:.0f}s (PROVISIONAL - rebuild when seg.npz exists)")
    return out


def load_seg(clip_id: str, device: str = "cuda", fallback: bool = False) -> tuple[dict, str]:
    """Real seg cache if present, else (optionally) the provisional teacher one."""
    if has_stage(clip_id, "seg"):
        return _fetch(clip_id, "seg"), "seg_cache"
    if fallback:
        return provisional_seg(clip_id, device), "provisional_segformer_teacher"
    return _fetch(clip_id, "seg"), "seg_cache"      # raises with the helpful message


def ensure_extra_cache(n_pairs: int, device: str, force: bool = False) -> Optional[int]:
    """Sample ``n_pairs`` (t-1, t) frame pairs from video/input.mp4 and cache depth+seg.

    Gives the traversability head scenes the 5 clips never show (different lighting,
    surfaces and clutter across the whole source video), which is the only reason the
    extras exist.  Returns the number of *current* frames cached, or None on failure.
    """
    if n_pairs <= 0:
        return 0
    if not VIDEO_IN.exists():
        print(f"[extra] {VIDEO_IN} not found - skipping extra frames")
        return None
    if (not force and has_stage(EXTRA_CLIP, "depth") and has_stage(EXTRA_CLIP, "seg")
            and has_stage(EXTRA_CLIP, "frames")):
        n = len(load_stage(EXTRA_CLIP, "frames")["rgb"]) // 2
        if n >= n_pairs:
            print(f"[extra] reusing cached extras ({n} pairs)")
            return n
    try:
        da = DepthAnythingRunner(device)
        tea = SegFormerTeacher(device)
    except Exception as exc:                     # noqa: BLE001 - offline model may be absent
        print(f"[extra] could not load offline teachers ({type(exc).__name__}: {exc}); "
              "continuing with clip frames only")
        return None

    cap = cv2.VideoCapture(str(VIDEO_IN))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 60:
        cap.release()
        print("[extra] input.mp4 too short - skipping")
        return None
    idxs = np.linspace(int(0.02 * total), int(0.98 * total), n_pairs).astype(int)
    rgbs, qs, labs, pms, ents = [], [], [], [], []
    t0 = time.time()
    for j, i in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(i - 1, 0)))
        ok1, f_prev = cap.read()
        ok2, f_cur = cap.read()
        if not (ok1 and ok2):
            continue
        for f in (f_prev, f_cur):
            f = cv2.resize(f, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)
            rgbs.append(f)
            qs.append(da(f))
            l, p, e = tea(f)
            labs.append(l)
            pms.append(p)
            ents.append(e)
        if (j + 1) % 8 == 0:
            print(f"[extra] {j+1}/{len(idxs)} pairs  {time.time()-t0:.0f}s", flush=True)
    cap.release()
    if len(rgbs) < 4:
        print("[extra] no usable frames")
        return None

    rgb = np.stack(rgbs)
    q = np.stack(qs).astype(np.float32)
    valid = np.broadcast_to(ego_mask(PROC_H, PROC_W), q.shape).copy()
    # metric depth via the shared ground-plane solve, exactly as for the clips
    depth = np.empty_like(q)
    vout = np.empty_like(valid)
    lab_a = np.stack(labs).astype(np.uint8)
    for i in range(len(q)):
        fit = G.fit_metric_ground(q[i], valid[i], lab_a[i])
        if not fit.ok:
            fit = G.GroundFit(a=1.0, b=0.0, normal=np.array([0, 1, 0], np.float32),
                              height=CFG.cam.height_above_ground_m, ok=False)
        d, v = G.depth_from_q(q[i], fit)
        depth[i] = d
        vout[i] = v & valid[i]
    save_stage(EXTRA_CLIP, "frames", rgb=rgb)
    save_stage(EXTRA_CLIP, "depth", depth=depth.astype(np.float16), valid=vout, q=q.astype(np.float16))
    save_stage(EXTRA_CLIP, "seg", label=lab_a, prob_max=np.stack(pms).astype(np.float16),
               entropy=np.stack(ents).astype(np.float16))
    print(f"[extra] cached {len(rgb)//2} pairs from input.mp4 in {time.time()-t0:.0f}s")
    return len(rgb) // 2


# ------------------------------------------------------------------ cache access

_STAGE_KEYS = {"depth": ("depth", "valid", "q"), "seg": ("label", "prob_max", "entropy")}


def _fetch(clip_id: str, stage: str) -> dict:
    if not has_stage(clip_id, stage):
        raise FileNotFoundError(
            f"cache stage '{stage}' for {clip_id} is missing.\n"
            f"  expected: work/cache/{clip_id}/{stage}.npz\n"
            f"  it is produced by another agent (depth -> models/depth.py, "
            f"seg -> models/seg_pidnet.py, odom -> perception/odometry.py).\n"
            f"  Re-run this script once it exists, or pass --wait-min to poll.")
    z = load_stage(clip_id, stage)
    for k in _STAGE_KEYS.get(stage, ()):
        if k not in z:
            raise KeyError(f"{clip_id}/{stage}.npz has keys {sorted(z)}; expected '{k}'")
    return z


def wait_for_caches(clip_ids, stages=("depth", "seg"), minutes: float = 0.0) -> list[str]:
    """Poll until every (clip, stage) exists or the budget runs out.  Returns missing list."""
    deadline = time.time() + minutes * 60.0
    while True:
        missing = [f"{c}/{s}" for c in clip_ids for s in stages if not has_stage(c, s)]
        if not missing or time.time() >= deadline:
            return missing
        print(f"[wait] still missing {len(missing)}: {missing[:6]}"
              f"{' ...' if len(missing) > 6 else ''}  "
              f"({(deadline-time.time())/60:.1f} min left)", flush=True)
        time.sleep(20)


# ------------------------------------------------------------------ dataset build

def _read_clip_rgb(clip_id: str, idxs: np.ndarray) -> dict[int, np.ndarray]:
    want = set(int(i) for i in idxs) | set(int(i) - 1 for i in idxs if i > 0)
    out = {}
    for i, f in read_frames(clip_id):
        if i in want:
            out[i] = f
        if i > max(want):
            break
    return out


def build_dataset(per_clip: int = 40, extra_pairs: int = 48, device: str = "cuda",
                  clip_ids=None, use_teacher: bool = True, wait_min: float = 0.0,
                  seg_fallback: bool = False, out_dir: Path = OUT_DIR) -> dict:
    clip_ids = list(clip_ids or CLIP_IDS)
    missing = wait_for_caches(clip_ids, ("depth", "seg"), wait_min)
    if missing:
        print(f"[cache] MISSING: {missing}"
              + ("  -> using the PROVISIONAL SegFormer teacher for terrain"
                 if seg_fallback else ""))
    usable = [c for c in clip_ids
              if has_stage(c, "depth") and (has_stage(c, "seg") or seg_fallback)]
    if not usable:
        raise FileNotFoundError(
            "no clip has both a depth and a seg cache yet. The depth/seg agents must "
            "run first (see the message above), or pass --seg-fallback to build "
            "provisional terrain labels from the SegFormer teacher.")

    n_extra = ensure_extra_cache(extra_pairs, device) or 0
    sources = [(c, per_clip) for c in usable]
    if n_extra:
        sources.append((EXTRA_CLIP, n_extra))

    teacher = None
    if use_teacher:
        try:
            teacher = SegFormerTeacher(device)
            print("[teacher] SegFormer-B0 loaded for the student/teacher disagreement target")
        except Exception as exc:                 # noqa: BLE001
            print(f"[teacher] unavailable ({type(exc).__name__}: {exc}); "
                  "seg-error target falls back to the student-confidence proxy")

    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.npy"):
        f.unlink()

    feats, travs, risks, uncs, uncws, meta = [], [], [], [], [], []
    cls_hist = np.zeros(N_TRAV + 1, np.int64)
    seg_modes: dict[str, int] = {}
    seg_sources: dict[str, str] = {}
    motion_src: dict[str, str] = {}

    for clip_id, n_want in sources:
        is_extra = clip_id == EXTRA_CLIP
        dz = _fetch(clip_id, "depth")
        sz, seg_src = load_seg(clip_id, device, fallback=seg_fallback and not is_extra)
        seg_sources[clip_id] = seg_src
        depth_all = np.asarray(dz["depth"], np.float32)
        valid_all = np.asarray(dz["valid"], bool)
        q_all = np.asarray(dz["q"], np.float32)
        lab_all = np.asarray(sz["label"], np.uint8)
        pm_all = np.asarray(sz["prob_max"], np.float32)
        en_all = np.asarray(sz["entropy"], np.float32)
        t_lab_all = np.asarray(sz["teacher_label"], np.uint8) if "teacher_label" in sz else None
        t_ent_all = np.asarray(sz["teacher_entropy"], np.float32) if "teacher_entropy" in sz else None
        n_frames = len(depth_all)

        if is_extra:
            rgb_all = np.asarray(load_stage(EXTRA_CLIP, "frames")["rgb"], np.uint8)
            cur_idx = np.arange(1, n_frames, 2)                  # pairs are [prev, cur, ...]
            dx = dy = dyaw = None
            motion_src[clip_id] = "orb_yaw + direct forward-speed search (input.mp4 frames have no VO track)"
        else:
            cur_idx = np.linspace(1, n_frames - 1, min(n_want, n_frames - 1)).astype(int)
            cur_idx = np.unique(cur_idx)
            frames = _read_clip_rgb(clip_id, cur_idx)
            rgb_all = None
            dx, dy, dyaw, src = load_relative_motion(clip_id, n_frames)
            motion_src[clip_id] = src if dx is not None else ("orb_yaw + direct forward-speed "
                                    "search (odom cache absent; target is a conservative lower bound)")

        fit_prev_cache: dict[int, "G.GroundFit"] = {}
        prior = None
        t0 = time.time()
        for k, i in enumerate(cur_idx):
            i = int(i)
            ip = i - 1
            if is_extra:
                rgb_c, rgb_p = rgb_all[i], rgb_all[ip]
            else:
                if i not in frames or ip not in frames:
                    continue
                rgb_c, rgb_p = frames[i], frames[ip]

            if "normal" in dz and "scale" in dz:
                # Reuse the depth stage's own solve rather than re-fitting: it is both
                # faster and guaranteed consistent with the depth map we were handed.
                nv = np.asarray(dz["normal"][i], np.float32)
                nv = nv / max(float(np.linalg.norm(nv)), 1e-9)
                fit_c = G.GroundFit(a=float(dz["scale"][i]), b=float(dz["shift"][i]),
                                    normal=nv, height=CFG.cam.height_above_ground_m,
                                    residual=float(dz["residual"][i]) if "residual" in dz else 0.0,
                                    ok=True)
            else:
                fit_c = G.fit_metric_ground(q_all[i], valid_all[i], lab_all[i], prior=prior)
            if fit_c.ok:
                prior = fit_c
            if "normal" in dz and "scale" in dz:
                np_ = np.asarray(dz["normal"][ip], np.float32)
                np_ = np_ / max(float(np.linalg.norm(np_)), 1e-9)
                fit_p = G.GroundFit(a=float(dz["scale"][ip]), b=float(dz["shift"][ip]),
                                    normal=np_, height=CFG.cam.height_above_ground_m, ok=True)
            else:
                fit_p = G.fit_metric_ground(q_all[ip], valid_all[ip], lab_all[ip])
            if not fit_p.ok:
                fit_p = fit_c

            hgt, slp, rgh, pts_veh = geometry_from_depth(depth_all[i], valid_all[i], fit_c)
            trav, risk, aux = pseudo_label_frame(
                depth_all[i], valid_all[i], hgt, slp, rgh, pts_veh,
                lab_all[i], pm_all[i], en_all[i], fit_ok=fit_c.ok)

            # ---- uncertainty targets ----
            gray_c = cv2.cvtColor(rgb_c, cv2.COLOR_BGR2GRAY)
            gray_p = cv2.cvtColor(rgb_p, cv2.COLOR_BGR2GRAY)
            if dx is not None:
                mx, my, mw, mok = float(dx[i]), float(dy[i]), float(dyaw[i]), True
            else:
                mx, my, mw, mok = relative_motion_fallback(
                    gray_p, gray_c, depth_all[ip], depth_all[i],
                    valid_all[ip], valid_all[i], fit_c)
            if mok and (abs(mx) + abs(my) + abs(mw)) > 1e-5:
                d_tgt, d_val = depth_error_target(
                    depth_all[i], valid_all[i], gray_c,
                    depth_all[ip], valid_all[ip], gray_p, fit_p, fit_c, mx, my, mw,
                    fit_residual=float(fit_c.residual))
            else:
                d_tgt = np.zeros((PROC_H, PROC_W), np.float32)
                d_val = np.zeros((PROC_H, PROC_W), bool)

            tl = t_lab_all[i] if t_lab_all is not None else None
            te = t_ent_all[i] if t_ent_all is not None else None
            if tl is None and teacher is not None and seg_src == "seg_cache":
                tl, _tp, te = teacher(rgb_c)
            s_tgt, s_val, mode = seg_error_target(lab_all[i], en_all[i], pm_all[i], tl, te)
            seg_modes[mode] = seg_modes.get(mode, 0) + 1

            # ---- pack at network resolution ----
            stack = build_feature_stack(rgb_c, depth_all[i], valid_all[i], hgt, slp, rgh,
                                        lab_all[i], pm_all[i], en_all[i])
            feats.append(resize_stack(stack).astype(np.float16))
            travs.append(cv2.resize(trav, (NET_W, NET_H), interpolation=cv2.INTER_NEAREST))
            risks.append(cv2.resize(risk, (NET_W, NET_H), interpolation=cv2.INTER_AREA).astype(np.float16))
            uncs.append(np.stack([
                cv2.resize(d_tgt, (NET_W, NET_H), interpolation=cv2.INTER_AREA),
                cv2.resize(s_tgt, (NET_W, NET_H), interpolation=cv2.INTER_AREA)]).astype(np.float16))
            uncws.append(np.stack([
                cv2.resize(d_val.astype(np.uint8), (NET_W, NET_H), interpolation=cv2.INTER_NEAREST),
                cv2.resize(s_val.astype(np.uint8), (NET_W, NET_H), interpolation=cv2.INTER_NEAREST)]))
            cls_hist += np.bincount(np.where(travs[-1] == 255, N_TRAV, travs[-1]).ravel(),
                                    minlength=N_TRAV + 1)
            meta.append(dict(clip_id=clip_id, idx=i, frac=aux["frac"].tolist(),
                             fit_ok=bool(fit_c.ok), motion=[mx, my, mw],
                             depth_tgt_valid=float(d_val.mean()),
                             depth_tgt_mean=float(d_tgt[d_val].mean()) if d_val.any() else 0.0,
                             seg_tgt_mean=float(s_tgt[s_val].mean()) if s_val.any() else 0.0))
        print(f"[{clip_id}] {len(cur_idx)} frames in {time.time()-t0:.0f}s", flush=True)

    n = len(feats)
    if n == 0:
        raise RuntimeError("no samples produced")
    np.save(out_dir / "feats.npy", np.stack(feats))
    np.save(out_dir / "trav.npy", np.stack(travs))
    np.save(out_dir / "risk.npy", np.stack(risks))
    np.save(out_dir / "unc.npy", np.stack(uncs))
    np.save(out_dir / "uncw.npy", np.stack(uncws))

    frac = cls_hist / max(cls_hist.sum(), 1)
    summary = dict(
        n_samples=n, net_hw=[NET_H, NET_W], in_ch=IN_CH,
        clips=[c for c, _ in sources],
        class_pixel_fraction={**{k: float(frac[i]) for i, k in
                                 enumerate(["safe", "risky", "obstacle", "unknown"])},
                              "ignore_255": float(frac[4])},
        seg_error_target_mode=seg_modes,
        terrain_label_source=seg_sources,
        relative_motion_source=motion_src,
        thresholds=THRESHOLDS,
        supervision="PSEUDO-LABELS from vehicle-envelope geometry + distilled terrain "
                    "semantics. NOT human annotation, NOT RELLIS-3D/GOOSE/ORFD ground truth.",
        samples=meta,
    )
    (out_dir / "meta.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "samples"}, indent=2))
    return summary


# ------------------------------------------------------------------ self test

def _synth_clip_cache(clip_id: str, n: int = 6, seed: int = 0):
    """Write *physically consistent* synthetic depth/seg/odom caches for standalone testing.

    A real ground plane at the assumed camera height, a 9 cm kerb on the right (which
    must come out OBSTACLE, since 9 cm > the 3 cm max step) and a grass verge on the
    left.  Used to verify the rule set before the real caches exist.
    """
    rng = np.random.default_rng(seed)
    m = G.ray_grid(PROC_H, PROC_W)
    my = m[..., 1]
    h_cam = CFG.cam.height_above_ground_m
    sky = my <= 0.018                                   # rays that never hit the plane

    q = np.empty((n, PROC_H, PROC_W), np.float32)
    depth = np.empty_like(q)
    valid = np.zeros((n, PROC_H, PROC_W), bool)
    lab = np.zeros((n, PROC_H, PROC_W), np.uint8)
    pm = np.full((n, PROC_H, PROC_W), 0.82, np.float32)
    en = np.full((n, PROC_H, PROC_W), 0.25, np.float32)

    kerb = np.zeros((PROC_H, PROC_W), bool)
    kerb[:, int(0.74 * PROC_W):] = True
    kerb &= ~sky
    grass = np.zeros((PROC_H, PROC_W), bool)
    grass[:, :int(0.20 * PROC_W)] = True
    grass &= ~sky

    for i in range(n):
        surf = np.full((PROC_H, PROC_W), h_cam, np.float32)     # distance camera->surface
        surf[kerb] = h_cam - 0.09                               # a 9 cm kerb
        d = surf / np.maximum(my, 1e-3)
        d += rng.normal(0, 0.004, d.shape).astype(np.float32)   # mild depth noise
        d = np.clip(d, 0.15, 25.0)
        d[sky] = np.nan
        depth[i] = d
        q[i] = np.nan_to_num(1.0 / d, nan=0.0)
        valid[i] = ego_mask(PROC_H, PROC_W) & ~sky & np.isfinite(d)
        lab[i, :] = 1
        lab[i][sky] = 0
        lab[i][grass] = 2
        lab[i][kerb] = 4
    save_stage(clip_id, "depth", depth=depth.astype(np.float16), valid=valid, q=q)
    save_stage(clip_id, "seg", label=lab, prob_max=pm, entropy=en)
    save_stage(clip_id, "odom", pose=np.stack([np.zeros(n), np.arange(n) * 0.04,
                                               np.zeros(n)], 1).astype(np.float32))
    return depth, valid, q, lab, pm, en


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-clip", type=int, default=40)
    ap.add_argument("--extra", type=int, default=48, help="frame pairs from video/input.mp4")
    ap.add_argument("--device", default=pick_device())
    ap.add_argument("--wait-min", type=float, default=0.0, help="poll for missing caches")
    ap.add_argument("--no-teacher", action="store_true")
    ap.add_argument("--seg-fallback", action="store_true",
                    help="build provisional terrain labels with the SegFormer teacher "
                         "when a clip's seg cache does not exist yet")
    ap.add_argument("--self-test", action="store_true", help="run on synthetic stand-ins only")
    a = ap.parse_args()

    if a.self_test:
        print("== pseudo-label self-test on SYNTHETIC stand-ins ==")
        cid = "_synth_trav"
        depth, valid, q, lab, pm, en = _synth_clip_cache(cid, n=4)
        fit = G.fit_metric_ground(q[1], valid[1], lab[1])
        print(f"ground fit ok={fit.ok} a={fit.a:.4f} b={fit.b:.4f} "
              f"inliers={fit.inliers} residual={fit.residual:.4f}")
        d_m, v_m = G.depth_from_q(q[1], fit) if fit.ok else (depth[1], valid[1])
        v_m = v_m & valid[1]
        hgt, slp, rgh, pts = geometry_from_depth(d_m, v_m, fit)
        trav, risk, aux = pseudo_label_frame(d_m, v_m, hgt, slp, rgh, pts,
                                             lab[1], pm[1], en[1], fit.ok)
        names = ["safe", "risky", "obstacle", "unknown"]
        tot = trav.size
        for c in range(4):
            print(f"  {names[c]:<9s} {(trav==c).sum()/tot*100:6.2f}%")
        print(f"  ignore255 {(trav==255).sum()/tot*100:6.2f}%")
        print(f"  risk mean {risk.mean():.3f} p95 {np.percentile(risk,95):.3f} max {risk.max():.3f}")
        print(f"  near-obstacle halo covers {aux['near_obst'].mean()*100:.1f}% of pixels")
        assert trav.dtype == np.uint8 and trav.shape == (PROC_H, PROC_W)
        assert 0.0 <= risk.min() and risk.max() <= 1.0

        dx, dy, dyaw, src = load_relative_motion(cid, 4)
        print(f"  relative motion source: {src}  dy[1]={dy[1]:.3f} m")
        gray = np.full((PROC_H, PROC_W), 120, np.uint8)
        dt, dv = depth_error_target(d_m, v_m, gray, depth[0], valid[0], gray, fit, fit,
                                    dx[1], dy[1], dyaw[1])
        print(f"  depth-error target valid {dv.mean()*100:.1f}% mean {dt[dv].mean():.4f}")
        stack = build_feature_stack(np.zeros((PROC_H, PROC_W, 3), np.uint8), d_m, v_m,
                                    hgt, slp, rgh, lab[1], pm[1], en[1])
        print(f"  feature stack {stack.shape} -> net {resize_stack(stack).shape}")
        shutil.rmtree((DATA_DIR.parent / "cache" / cid), ignore_errors=True)
        print("SELF-TEST OK")
    else:
        build_dataset(a.per_clip, a.extra, a.device, wait_min=a.wait_min,
                      use_teacher=not a.no_teacher, seg_fallback=a.seg_fallback)
