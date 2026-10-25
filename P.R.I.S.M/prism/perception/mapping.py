"""Rolling 2.5D BEV local map with ego-motion compensation and observation age.

What this module is
-------------------
`MappingStage` turns per-frame monocular geometry (metric depth + terrain semantics +
traversability + confidence) into a persistent, top-down 2.5D grid around the vehicle:

    128 rows forward (7.68 m) x 192 cols lateral (+/-5.76 m) at 0.06 m/cell
    row 0   = farthest forward cell
    row 127 = the cell the vehicle is standing in
    col 96  = the centreline

Every frame the map is *rolled*: the previous grid is transported into the new vehicle
frame with an affine warp driven by the visual-odometry delta pose (`d_trans`, `d_yaw`),
its confidence is decayed by `CFG.bev.decay_per_frame`, its observation age is
incremented, and the new observation is fused in. Cells that have not been re-observed
for `CFG.bev.max_age_frames` frames are *thrown away* and go back to UNKNOWN. This is
the point of the module: the map remembers what it saw a second ago (so the vehicle can
plan through terrain that has left the field of view) but it never pretends that memory
is a measurement.

Safety rules encoded here (these are load-bearing, do not "improve" them away)
-----------------------------------------------------------------------------
* A cell that has never been hit by a valid depth sample is UNKNOWN, not free. Missing
  geometry is never turned into drivable space.
* A cell whose fused confidence is below `CFG.safety.conf_unknown` is reported as
  UNKNOWN regardless of what the traversability head guessed.
* A cell that is stale (age > `CFG.bev.max_age_frames`) is reset to UNKNOWN.
* Per-cell height is a **robust order statistic** (a high percentile of the samples that
  land in the cell), never a mean. A mean averages an obstacle top together with the
  ground in front of it and makes a 30 cm box look like a 10 cm bump.

Metric scale
------------
Everything metric here inherits from `CFG.cam.height_above_ground_m` (an *assumed*
camera height, 0.12 m), through the depth stage's inverse-depth alignment. It is an
assumption, not a calibration. Renderers must say so.

Derived products for the planner
--------------------------------
`MappingStage` also computes, and attaches to the returned `BEVMap` as extra
attributes (the dataclass in `types.py` is frozen contract, so these ride along):

    bev.height_step   (H,W) float32  height above the *local* ground estimate, NaN unknown
    bev.local_ground  (H,W) float32  the local ground surface it was measured against
    bev.inflated      (H,W) bool     obstacle mask dilated by the vehicle half-width
    bev.observed      (H,W) bool     cells with any live evidence at all

The same products are available as pure functions (`height_step_map`,
`inflate_obstacles`) so the planner can recompute them from a cached `BEVMap`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..config import (CFG, N_TERRAIN, N_TRAV, SAFE, RISKY, OBSTACLE, UNKNOWN,
                      TERRAIN_DRIVE_PRIOR)
from ..types import BEVMap, FramePacket
from ..io_utils import ego_mask
from . import geometry as geo

__all__ = ["MappingStage", "height_step_map", "inflate_obstacles",
           "local_ground_estimate", "nominal_ground_fit"]


# --------------------------------------------------------------------------- helpers

def nominal_ground_fit(pitch_deg: Optional[float] = None,
                       h_cam: Optional[float] = None) -> geo.GroundFit:
    """The documented-assumption ground plane: camera pitched `pitch_deg` nose-down.

    Used only as a last-resort fallback when no upstream ground normal is available.
    The camera frame is X right, Y down, Z forward; a nose-down pitch tilts the ground
    normal (which points *down*, i.e. +Y) toward -Z.
    """
    p = np.deg2rad(CFG.cam.pitch_deg if pitch_deg is None else pitch_deg)
    h = CFG.cam.height_above_ground_m if h_cam is None else float(h_cam)
    # rotate the down-axis (0,1,0) about the camera X axis by the pitch
    n = np.array([0.0, np.cos(p), np.sin(p)], np.float32)
    n /= np.linalg.norm(n)
    return geo.GroundFit(a=1.0, b=0.0, normal=n, height=h, ok=True)


def _segment_stats(cell: np.ndarray, val: np.ndarray, n_cells: int,
                   quantiles=(0.20, 0.80)) -> tuple[list[np.ndarray], np.ndarray]:
    """Per-cell order statistics of `val`, fully vectorised.

    Returns (list of (n_cells,) float32 quantile grids, (n_cells,) int64 counts).
    Cells with no samples are NaN. One sort serves all requested quantiles, which is why
    this is cheap enough to run 1500 times.
    """
    counts = np.bincount(cell, minlength=n_cells)
    out = [np.full(n_cells, np.nan, np.float32) for _ in quantiles]
    if cell.size == 0:
        return out, counts
    # Sort by cell, then by value within the cell. `np.lexsort` does this directly but
    # costs two passes; packing both keys into one int64 lets numpy's stable integer
    # sort (radix) do it in one, which is ~3x faster at 130k points. The value key is
    # quantised to 20 bits over the configured z range - about 1.3 um, i.e. far below
    # any real depth precision - and only the *ordering* is quantised: the values
    # returned are the original floats.
    lo, hi = CFG.bev.z_min - 1.0, CFG.bev.z_max + 1.0
    q = np.clip((val - lo) * ((1 << 20) - 1) / (hi - lo), 0, (1 << 20) - 1)
    order = np.argsort((cell << 20) | q.astype(np.int64), kind="stable")
    vs = val[order]
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    nz = np.nonzero(counts)[0]
    c = counts[nz]
    s = starts[nz]
    for k, q in enumerate(quantiles):
        j = np.minimum((q * (c - 1) + 0.5).astype(np.int64), c - 1)
        out[k][nz] = vs[s + j]
    return out, counts


def _warp_multi(src: np.ndarray, M: np.ndarray, size: tuple[int, int],
                border: float = 0.0, interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    """warpAffine for arbitrary channel counts (OpenCV caps at 4 per call)."""
    if src.ndim == 2:
        return cv2.warpAffine(src, M, size, flags=interp | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    chans = []
    for i in range(0, src.shape[2], 4):
        blk = np.ascontiguousarray(src[:, :, i:i + 4])
        w = cv2.warpAffine(blk, M, size, flags=interp | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(border,) * 4)
        if w.ndim == 2:
            w = w[:, :, None]
        chans.append(w)
    return np.concatenate(chans, axis=2)


def ego_motion_matrix(d_trans: float, d_yaw: float) -> np.ndarray:
    """2x3 affine mapping **new** BEV pixel -> **old** BEV pixel (for WARP_INVERSE_MAP).

    Vehicle frame: x right, y forward, yaw CCW measured from +y. Over one frame the
    vehicle advances `d_trans` metres along a circular arc and turns by `d_yaw`; the
    midpoint heading is used so a turning step is transported correctly.

    A point with coordinates p_new in the new frame had coordinates
        p_old = R(d_yaw) @ p_new + t,    t = d_trans * (-sin(d_yaw/2), cos(d_yaw/2))
    and the pixel<->metre map is p = A u + c with A = diag(res, -res).
    """
    b = CFG.bev
    res = b.res_m
    c_, s_ = np.cos(d_yaw), np.sin(d_yaw)
    R = np.array([[c_, -s_], [s_, c_]], np.float64)
    t = d_trans * np.array([-np.sin(0.5 * d_yaw), np.cos(0.5 * d_yaw)], np.float64)
    A = np.array([[res, 0.0], [0.0, -res]], np.float64)
    c = np.array([(0.5 - b.n_lateral) * res, (b.n_forward - 1 + 0.5) * res], np.float64)
    Ai = np.linalg.inv(A)
    lin = Ai @ R @ A
    off = Ai @ (R @ c + t - c)
    return np.concatenate([lin, off[:, None]], axis=1).astype(np.float32)


def local_ground_estimate(height: np.ndarray, win_m: float = 0.90) -> np.ndarray:
    """Local supporting-surface height per cell (morphological opening of the surface).

    Unobserved cells are filled with 0.0 — the fitted ground plane is z = 0 by
    construction, so that is the correct prior for "what the ground would be here",
    and it keeps the opening from eating real terrain at map edges.
    """
    k = max(3, int(round(win_m / CFG.bev.res_m)) | 1)
    filled = np.nan_to_num(height, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # opening = min-filter then max-filter: removes positive structures smaller than the
    # kernel (obstacles) and keeps the underlying surface (slopes, ramps, kerb tops).
    ground = cv2.morphologyEx(filled, cv2.MORPH_OPEN, ker)
    return cv2.GaussianBlur(ground, (0, 0), 1.4).astype(np.float32)


def height_step_map(bev: BEVMap, win_m: float = 0.90) -> tuple[np.ndarray, np.ndarray]:
    """(height_step, local_ground): height above the local ground, NaN where unobserved."""
    ground = local_ground_estimate(bev.height, win_m)
    step = bev.height - ground
    step[~np.isfinite(bev.height)] = np.nan
    return step.astype(np.float32), ground


def inflate_obstacles(bev: BEVMap, height_step: Optional[np.ndarray] = None,
                      margin_m: Optional[float] = None) -> np.ndarray:
    """Obstacle mask dilated by the vehicle half-width + corridor margin.

    A cell is an obstacle if the traversability head says so, or if the terrain steps up
    over the chassis clearance limit. Unknown cells are *not* inflated (they are handled
    separately by the planner) but they are not free either.
    """
    if height_step is None:
        height_step, _ = height_step_map(bev)
    occ = (bev.trav == OBSTACLE)
    occ |= np.isfinite(height_step) & (height_step > CFG.ugv.max_step_m)
    occ |= np.isfinite(height_step) & (height_step < -max(CFG.ugv.clearance_m * 2.0, 0.09))
    m = CFG.ugv.width_m * 0.5 + (CFG.safety.corridor_margin_m if margin_m is None else margin_m)
    r = max(1, int(np.ceil(m / CFG.bev.res_m)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(occ.astype(np.uint8), ker).astype(bool)


# --------------------------------------------------------------------------- observation

@dataclass
class _Obs:
    """One frame's raw BEV observation, before temporal fusion."""
    height: np.ndarray        # (H,W) float32, robust top surface, NaN = not seen
    support: np.ndarray       # (H,W) float32, robust low surface (ground support)
    prob: np.ndarray          # (H,W,4) float32 traversability
    terrain: np.ndarray       # (H,W,7) float32 weighted votes
    conf: np.ndarray          # (H,W) float32 observation strength 0..1
    count: np.ndarray         # (H,W) float32 samples per cell
    mask: np.ndarray          # (H,W) bool, cell was hit this frame
    n_points: int = 0


# --------------------------------------------------------------------------- stage

class MappingStage:
    """Rolling 2.5D BEV mapper. `reset()` between clips."""

    #: samples with fewer than this many hits in a cell are down-weighted (soft, not hard)
    HIT_SATURATION = 3.0
    #: Quantile used for the cell's representative surface height.
    #: A 12 cm camera looking at a 30 cm box sees only its *front face*, never its top,
    #: so the tallest thing the sensor can report for that cell is the top of the face.
    #: A high order statistic recovers it (0.285 m measured vs 0.30 m truth on the
    #: synthetic scene) while still rejecting the top 5% as outliers. A mean would
    #: report 0.15 m and drive the vehicle into the box.
    TOP_Q = 0.95
    #: quantile used for the cell's supporting surface
    SUPPORT_Q = 0.20

    def __init__(self, device: str = "cuda", pixel_stride: int = 1,
                 max_range_m: Optional[float] = None, **kw):
        self.device = device                      # unused: this stage is numpy/OpenCV only
        self.pixel_stride = int(max(1, pixel_stride))
        self.max_range_m = float(max_range_m) if max_range_m is not None else \
            CFG.bev.range_forward_m * 1.35
        self.H, self.W = CFG.bev.H, CFG.bev.W
        self._ground_normal: Optional[np.ndarray] = None
        #: "learned" once a traversability head has been consumed, "geometric" if the
        #: stage had to fall back on geometry alone. Renderers must surface this.
        self.trav_source = "none"
        self.last_stats: dict = {}
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Clear all temporal state. Mandatory between clips."""
        H, W = self.H, self.W
        self._height = np.full((H, W), np.nan, np.float32)
        self._support = np.full((H, W), np.nan, np.float32)
        self._prob = np.zeros((H, W, N_TRAV), np.float32)
        self._prob[..., UNKNOWN] = 1.0
        self._terrain = np.zeros((H, W, N_TERRAIN), np.float32)
        self._conf = np.zeros((H, W), np.float32)
        self._age = np.full((H, W), 1e4, np.float32)
        self._hits = np.zeros((H, W), np.float32)
        self._ground_normal = None
        self._frame = 0
        self.trav_source = "none"
        self.last_stats = {}

    # ------------------------------------------------------------------ main entry
    def __call__(self, packet: FramePacket) -> FramePacket:
        t0 = time.perf_counter()

        d_trans = float(getattr(packet.odom, "d_trans", 0.0) or 0.0) if packet.odom else 0.0
        d_yaw = float(getattr(packet.odom, "d_yaw", 0.0) or 0.0) if packet.odom else 0.0
        tracking_ok = bool(getattr(packet.odom, "tracking_ok", True)) if packet.odom else True
        if not np.isfinite(d_trans):
            d_trans = 0.0
        if not np.isfinite(d_yaw):
            d_yaw = 0.0
        # a VO dropout must not smear the map around: hold position, decay harder
        if not tracking_ok:
            d_trans, d_yaw = 0.0, 0.0

        t_warp = time.perf_counter()
        self._roll(d_trans, d_yaw, extra_decay=1.0 if tracking_ok else 0.90)
        t_warp = (time.perf_counter() - t_warp) * 1e3

        t_obs = time.perf_counter()
        obs = self._observe(packet)
        t_obs = (time.perf_counter() - t_obs) * 1e3

        t_fuse = time.perf_counter()
        if obs is not None:
            self._fuse(obs)
        t_fuse = (time.perf_counter() - t_fuse) * 1e3

        bev = self._publish()
        packet.bev = bev
        self._frame += 1

        dt = (time.perf_counter() - t0) * 1e3
        packet.timings_ms["bev"] = dt
        self.last_stats = dict(ms_total=dt, ms_warp=t_warp, ms_observe=t_obs,
                               ms_fuse=t_fuse, n_points=0 if obs is None else obs.n_points,
                               d_trans=d_trans, d_yaw=d_yaw, **cell_stats(bev))
        return packet

    # ------------------------------------------------------------------ rolling
    def _roll(self, d_trans: float, d_yaw: float, extra_decay: float = 1.0) -> None:
        """Transport the map into the new vehicle frame, decay it and age it."""
        b = CFG.bev
        moved = abs(d_trans) > 1e-5 or abs(d_yaw) > 1e-6
        if moved:
            M = ego_motion_matrix(d_trans, d_yaw)
            size = (self.W, self.H)

            hv = np.isfinite(self._height).astype(np.float32)
            hf = np.nan_to_num(self._height, nan=0.0)
            sf = np.nan_to_num(self._support, nan=0.0)
            # bundle everything into one multi-channel warp; height/support carry their
            # own validity channel so linear interpolation cannot invent surfaces
            stack = np.concatenate([
                hf[..., None], sf[..., None], hv[..., None],
                self._prob, self._terrain,
                self._conf[..., None], self._hits[..., None],
            ], axis=2).astype(np.float32)
            w = _warp_multi(stack, M, size, border=0.0, interp=cv2.INTER_LINEAR)
            # age warps nearest-neighbour: interpolating "how old is this" is meaningless
            age = _warp_multi(self._age, M, size, border=1e4, interp=cv2.INTER_NEAREST)

            i = 0
            hf = w[..., i]; i += 1
            sf = w[..., i]; i += 1
            hv = w[..., i]; i += 1
            self._prob = np.ascontiguousarray(w[..., i:i + N_TRAV]); i += N_TRAV
            self._terrain = np.ascontiguousarray(w[..., i:i + N_TERRAIN]); i += N_TERRAIN
            self._conf = np.ascontiguousarray(w[..., i]); i += 1
            self._hits = np.ascontiguousarray(w[..., i]); i += 1
            self._age = age

            good = hv > 0.5
            self._height = np.where(good, hf / np.maximum(hv, 1e-3), np.nan).astype(np.float32)
            self._support = np.where(good, sf / np.maximum(hv, 1e-3), np.nan).astype(np.float32)
            self._prob[~good] = 0.0
            self._prob[~good, UNKNOWN] = 1.0
            # cells scrolled in from outside the previous map have no evidence at all
            self._conf[~good] = 0.0
            self._hits[~good] = 0.0

        self._conf *= (b.decay_per_frame * extra_decay)
        self._hits *= b.decay_per_frame
        self._terrain *= b.decay_per_frame
        self._age += 1.0
        self._retire_stale()

    def _retire_stale(self) -> None:
        """Cells nobody has seen recently go back to UNKNOWN. This is the honest bit."""
        stale = (self._age > CFG.bev.max_age_frames) | (self._conf < 1e-3)
        if stale.any():
            self._height[stale] = np.nan
            self._support[stale] = np.nan
            self._prob[stale] = 0.0
            self._prob[stale, UNKNOWN] = 1.0
            self._terrain[stale] = 0.0
            self._conf[stale] = 0.0
            self._hits[stale] = 0.0
            self._age[stale] = 1e4

    # ------------------------------------------------------------------ observation
    def _vehicle_points(self, packet: FramePacket):
        """(N,3) vehicle-frame points + the pixel indices they came from."""
        if packet.geom is not None and getattr(packet.geom, "points_veh", None) is not None:
            pv = np.asarray(packet.geom.points_veh, np.float32)
            h, w = pv.shape[:2]
            valid = np.isfinite(pv).all(axis=2)
            if packet.depth is not None and packet.depth.valid is not None:
                valid &= np.asarray(packet.depth.valid, bool)
            return pv, valid, (h, w)

        if packet.depth is None or packet.depth.depth_m is None:
            return None, None, None
        d = np.asarray(packet.depth.depth_m, np.float32)
        h, w = d.shape
        valid = np.isfinite(d) & (d > 0.05) & (d < self.max_range_m)
        if packet.depth.valid is not None:
            valid &= np.asarray(packet.depth.valid, bool)

        pc = geo.unproject(d, CFG.cam.K if (w, h) == (CFG.cam.width, CFG.cam.height)
                           else _scaled_K(w, h))
        fit = self._resolve_fit(packet, pc, valid)
        pv = geo.to_vehicle(pc, fit)
        return pv, valid, (h, w)

    def _resolve_fit(self, packet: FramePacket, points_cam: np.ndarray,
                     valid: np.ndarray) -> geo.GroundFit:
        """Ground normal, in preference order: upstream -> own robust refit -> config.

        When the depth stage published the normal it actually fitted, that value is used
        **verbatim**: `geometry.fit_metric_ground` already smooths it temporally, and
        every other stage back-projects with it, so re-smoothing here would put the map
        in a slightly different vehicle frame from the rest of the system.
        """
        h_cam = CFG.cam.height_above_ground_m
        n, from_upstream = None, False
        for src in (packet.depth, packet.geom):
            if src is None:
                continue
            for attr in ("ground_normal", "normal"):
                v = getattr(src, attr, None)
                if v is not None and np.isfinite(np.asarray(v, np.float64)).all() \
                        and np.linalg.norm(np.asarray(v, np.float64)) > 1e-6:
                    n, from_upstream = np.asarray(v, np.float32).reshape(3), True
                    break
            if n is not None:
                break
        if n is None:
            n = self._refit_normal(points_cam, valid)
        if n is None:
            n = (self._ground_normal if self._ground_normal is not None
                 else nominal_ground_fit().normal)
        n = np.asarray(n, np.float32)
        n = n / (np.linalg.norm(n) + 1e-9)
        if n[1] < 0:
            n = -n
        # only our own per-frame refit needs smoothing; upstream normals are already stable
        if not from_upstream and self._ground_normal is not None:
            n = 0.75 * self._ground_normal + 0.25 * n
            n /= (np.linalg.norm(n) + 1e-9)
        self._ground_normal = n.astype(np.float32)
        return geo.GroundFit(a=1.0, b=0.0, normal=self._ground_normal, height=h_cam, ok=True)

    def _refit_normal(self, points_cam: np.ndarray, valid: np.ndarray) -> Optional[np.ndarray]:
        """IRLS fit of n.P = h_cam over lower-image points; returns the unit normal.

        The scale is *not* re-estimated here (it belongs to the depth stage); only the
        plane direction, which is what `to_vehicle` needs.
        """
        h, w = valid.shape
        band = np.zeros((h, w), bool)
        band[int(0.55 * h):, :] = True
        m = valid & band & ego_mask(h, w)
        idx = np.nonzero(m.ravel())[0]
        if idx.size < 500:
            return None
        rng = np.random.default_rng(7)
        if idx.size > 4000:
            idx = rng.choice(idx, 4000, replace=False)
        P = points_cam.reshape(-1, 3)[idx].astype(np.float64)
        rr = np.linalg.norm(P, axis=1)
        P = P[(rr > 0.2) & (rr < 12.0)]
        if P.shape[0] < 400:
            return None
        h_cam = CFG.cam.height_above_ground_m
        wts = np.ones(P.shape[0])
        u = np.array([0.0, 1.0 / h_cam, 0.0])
        for _ in range(4):
            A = P * wts[:, None]
            bb = np.ones(P.shape[0]) * wts
            try:
                u, *_ = np.linalg.lstsq(A, bb, rcond=None)
            except np.linalg.LinAlgError:
                return None
            r = P @ u - 1.0
            s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
            wts = 1.0 / np.sqrt(1.0 + (r / (2.0 * s)) ** 2)
        nu = np.linalg.norm(u)
        if not np.isfinite(nu) or nu < 1e-6:
            return None
        n = (u / nu).astype(np.float32)
        if n[1] < 0:
            n = -n
        # reject nonsense (camera cannot be looking more than ~40 deg off level)
        if np.degrees(np.arccos(np.clip(float(n[1]), -1, 1))) > 40.0:
            return None
        return n

    def _observe(self, packet: FramePacket) -> Optional[_Obs]:
        pv, valid, shape = self._vehicle_points(packet)
        if pv is None:
            return None
        h, w = shape
        valid = valid & ego_mask(h, w)

        st = self.pixel_stride
        if st > 1:
            pv = pv[::st, ::st]
            valid = valid[::st, ::st]

        pf = pv.reshape(-1, 3)
        vf = valid.reshape(-1)
        with np.errstate(invalid="ignore"):      # NaN depth is expected and filtered by `keep`
            row, col, keep = geo.bev_indices(pf, vf)
        z = pf[:, 2]
        keep &= np.isfinite(z) & (z > CFG.bev.z_min) & (z < CFG.bev.z_max)
        idx = np.nonzero(keep)[0]
        n_cells = self.H * self.W
        if idx.size < 20:
            return None

        cell = (row[idx] * self.W + col[idx]).astype(np.int64)
        zc = z[idx].astype(np.float32)

        # --- per-pixel weight: confidence x range attenuation ------------------
        wgt = np.ones(idx.size, np.float32)
        if packet.unc is not None and getattr(packet.unc, "fused_conf", None) is not None:
            c = np.asarray(packet.unc.fused_conf, np.float32)
            if c.shape[:2] != (h, w):
                c = cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR)
            if st > 1:
                c = c[::st, ::st]
            wgt *= np.clip(c.reshape(-1)[idx], 0.02, 1.0)
        rng_m = np.sqrt(pf[idx, 0] ** 2 + pf[idx, 1] ** 2)
        wgt *= (1.0 / (1.0 + (rng_m / 4.0) ** 2)).astype(np.float32)   # far = less trusted

        # --- robust height ------------------------------------------------------
        (sup, top), counts = _segment_stats(cell, zc, n_cells,
                                            (self.SUPPORT_Q, self.TOP_Q))
        cnt = counts.astype(np.float32)
        wsum = np.bincount(cell, weights=wgt, minlength=n_cells).astype(np.float32)

        # --- terrain votes (before traversability: the geometric fallback uses them) --
        terr = np.zeros((n_cells, N_TERRAIN), np.float32)
        if packet.seg is not None and getattr(packet.seg, "label", None) is not None:
            lab = np.asarray(packet.seg.label, np.uint8)
            if lab.shape[:2] != (h, w):
                lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
            if st > 1:
                lab = lab[::st, ::st]
            lf = lab.reshape(-1)[idx].astype(np.int64)
            ok = lf < N_TERRAIN
            for k in range(N_TERRAIN):
                sel = ok & (lf == k)
                if sel.any():
                    terr[:, k] = np.bincount(cell[sel], weights=wgt[sel], minlength=n_cells)

        # --- traversability ------------------------------------------------------
        prob = np.zeros((n_cells, N_TRAV), np.float32)
        tp = None
        if packet.trav is not None and getattr(packet.trav, "prob", None) is not None:
            tp = np.asarray(packet.trav.prob, np.float32)
            if tp.ndim == 3 and tp.shape[0] == N_TRAV:
                tp = np.moveaxis(tp, 0, -1)
            if tp.shape[:2] != (h, w):
                tp = cv2.resize(tp, (w, h), interpolation=cv2.INTER_LINEAR)
            if st > 1:
                tp = tp[::st, ::st]
            tpf = tp.reshape(-1, N_TRAV)[idx]
            for k in range(N_TRAV):
                prob[:, k] = np.bincount(cell, weights=wgt * tpf[:, k], minlength=n_cells)
            self.trav_source = "learned"
        elif packet.trav is not None and getattr(packet.trav, "label", None) is not None:
            lab = np.asarray(packet.trav.label, np.uint8)
            if lab.shape[:2] != (h, w):
                lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
            if st > 1:
                lab = lab[::st, ::st]
            lf = np.clip(lab.reshape(-1)[idx], 0, N_TRAV - 1).astype(np.int64)
            for k in range(N_TRAV):
                sel = lf == k
                if sel.any():
                    prob[:, k] = np.bincount(cell[sel], weights=wgt[sel], minlength=n_cells)
            self.trav_source = "learned"
        else:
            # No learned traversability head cached. Rather than report a blank map,
            # fall back to the classical *geometric* rule (step height vs chassis
            # clearance, plus the terrain drive prior if segmentation exists). This is
            # not guessing about unobserved space - it is real evidence from the
            # geometry we did measure - but it is weaker than the learned head, so it
            # carries a permanent UNKNOWN floor and the stage flags itself as degraded.
            prob = self._geometric_prob(top, sup, counts, terr)
            self.trav_source = "geometric"
        if self.trav_source != "geometric":
            prob /= np.maximum(wsum, 1e-6)[:, None]
        s = prob.sum(axis=1, keepdims=True)
        prob = np.where(s > 1e-6, prob / np.maximum(s, 1e-6), 0.0).astype(np.float32)

        # --- observation strength -------------------------------------------------
        mean_w = wsum / np.maximum(cnt, 1.0)
        sat = cnt / (cnt + self.HIT_SATURATION)          # 1 sample != a measurement
        obs_conf = np.clip(mean_w * sat, 0.0, 1.0).astype(np.float32)

        R = (self.H, self.W)
        return _Obs(
            height=top.reshape(R), support=sup.reshape(R),
            prob=prob.reshape(*R, N_TRAV), terrain=terr.reshape(*R, N_TERRAIN),
            conf=obs_conf.reshape(R), count=cnt.reshape(R),
            mask=(counts > 0).reshape(R), n_points=int(idx.size),
        )

    # ------------------------------------------------------------------ geometric trav
    #: geometric-fallback thresholds. `CFG.ugv.max_step_m` is 3 cm, which is below the
    #: noise floor of monocular depth at these ranges, so the safe/risky boundary is
    #: floored at 3.5 cm and the obstacle boundary at twice the chassis clearance.
    GEO_STEP_SAFE = max(CFG.ugv.max_step_m, 0.035)
    GEO_STEP_OBST = max(CFG.ugv.clearance_m * 2.2, 0.10)
    #: mass permanently reserved for UNKNOWN when traversability came from geometry
    #: alone, so a heuristic can never present itself as a confident classification
    GEO_UNKNOWN_FLOOR = 0.12

    def _geometric_prob(self, top: np.ndarray, sup: np.ndarray,
                        counts: np.ndarray, terr: np.ndarray) -> np.ndarray:
        """Classical geometric traversability, used when no learned head is cached.

        Two geometric cues per cell: the step of its surface above the *local* ground
        (a kerb, a bank, a wall) and the vertical relief inside the cell itself (a
        surface too steep or too broken to be a driving surface). Combined, where
        available, with the prism-7 terrain drive prior from segmentation.
        """
        H, W = self.H, self.W
        seen = counts > 0
        top_g = np.where(seen, top, np.nan).reshape(H, W)
        ground = local_ground_estimate(top_g, win_m=0.90)
        step = np.abs(np.nan_to_num(top_g - ground, nan=0.0)).reshape(-1)
        relief = np.abs(np.nan_to_num(top - sup, nan=0.0))
        s = np.maximum(step, relief * 0.75)

        a, b = self.GEO_STEP_SAFE, self.GEO_STEP_OBST
        p_obst = np.clip((s - 0.8 * b) / (1.0 * b), 0.0, 1.0)
        p_safe = np.clip(1.0 - (s - 0.6 * a) / (1.2 * a), 0.0, 1.0) * (1.0 - p_obst)
        p_risky = np.clip(1.0 - p_safe - p_obst, 0.0, 1.0)

        tsum = terr.sum(axis=1)
        has_t = tsum > 1e-6
        if has_t.any():
            drive = np.zeros_like(tsum)
            drive[has_t] = (terr[has_t] @ TERRAIN_DRIVE_PRIOR) / tsum[has_t]
            # a low drive prior (rough vegetation, water, a wall) demotes "safe"
            keep = 0.30 + 0.70 * drive
            moved = p_safe * (1.0 - keep)
            p_safe = p_safe * keep
            p_risky = p_risky + moved * (0.55 + 0.45 * drive)
            p_obst = p_obst + moved * (0.45 - 0.45 * drive)

        prob = np.zeros((counts.size, N_TRAV), np.float32)
        scale = 1.0 - self.GEO_UNKNOWN_FLOOR
        prob[:, SAFE] = p_safe * scale
        prob[:, RISKY] = p_risky * scale
        prob[:, OBSTACLE] = p_obst * scale
        prob[:, UNKNOWN] = self.GEO_UNKNOWN_FLOOR
        prob[~seen] = 0.0
        prob[~seen, UNKNOWN] = 1.0
        return prob

    # ------------------------------------------------------------------ fusion
    def _fuse(self, obs: _Obs) -> None:
        """Confidence-weighted fusion of a new observation into the rolled map."""
        m = obs.mask
        w_obs = np.where(m, obs.conf, 0.0).astype(np.float32)
        c_prev = self._conf

        # height / support: weighted mix of whichever sides are finite
        for name, new in (("_height", obs.height), ("_support", obs.support)):
            prev = getattr(self, name)
            wp = np.where(np.isfinite(prev), c_prev, 0.0)
            wn = np.where(np.isfinite(new), w_obs, 0.0)
            den = wp + wn
            mixed = (np.nan_to_num(prev) * wp + np.nan_to_num(new) * wn) / np.maximum(den, 1e-6)
            mixed[den <= 1e-6] = np.nan
            setattr(self, name, mixed.astype(np.float32))

        # traversability: confidence-weighted mixture (a Bayesian-ish update where the
        # weight is the evidence mass, so a strong observation dominates a stale prior)
        den = (c_prev + w_obs)[..., None]
        self._prob = ((self._prob * c_prev[..., None] + obs.prob * w_obs[..., None])
                      / np.maximum(den, 1e-6)).astype(np.float32)
        empty = den[..., 0] <= 1e-6
        if empty.any():
            self._prob[empty] = 0.0
            self._prob[empty, UNKNOWN] = 1.0

        self._terrain += obs.terrain

        # confidence: noisy-OR accumulation -> repeated consistent looks approach 1
        self._conf = np.clip(c_prev + w_obs * (1.0 - c_prev), 0.0, 1.0).astype(np.float32)
        self._hits = np.minimum(self._hits + obs.count, 4095.0).astype(np.float32)
        self._age[m] = 0.0

    # ------------------------------------------------------------------ publish
    def _publish(self) -> BEVMap:
        conf = self._conf.copy()
        age = self._age.copy()
        hits = self._hits.copy()
        height = self._height.copy()
        prob = np.moveaxis(self._prob, -1, 0).copy()

        observed = (hits > 0.05) & (age <= CFG.bev.max_age_frames) & np.isfinite(height)
        # SAFETY: never observed, stale, or low confidence -> UNKNOWN, never free space
        unknown = (~observed) | (conf < CFG.safety.conf_unknown)

        trav = np.argmax(prob, axis=0).astype(np.uint8)
        trav[unknown] = UNKNOWN
        prob[:, unknown] = 0.0
        prob[UNKNOWN, unknown] = 1.0
        height[~observed] = np.nan

        tsum = self._terrain.sum(axis=-1)
        terrain = np.argmax(self._terrain, axis=-1).astype(np.uint8)
        terrain[(tsum <= 1e-6) | unknown] = 0        # 0 = sky, used here as "no vote"

        bev = BEVMap(height=height, trav_prob=np.ascontiguousarray(prob), trav=trav,
                     conf=conf, age=np.minimum(age, 9999.0), hits=hits, terrain=terrain)

        step, ground = height_step_map(bev)
        step[~observed] = np.nan
        # extra products the planner needs; BEVMap is contract-frozen so they ride along
        bev.trav_source = self.trav_source
        bev.height_step = step
        bev.local_ground = ground
        bev.observed = observed
        bev.inflated = inflate_obstacles(bev, step)
        return bev


def _scaled_K(w: int, h: int) -> np.ndarray:
    K = CFG.cam.K.copy()
    sx, sy = w / float(CFG.cam.width), h / float(CFG.cam.height)
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    return K


def cell_stats(bev: BEVMap) -> dict:
    """Fractions of the grid in each traversability state plus mean live confidence."""
    n = float(bev.trav.size)
    live = bev.trav != UNKNOWN
    return dict(
        frac_safe=float((bev.trav == SAFE).sum() / n),
        frac_risky=float((bev.trav == RISKY).sum() / n),
        frac_obstacle=float((bev.trav == OBSTACLE).sum() / n),
        frac_unknown=float((bev.trav == UNKNOWN).sum() / n),
        mean_conf=float(bev.conf[live].mean()) if live.any() else 0.0,
        mean_age=float(np.minimum(bev.age, CFG.bev.max_age_frames)[live].mean()) if live.any() else 0.0,
    )


# =========================================================================== self-test

def synthetic_scene(t_forward_m: float = 0.0, yaw_rad: float = 0.0,
                    box_h: float = 0.30, w: int = 640, h: int = 360):
    """A ray-traced synthetic scene with exactly known geometry.

    Ground plane at z = 0, a `box_h` tall box at x in [0.30, 0.60], y in [2.00, 2.30],
    and a 0.12 m deep ditch spanning the full width at y in [3.00, 3.40]. The camera sits
    `CFG.cam.height_above_ground_m` above the ground, pitched by `CFG.cam.pitch_deg`,
    displaced `t_forward_m` forward and rotated `yaw_rad`.

    Returns (depth_m, valid, trav_label, seg_label, truth) where `truth` carries the
    box footprint in the *current* vehicle frame so the BEV can be checked against it.
    """
    K = _scaled_K(w, h)
    m = geo.ray_grid(h, w, K).reshape(-1, 3).astype(np.float64)
    fit = nominal_ground_fit()
    R = geo.vehicle_basis(fit.normal)                     # cam -> vehicle
    d = m @ R.T                                           # ray dirs in vehicle frame
    o = np.array([0.0, 0.0, fit.height])

    # world-frame box/ditch, expressed in the *current* vehicle frame
    def to_local(x0, x1, y0, y1):
        y0, y1 = y0 - t_forward_m, y1 - t_forward_m
        c, s = np.cos(-yaw_rad), np.sin(-yaw_rad)
        cs = np.array([[c, -s], [s, c]])
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) @ cs.T
        return pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()

    bx0, bx1, by0, by1 = to_local(0.30, 0.60, 2.00, 2.30)
    dx0, dx1, dy0, dy1 = to_local(-3.00, 3.00, 3.00, 3.40)
    dz = -0.12

    D = np.full(m.shape[0], np.inf)

    def hit_plane(axis, value, cond):
        """Intersect rays with the plane {p[axis] == value}; cond(p) accepts the hit."""
        den = d[:, axis]
        ok = np.abs(den) > 1e-9
        t = np.full(den.shape, np.inf)
        t[ok] = (value - o[axis]) / den[ok]
        t[t <= 1e-3] = np.inf
        with np.errstate(invalid="ignore"):
            p = o + np.where(np.isfinite(t), t, 0.0)[:, None] * d
        good = np.isfinite(t) & cond(p)
        np.minimum(D, np.where(good, t, np.inf), out=D)

    inside_box = lambda p: (p[:, 0] >= bx0) & (p[:, 0] <= bx1) & (p[:, 1] >= by0) & (p[:, 1] <= by1)
    inside_dit = lambda p: (p[:, 0] >= dx0) & (p[:, 0] <= dx1) & (p[:, 1] >= dy0) & (p[:, 1] <= dy1)

    hit_plane(2, 0.0, lambda p: (~inside_box(p)) & (~inside_dit(p)))          # ground
    hit_plane(2, box_h, inside_box)                                           # box top
    hit_plane(2, dz, inside_dit)                                              # ditch floor
    hit_plane(1, by0, lambda p: (p[:, 0] >= bx0) & (p[:, 0] <= bx1)
              & (p[:, 2] >= 0) & (p[:, 2] <= box_h))                          # box front
    hit_plane(0, bx0, lambda p: (p[:, 1] >= by0) & (p[:, 1] <= by1)
              & (p[:, 2] >= 0) & (p[:, 2] <= box_h))                          # box left
    hit_plane(0, bx1, lambda p: (p[:, 1] >= by0) & (p[:, 1] <= by1)
              & (p[:, 2] >= 0) & (p[:, 2] <= box_h))                          # box right
    hit_plane(1, dy0, lambda p: ((p[:, 0] >= dx0) & (p[:, 0] <= dx1)      # ditch near wall
                                 & (p[:, 2] >= dz) & (p[:, 2] <= 0.0)))

    valid = np.isfinite(D) & (D < 30.0)
    P = o + np.where(valid, D, 0.0)[:, None] * d
    # optical-axis depth == the ray parameter, because ray_grid has m_z == 1
    depth = np.where(valid, D, np.nan).reshape(h, w).astype(np.float32)
    valid = valid.reshape(h, w) & ego_mask(h, w)

    z = P[:, 2]
    on_box = inside_box(P) & (z > 0.02)
    trav = np.where(on_box, OBSTACLE, SAFE).astype(np.uint8).reshape(h, w)
    seg = np.where(on_box, 4, 1).astype(np.uint8).reshape(h, w)
    truth = dict(box=(bx0, bx1, by0, by1), box_h=box_h,
                 ditch=(dx0, dx1, dy0, dy1), ditch_z=dz)
    return depth, valid, trav, seg, truth


def _packet_from_scene(i: int, depth, valid, trav, seg, d_trans=0.0, d_yaw=0.0) -> FramePacket:
    from ..types import DepthResult, SegResult, TraversabilityResult, UncertaintyResult, OdometryResult
    h, w = depth.shape
    prob = np.zeros((N_TRAV, h, w), np.float32)
    prob[trav, np.arange(h)[:, None], np.arange(w)[None, :]] = 1.0
    pk = FramePacket(clip_id="synthetic", idx=i, t=i / 30.0)
    pk.depth = DepthResult(rel_inv=np.zeros_like(depth), depth_m=depth, valid=valid)
    pk.seg = SegResult(label=seg, prob_max=np.full((h, w), 0.9, np.float32),
                       entropy=np.zeros((h, w), np.float32))
    pk.trav = TraversabilityResult(prob=prob, label=trav,
                                   risk=(trav == OBSTACLE).astype(np.float32))
    pk.unc = UncertaintyResult(depth_conf=np.full((h, w), 0.9, np.float32),
                               seg_conf=np.full((h, w), 0.9, np.float32),
                               fused_conf=np.full((h, w), 0.9, np.float32), mean_conf=0.9)
    pk.odom = OdometryResult(d_trans=d_trans, d_yaw=d_yaw, tracking_ok=True)
    return pk


def _self_test() -> None:
    print("=" * 74)
    print("MappingStage self-test on a ray-traced synthetic scene")
    print("=" * 74)
    b = CFG.bev
    print(f"grid {b.H} rows x {b.W} cols @ {b.res_m} m  ->  "
          f"{b.range_forward_m:.2f} m forward, +/-{b.range_lateral_m:.2f} m lateral")
    print(f"metric scale anchored to assumed camera height {CFG.cam.height_above_ground_m} m")

    st = MappingStage(device="cpu")
    depth, valid, trav, seg, truth = synthetic_scene(0.0)
    print(f"\nsynthetic depth: {depth.shape} valid={valid.mean():.1%} "
          f"range {np.nanmin(depth):.2f}..{np.nanmax(depth):.2f} m")

    rng = np.random.default_rng(3)
    noise = 0.02          # 2% multiplicative depth noise, so this is not a noiseless toy
    times = []
    for i in range(20):
        d = (depth * (1.0 + noise * rng.standard_normal(depth.shape))).astype(np.float32)
        pk = _packet_from_scene(i, d, valid, trav, seg, d_trans=0.0, d_yaw=0.0)
        t0 = time.perf_counter()
        st(pk)
        times.append((time.perf_counter() - t0) * 1e3)
    bev = pk.bev
    print(f"\n20 static frames (+{noise:.0%} depth noise): {np.mean(times):.1f} ms/frame "
          f"(median {np.median(times):.1f}, warp {st.last_stats['ms_warp']:.1f}, "
          f"observe {st.last_stats['ms_observe']:.1f}, fuse {st.last_stats['ms_fuse']:.1f})")

    # --- does a 30 cm box land in the right cells at the right height? -----------
    bx0, bx1, by0, by1 = truth["box"]
    rows = np.arange(b.H)[:, None] * np.ones((1, b.W))
    cols = np.ones((b.H, 1)) * np.arange(b.W)[None, :]
    X, Y = geo.bev_to_veh(rows, cols)
    box_cells = (X >= bx0) & (X <= bx1) & (Y >= by0) & (Y <= by1)
    seen = box_cells & np.isfinite(bev.height)
    hs = bev.height[seen]
    print(f"\nBOX truth: x[{bx0:.2f},{bx1:.2f}] y[{by0:.2f},{by1:.2f}] h={truth['box_h']:.2f} m")
    print(f"  footprint cells      : {int(box_cells.sum())}  observed {int(seen.sum())}")
    print(f"  measured height      : mean {hs.mean():.3f}  median {np.median(hs):.3f}  "
          f"p10 {np.percentile(hs, 10):.3f}  p90 {np.percentile(hs, 90):.3f} m")
    err = abs(float(np.median(hs)) - truth["box_h"])
    print(f"  |median - truth|     : {err * 100:.1f} cm   -> {'PASS' if err < 0.03 else 'FAIL'}")
    step = bev.height_step
    print(f"  height_step median   : {np.nanmedian(step[seen]):.3f} m "
          f"(local ground {np.nanmedian(bev.local_ground[seen]):.3f} m)")
    obst = (bev.trav[seen] == OBSTACLE).mean()
    print(f"  labelled OBSTACLE    : {obst:.1%}  -> {'PASS' if obst > 0.7 else 'FAIL'}")
    print(f"  inflated footprint   : {int(bev.inflated.sum())} cells "
          f"(dilated by {CFG.ugv.width_m / 2 + CFG.safety.corridor_margin_m:.3f} m)")

    # --- ground cells in front of the box must be flat and SAFE -------------------
    near = (Y > 0.5) & (Y < 1.6) & (np.abs(X) < 0.8) & np.isfinite(bev.height)
    print(f"\nGROUND (0.5-1.6 m ahead): height median {np.nanmedian(bev.height[near]) * 100:+.1f} cm, "
          f"p95 {np.nanpercentile(bev.height[near], 95) * 100:+.1f} cm "
          f"-> {'PASS' if abs(np.nanmedian(bev.height[near])) < 0.02 else 'FAIL'}")
    print(f"  labelled SAFE        : {(bev.trav[near] == SAFE).mean():.1%}")

    # --- the ditch floor is occluded from a 12 cm camera: must stay UNKNOWN -------
    dx0, dx1, dy0, dy1 = truth["ditch"]
    dit = (Y > dy0 + 0.1) & (Y < dy1) & (np.abs(X) < 1.0)
    unk = (bev.trav[dit] == UNKNOWN).mean()
    print(f"\nDITCH {dy0:.1f}-{dy1:.1f} m (floor occluded from a "
          f"{CFG.cam.height_above_ground_m * 100:.0f} cm camera):")
    print(f"  cells UNKNOWN        : {unk:.1%}  -> "
          f"{'PASS (never invented free space)' if unk > 0.5 else 'FAIL'}")

    # --- ego-motion compensation --------------------------------------------------
    st.reset()
    for i in range(6):
        pk = _packet_from_scene(i, depth, valid, trav, seg, d_trans=0.0)
        st(pk)
    before = _box_row(pk.bev, truth)
    # now roll 1.0 m forward with NO new observations: the memory must slide toward ego
    blank = np.full_like(depth, np.nan)
    bvalid = np.zeros_like(valid)
    for i in range(10):
        pk = _packet_from_scene(100 + i, blank, bvalid, trav, seg, d_trans=0.10)
        st(pk)
    after = _box_row(pk.bev, truth)
    exp = before + 1.0 / b.res_m
    print(f"\nEGO-MOTION: box row before {before:.1f} -> after 1.00 m of pure memory "
          f"{after:.1f} (expected {exp:.1f})")
    print(f"  row error            : {abs(after - exp):.2f} cells "
          f"({abs(after - exp) * b.res_m * 100:.1f} cm) -> "
          f"{'PASS' if abs(after - exp) < 2.0 else 'FAIL'}")
    print(f"  confidence after 10 decayed frames: {np.nanmax(pk.bev.conf):.3f} "
          f"(decay {b.decay_per_frame}^10 = {b.decay_per_frame ** 10:.3f})")

    # --- staleness ----------------------------------------------------------------
    st.reset()
    for i in range(3):
        st(_packet_from_scene(i, depth, valid, trav, seg))
    live0 = float((st._age < 1e3).mean())
    for i in range(CFG.bev.max_age_frames + 5):
        pk = _packet_from_scene(50 + i, blank, bvalid, trav, seg, d_trans=0.0)
        st(pk)
    live1 = float((pk.bev.trav != UNKNOWN).mean())
    print(f"\nSTALENESS: live cells {live0:.1%} -> {live1:.1%} after "
          f"{CFG.bev.max_age_frames + 5} unobserved frames -> "
          f"{'PASS (map forgets)' if live1 < 0.01 else 'FAIL'}")

    # --- yaw compensation ---------------------------------------------------------
    st.reset()
    for i in range(5):
        st(_packet_from_scene(i, depth, valid, trav, seg))
    b0 = _box_col(st._publish(), truth)
    for i in range(10):
        pk = _packet_from_scene(200 + i, blank, bvalid, trav, seg, d_yaw=np.deg2rad(1.0))
        st(pk)
    b1 = _box_col(pk.bev, truth)
    print(f"\nYAW: 10 deg CCW turn moves the remembered box from col {b0:.1f} to {b1:.1f} "
          f"(expected to move right/+, got {'+' if b1 > b0 else '-'}) -> "
          f"{'PASS' if b1 > b0 else 'FAIL'}")

    print("\nstats:", {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in cell_stats(bev).items()})
    print("done.")


def _box_row(bev: BEVMap, truth) -> float:
    m = (bev.height > truth["box_h"] * 0.5) & np.isfinite(bev.height)
    return float(np.nonzero(m)[0].mean()) if m.any() else float("nan")


def _box_col(bev: BEVMap, truth) -> float:
    m = (bev.height > truth["box_h"] * 0.5) & np.isfinite(bev.height)
    return float(np.nonzero(m)[1].mean()) if m.any() else float("nan")


if __name__ == "__main__":
    _self_test()
