"""Short-lived obstacle layer for things that move: people, animals, vehicles.

The rolling 2.5D map is built for terrain, which stays where it is.  A person who
walks across the trail breaks both of its assumptions: the cells they occupy
must block the vehicle *immediately* (no multi-frame confidence build-up), and
once they have walked away those cells must not stay blocked for the
`CFG.bev.max_age_frames` (1.5 s) a terrain observation is remembered for.

So prism-7 class 6 (`dynamic`) pixels are handled here, separately:

1. back-project the dynamic pixels with metric depth into the vehicle frame and
   drop them into BEV cells;
2. dilate those cells by the vehicle half-width plus `inflate_m` - a moving
   thing gets a wider berth than a rock;
3. give every cell an expiry time `t + ttl_s` (0.8 s: a dynamic cell expires in
   under one second unless it is seen again);
4. carry the layer along with ego-motion exactly like the terrain map;
5. `apply()` writes the active cells into the planner's world state as certain
   OBSTACLE, and `changed` tells the goal layer to replan with D* Lite.

Nothing here makes a moving obstacle "free" early: a cell is cleared only by its
own expiry, never by a frame in which the segmentation happened to miss it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..config import CFG, TERRAIN_CLASSES, OBSTACLE
from . import geometry as geo

DYNAMIC_CLASS = TERRAIN_CLASSES.index("dynamic")


@dataclass
class DynamicUpdate:
    active: np.ndarray            # (H, W) bool BEV cells currently blocked by a mover
    new_cells: int = 0            # cells that became active this update
    expired_cells: int = 0        # cells that expired this update
    n_pixels: int = 0             # dynamic pixels seen in the image this frame
    nearest_m: float = float("inf")   # closest active cell, metres from the vehicle

    @property
    def changed(self) -> bool:
        return self.new_cells > 0 or self.expired_cells > 0


class DynamicObstacleLayer:
    """BEV-aligned expiry grid for moving obstacles.  `reset()` between runs."""

    def __init__(self, ttl_s: Optional[float] = None, inflate_m: Optional[float] = None,
                 max_range_m: Optional[float] = None, min_pixels: Optional[int] = None,
                 min_cells: Optional[int] = None):
        from ..nav.nav_config import NAV
        d = NAV.dynamic
        self.ttl_s = float(d.ttl_s if ttl_s is None else ttl_s)
        self.inflate_m = float(d.inflate_m if inflate_m is None else inflate_m)
        self.max_range_m = float(d.max_range_m if max_range_m is None else max_range_m)
        self.min_pixels = int(d.min_pixels if min_pixels is None else min_pixels)
        self.min_cells = int(d.min_cells if min_cells is None else min_cells)
        b = CFG.bev
        self.H, self.W = b.H, b.W
        r = int(np.ceil((CFG.ugv.width_m * 0.5 + self.inflate_m) / b.res_m))
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        rr, cc = np.meshgrid(np.arange(self.H, dtype=np.float32),
                             np.arange(self.W, dtype=np.float32), indexing="ij")
        bx, by = geo.bev_to_veh(rr, cc)
        self._range = np.hypot(bx, by).astype(np.float32)
        self.reset()

    def reset(self) -> None:
        self.until = np.full((self.H, self.W), -np.inf, np.float64)
        self.t = -np.inf
        self.last: Optional[DynamicUpdate] = None

    # ------------------------------------------------------------------ helpers
    def _roll(self, d_trans: float, d_yaw: float) -> None:
        if abs(d_trans) < 1e-5 and abs(d_yaw) < 1e-6:
            return
        from .mapping import ego_motion_matrix
        M = ego_motion_matrix(d_trans, d_yaw)
        # warpAffine has no float64 path for every flag combination; expiry times are
        # stored relative to "now" in float32 for the warp and restored after.
        rel = np.where(np.isfinite(self.until), self.until - self.t, -1.0).astype(np.float32)
        rel = cv2.warpAffine(rel, M, (self.W, self.H),
                             flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)
        self.until = np.where(rel > 0, rel.astype(np.float64) + self.t, -np.inf)

    @staticmethod
    def points_from_depth(depth_m: np.ndarray, valid: np.ndarray,
                          fit: Optional[geo.GroundFit] = None) -> np.ndarray:
        """(H, W, 3) vehicle-frame points from metric depth and a ground fit."""
        if fit is None:
            from .mapping import nominal_ground_fit
            fit = nominal_ground_fit()
        d = np.where(valid, depth_m, np.nan).astype(np.float32)
        return geo.to_vehicle(geo.unproject(d), fit)

    # ------------------------------------------------------------------ main
    def update(self, t: float, seg_label: Optional[np.ndarray],
               points_veh: Optional[np.ndarray], valid: Optional[np.ndarray] = None,
               d_trans: float = 0.0, d_yaw: float = 0.0) -> DynamicUpdate:
        """Advance to time `t` (seconds), roll by ego-motion, add new movers."""
        if np.isfinite(self.t):
            self._roll(float(d_trans), float(d_yaw))
        prev_active = self.until > self.t if np.isfinite(self.t) else np.zeros_like(self.until, bool)
        self.t = float(t)
        still = self.until > self.t
        expired = int(np.count_nonzero(prev_active & ~still))

        new_cells, n_pix = 0, 0
        if seg_label is not None and points_veh is not None:
            dyn = np.asarray(seg_label) == DYNAMIC_CLASS
            if valid is not None:
                dyn &= np.asarray(valid, bool)
            n_pix = int(dyn.sum())
            if n_pix >= self.min_pixels:
                pv = points_veh[dyn]
                rng = np.hypot(pv[:, 0], pv[:, 1])
                keep = np.isfinite(rng) & (rng < self.max_range_m)
                row, col, ok = geo.bev_indices(pv[keep][None], np.ones((1, int(keep.sum())), bool))
                row, col = row[ok], col[ok]
                if row.size:
                    hit = np.zeros((self.H, self.W), np.uint8)
                    hit[row, col] = 1
                    if int(hit.sum()) >= self.min_cells:
                        grown = cv2.dilate(hit, self._kernel) > 0
                        new_cells = int(np.count_nonzero(grown & ~still))
                        self.until[grown] = self.t + self.ttl_s
                        still = self.until > self.t

        nearest = float(self._range[still].min()) if still.any() else float("inf")
        self.last = DynamicUpdate(active=still, new_cells=new_cells, expired_cells=expired,
                                  n_pixels=n_pix, nearest_m=nearest)
        return self.last

    def apply(self, state: np.ndarray, active: Optional[np.ndarray] = None) -> np.ndarray:
        """Copy of an (8, H, W) world state with active movers written in as OBSTACLE."""
        from ..nav import bev_utils as bu
        a = (self.until > self.t) if active is None else active
        if not a.any():
            return state
        out = state.copy()
        out[bu.CH_OBS][a] = 1.0
        out[bu.CH_SAFE:bu.CH_UNK + 1, a] = 0.0
        out[bu.CH_SAFE + OBSTACLE][a] = 1.0
        out[bu.CH_CONF][a] = 1.0
        out[bu.CH_AGE][a] = 0.0
        return out

    def active_world_points(self, pose) -> tuple[np.ndarray, np.ndarray]:
        """World-frame centres of the active cells (for the D* Lite grid)."""
        a = self.until > self.t
        if not a.any():
            return np.zeros(0), np.zeros(0)
        rr, cc = np.nonzero(a)
        xv, yv = geo.bev_to_veh(rr.astype(np.float32), cc.astype(np.float32))
        px, py, th = float(pose[0]), float(pose[1]), float(pose[2])
        c, s = np.cos(th), np.sin(th)
        return px + c * xv - s * yv, py + s * xv + c * yv
