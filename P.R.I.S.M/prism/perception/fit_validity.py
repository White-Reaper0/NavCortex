"""Where the ground fit is unreliable, depth is marked invalid (-> UNKNOWN).

Every metric number prism produces comes from one ground-plane fit per frame
(`geometry.fit_metric_ground`).  When that fit is wrong, every height and every
distance derived from it is wrong *with full confidence*.  This module decides
where the fit can be trusted, and everything it rejects is dropped from
`DepthResult.valid`, which the 2.5D map turns into UNKNOWN - slow down or
reroute, never "free".

Two checks
----------
1. **Frame level.**  If the fit fails this frame, the previous good fit is
   reused for at most `max_stale_frames` (the scene barely changes in 0.5 s);
   after that the whole frame is invalid.  With no previous good fit at all, the
   frame is invalid from the start - the old behaviour silently used a nominal
   plane and published its depths as valid.

2. **Region level.**  The lower half of the image is split into tiles.  In each
   tile, pixels the terrain model calls ground (trail, grass) should agree with
   the fitted plane.  Where their median relative inverse-depth residual is
   larger than `tile_tol`, the plane does not describe that patch of ground
   (a slope change, a ditch, a bad depth patch) and the tile is invalid.
   Non-ground pixels are not tested - an obstacle is supposed to be off-plane -
   but they sit on the same fit, so an unreliable tile is dropped as a whole.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import GroundFit, ray_grid

GROUND_CLASSES = (1, 2)       # prism-7 trail, grass


@dataclass
class FitValidity:
    valid: np.ndarray                 # (H, W) bool, AND this into DepthResult.valid
    frame_ok: bool
    reason: str = ""
    bad_tiles: int = 0
    n_tiles: int = 0


def region_validity(q: np.ndarray, base_valid: np.ndarray, seg_label: Optional[np.ndarray],
                    fit: GroundFit, K: Optional[np.ndarray] = None,
                    tiles: tuple[int, int] = (8, 4), tile_tol: float = 0.30,
                    min_px: int = 150) -> tuple[np.ndarray, int, int]:
    """(valid mask, n_bad_tiles, n_tested_tiles) from per-tile plane residuals."""
    H, W = q.shape
    valid = np.ones((H, W), bool)
    if seg_label is None or seg_label.shape != q.shape:
        return valid, 0, 0
    m = ray_grid(H, W, K)
    inv_plane = m @ (np.asarray(fit.normal, np.float32) / float(fit.height))   # 1/D of the plane
    inv_pred = fit.a * q + fit.b
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(inv_plane - inv_pred) / np.maximum(np.abs(inv_plane), 1e-6)
    ground = base_valid & np.isin(seg_label, GROUND_CLASSES) & (inv_plane > 0)
    y0 = H // 2
    nx, ny = tiles
    xs = np.linspace(0, W, nx + 1).astype(int)
    ys = np.linspace(y0, H, ny + 1).astype(int)
    bad = tested = 0
    for i in range(ny):
        for j in range(nx):
            sl = (slice(ys[i], ys[i + 1]), slice(xs[j], xs[j + 1]))
            g = ground[sl]
            if int(g.sum()) < min_px:
                continue
            tested += 1
            if float(np.median(rel[sl][g])) > tile_tol:
                valid[sl] = False
                bad += 1
    return valid, bad, tested


class FitValidityTracker:
    """Frame- and region-level ground-fit validity.  `reset()` between clips."""

    def __init__(self, max_stale_frames: int = 15, tile_tol: float = 0.30):
        self.max_stale_frames = int(max_stale_frames)
        self.tile_tol = float(tile_tol)
        self.reset()

    def reset(self) -> None:
        self.stale = 0
        self.have_good = False

    def __call__(self, q: np.ndarray, base_valid: np.ndarray,
                 seg_label: Optional[np.ndarray], fit: GroundFit,
                 fit_ok_this_frame: bool) -> FitValidity:
        H, W = q.shape
        if fit_ok_this_frame:
            self.stale = 0
            self.have_good = True
        else:
            self.stale += 1
        if not self.have_good:
            return FitValidity(np.zeros((H, W), bool), False,
                               "no reliable ground fit yet: whole frame UNKNOWN")
        if self.stale > self.max_stale_frames:
            return FitValidity(np.zeros((H, W), bool), False,
                               f"ground fit failed for {self.stale} frames "
                               f"(> {self.max_stale_frames}): whole frame UNKNOWN")
        v, bad, n = region_validity(q, base_valid, seg_label, fit, tile_tol=self.tile_tol)
        why = "" if not bad else f"{bad}/{n} ground tiles disagree with the plane: UNKNOWN"
        if self.stale:
            why = (why + "; " if why else "") + f"reusing the last good fit ({self.stale} frames old)"
        return FitValidity(v, True, why, bad, n)
