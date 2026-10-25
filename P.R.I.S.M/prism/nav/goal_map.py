"""World-fixed coarse cost grid: "the ground seen so far", for D* Lite.

The rolling 2.5D map (`perception/mapping.py`) is vehicle-centred and forgets a
cell after `CFG.bev.max_age_frames`.  That is right for the arc planner - it only
needs the next two seconds - but it cannot remember that the corridor on the left
was a dead end ten seconds ago.  `GlobalCostGrid` folds every BEV state into a
world-frame grid anchored at the start pose, keeps the most recent evidence per
cell, and turns it into a traversal cost per metre for `dstar_lite.DStarLite`:

    never observed / forgotten     cost_unknown   (finite: unseen ground COSTS,
    observed with low confidence   cost_unknown    it is never free, and never
                                                   treated as a wall either)
    observed safe                  cost_safe
    observed risky                 cost_risky
    observed obstacle / step       inf             (then dilated by the footprint)
    active dynamic obstacle        inf             (expires after ttl_s)

World frame = the odometry frame: x right, y forward at the start pose, yaw CCW
from +y, metres.  It inherits VO drift; evidence older than `forget_s` reverts to
unknown so an old, drifted obstacle cannot wall the vehicle in forever.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..config import CFG
from ..perception.geometry import bev_to_veh
from . import bev_utils as bu
from .nav_config import NAV, GlobalGridConfig

#: per-observation codes, ordered by severity: within one frame the worst wins
_UNSEEN, _SAFE, _RISKY, _UNKNOWN, _BLOCKED = 0, 1, 2, 3, 4


class GlobalCostGrid:
    def __init__(self, cfg: Optional[GlobalGridConfig] = None, forget_s: float = 30.0,
                 origin_xy: tuple[float, float] = (0.0, 0.0)):
        self.cfg = cfg or NAV.grid
        self.res = float(self.cfg.res_m)
        self.n = int(np.ceil(self.cfg.size_m / self.res))
        self.forget_s = float(forget_s)
        self.ox, self.oy = float(origin_xy[0]), float(origin_xy[1])
        # vehicle-frame centres of every BEV cell, computed once
        b = CFG.bev
        rr, cc = np.meshgrid(np.arange(b.H, dtype=np.float32),
                             np.arange(b.W, dtype=np.float32), indexing="ij")
        self._bx, self._by = bev_to_veh(rr, cc)
        r = int(np.ceil(self.cfg.inflate_m / self.res))
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        self.reset()

    def reset(self) -> None:
        self.code = np.zeros((self.n, self.n), np.uint8)       # last evidence
        self.stamp = np.full((self.n, self.n), -np.inf)         # when it was seen
        self.dyn_until = np.full((self.n, self.n), -np.inf)     # dynamic expiry
        self._cost: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ frames
    def world_to_cell(self, x, y):
        """World metres -> (row, col).  Row grows with +y (forward at the start)."""
        half = self.n * self.res * 0.5
        col = np.floor((np.asarray(x) - self.ox + half) / self.res).astype(np.int64)
        row = np.floor((np.asarray(y) - self.oy + half) / self.res).astype(np.int64)
        return row, col

    def cell_to_world(self, row, col):
        half = self.n * self.res * 0.5
        x = (np.asarray(col, np.float64) + 0.5) * self.res - half + self.ox
        y = (np.asarray(row, np.float64) + 0.5) * self.res - half + self.oy
        return x, y

    def inside(self, row, col):
        row, col = np.asarray(row), np.asarray(col)
        return (row >= 0) & (row < self.n) & (col >= 0) & (col < self.n)

    @staticmethod
    def veh_to_world(xv, yv, pose):
        """Vehicle-frame metres -> world metres for pose (x, y, yaw)."""
        px, py, th = float(pose[0]), float(pose[1]), float(pose[2])
        c, s = np.cos(th), np.sin(th)
        return px + c * xv - s * yv, py + s * xv + c * yv

    @staticmethod
    def world_to_veh(xw, yw, pose):
        px, py, th = float(pose[0]), float(pose[1]), float(pose[2])
        c, s = np.cos(th), np.sin(th)
        dx, dy = np.asarray(xw) - px, np.asarray(yw) - py
        return c * dx + s * dy, -s * dx + c * dy

    # ------------------------------------------------------------------ evidence
    def classify_state(self, state: np.ndarray, step_map: Optional[np.ndarray] = None) -> np.ndarray:
        """(H, W) uint8 evidence code per BEV cell, same rules as the arc planner."""
        g = self.cfg
        step = bu.height_step_map(state) if step_map is None else step_map
        obs = state[bu.CH_OBS] > 0.5
        code = np.full(obs.shape, _UNSEEN, np.uint8)
        low = obs & ((state[bu.CH_CONF] < g.min_conf) | (state[bu.CH_UNK] > 0.5))
        blocked = obs & ((state[bu.CH_OBST] > g.obstacle_p) |
                         (np.isfinite(step) & (step > CFG.ugv.clearance_m)))
        risky = obs & (state[bu.CH_RISKY] > g.risky_p)
        code[obs] = _SAFE
        code[risky] = _RISKY
        code[low] = _UNKNOWN
        code[blocked] = _BLOCKED          # an obstacle is never downgraded by low confidence
        return code

    def integrate(self, state: np.ndarray, pose, t: float,
                  step_map: Optional[np.ndarray] = None) -> int:
        """Fold one BEV world state, observed at `pose`, into the grid.

        Returns the number of global cells whose evidence changed.
        """
        code = self.classify_state(state, step_map)
        m = code != _UNSEEN
        if not m.any():
            return 0
        xw, yw = self.veh_to_world(self._bx[m], self._by[m], pose)
        r, c = self.world_to_cell(xw, yw)
        ok = self.inside(r, c)
        r, c, k = r[ok], c[ok], code[m][ok]
        if r.size == 0:
            return 0
        frame = np.zeros((self.n, self.n), np.uint8)
        np.maximum.at(frame, (r, c), k)          # worst evidence in this frame wins
        hit = frame > 0
        changed = int(np.count_nonzero(hit & (frame != self.code)))
        self.code[hit] = frame[hit]
        self.stamp[hit] = float(t)
        self._cost = None
        return changed

    def mark_dynamic(self, xw: np.ndarray, yw: np.ndarray, until: float) -> int:
        """Block world points until time `until` (moving obstacles)."""
        r, c = self.world_to_cell(xw, yw)
        ok = self.inside(r, c)
        r, c = r[ok], c[ok]
        if r.size == 0:
            return 0
        new = int(np.count_nonzero(self.dyn_until[r, c] < until - 1e-6))
        self.dyn_until[r, c] = np.maximum(self.dyn_until[r, c], until)
        self._cost = None
        return new

    # ------------------------------------------------------------------ cost
    def cost_grid(self, t: float) -> np.ndarray:
        """(n, n) float64 cost per metre; inf = blocked (footprint-inflated)."""
        g = self.cfg
        code = self.code.copy()
        code[(t - self.stamp) > self.forget_s] = _UNSEEN
        cost = np.full(code.shape, g.cost_unknown, np.float64)
        cost[code == _SAFE] = g.cost_safe
        cost[code == _RISKY] = g.cost_risky
        blocked = (code == _BLOCKED) | (self.dyn_until > t)
        if blocked.any():
            blocked = cv2.dilate(blocked.astype(np.uint8), self._kernel) > 0
            # soft inflation (as in a Nav2 costmap's inflation layer): cost rises
            # towards the inflated boundary so routes keep off walls instead of
            # grazing them, which would leave the arc planner no feasible footprint
            if g.soft_inflate_m > 0:
                d = cv2.distanceTransform((~blocked).astype(np.uint8), cv2.DIST_L2, 3) * self.res
                near = np.clip(1.0 - d / g.soft_inflate_m, 0.0, 1.0)
                cost *= 1.0 + g.soft_inflate_gain * near
            cost[blocked] = np.inf
        self._cost = cost
        return cost

    @property
    def min_cost(self) -> float:
        g = self.cfg
        return float(min(g.cost_safe, g.cost_risky, g.cost_unknown))

    def blocked_on_segment(self, a_xy, b_xy, t: float, cost: Optional[np.ndarray] = None) -> bool:
        """True if any inflated-blocked global cell lies on the straight segment a->b."""
        cost = self.cost_grid(t) if cost is None else cost
        d = float(np.hypot(b_xy[0] - a_xy[0], b_xy[1] - a_xy[1]))
        n = max(2, int(np.ceil(d / (0.5 * self.res))))
        s = np.linspace(0.0, 1.0, n)
        xs = a_xy[0] + s * (b_xy[0] - a_xy[0])
        ys = a_xy[1] + s * (b_xy[1] - a_xy[1])
        r, c = self.world_to_cell(xs, ys)
        ok = self.inside(r, c)
        return bool(np.any(~np.isfinite(cost[r[ok], c[ok]])))

    def recall_into_state(self, state: np.ndarray, pose, t: float,
                          conf: float = 0.5, max_age_s: float = 15.0) -> tuple[np.ndarray, int]:
        """Fill BEV cells the rolling map has forgotten from remembered evidence.

        The rolling map drops a cell 1.5 s after it leaves the camera's view, so the
        ground the vehicle just drove over reads UNKNOWN as soon as it turns round -
        and every turn-around arc is rejected as "62% unknown".  This writes the
        global grid's remembered evidence (no older than `max_age_s`) into cells that
        are unobserved *locally*, at a deliberately reduced confidence (`conf`, below
        the supervisor's `conf_slow`), so the vehicle may turn back over ground it
        has seen but slows while it relies on memory.  Live observations are never
        overwritten, and remembered obstacles come back as obstacles.
        """
        obs = state[bu.CH_OBS] > 0.5
        miss = ~obs
        if not miss.any():
            return state, 0
        xw, yw = self.veh_to_world(self._bx[miss], self._by[miss], pose)
        r, c = self.world_to_cell(xw, yw)
        ok = self.inside(r, c)
        code = np.zeros(r.shape, np.uint8)
        age = np.full(r.shape, np.inf)
        code[ok] = self.code[r[ok], c[ok]]
        age[ok] = t - self.stamp[r[ok], c[ok]]
        use = (code != _UNSEEN) & (code != _UNKNOWN) & (age <= max_age_s)
        if not use.any():
            return state, 0
        out = state.copy()
        rows, cols = np.nonzero(miss)
        rows, cols, code, age = rows[use], cols[use], code[use], age[use]
        out[bu.CH_OBS, rows, cols] = 1.0
        out[bu.CH_SAFE:bu.CH_UNK + 1, rows, cols] = 0.0
        safe, risky, blk = code == _SAFE, code == _RISKY, code == _BLOCKED
        out[bu.CH_SAFE, rows[safe], cols[safe]] = 0.85
        out[bu.CH_RISKY, rows[safe], cols[safe]] = 0.10
        out[bu.CH_UNK, rows[safe], cols[safe]] = 0.05
        out[bu.CH_RISKY, rows[risky], cols[risky]] = 0.80
        out[bu.CH_UNK, rows[risky], cols[risky]] = 0.20
        out[bu.CH_OBST, rows[blk], cols[blk]] = 0.95
        out[bu.CH_UNK, rows[blk], cols[blk]] = 0.05
        out[bu.CH_CONF, rows, cols] = float(conf)
        out[bu.CH_AGE, rows, cols] = 1.0
        # The global grid keeps no metric height, so recalled cells are written at
        # local-ground height (0 m in CH_HEIGHT's normalised encoding - *not* 0.0,
        # which would decode to mid-range, a phantom step); remembered obstacles
        # are carried by the obstacle posterior instead.
        b = CFG.bev
        out[bu.CH_HEIGHT, rows, cols] = (0.0 - b.z_min) / (b.z_max - b.z_min) * 2.0 - 1.0
        return out, int(use.sum())

    def stats(self, t: float) -> dict:
        cost = self._cost if self._cost is not None else self.cost_grid(t)
        seen = self.code != _UNSEEN
        return dict(cells=int(self.n * self.n), seen=int(seen.sum()),
                    blocked=int((~np.isfinite(cost)).sum()),
                    dynamic=int((self.dyn_until > t).sum()))
