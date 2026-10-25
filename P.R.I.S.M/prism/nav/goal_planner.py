"""Point A -> Point B: arcs head for B on open ground, D* Lite when boxed in.

How one control cycle works
---------------------------
1. Fold the current BEV world state into the world-fixed `GlobalCostGrid`
   ("the ground seen so far"), plus any active dynamic obstacles.
2. If the vehicle is within `arrive_tol_m` of Point B -> ARRIVED (stop).
3. **Open ground**: if the straight segment towards B is observed, unblocked in
   the local map and unblocked in the global grid, the 17 candidate arcs are
   simply scored for progress towards B itself.
4. **Boxed in**: otherwise D* Lite searches from B back to the vehicle over the
   global grid.  Unseen ground costs `cost_unknown` per metre - more than seen
   safe ground, so the search prefers what it has seen, but never infinite, so a
   route through unexplored ground is still a route.  The arcs are then scored
   for progress towards a look-ahead point on that path.
   D* Lite is incremental: when cells change (a dead end is seen, a person steps
   in, a dynamic cell expires) or the vehicle moves, it repairs only what changed.
5. Cells the rolling map has already forgotten (it drops a cell 1.5 s after it
   leaves the camera's view) are refilled from the global grid's memory at a
   reduced confidence (`GoalStatus.state`), so the vehicle can turn round over
   ground it has seen - slowly, because remembered ground is below `conf_slow`.
6. The arc planner + safety supervisor then decide the actual command, exactly
   as before.  The goal layer *chooses a direction*; it never overrides a rule.

What this is not
----------------
The recorded footage has no Point B and no ground-truth map, so this layer is
exercised by `tools/sim_goal_nav.py` (a 2-D kinematic check with a known map)
and by the Gazebo Harmonic world in `ros2/`, not by the five demo clips.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import CFG
from . import bev_utils as bu
from .dstar_lite import DStarLite
from .goal_map import GlobalCostGrid
from .nav_config import NAV, NavConfig


@dataclass
class GoalStatus:
    """Outcome of one goal cycle.  `state` is the world state the arc planner should
    use: the input state plus remembered ground from the global grid."""
    mode: str = "idle"                  # idle | open | dstar | searching | arrived | no_route
    goal_world: Optional[tuple[float, float]] = None
    dist_m: float = float("inf")
    bearing_rad: float = 0.0            # to Point B, vehicle frame, +CCW from forward
    steer_xy: Optional[np.ndarray] = None   # vehicle-frame point the arcs head for
    steer_yaw: float = 0.0
    path_world: Optional[np.ndarray] = None  # (N, 2) D* Lite path, world metres
    path_cost: float = float("inf")
    replanned: bool = False
    expansions: int = 0
    plan_ms: float = 0.0
    reason: str = ""
    grid: dict = field(default_factory=dict)
    recalled_cells: int = 0
    state: Optional[np.ndarray] = None


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class GoalPlanner:
    """Goal layer between the 2.5D map and the arc planner."""

    def __init__(self, cfg: Optional[NavConfig] = None, start_pose=(0.0, 0.0, 0.0),
                 recall: bool = True, recall_conf: float = 0.5):
        self.cfg = cfg or NAV
        self.recall, self.recall_conf = bool(recall), float(recall_conf)
        self.grid = GlobalCostGrid(self.cfg.grid, origin_xy=(float(start_pose[0]),
                                                             float(start_pose[1])))
        self.goal: Optional[tuple[float, float]] = None
        self._ds: Optional[DStarLite] = None
        self._last_plan_t = -math.inf
        self.status = GoalStatus()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        self.grid.reset()
        self._ds = None
        self._last_plan_t = -math.inf
        self.status = GoalStatus(goal_world=self.goal)

    def set_goal(self, x: float, y: float) -> None:
        """Point B in the odometry/world frame (metres)."""
        r, c = self.grid.world_to_cell(x, y)
        if not bool(self.grid.inside(r, c)):
            raise ValueError(f"Point B ({x:.1f}, {y:.1f}) m is outside the "
                             f"{self.grid.n * self.grid.res:.0f} m global grid")
        self.goal = (float(x), float(y))
        self._ds = None
        self._last_plan_t = -math.inf
        self.status = GoalStatus(mode="idle", goal_world=self.goal)

    # ------------------------------------------------------------------ helpers
    def _local_open(self, state: np.ndarray, step_map: np.ndarray,
                    tx: float, ty: float) -> bool:
        """Is the straight corridor to (tx, ty) - clipped to the map - observed and clear?"""
        b = CFG.bev
        d = math.hypot(tx, ty)
        if d < 1e-3:
            return True
        L = min(d, b.range_forward_m * 0.95)
        n = max(2, int(L / b.res_m))
        s = np.linspace(b.res_m, L, n)
        ux, uy = tx / d, ty / d
        half = CFG.ugv.width_m * 0.5 + CFG.safety.corridor_margin_m
        offs = np.linspace(-half, half, max(2, int(2 * half / b.res_m) + 1))
        xs = (s[:, None] * ux + offs[None, :] * uy).ravel()
        ys = (s[:, None] * uy - offs[None, :] * ux).ravel()
        col = np.floor(xs / b.res_m).astype(np.int64) + b.n_lateral
        row = (b.n_forward - 1) - np.floor(ys / b.res_m).astype(np.int64)
        ins = (col >= 0) & (col < b.W) & (row >= 0) & (row < b.H)
        if ins.mean() < 0.5:                      # the segment leaves the local map sideways
            return False
        r, c = row[ins], col[ins]
        obs = state[bu.CH_OBS][r, c] > 0.5
        if obs.mean() < self.cfg.goal.open_min_observed:
            return False
        stp = step_map[r, c]
        blocked = (state[bu.CH_OBST][r, c] > 0.5) | (np.isfinite(stp) & (stp > CFG.ugv.clearance_m))
        return not bool(blocked.any())

    def _ensure_search(self, cost: np.ndarray, start_rc, t: float) -> tuple[bool, bool]:
        """Create or repair the D* Lite search.  Returns (reachable, replanned)."""
        gr, gc = self.grid.world_to_cell(*self.goal)
        goal_rc = (int(gr), int(gc))
        c = cost.copy()
        # The vehicle must always be able to plan *out of* where it stands, and
        # Point B must be enterable: inflation around a nearby wall or a person
        # would otherwise seal the start and make every route "impossible".  This
        # only shapes the global route - whether moving is actually safe is still
        # decided by the arc planner's footprint checks and the supervisor.
        rad = int(np.ceil((CFG.ugv.width_m * 0.5 + self.cfg.grid.inflate_m) / self.grid.res))
        for (r0, c0) in (start_rc, goal_rc):
            sl = (slice(max(r0 - rad, 0), r0 + rad + 1), slice(max(c0 - rad, 0), c0 + rad + 1))
            blk = ~np.isfinite(c[sl])
            c[sl][blk] = self.cfg.grid.cost_risky * 2.0
        if self._ds is None:
            self._ds = DStarLite(c, start_rc, goal_rc, res_m=self.grid.res,
                                 min_cost=self.grid.min_cost)
            ok = self._ds.compute(self.cfg.goal.max_expansions)
            self._last_plan_t = t
            return ok, True
        self._ds.move_start(start_rc)
        # Blocking changes (a wall seen, a person stepping in, a dynamic cell
        # expiring) are applied at once.  Cost refinements on cells that stay
        # passable (unknown -> seen-safe as the camera sweeps) are batched every
        # `replan_period_s`: applying thousands of them every frame made each D* Lite
        # repair cost hundreds of milliseconds for no change in the route.
        old = self._ds.cost
        flip = np.isfinite(old) != np.isfinite(c)
        periodic = (t - self._last_plan_t) >= self.cfg.goal.replan_period_s
        if periodic:
            n = self._ds.set_cost_grid(c)
            self._last_plan_t = t
        elif flip.any():
            rr, cc = np.nonzero(flip)
            n = self._ds.update_costs(zip(rr.tolist(), cc.tolist()), c[rr, cc].tolist())
        else:
            n = 0
        ok = self._ds.compute(self.cfg.goal.max_expansions)
        return ok, bool(n) or self._ds.expansions > 0

    # ------------------------------------------------------------------ main
    def step(self, state: np.ndarray, pose, t: float,
             dynamic_world: Optional[tuple[np.ndarray, np.ndarray]] = None,
             dynamic_until: Optional[float] = None,
             step_map: Optional[np.ndarray] = None,
             tracking_ok: bool = True) -> GoalStatus:
        """One goal cycle.  `pose` = (x, y, yaw) in the world frame."""
        t0 = time.perf_counter()
        step_map = bu.height_step_map(state) if step_map is None else step_map
        if tracking_ok:
            # a lost VO pose cannot place the observation in the world: skip it
            self.grid.integrate(state, pose, t, step_map)
        if dynamic_world is not None and len(dynamic_world[0]):
            until = t + self.cfg.dynamic.ttl_s if dynamic_until is None else dynamic_until
            self.grid.mark_dynamic(dynamic_world[0], dynamic_world[1], until)

        st = GoalStatus(goal_world=self.goal)
        if self.recall and tracking_ok:
            st.state, st.recalled_cells = self.grid.recall_into_state(
                state, pose, t, conf=self.recall_conf)
        else:
            st.state = state
        if self.goal is None:
            st.mode, st.reason = "idle", "no Point B set: heading straight ahead"
            st.steer_yaw = 0.0
            self.status = st
            return st

        gx, gy = self.grid.world_to_veh(self.goal[0], self.goal[1], pose)
        gx, gy = float(gx), float(gy)
        st.dist_m = math.hypot(gx, gy)
        st.bearing_rad = math.atan2(-gx, gy)          # yaw CCW from +y
        if st.dist_m <= self.cfg.goal.arrive_tol_m:
            st.mode = "arrived"
            st.reason = (f"G0 ARRIVED: {st.dist_m:.2f} m from Point B "
                         f"<= {self.cfg.goal.arrive_tol_m:.2f} m tolerance")
            st.plan_ms = (time.perf_counter() - t0) * 1e3
            self.status = st
            return st

        cost = self.grid.cost_grid(t)
        pos = (float(pose[0]), float(pose[1]))
        open_local = self._local_open(state, step_map, gx, gy)
        open_global = not self.grid.blocked_on_segment(pos, self.goal, t, cost)
        if open_local and open_global:
            st.mode = "open"
            st.steer_xy = np.array([gx, gy], np.float32)
            st.steer_yaw = st.bearing_rad
            st.reason = (f"open ground: heading straight for Point B, "
                         f"{st.dist_m:.1f} m at {math.degrees(st.bearing_rad):+.0f} deg")
        else:
            sr, sc = self.grid.world_to_cell(*pos)
            ok, replanned = self._ensure_search(cost, (int(sr), int(sc)), t)
            st.replanned = replanned
            st.expansions = self._ds.expansions
            st.path_cost = self._ds.path_cost()
            if not ok and not self._ds.converged:
                st.mode = "searching"
                st.reason = (f"G2 WAIT: D* Lite still searching ({self._ds.expansions} "
                             f"cells this cycle, resumes next cycle)")
            elif not ok:
                st.mode = "no_route"
                st.reason = ("G1 NO ROUTE: D* Lite finds no path to Point B over the "
                             "ground seen so far - holding until something changes "
                             "(a mover leaves, a cell expires)")
            else:
                path = self._ds.path()
                pr = np.array([p[0] for p in path]); pc = np.array([p[1] for p in path])
                wx, wy = self.grid.cell_to_world(pr, pc)
                st.path_world = np.stack([wx, wy], 1).astype(np.float32)
                vx, vy = self.grid.world_to_veh(wx, wy, pose)
                d = np.hypot(vx, vy)
                ahead = np.nonzero(d >= self.cfg.goal.lookahead_m)[0]
                k = int(ahead[0]) if ahead.size else len(path) - 1
                st.steer_xy = np.array([vx[k], vy[k]], np.float32)
                st.steer_yaw = math.atan2(-float(vx[k]), float(vy[k]))
                why = "blocked locally" if not open_local else "blocked further on"
                st.mode = "dstar"
                st.reason = (f"boxed in ({why}): D* Lite route {st.path_cost:.1f} cost-m, "
                             f"{len(path)} cells, steering {math.degrees(st.steer_yaw):+.0f} deg"
                             f"{' (replanned)' if replanned else ''}")
        st.grid = self.grid.stats(t)
        st.plan_ms = (time.perf_counter() - t0) * 1e3
        self.status = st
        return st
