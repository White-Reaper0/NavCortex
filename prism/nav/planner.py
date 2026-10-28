"""Sampling-based rollout planner over the prism BEV world state.

What this is (and is not)
-------------------------
This is a **lightweight sampling-based rollout planner**: it enumerates a fixed
set of candidate control arcs (the six discrete `CFG.ACTIONS` plus a small fan of
steering perturbations around each), integrates each one forward with a unicycle
motion model over `CFG.safety.horizon_s` in `CFG.safety.n_rollout_steps` steps,
sweeps the vehicle footprint along the result through the BEV map, and scores the
outcome.  That is the same *idea* as Nav2's MPPI controller - predict many
candidate motions, evaluate them against a costmap, pick the best - but it is a
deterministic small-fan evaluation with a hand-written cost, **not** MPPI: there
is no importance-weighted stochastic sampling and no iterative control-sequence
refinement.  Please describe it that way.

Cost terms (all normalised, weights in `PlannerWeights`):
  * world-model predicted collision risk for the candidate's base action
  * geometric sweep risk from the BEV map (obstacle posterior + height step)
  * unknown / stale fraction under the swept footprint
  * 1 - mean confidence under the swept footprint
  * inverse lateral clearance to the nearest blocking cell
  * progress along the goal direction (negative cost - this is what stops the
    planner from preferring to sit still)
  * a small turn-rate penalty so it does not weave

Hard feasibility (sets `Trajectory.feasible = False` and writes a specific
`reject_reason` used verbatim on the dashboard):
  R-P1 height step above the footprint exceeds `CFG.ugv.clearance_m`
  R-P2 an OBSTACLE cell lies under the swept footprint
  R-P3 the corridor is more unknown than `CFG.safety.unknown_frac_stop`
  R-P4 free distance ahead is shorter than the braking distance at that speed
  R-P5 the rollout leaves the mapped area
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..config import CFG, ACTIONS, N_ACTIONS
from ..types import FramePacket, Trajectory
from . import bev_utils as bu


@dataclass
class PlannerWeights:
    wm_risk: float = 2.4
    sweep_risk: float = 2.0
    unknown: float = 1.3
    low_conf: float = 0.8
    clearance: float = 0.55
    progress: float = 1.6
    turn: float = 0.18
    step: float = 1.1
    #: cost of standing still while a goal is set (only with `goal_xy`).  Without
    #: it, a goal *behind* the vehicle deadlocks it: every turning arc earns little
    #: progress and pays turn/clearance terms, so STOP always scores lowest.  It is
    #: a cost, not a constraint - infeasible arcs stay rejected and every
    #: supervisor gate still applies - so the vehicle still stops when no safe arc
    #: exists or the remaining arcs are worse than idling by more than this.
    idle: float = 2.0
    #: with a goal point: penalty for ending the arc facing away from it, in
    #: [0, heading] as (1 - cos(angle off)) / 2.  Distance-reduction alone barely
    #: separates "U-turn towards a point behind" from "creep forward away from it".
    heading: float = 1.5


#: (base action, extra angular offset rad/s) fan.  Small deterministic spread
#: around each discrete action so the planner can shade a turn rather than only
#: choosing between six fixed arcs.
STEER_FAN: dict[int, tuple[float, ...]] = {
    0: (0.0, -0.30, +0.30, -0.60, +0.60),   # FORWARD
    1: (0.0, -0.35, +0.35),                 # LEFT
    2: (0.0, -0.35, +0.35),                 # RIGHT
    3: (0.0, -0.45, +0.45),                 # SLOW
    4: (0.0, -3.20),                        # REROUTE: + and - evasive arcs
    5: (0.0,),                              # STOP
}


def braking_distance(speed_mps: float) -> float:
    """Stopping space required at a commanded speed.

    `CFG.ugv.brake_distance_m` is the space needed at `max_speed_mps`; braking
    distance scales with v^2 for a constant deceleration, which is the model used
    here.  These are vehicle-envelope assumptions at the scale implied by the
    assumed camera height, not a measured brake test.
    """
    v = max(0.0, float(speed_mps))
    return CFG.ugv.brake_distance_m * (v / max(CFG.ugv.max_speed_mps, 1e-6)) ** 2


class Planner:
    """Enumerate + score candidate arcs against the BEV world state."""

    def __init__(self, weights: Optional[PlannerWeights] = None,
                 margin: float = CFG.safety.corridor_margin_m):
        self.w = weights or PlannerWeights()
        self.margin = float(margin)
        self.dt = CFG.safety.horizon_s / CFG.safety.n_rollout_steps
        self.n_steps = CFG.safety.n_rollout_steps
        self.candidates: list[tuple[int, float, float]] = []
        for a in range(N_ACTIONS):
            v, w0 = bu.action_motion(a)
            for dw in STEER_FAN.get(a, (0.0,)):
                self.candidates.append((a, v, w0 + dw))

    # ------------------------------------------------------------------
    def rollout(self, v: float, w: float):
        return bu.arc_poses(v, w, self.dt, self.n_steps)

    def plan(self, state: np.ndarray,
             wm_risk: Optional[Sequence[float]] = None,
             goal_yaw: float = 0.0,
             goal_xy: Optional[Sequence[float]] = None) -> list[Trajectory]:
        """Score every candidate arc.  `wm_risk` is one risk in [0,1] per action.

        `goal_yaw` is the desired heading in the vehicle frame (0 = straight
        ahead, +CCW).  On the recorded footage there is no Point B, so the goal
        direction is "keep going forward along the trail", i.e. 0.

        `goal_xy` is the point the arcs should head for, in the vehicle frame
        (Point B itself on open ground, or the D* Lite look-ahead point when the
        vehicle is boxed in; see `goal_planner.py`).  When given, progress is the
        reduction in distance to that point, so an arc that would overshoot it
        earns nothing for the overshoot.
        """
        wmr = np.zeros(N_ACTIONS, np.float32) if wm_risk is None \
            else np.clip(np.asarray(wm_risk, np.float32).ravel()[:N_ACTIONS], 0, 1)
        gdir = np.array([-np.sin(goal_yaw), np.cos(goal_yaw)], np.float32)
        gxy = None if goal_xy is None else np.asarray(goal_xy, np.float32).ravel()[:2]
        g0 = float(np.linalg.norm(gxy)) if gxy is not None else 0.0
        step_map = bu.height_step_map(state)      # computed once for all candidates

        trajs: list[Trajectory] = []
        for a, v, w in self.candidates:
            xy, yaw = self.rollout(v, w)
            m = bu.sweep(state, xy, yaw, self.margin, step_map)
            clr = bu.sweep_clearance_series(state, xy, step_map)

            if gxy is None:
                progress = float(np.dot(xy[-1], gdir))
            else:
                progress = g0 - float(np.min(np.linalg.norm(xy - gxy[None], axis=1)))
            step_norm = float(np.clip(m.max_step / max(CFG.ugv.clearance_m, 1e-3), 0, 2))
            sweep_risk = float(np.max(m.per_step_risk))
            inv_clear = float(1.0 / max(m.min_clearance, 0.08)) if m.min_clearance < 2.0 else 0.0

            cost = (self.w.wm_risk * float(wmr[a])
                    + self.w.sweep_risk * sweep_risk
                    + self.w.unknown * m.unknown_frac
                    + self.w.low_conf * (1.0 - m.mean_conf)
                    + self.w.clearance * inv_clear
                    + self.w.step * step_norm
                    + self.w.turn * abs(w)
                    - self.w.progress * progress)

            feasible, reason = True, ""
            brake = braking_distance(v)
            if m.max_step > CFG.ugv.clearance_m:
                feasible = False
                reason = (f"R-P1 height step {m.max_step:.3f} m > clearance "
                          f"{CFG.ugv.clearance_m:.3f} m at {m.max_step_dist:.2f} m ahead")
            elif m.obstacle_frac > 0.5:
                feasible = False
                reason = (f"R-P2 OBSTACLE cell under footprint at "
                          f"{m.free_distance:.2f} m ahead (p={m.obstacle_frac:.2f})")
            elif m.unknown_frac > CFG.safety.unknown_frac_stop:
                feasible = False
                reason = (f"R-P3 corridor {m.unknown_frac*100:.0f}% unknown > "
                          f"{CFG.safety.unknown_frac_stop*100:.0f}% limit")
            elif v > 0.05 and m.free_distance < brake:
                feasible = False
                reason = (f"R-P4 free distance {m.free_distance:.2f} m < braking "
                          f"distance {brake:.2f} m at {v:.2f} m/s")
            elif m.off_map_frac > 0.55:
                feasible = False
                reason = (f"R-P5 {m.off_map_frac*100:.0f}% of the rollout leaves the "
                          f"{CFG.bev.range_forward_m:.1f} m map")

            if gxy is not None:
                if v < 0.05:
                    cost += self.w.idle
                elif g0 - progress > 0.30:
                    # only arcs that do not pass through the target are judged on
                    # where they end up facing; one that reaches it has done its job
                    to = gxy - xy[-1]
                    if float(np.hypot(to[0], to[1])) > 0.15:
                        want = float(np.arctan2(-to[0], to[1]))
                        cost += self.w.heading * 0.5 * (1.0 - float(np.cos(want - yaw[-1])))
            if not feasible:
                cost += 100.0

            trajs.append(Trajectory(
                action=a,
                xy=xy.astype(np.float32),
                yaw=yaw.astype(np.float32),
                clearance=clr.astype(np.float32),
                max_step=float(m.max_step),
                unknown_frac=float(m.unknown_frac),
                mean_conf=float(m.mean_conf),
                collision_risk=float(max(sweep_risk, wmr[a])),
                cost=float(cost),
                feasible=bool(feasible),
                reject_reason=reason,
            ))
        return trajs

    # ------------------------------------------------------------------
    @staticmethod
    def best(trajs: Sequence[Trajectory]) -> tuple[int, Optional[Trajectory]]:
        """Index of the lowest-cost feasible trajectory (-1 if none feasible)."""
        feas = [i for i, t in enumerate(trajs) if t.feasible]
        if not feas:
            return -1, None
        i = min(feas, key=lambda j: trajs[j].cost)
        return i, trajs[i]

    @staticmethod
    def best_for_action(trajs: Sequence[Trajectory], action: int) -> tuple[int, Optional[Trajectory]]:
        """Lowest-cost trajectory whose base action is `action`, feasible or not."""
        cand = [i for i, t in enumerate(trajs) if t.action == action]
        if not cand:
            return -1, None
        feas = [i for i in cand if trajs[i].feasible]
        pool = feas if feas else cand
        i = min(pool, key=lambda j: trajs[j].cost)
        return i, trajs[i]


class PlannerStage:
    """Contract-shaped wrapper filling `packet.trajectories`."""

    name = "planner"

    def __init__(self, device: str = "cpu", **kw):
        self.planner = Planner(**kw)

    def reset(self) -> None:
        pass

    def __call__(self, packet: FramePacket) -> FramePacket:
        if packet.bev is None:
            return packet
        st = bu.pack_state(packet.bev.height, packet.bev.trav_prob, packet.bev.conf,
                           packet.bev.age, getattr(packet.bev, "hits", None))
        wmr = [float(p.risk_total) for p in packet.wm_preds] if packet.wm_preds else None
        packet.trajectories = self.planner.plan(st, wmr)
        return packet


if __name__ == "__main__":
    import time
    pl = Planner()
    print(f"planner: {len(pl.candidates)} candidate arcs, "
          f"{pl.n_steps} steps x {pl.dt:.3f} s = {CFG.safety.horizon_s:.1f} s horizon, "
          f"footprint {CFG.ugv.width_m:.2f} m + 2x{CFG.safety.corridor_margin_m:.2f} m margin")
    for scene, kw in [("clear", {}), ("wall @1.0m h=0.10", dict(kind="wall", wall_dist_m=1.0)),
                      ("wall_left @1.2m", dict(kind="wall_left", wall_dist_m=1.2)),
                      ("blind", dict(kind="blind"))]:
        st = bu.synthetic_state(**({"kind": "clear"} | kw))
        t0 = time.perf_counter()
        trajs = pl.plan(st)
        dt = (time.perf_counter() - t0) * 1e3
        bi, bt = pl.best(trajs)
        nf = sum(t.feasible for t in trajs)
        print(f"\n{scene:22s}  {dt:5.1f} ms  feasible {nf}/{len(trajs)}")
        if bt is not None:
            print(f"   best: {ACTIONS[bt.action]:8s} cost {bt.cost:+.2f} "
                  f"step {bt.max_step:.3f} m  unk {bt.unknown_frac:.2f} "
                  f"conf {bt.mean_conf:.2f} risk {bt.collision_risk:.2f}")
        else:
            print("   best: NONE FEASIBLE")
        for t in trajs[:6]:
            flag = "ok " if t.feasible else "REJ"
            print(f"   {flag} {ACTIONS[t.action]:8s} cost {t.cost:+7.2f}  {t.reject_reason}")
