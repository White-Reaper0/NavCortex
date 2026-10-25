"""prism safety supervisor - the gate that sits over the RL policy.

The RL policy proposes; the supervisor disposes.  Every rule below is derived
from `CFG.safety` and `CFG.ugv`, is evaluated in a fixed priority order, and
writes a specific human-readable `Decision.reason` naming the rule number and the
numbers that triggered it, e.g.

    "R3 STOP: height step 0.087 m > clearance 0.045 m at 1.4 m ahead"

That string is what the final dashboard shows, so it must always say *why*.

Rule order (first match wins)
-----------------------------
R1  VO tracking lost  -> STOP, goal pursuit suspended.  A monocular stack with no
    pose has no idea where the map is relative to the vehicle, so per the
    reference design it must not keep driving on a stale map.
R2  No feasible candidate trajectory at all -> STOP.
R3  Terrain height step under the footprint above `ugv.clearance_m` -> that region
    is non-traversable.  If some other arc is feasible -> REROUTE, else STOP.
R4  World-model / sweep collision risk:  >= `risk_stop` -> STOP,
    >= `risk_reroute` -> REROUTE, >= `risk_slow` -> SLOW.
R5  Unknown fraction in the corridor: >= `unknown_frac_stop` -> STOP,
    >= `unknown_frac_slow` -> SLOW.
R6  Confidence: mean confidence < `conf_unknown` -> the corridor is effectively
    UNKNOWN -> REROUTE if another arc is better, else SLOW.  Below `conf_slow`
    -> SLOW.  Never proceed at speed through a low-confidence region.
R7  Stopping space: the free distance ahead must exceed the braking distance at
    the commanded speed.  If not, the speed is reduced to whatever the free
    distance supports; if that falls below a usable crawl -> STOP.
R8  Otherwise GO.

Confidence maps are confidence, not calibrated collision probability, and the
thresholds above are policy choices - they are not derived from a measured
collision rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..config import (CFG, ACTIONS, N_ACTIONS, DECISIONS, GO, SLOW, REROUTE, STOP)
from ..types import Decision, FramePacket, Trajectory
from . import bev_utils as bu
from .planner import Planner, braking_distance

#: crawl speed below which "slow down" is not meaningful and we simply stop
MIN_USEFUL_SPEED = 0.12
SLOW_FACTOR = 0.42


@dataclass
class SupervisorReport:
    """Everything the renderers need to explain one decision."""
    decision: Decision
    best_idx: int = -1
    policy_action: int = -1
    overrode: bool = False
    rule: str = ""
    checks: dict = None


class Supervisor:
    """Safety gate over the policy's chosen action."""

    name = "supervisor"

    def __init__(self, planner: Optional[Planner] = None):
        self.planner = planner or Planner()
        self.last: Optional[SupervisorReport] = None

    def reset(self) -> None:
        self.last = None

    # ------------------------------------------------------------------
    @staticmethod
    def _speed_for(action: int) -> float:
        from ..config import ACTION_CMD
        return float(min(ACTION_CMD[int(action)][0], CFG.ugv.max_speed_mps))

    @staticmethod
    def _cmd_of(traj: Trajectory, dt: float) -> tuple[float, float]:
        """Recover the (v, w) actually used for a candidate arc, including its
        steering-fan offset, from the integrated path."""
        v = float(np.linalg.norm(traj.xy[0])) / dt
        w = float(traj.yaw[0]) / dt
        return v, w

    def _free_ahead(self, state: np.ndarray, traj: Trajectory) -> float:
        """Free distance along the candidate arc, extended to the edge of the BEV map.

        The planner's own `free_distance` is truncated at the 2 s horizon, which
        would make the stopping-space rule blind to anything just beyond it.  This
        re-integrates the same arc out to `CFG.bev.range_forward_m`.
        """
        v, w = self._cmd_of(traj, self.planner.dt)
        if v < 1e-3:
            return 0.0
        n = int(min(80, max(self.planner.n_steps,
                            np.ceil(CFG.bev.range_forward_m / (v * self.planner.dt)))))
        xy, yaw = bu.arc_poses(v, w, self.planner.dt, n)
        return float(bu.sweep(state, xy, yaw, self.planner.margin,
                              bu.height_step_map(state)).free_distance)

    def decide(self,
               state: np.ndarray,
               trajs: Sequence[Trajectory],
               wm_risk: Optional[Sequence[float]] = None,
               policy_action: int = -1,
               tracking_ok: bool = True,
               track_quality: float = 1.0,
               mean_conf_map: Optional[float] = None) -> SupervisorReport:
        """Gate the policy's action against the BEV map and the world model."""
        wmr = np.zeros(N_ACTIONS, np.float32) if wm_risk is None \
            else np.clip(np.asarray(wm_risk, np.float32).ravel()[:N_ACTIONS], 0, 1)

        # candidate the policy wants, and the planner's own best
        pi_idx, pi_traj = (Planner.best_for_action(trajs, policy_action)
                           if 0 <= policy_action < N_ACTIONS else (-1, None))
        pl_idx, pl_traj = Planner.best(trajs)

        chosen_idx, chosen = (pi_idx, pi_traj) if (pi_traj is not None and pi_traj.feasible) \
            else (pl_idx, pl_traj)
        overrode = bool(policy_action >= 0 and chosen is not None and
                        chosen.action != policy_action)

        checks = {}
        d = Decision(policy_source="supervisor" if policy_action < 0 else
                     ("supervisor" if overrode else "rl+supervisor"))

        # ---------------------------------------------------------- R1
        if not tracking_ok:
            d.kind, d.action, d.speed_mps = STOP, ACTIONS.index("STOP"), 0.0
            d.reason = (f"R1 STOP: visual odometry tracking lost "
                        f"(track quality {track_quality:.2f}) - goal pursuit suspended, "
                        f"the local map cannot be trusted to be registered to the vehicle")
            d.risk = 1.0
            return self._finish(d, chosen_idx, policy_action, True, "R1", checks, trajs, state)

        # ---------------------------------------------------------- R2
        if chosen is None:
            worst = min(trajs, key=lambda t: t.cost) if trajs else None
            why = worst.reject_reason if worst is not None else "no candidates generated"
            d.kind, d.action, d.speed_mps = STOP, ACTIONS.index("STOP"), 0.0
            d.reason = f"R2 STOP: no feasible trajectory out of {len(trajs)} candidates - {why}"
            d.risk = 1.0
            d.unknown_frac = float(np.mean([t.unknown_frac for t in trajs])) if trajs else 1.0
            return self._finish(d, -1, policy_action, policy_action >= 0, "R2", checks, trajs, state)

        # ---------------------------------------------------------- R2b
        # The stationary arc is always "feasible"; if it is the *best* one then
        # every arc that actually moves was rejected, and that is a STOP.
        if chosen.action == ACTIONS.index("STOP"):
            movers = [t for t in trajs if t.action != ACTIONS.index("STOP")]
            rejected = [t for t in movers if not t.feasible]
            d.kind, d.action, d.speed_mps = STOP, ACTIONS.index("STOP"), 0.0
            d.risk = float(max([t.collision_risk for t in movers], default=1.0))
            d.unknown_frac = float(np.mean([t.unknown_frac for t in movers])) if movers else 1.0
            if rejected:
                why = min(rejected, key=lambda t: t.cost).reject_reason
                d.reason = (f"R2 STOP: {len(rejected)}/{len(movers)} moving arcs rejected "
                            f"and standing still is the lowest-cost option left - {why}")
            else:
                d.reason = (f"R2 STOP: policy requested STOP and all {len(movers)} moving "
                            f"arcs were feasible - the supervisor restricts motion, it "
                            f"never forces it")
            return self._finish(d, chosen_idx, policy_action,
                                policy_action >= 0 and policy_action != ACTIONS.index("STOP"),
                                "R2", checks, trajs, state)

        # from here on we have a feasible arc that moves
        risk = float(max(chosen.collision_risk, wmr[chosen.action]))
        unk = float(chosen.unknown_frac)
        conf = float(chosen.mean_conf if mean_conf_map is None
                     else min(chosen.mean_conf, mean_conf_map))
        d.risk, d.unknown_frac, d.confidence = risk, unk, conf
        d.action = chosen.action
        d.speed_mps = self._speed_for(chosen.action)
        checks = {"risk": risk, "unknown_frac": unk, "conf": conf,
                  "max_step": chosen.max_step, "cost": chosen.cost}

        # ---------------------------------------------------------- R3
        blocked = [t for t in trajs if not t.feasible and
                   t.max_step > CFG.ugv.clearance_m]
        if chosen.max_step > CFG.ugv.clearance_m:
            d.kind, d.action, d.speed_mps = STOP, ACTIONS.index("STOP"), 0.0
            d.reason = (f"R3 STOP: height step {chosen.max_step:.3f} m > clearance "
                        f"{CFG.ugv.clearance_m:.3f} m under the footprint")
            return self._finish(d, chosen_idx, policy_action, True, "R3", checks, trajs, state)
        if blocked and chosen.action in (ACTIONS.index("LEFT"), ACTIONS.index("RIGHT"),
                                         ACTIONS.index("REROUTE")):
            step = max(t.max_step for t in blocked)
            d.kind = REROUTE
            d.reason = (f"R3 REROUTE: {len(blocked)} arcs blocked by a "
                        f"{step:.3f} m step > clearance {CFG.ugv.clearance_m:.3f} m; "
                        f"steering to {ACTIONS[chosen.action]}")
            d.speed_mps = min(d.speed_mps, CFG.ugv.max_speed_mps * 0.6)
            return self._finish(d, chosen_idx, policy_action, overrode, "R3", checks, trajs, state)

        # ---------------------------------------------------------- R4
        if risk >= CFG.safety.risk_stop:
            d.kind, d.action, d.speed_mps = STOP, ACTIONS.index("STOP"), 0.0
            d.reason = (f"R4 STOP: predicted collision risk {risk:.2f} >= "
                        f"risk_stop {CFG.safety.risk_stop:.2f} on the best arc "
                        f"({ACTIONS[chosen.action]})")
            return self._finish(d, chosen_idx, policy_action, True, "R4", checks, trajs, state)
        if risk >= CFG.safety.risk_reroute:
            alt = self._best_low_risk(trajs, wmr, CFG.safety.risk_reroute)
            if alt is not None:
                chosen_idx, chosen = alt
                d.action = chosen.action
                d.speed_mps = min(self._speed_for(chosen.action), CFG.ugv.max_speed_mps * 0.7)
                overrode = bool(policy_action >= 0 and chosen.action != policy_action)
            d.kind = REROUTE
            d.reason = (f"R4 REROUTE: predicted collision risk {risk:.2f} >= "
                        f"risk_reroute {CFG.safety.risk_reroute:.2f}; "
                        f"steering to {ACTIONS[chosen.action]} "
                        f"(risk {chosen.collision_risk:.2f})")
            return self._finish(d, chosen_idx, policy_action, overrode, "R4", checks, trajs, state)
        if risk >= CFG.safety.risk_slow:
            d.kind = SLOW
            d.speed_mps *= SLOW_FACTOR
            d.reason = (f"R4 SLOW: predicted collision risk {risk:.2f} >= "
                        f"risk_slow {CFG.safety.risk_slow:.2f} - speed capped to "
                        f"{d.speed_mps:.2f} m/s")
            return self._finish(d, chosen_idx, policy_action, overrode, "R4", checks, trajs, state)

        # ---------------------------------------------------------- R5
        if unk >= CFG.safety.unknown_frac_stop:
            d.kind, d.action, d.speed_mps = STOP, ACTIONS.index("STOP"), 0.0
            d.reason = (f"R5 STOP: {unk*100:.0f}% of the corridor is unknown/stale >= "
                        f"{CFG.safety.unknown_frac_stop*100:.0f}% limit")
            return self._finish(d, chosen_idx, policy_action, True, "R5", checks, trajs, state)
        if unk >= CFG.safety.unknown_frac_slow:
            d.kind = SLOW
            d.speed_mps *= SLOW_FACTOR
            d.reason = (f"R5 SLOW: {unk*100:.0f}% of the corridor is unknown/stale >= "
                        f"{CFG.safety.unknown_frac_slow*100:.0f}% - speed capped to "
                        f"{d.speed_mps:.2f} m/s")
            return self._finish(d, chosen_idx, policy_action, overrode, "R5", checks, trajs, state)

        # ---------------------------------------------------------- R6
        if conf < CFG.safety.conf_unknown:
            alt = self._best_conf(trajs, CFG.safety.conf_slow)
            if alt is not None and alt[1].action != chosen.action:
                chosen_idx, chosen = alt
                d.action = chosen.action
                overrode = bool(policy_action >= 0 and chosen.action != policy_action)
                d.kind = REROUTE
                d.speed_mps = min(self._speed_for(chosen.action) * SLOW_FACTOR,
                                  CFG.ugv.max_speed_mps * 0.5)
                d.reason = (f"R6 REROUTE: mean confidence {conf:.2f} < conf_unknown "
                            f"{CFG.safety.conf_unknown:.2f} - that region is treated as "
                            f"UNKNOWN; steering to {ACTIONS[chosen.action]} "
                            f"(conf {chosen.mean_conf:.2f})")
            else:
                d.kind = SLOW
                d.speed_mps = min(d.speed_mps * SLOW_FACTOR, 0.30)
                d.reason = (f"R6 SLOW: mean confidence {conf:.2f} < conf_unknown "
                            f"{CFG.safety.conf_unknown:.2f} - corridor treated as UNKNOWN, "
                            f"crawling at {d.speed_mps:.2f} m/s")
            return self._finish(d, chosen_idx, policy_action, overrode, "R6", checks, trajs, state)
        if conf < CFG.safety.conf_slow:
            d.kind = SLOW
            d.speed_mps *= SLOW_FACTOR
            d.reason = (f"R6 SLOW: mean confidence {conf:.2f} < conf_slow "
                        f"{CFG.safety.conf_slow:.2f} - speed capped to {d.speed_mps:.2f} m/s")
            return self._finish(d, chosen_idx, policy_action, overrode, "R6", checks, trajs, state)

        # ---------------------------------------------------------- R7
        free = self._free_ahead(state, chosen)
        need = braking_distance(d.speed_mps)
        checks["free_distance"] = free
        checks["brake_distance"] = need
        if need > free:
            v_ok = CFG.ugv.max_speed_mps * float(np.sqrt(
                max(free, 0.0) / max(CFG.ugv.brake_distance_m, 1e-6)))
            if v_ok < MIN_USEFUL_SPEED:
                d.kind, d.action, d.speed_mps = STOP, ACTIONS.index("STOP"), 0.0
                d.reason = (f"R7 STOP: only {free:.2f} m of free space ahead, less than the "
                            f"{braking_distance(MIN_USEFUL_SPEED):.2f} m needed to stop even "
                            f"at a {MIN_USEFUL_SPEED:.2f} m/s crawl")
            else:
                d.kind = SLOW
                d.speed_mps = float(min(d.speed_mps, v_ok))
                d.reason = (f"R7 SLOW: free distance {free:.2f} m < braking distance "
                            f"{need:.2f} m - speed reduced to {d.speed_mps:.2f} m/s "
                            f"(stopping space now {braking_distance(d.speed_mps):.2f} m)")
            return self._finish(d, chosen_idx, policy_action, overrode or d.kind == STOP,
                                "R7", checks, trajs, state)

        # ---------------------------------------------------------- R8
        # the decision kind follows the action family: a deliberate SLOW or an
        # evasive REROUTE arc is not a plain GO even when every gate passes.
        if chosen.action == ACTIONS.index("SLOW"):
            d.kind = SLOW
        elif chosen.action == ACTIONS.index("REROUTE"):
            d.kind = REROUTE
        else:
            d.kind = GO
        d.reason = (f"R8 {DECISIONS[d.kind]}: {ACTIONS[chosen.action]} at "
                    f"{d.speed_mps:.2f} m/s - all gates clear (risk {risk:.2f} < "
                    f"{CFG.safety.risk_slow:.2f}, unknown {unk*100:.0f}% < "
                    f"{CFG.safety.unknown_frac_slow*100:.0f}%, conf {conf:.2f} >= "
                    f"{CFG.safety.conf_slow:.2f}, {free:.2f} m free vs "
                    f"{need:.2f} m braking)")
        return self._finish(d, chosen_idx, policy_action, overrode, "R8", checks, trajs, state)

    # ------------------------------------------------------------------
    @staticmethod
    def _best_low_risk(trajs, wmr, thresh):
        cand = [(i, t) for i, t in enumerate(trajs)
                if t.feasible and max(t.collision_risk, wmr[t.action]) < thresh]
        if not cand:
            return None
        return min(cand, key=lambda it: it[1].cost)

    @staticmethod
    def _best_conf(trajs, thresh):
        cand = [(i, t) for i, t in enumerate(trajs) if t.feasible and t.mean_conf >= thresh]
        if not cand:
            return None
        return min(cand, key=lambda it: it[1].cost)

    def _finish(self, d, idx, policy_action, overrode, rule, checks, trajs, state):
        if policy_action < 0:
            d.policy_source = "supervisor"
        elif overrode or d.action != policy_action:
            d.policy_source = "supervisor"
        elif d.kind == GO:
            d.policy_source = "rl"
        else:
            d.policy_source = "rl+supervisor"
        d.speed_mps = float(np.clip(d.speed_mps, 0.0, CFG.ugv.max_speed_mps))
        rep = SupervisorReport(decision=d, best_idx=int(idx),
                               policy_action=int(policy_action),
                               overrode=bool(policy_action >= 0 and d.action != policy_action),
                               rule=rule, checks=checks or {})
        self.last = rep
        return rep

    # ------------------------------------------------------------------
    def __call__(self, packet: FramePacket) -> FramePacket:
        if packet.bev is None or not packet.trajectories:
            return packet
        st = bu.pack_state(packet.bev.height, packet.bev.trav_prob, packet.bev.conf,
                           packet.bev.age, getattr(packet.bev, "hits", None))
        wmr = [float(p.risk_total) for p in packet.wm_preds] if packet.wm_preds else None
        ok = packet.odom.tracking_ok if packet.odom is not None else True
        tq = packet.odom.track_quality if packet.odom is not None else 1.0
        mc = packet.unc.mean_conf if packet.unc is not None else None
        pa = getattr(packet, "_policy_action", -1)
        rep = self.decide(st, packet.trajectories, wmr, pa, ok, tq, mc)
        packet.decision = rep.decision
        return packet


if __name__ == "__main__":
    pl = Planner()
    sup = Supervisor(pl)
    print("supervisor rule check on synthetic scenes "
          f"(clearance {CFG.ugv.clearance_m:.3f} m, risk_stop {CFG.safety.risk_stop:.2f}, "
          f"conf_unknown {CFG.safety.conf_unknown:.2f})\n")
    cases = [
        ("clear corridor",                 dict(kind="clear"), True, None),
        ("10 cm wall 1.0 m ahead",         dict(kind="wall", wall_dist_m=1.0,
                                                wall_height_m=0.10), True, None),
        ("10 cm wall 2.5 m ahead",         dict(kind="wall", wall_dist_m=2.5,
                                                wall_height_m=0.10), True, None),
        ("wall only on the left, 1.2 m",   dict(kind="wall_left", wall_dist_m=1.2,
                                                wall_height_m=0.12), True, None),
        ("5.5 cm kerb 1.5 m ahead",        dict(kind="kerb", wall_dist_m=1.5), True, None),
        ("far field low confidence",       dict(kind="low_conf"), True, None),
        ("blind (everything unknown)",     dict(kind="blind"), True, None),
        ("clear corridor, VO LOST",        dict(kind="clear"), False, None),
        ("clear corridor, policy=STOP",    dict(kind="clear"), True, 5),
        ("wall 1.0 m, policy=FORWARD",     dict(kind="wall", wall_dist_m=1.0,
                                                wall_height_m=0.10), True, 0),
    ]
    ok_all = True
    for name, kw, track, pa in cases:
        st = bu.synthetic_state(**kw)
        trajs = pl.plan(st)
        rep = sup.decide(st, trajs, None, policy_action=-1 if pa is None else pa,
                         tracking_ok=track)
        d = rep.decision
        ov = "  <OVERRIDE>" if rep.overrode else ""
        print(f"{name:32s} -> {DECISIONS[d.kind]:7s} {ACTIONS[d.action]:8s} "
              f"{d.speed_mps:4.2f} m/s  [{d.policy_source}]{ov}")
        print(f"{'':32s}    {d.reason}")
    # assertions the demo depends on
    st = bu.synthetic_state("wall", wall_dist_m=1.0, wall_height_m=0.10)
    r = sup.decide(st, pl.plan(st), None, policy_action=0)
    assert r.decision.kind == STOP, f"expected STOP for a wall 1 m ahead, got {DECISIONS[r.decision.kind]}"
    assert r.overrode, "supervisor must override a FORWARD policy into the wall"
    st = bu.synthetic_state("clear")
    r = sup.decide(st, pl.plan(st), None, policy_action=0)
    assert r.decision.kind == GO, f"expected GO on a clear corridor, got {DECISIONS[r.decision.kind]}"
    st = bu.synthetic_state("clear")
    r = sup.decide(st, pl.plan(st), None, policy_action=0, tracking_ok=False)
    assert r.decision.kind == STOP and r.rule == "R1"
    print("\nassertions passed: wall 1 m ahead -> STOP + override, clear -> GO, VO lost -> STOP")
