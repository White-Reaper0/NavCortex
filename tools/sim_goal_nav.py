"""Closed-loop Point A -> Point B check in a 2-D kinematic world with a known map.

What this is
------------
A fast, dependency-light stand-in for the Gazebo Harmonic runs in `ros2/`: the
*real* prism runtime (`prism/runtime.py`: `GoalPlanner` + D* Lite, the
17-arc `Planner`, the `Supervisor` safety gate and the `DynamicObstacleLayer`)
drives a unicycle through a ground-truth occupancy map, and every episode is
scored on the same three numbers the presentation names: **success rate,
collisions and interventions**.

What this is not
----------------
There is no camera, no depth network and no terrain model in the loop.  The
2.5D world state is rendered *from the ground-truth map* through a 92 deg,
6.5 m view cone with occlusion and the same 1.5 s observation memory as the
mapping stage.  It tests decision-making and replanning, not perception, and
its numbers must never be quoted as perception accuracy or as field results.

Scenarios
---------
open        empty field, B 8 m ahead                     (arcs head straight for B)
wall        a 3 m wall across the straight line          (detour)
dead_end    a U-shaped pocket opening towards the start  (D* Lite must route round it)
pedestrian  open field, a person walks across the path   (dynamic cells, expiry, replan)
slalom      three staggered walls                        (repeated replanning)
trap        a long corridor towards B with a closed far end, out of view range
            until the vehicle is committed: it must turn round and go outside

Usage
-----
    python tools/sim_goal_nav.py                 # all scenarios, 3 seeds each
    python tools/sim_goal_nav.py --scenario dead_end --seeds 1 --verbose
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prism.config import CFG, DECISIONS, N_TRAV, SAFE, OBSTACLE              # noqa: E402
from prism.nav import bev_utils as bu                                         # noqa: E402
from prism.perception.geometry import bev_to_veh                              # noqa: E402
from prism.runtime import PrismNavigator                                    # noqa: E402
from prism.types import BEVMap, OdometryResult                                # noqa: E402

RES = 0.05                    # ground-truth map resolution, metres
SIZE = 32.0                   # ground-truth map side, metres (centred on the start)
DT = 0.1                      # control period, seconds (10 Hz)
VIEW_RANGE = 6.5
OBST_H = 0.20                 # every wall is 20 cm tall: far above the 4.5 cm clearance


# ---------------------------------------------------------------------- world
@dataclass
class Walker:
    x0: float
    y0: float
    vx: float
    vy: float
    t_start: float
    radius: float = 0.18

    def pos(self, t: float):
        dt = max(0.0, t - self.t_start)
        return self.x0 + self.vx * dt, self.y0 + self.vy * dt


@dataclass
class World:
    occ: np.ndarray                         # (n, n) bool static obstacles
    goal: tuple[float, float]
    walkers: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.occ.shape[0]

    def cell(self, x, y):
        half = SIZE * 0.5
        c = np.floor((np.asarray(x) + half) / RES).astype(np.int64)
        r = np.floor((np.asarray(y) + half) / RES).astype(np.int64)
        return r, c

    def occupied(self, x, y, t: float = 0.0) -> np.ndarray:
        r, c = self.cell(x, y)
        ins = (r >= 0) & (r < self.n) & (c >= 0) & (c < self.n)
        out = np.ones(np.shape(r), bool)              # off-map counts as blocked
        out[ins] = self.occ[r[ins], c[ins]]
        return out

    def walker_hit(self, x, y, t: float) -> np.ndarray:
        hit = np.zeros(np.shape(x), bool)
        for w in self.walkers:
            wx, wy = w.pos(t)
            hit |= np.hypot(np.asarray(x) - wx, np.asarray(y) - wy) < w.radius
        return hit


def _wall(occ, x0, y0, x1, y1, th=0.10):
    half = SIZE * 0.5
    n = occ.shape[0]
    L = math.hypot(x1 - x0, y1 - y0)
    for s in np.linspace(0, 1, max(2, int(L / (RES * 0.5)))):
        x, y = x0 + s * (x1 - x0), y0 + s * (y1 - y0)
        r0 = int((y - th / 2 + half) / RES); r1 = int((y + th / 2 + half) / RES) + 1
        c0 = int((x - th / 2 + half) / RES); c1 = int((x + th / 2 + half) / RES) + 1
        occ[max(r0, 0):min(r1, n), max(c0, 0):min(c1, n)] = True


def make_world(name: str, rng: np.random.Generator) -> World:
    n = int(SIZE / RES)
    occ = np.zeros((n, n), bool)
    j = float(rng.uniform(-0.3, 0.3))                 # small per-seed variation
    goal = (0.0 + j, 8.0)
    walkers = []
    if name == "open":
        pass
    elif name == "wall":
        _wall(occ, -1.5 + j, 4.0, 1.5 + j, 4.0)
    elif name == "dead_end":
        # U-pocket 2.4 m wide, 2.5 m deep, open towards the start, B behind it
        _wall(occ, -1.2 + j, 5.5, 1.2 + j, 5.5)
        _wall(occ, -1.2 + j, 3.0, -1.2 + j, 5.5)
        _wall(occ, 1.2 + j, 3.0, 1.2 + j, 5.5)
    elif name == "pedestrian":
        # crosses the straight line to B about 1 m in front of the vehicle
        walkers.append(Walker(x0=-2.5, y0=4.3 + j, vx=0.9, vy=0.0, t_start=0.8))
    elif name == "trap":
        # a 1.8 m corridor towards B whose far end is closed; the closed end is out of
        # view range until the vehicle is well inside, so it has to turn round
        goal = (0.0 + j, 13.0)
        _wall(occ, -0.9 + j, 1.0, -0.9 + j, 11.0)
        _wall(occ, 0.9 + j, 1.0, 0.9 + j, 11.0)
        _wall(occ, -0.9 + j, 11.0, 0.9 + j, 11.0)
    elif name == "slalom":
        _wall(occ, -2.0, 2.5, 0.4 + j, 2.5)
        _wall(occ, -0.4 + j, 4.5, 2.0, 4.5)
        _wall(occ, -2.0, 6.5, 0.4 + j, 6.5)
    else:
        raise ValueError(name)
    # keep a border so "off the map" is visible as a wall, not a hole
    occ[:2, :] = occ[-2:, :] = occ[:, :2] = occ[:, -2:] = True
    return World(occ=occ, goal=goal, walkers=walkers)


# ---------------------------------------------------------------------- sensing
class ConeSensor:
    """Ground-truth -> (8, H, W) world state through a view cone, with memory."""

    def __init__(self):
        b = CFG.bev
        rr, cc = np.meshgrid(np.arange(b.H, dtype=np.float32),
                             np.arange(b.W, dtype=np.float32), indexing="ij")
        self.bx, self.by = bev_to_veh(rr, cc)
        fov = math.radians(CFG.cam.hfov_deg * 0.5)
        self.cone = (self.by > 0.05) & (np.abs(self.bx) < np.tan(fov) * self.by + 0.15) & \
                    (np.hypot(self.bx, self.by) < VIEW_RANGE)
        self.ang = np.arctan2(self.bx, self.by)
        self.rng = np.hypot(self.bx, self.by)
        self.max_age_s = CFG.bev.max_age_frames / 30.0
        self.mem_t: dict[tuple[int, int], float] = {}
        self.seen_t = None                 # world-cell -> last seen time grid

    def reset(self, world: World):
        self.seen_t = np.full(world.occ.shape, -np.inf)

    def observe(self, world: World, pose, t: float):
        """Returns (BEVMap, dyn_points_veh) for this pose."""
        px, py, th = pose
        c, s = math.cos(th), math.sin(th)
        wx = px + c * self.bx - s * self.by
        wy = py + s * self.bx + c * self.by
        occ = world.occupied(wx, wy)
        # occlusion: per 1-degree bearing bin, nothing beyond the nearest obstacle
        bins = np.clip(((self.ang + math.pi) / math.radians(1.0)).astype(np.int64), 0, 359)
        near = np.full(360, np.inf)
        hit = self.cone & occ
        np.minimum.at(near, bins[hit], self.rng[hit])
        visible = self.cone & (self.rng <= near[bins] + 0.08)
        r, cidx = world.cell(wx, wy)
        ins = (r >= 0) & (r < world.n) & (cidx >= 0) & (cidx < world.n)
        vis_ins = visible & ins
        self.seen_t[r[vis_ins], cidx[vis_ins]] = t
        # what the map remembers: anything seen within the mapping memory window
        last = np.full(self.bx.shape, -np.inf)
        last[ins] = self.seen_t[r[ins], cidx[ins]]
        age_s = t - last
        observed = age_s <= self.max_age_s

        H, W = self.bx.shape
        height = np.full((H, W), np.nan, np.float32)
        tp = np.zeros((N_TRAV, H, W), np.float32)
        conf = np.zeros((H, W), np.float32)
        age = np.full((H, W), 1e4, np.float32)
        hits = np.zeros((H, W), np.float32)
        free = observed & ~occ
        ob = observed & occ
        height[free] = 0.0
        height[ob] = OBST_H
        tp[SAFE][free] = 0.92
        tp[1][free] = 0.06
        tp[3][free] = 0.02
        tp[OBSTACLE][ob] = 0.95
        tp[1][ob] = 0.05
        conf[observed] = np.clip(0.9 - 0.04 * self.rng[observed], 0.5, 0.9)
        age[observed] = (age_s[observed] * 30.0).astype(np.float32)
        hits[observed] = 4.0
        trav = np.argmax(tp, 0).astype(np.uint8)
        bev = BEVMap(height=height, trav_prob=tp, trav=trav, conf=conf, age=age, hits=hits,
                     terrain=np.zeros((H, W), np.uint8))

        # moving obstacles are not in the static map: they reach the planner only
        # through the dynamic layer, as dynamic-class pixels with depth would
        dyn = []
        for w in world.walkers:
            wxp, wyp = w.pos(t)
            dx, dy = wxp - px, wyp - py
            xv, yv = c * dx + s * dy, -s * dx + c * dy
            rng_ = math.hypot(xv, yv)
            if yv > 0.05 and abs(xv) < math.tan(math.radians(CFG.cam.hfov_deg / 2)) * yv \
                    and rng_ < VIEW_RANGE:
                b = int(np.clip((math.atan2(xv, yv) + math.pi) / math.radians(1.0), 0, 359))
                if rng_ <= near[b] + 0.08:
                    a = np.linspace(0, 2 * math.pi, 40, endpoint=False)
                    pts = np.stack([xv + w.radius * np.cos(a), yv + w.radius * np.sin(a),
                                    np.full_like(a, 0.8)], 1)
                    dyn.append(pts)
        dyn_pts = np.concatenate(dyn, 0).astype(np.float32) if dyn else None
        return bev, dyn_pts


# ---------------------------------------------------------------------- episode
def footprint_pts(pose, n=5):
    px, py, th = pose
    hw, hl = CFG.ugv.width_m / 2, CFG.ugv.length_m / 2
    gx, gy = np.meshgrid(np.linspace(-hw, hw, n), np.linspace(-hl, hl, n))
    c, s = math.cos(th), math.sin(th)
    return px + c * gx - s * gy, py + s * gx + c * gy


def run_episode(name: str, seed: int, timeout_s: float = 90.0, stuck_s: float = 6.0,
                verbose: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    world = make_world(name, rng)
    sensor = ConeSensor()
    sensor.reset(world)

    true_pose = [0.0, 0.0, 0.0]
    motion = {"d": 0.0, "dy": 0.0}

    def perception(packet):
        """Stand-in for the camera stack: ground truth through the view cone."""
        bev, dyn_pts = sensor.observe(world, true_pose, packet.t)
        packet.bev = bev
        packet.odom = OdometryResult(d_trans=motion["d"], d_yaw=motion["dy"],
                                     tracking_ok=True, track_quality=1.0)
        packet.dynamic_points_veh = dyn_pts
        return packet

    nav = PrismNavigator(perception=perception, use_loop_closure=False)
    nav.set_goal(*world.goal)

    t = 0.0
    collisions, in_contact, interventions = 0, False, 0
    replans, dstar_steps, max_plan_ms, max_cycle_ms = 0, 0, 0.0, 0.0
    cycle_ms: list = []
    anchor, last_progress_t = (0.0, 0.0), 0.0
    decisions = {k: 0 for k in DECISIONS}
    rules: dict[str, int] = {}
    path_len = 0.0
    outcome = "timeout"

    while t < timeout_s:
        out = nav.step(None, t)
        gs = out.goal
        max_plan_ms = max(max_plan_ms, gs.plan_ms)
        max_cycle_ms = max(max_cycle_ms, out.timings_ms.get("total", 0.0))
        cycle_ms.append(out.timings_ms.get("total", 0.0))
        replans += int(gs.replanned)
        dstar_steps += int(gs.mode == "dstar")
        decisions[out.kind] += 1
        rules[out.rule] = rules.get(out.rule, 0) + 1
        if gs.mode == "arrived":
            outcome = "success"
            break
        v, w = out.v_mps, out.w_radps
        if verbose and int(round(t / DT)) % 10 == 0:
            print(f"  t={t:5.1f} pose=({true_pose[0]:+.2f},{true_pose[1]:+.2f},"
                  f"{math.degrees(true_pose[2]):+4.0f}) {gs.mode:8s} dist={gs.dist_m:4.1f} "
                  f"{out.kind:7s} v={v:.2f} w={w:+.2f}  {out.decision.reason[:70]}")

        xy, yaw = bu.arc_poses(v, w, DT, 1, *true_pose)
        nx, ny, nth = float(xy[0, 0]), float(xy[0, 1]), float(yaw[0])
        step = math.hypot(nx - true_pose[0], ny - true_pose[1])
        motion["d"], motion["dy"] = step, nth - true_pose[2]   # chord + heading change
        path_len += step
        true_pose[:] = [nx, ny, nth]
        t += DT

        fx, fy = footprint_pts(true_pose)
        hit = bool(world.occupied(fx, fy).any() or world.walker_hit(fx, fy, t).any())
        if hit and not in_contact:
            collisions += 1
        in_contact = hit

        # stuck = has not moved 0.25 m in `stuck_s` seconds.  (Not "has not got closer
        # to B": the right route out of a dead end leads away from B first.)
        if math.hypot(true_pose[0] - anchor[0], true_pose[1] - anchor[1]) > 0.25:
            anchor, last_progress_t = (true_pose[0], true_pose[1]), t
        if t - last_progress_t > stuck_s:
            # an operator would have to step in: count it and end the episode
            interventions += 1
            outcome = "stuck"
            break

    final = math.hypot(world.goal[0] - true_pose[0], world.goal[1] - true_pose[1])
    drift = math.hypot(nav.pose[0] - true_pose[0], nav.pose[1] - true_pose[1])
    return dict(scenario=name, seed=seed, outcome=outcome, success=outcome == "success",
                time_s=round(t, 1), path_m=round(path_len, 2), final_dist_m=round(final, 2),
                collisions=collisions, interventions=interventions,
                dstar_cycles=dstar_steps, replans=replans,
                max_goal_plan_ms=round(max_plan_ms, 1), max_cycle_ms=round(max_cycle_ms, 1),
                median_cycle_ms=round(float(np.median(cycle_ms)), 1) if cycle_ms else 0.0,
                pose_error_m=round(drift, 3), decisions=decisions, rules=rules)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenario", nargs="*", default=["open", "wall", "dead_end",
                                                      "pedestrian", "slalom", "trap"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "logs" / "sim_goal_nav.json"))
    a = ap.parse_args(argv)

    t0 = time.perf_counter()
    eps = []
    for name in a.scenario:
        for seed in range(a.seeds):
            r = run_episode(name, seed, verbose=a.verbose)
            eps.append(r)
            print(f"{name:11s} seed {seed}: {r['outcome']:8s} {r['time_s']:5.1f} s "
                  f"path {r['path_m']:5.2f} m  collisions {r['collisions']}  "
                  f"interventions {r['interventions']}  D* cycles {r['dstar_cycles']:3d} "
                  f"(replans {r['replans']})  cycle med/max {r['median_cycle_ms']:.0f}/{r['max_cycle_ms']:.0f} ms  "
                  f"pose err {r['pose_error_m']:.3f} m")
    n = len(eps)
    summary = dict(
        what=("2-D kinematic closed loop over a ground-truth map rendered through a "
              "92 deg / 6.5 m view cone. Real prism goal layer, D* Lite, 17-arc "
              "planner, supervisor and dynamic layer; NO camera / depth / terrain "
              "network in the loop. Tests decisions, not perception."),
        episodes=n,
        success_rate=round(sum(e["success"] for e in eps) / max(n, 1), 3),
        collisions=int(sum(e["collisions"] for e in eps)),
        interventions=int(sum(e["interventions"] for e in eps)),
        wall_s=round(time.perf_counter() - t0, 1),
        per_episode=eps,
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=1))
    print(f"\n{n} episodes: success {summary['success_rate']*100:.0f}%  "
          f"collisions {summary['collisions']}  interventions {summary['interventions']}  "
          f"-> {a.out}")
    return summary


if __name__ == "__main__":
    main()
