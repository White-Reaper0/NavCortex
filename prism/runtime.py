"""prism end to end in one module: camera frame in, velocity command out.

    nav = PrismNavigator(device="cuda")           # builds every stage
    nav.set_goal(0.0, 8.0)                          # Point B, metres, odometry frame
    out = nav.step(bgr_frame, t)                    # one control cycle
    send(out.v_mps, out.w_radps)                    # GO / SLOW / REROUTE / STOP

This is the module the ROS 2 node (`ros2/prism_ros`) wraps for Gazebo Harmonic
and for the rover.  The offline pipeline in `pipeline.py` / `training/run_*`
computes the same stages clip by clip into caches so the demo videos can be
rendered; this runs them *live*, frame by frame, in the order below.

    1  terrain     PIDNet-S, 7 classes            (models/segmentation.py)
    2  depth       Depth Anything V2 + ground fit (models/depth.py)
                   unreliable fit -> region invalid -> UNKNOWN (perception/fit_validity.py)
    3  trav + trust 17-channel MobileNetV3 head, 8 MC-dropout passes
    4  odometry    ORB + MAGSAC, depth-anchored scale (perception/odometry.py)
    4b IMU         rover only: scale + heading drift (perception/imu_fusion.py)
    4c loops       rover only: VPR + PnP check + pose graph (perception/loop_closure.py)
    5  2.5D map    6 cm cells: height, risk, confidence, age (perception/mapping.py)
    5b movers      dynamic cells, inflated, expire < 1 s (perception/dynamic_layer.py)
    6  goal        arcs head for B; D* Lite when boxed in (nav/goal_planner.py)
    7  arcs        17 candidate arcs scored (nav/planner.py)
    8  supervisor  ordered safety gate, rule + reason on every command (nav/supervisor.py)

Learned components only *suggest*: the PPO policy is off by default
(`use_policy=False`), and even when on, the supervisor decides.

Perception can be injected (`perception=callable`) - the closed-loop simulator and
the unit tests drive the navigation half with a stand-in that fills `packet.bev`
and `packet.odom`, without any network weights.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from .config import PROC_W, PROC_H, ACTIONS, DECISIONS, STOP
from .types import Decision, FramePacket, OdometryResult
from .nav import bev_utils as bu
from .nav.goal_planner import GoalPlanner, GoalStatus
from .nav.planner import Planner
from .nav.supervisor import Supervisor, SupervisorReport
from .perception.dynamic_layer import DynamicObstacleLayer
from .perception.imu_fusion import ImuFusion


@dataclass
class NavOutput:
    v_mps: float
    w_radps: float
    decision: Decision
    rule: str
    goal: GoalStatus
    pose: tuple[float, float, float]           # fused world pose (x, y, yaw)
    packet: Optional[FramePacket] = None
    timings_ms: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return DECISIONS[self.decision.kind]

    def as_dict(self) -> dict:
        return dict(v=self.v_mps, w=self.w_radps, decision=self.kind,
                    action=ACTIONS[self.decision.action], rule=self.rule,
                    reason=self.decision.reason, risk=self.decision.risk,
                    confidence=self.decision.confidence,
                    unknown_frac=self.decision.unknown_frac,
                    goal_mode=self.goal.mode, goal_dist_m=self.goal.dist_m,
                    goal_reason=self.goal.reason, pose=list(self.pose),
                    timings_ms=self.timings_ms)


class PrismNavigator:
    def __init__(self, device: str = "cuda", vehicle: Optional[str] = None,
                 perception: Optional[Callable[[FramePacket], FramePacket]] = None,
                 use_world_model: bool = False, use_policy: bool = False,
                 use_loop_closure: bool = True, use_vpr: bool = True):
        from .vehicles import apply_vehicle_profile
        self.vehicle = apply_vehicle_profile(vehicle)       # before any stage is built
        self.device = device
        self._perception = perception
        self.stages: dict = {}
        if perception is None:
            self._build_perception(use_vpr)
        self.wm = self.policy = None
        if use_world_model or use_policy:
            from .models.world_model import WorldModelStage
            self.wm = WorldModelStage(device=device)
        if use_policy:
            from .nav.rl_env import PolicyStage
            self.policy = PolicyStage(device="cpu")
        self.planner = Planner()
        self.supervisor = Supervisor(self.planner)
        self.goal = GoalPlanner()
        self.dynamic = DynamicObstacleLayer()
        self.imu = ImuFusion()
        self.loops = None
        if use_loop_closure:
            from .perception.loop_closure import LoopClosureManager
            self.loops = LoopClosureManager()
        self.reset()

    # ------------------------------------------------------------------ build
    def _build_perception(self, use_vpr: bool) -> None:
        from .models.segmentation import SegStage
        from .models.depth import DepthStage
        from .models.traversability import TraversabilityStage
        from .models.uncertainty import UncertaintyStage
        from .perception.odometry import OdometryStage
        from .perception.mapping import MappingStage
        d = self.device
        self.stages["seg"] = SegStage(device=d)
        self.stages["depth"] = DepthStage(device=d)
        self.stages["trav"] = TraversabilityStage(device=d)
        self.stages["unc"] = UncertaintyStage(device=d, trav_stage=self.stages["trav"])
        self.stages["odom"] = OdometryStage(device=d, use_depth_cache=False)
        if use_vpr:
            from .models.vpr import VPRStage
            self.stages["vpr"] = VPRStage(device=d)
        self.stages["bev"] = MappingStage(device=d)

    def _run_perception(self, p: FramePacket) -> FramePacket:
        s = self.stages
        p = s["seg"](p)
        p = s["depth"](p)
        p = s["trav"](p)
        p = s["unc"](p)
        p = s["odom"](p)
        self._fuse_imu(p)
        if "vpr" in s:
            p = s["vpr"](p)
        p = s["bev"](p)
        return p

    # ------------------------------------------------------------------ control
    def reset(self) -> None:
        for st in self.stages.values():
            if hasattr(st, "reset"):
                st.reset()
        for st in (self.wm, self.policy, self.supervisor, self.dynamic, self.imu):
            if st is not None and hasattr(st, "reset"):
                st.reset()
        if self.loops is not None:
            self.loops.reset()
        self.goal.reset()
        self.idx = 0
        self.pose = np.zeros(3)
        self._t_prev: Optional[float] = None
        self._last_motion = (0.0, 0.0)
        self.last: Optional[NavOutput] = None

    def set_goal(self, x: float, y: float) -> None:
        """Point B in the odometry frame: x right, y forward of the start pose, metres."""
        self.goal.set_goal(x, y)

    def add_imu(self, t: float, gyro, accel) -> None:
        """Rover IMU sample, already in the prism vehicle frame (x right, y fwd, z up)."""
        self.imu.add_imu(t, gyro, accel)

    def _fuse_imu(self, p: FramePacket) -> None:
        o = p.odom
        if o is None or not self.imu.has_imu:
            return
        dt = 1.0 / 30.0 if self._t_prev is None else max(p.t - self._t_prev, 1e-3)
        stopped = None if self.last is None else bool(self.last.v_mps == 0.0 and
                                                       self.last.w_radps == 0.0)
        m = self.imu.fuse(p.t, o.d_trans, o.d_yaw, dt, o.tracking_ok, commanded_stop=stopped)
        o.d_trans, o.d_yaw = float(m.d_trans), float(m.d_yaw)

    def _integrate_pose(self, odom: Optional[OdometryResult]) -> None:
        if odom is None:
            return
        d, dy = float(odom.d_trans or 0.0), float(odom.d_yaw or 0.0)
        if not (math.isfinite(d) and math.isfinite(dy)):
            return
        th = self.pose[2] + 0.5 * dy                         # midpoint heading, as mapping.py
        self.pose[0] += -math.sin(th) * d
        self.pose[1] += math.cos(th) * d
        self.pose[2] = (self.pose[2] + dy + math.pi) % (2 * math.pi) - math.pi
        self._last_motion = (d, dy)

    def _loop_closure(self, p: FramePacket) -> None:
        if self.loops is None:
            return
        gray = depth = None
        if p.rgb is not None and p.depth is not None:
            gray = cv2.cvtColor(p.rgb, cv2.COLOR_BGR2GRAY)
            depth = np.where(p.depth.valid, p.depth.depth_m, np.nan) \
                if p.depth.valid is not None else p.depth.depth_m
        self.loops.maybe_keyframe(p.idx, self.pose, gray, depth)
        pl = p.place
        vpr = self.stages.get("vpr")
        if pl is not None and pl.is_revisit and pl.best_match_idx >= 0 and vpr is not None:
            frames = getattr(vpr, "_db_frame", [])
            if 0 <= pl.best_match_idx < len(frames):
                res = self.loops.on_revisit(int(frames[pl.best_match_idx]), gray, self.pose)
                if res is not None:
                    self.pose = np.asarray(res[0], float)

    # ------------------------------------------------------------------ main
    def step(self, bgr: Optional[np.ndarray], t: float,
             packet: Optional[FramePacket] = None) -> NavOutput:
        tm: dict = {}
        t0 = time.perf_counter()
        if packet is None:
            rgb = None
            if bgr is not None:
                rgb = bgr if bgr.shape[:2] == (PROC_H, PROC_W) else \
                    cv2.resize(bgr, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)
            packet = FramePacket(clip_id="live", idx=self.idx, t=float(t), rgb=rgb)
        if self._perception is not None:
            packet = self._perception(packet)
            self._fuse_imu(packet)
        else:
            packet = self._run_perception(packet)
        tm["perception"] = (time.perf_counter() - t0) * 1e3
        tm.update({f"stage_{k}": v for k, v in packet.timings_ms.items()})

        odom = packet.odom
        tracking_ok = bool(odom.tracking_ok) if odom is not None else True
        self._integrate_pose(odom)
        self._loop_closure(packet)

        a = time.perf_counter()
        if packet.bev is None:
            raise RuntimeError("perception did not produce a BEV map")
        b = packet.bev
        state = bu.pack_state(b.height, b.trav_prob, b.conf, b.age, getattr(b, "hits", None))

        # moving obstacles: dynamic-class pixels with depth -> inflated, expiring cells
        pts = valid = label = None
        inj = getattr(packet, "dynamic_points_veh", None)
        if inj is not None:
            # hook for simulators / detectors that already output 3-D points of movers
            from .perception.dynamic_layer import DYNAMIC_CLASS
            pts = np.asarray(inj, np.float32)[None]
            label = np.full(pts.shape[:2], DYNAMIC_CLASS, np.uint8)
        elif packet.seg is not None and packet.depth is not None:
            label = packet.seg.label
            valid = packet.depth.valid if packet.depth.valid is not None \
                else np.isfinite(packet.depth.depth_m)
            fit = getattr(self.stages.get("depth"), "last_fit", None)
            pts = DynamicObstacleLayer.points_from_depth(packet.depth.depth_m, valid, fit)
        upd = self.dynamic.update(float(t), label, pts, valid, *self._last_motion)
        state = self.dynamic.apply(state, upd.active)
        tm["dynamic"] = (time.perf_counter() - a) * 1e3

        a = time.perf_counter()
        step_map = bu.height_step_map(state)
        gs = self.goal.step(state, tuple(self.pose), float(t),
                            dynamic_world=self.dynamic.active_world_points(self.pose),
                            dynamic_until=float(t) + self.dynamic.ttl_s,
                            step_map=step_map, tracking_ok=tracking_ok)
        tm["goal"] = (time.perf_counter() - a) * 1e3

        a = time.perf_counter()
        state = gs.state if gs.state is not None else state
        wmr = None
        policy_action = -1
        if self.wm is not None:
            lat, preds = self.wm.predict(state)
            wmr = [float(pp.risk_total) for pp in preds]
            if self.policy is not None:
                from .nav.rl_env import state_context
                ctx = state_context(state[None])[0]
                obs = self.policy.build_obs(lat, ctx, bu.mean_traversability(state),
                                            float(np.min(wmr)))
                policy_action, _ = self.policy.act(obs)
        trajs = self.planner.plan(state, wmr, gs.steer_yaw, gs.steer_xy)
        tq = float(odom.track_quality) if odom is not None else 1.0
        mc = packet.unc.mean_conf if packet.unc is not None else None
        rep: SupervisorReport = self.supervisor.decide(state, trajs, wmr, policy_action,
                                                       tracking_ok, tq, mc)
        d = rep.decision
        rule = rep.rule
        # goal-level stops sit in front of the gate's own rules
        if gs.mode in ("arrived", "no_route", "searching"):
            d = Decision(kind=STOP, action=ACTIONS.index("STOP"), speed_mps=0.0,
                         reason=gs.reason, risk=d.risk, confidence=d.confidence,
                         unknown_frac=d.unknown_frac, policy_source="supervisor")
            rule = {"arrived": "G0", "no_route": "G1", "searching": "G2"}[gs.mode]
        v = w = 0.0
        if d.kind != STOP and rep.best_idx >= 0:
            v0, w0 = Supervisor._cmd_of(trajs[rep.best_idx], self.planner.dt)
            v = float(min(d.speed_mps, v0))
            w = float(w0 * (v / v0)) if v0 > 1e-6 else 0.0
        tm["decide"] = (time.perf_counter() - a) * 1e3
        tm["total"] = (time.perf_counter() - t0) * 1e3

        packet.trajectories = trajs
        packet.decision = d
        self._t_prev = float(t)
        self.idx += 1
        self.last = NavOutput(v, w, d, rule, gs, tuple(float(x) for x in self.pose),
                              packet, tm)
        return self.last
