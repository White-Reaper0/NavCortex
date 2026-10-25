"""Run the world model, RL policy, planner and supervisor over all five clips.

    python -m prism.training.run_nav_infer [--clips clip_01 ...] [--device cuda]

Reads the `bev` cache (and `odom` / `unc` when present) and writes
`save_stage(clip_id, "plan", ...)`.  Everything the stage-09 renderer, the
stage-10 renderer and the final dashboard need about prediction and decision
lives in that one file; the exact key list is printed at the end of every run and
is reproduced below so the lead can wire it up without reading this source.

`plan` cache keys
-----------------
N = frames, A = 6 actions, T = 6 world-model horizon steps,
C = number of planner candidate arcs, S = 12 planner rollout steps.

    world model
      wm_latent            (N, 96)          float16  encoder latent of the BEV state
      wm_occ               (N, A, T, 16, 16) float16 predicted obstacle occupancy
      wm_trav              (N, A, T)        float16  predicted mean traversability
      wm_risk              (N, A, T)        float16  predicted collision risk per step
      wm_risk_total        (N, A)           float32  peak risk over the horizon
      wm_latency_ms        (N,)             float32  measured per-frame rollout latency

    planner candidates
      cand_action          (C,)             int8     base action index of each candidate
      cand_v               (C,)             float32  commanded linear speed, m/s
      cand_w               (C,)             float32  commanded yaw rate, rad/s
      traj_xy              (N, C, S, 2)     float16  path in the vehicle frame, metres
      traj_yaw             (N, C, S)        float16  heading along the path, rad
      traj_clearance       (N, C, S)        float16  lateral clearance to blocking cells, m
      traj_cost            (N, C)           float32  total planner cost (+100 if rejected)
      traj_risk            (N, C)           float32  max(sweep risk, world-model risk)
      traj_max_step        (N, C)           float32  worst height step under the footprint, m
      traj_unknown_frac    (N, C)           float32
      traj_mean_conf       (N, C)           float32
      traj_feasible        (N, C)           bool
      traj_reject_reason   (N, C)           <U96     "" when feasible
      best_traj_idx        (N,)             int16    committed candidate, -1 if none

    policy
      policy_action        (N,)             int8     PPO proposal, -1 if unavailable
      policy_probs         (N, A)           float32  action probabilities
      policy_source        (N,)             <U24     "rl" | "rl+supervisor" | "supervisor"
                                            - which brain the executed action came from
      policy_ckpt          (N,)             <U96     where the policy itself was loaded from

    decision (the supervisor's verdict - this is what the dashboard shows)
      decision_kind        (N,)             int8     index into CFG.DECISIONS
      decision_action      (N,)             int8     index into CFG.ACTIONS
      decision_speed       (N,)             float32  commanded speed, m/s
      decision_risk        (N,)             float32
      decision_conf        (N,)             float32
      decision_unknown_frac(N,)             float32
      decision_rule        (N,)             <U4      "R1".."R8", the rule that fired
      decision_reason      (N,)             <U200    human-readable, shown verbatim
      override             (N,)             bool     supervisor changed the policy's action

    meta                   ()               json string: model sizes, device, timings,
                                            action names, data provenance
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from ..config import CFG, CLIP_IDS, ACTIONS, N_ACTIONS, DECISIONS
from ..io_utils import save_stage
from ..models.world_model import WorldModelStage
from ..nav import bev_utils as bu
from ..nav.planner import Planner
from ..nav.supervisor import Supervisor
from ..nav.rl_env import PolicyStage, state_context

REASON_DT = "<U200"
REJECT_DT = "<U96"


def run_clip(clip_id: str, wms: WorldModelStage, planner: Planner,
             sup: Supervisor, pol: PolicyStage, verbose: bool = True) -> dict | None:
    states = bu.load_bev_states(clip_id)
    if states is None:
        print(f"  [{clip_id}] no `bev` cache - skipped")
        return None
    odom = bu.load_odom(clip_id)
    N = states.shape[0]
    A, T = N_ACTIONS, wms.horizon
    C, S = len(planner.candidates), planner.n_steps

    out = {
        "wm_latent": np.zeros((N, CFG.wm.state_dim), np.float16),
        "wm_occ": np.zeros((N, A, T, bu.OCC_G, bu.OCC_G), np.float16),
        "wm_trav": np.zeros((N, A, T), np.float16),
        "wm_risk": np.zeros((N, A, T), np.float16),
        "wm_risk_total": np.zeros((N, A), np.float32),
        "wm_latency_ms": np.zeros(N, np.float32),
        "cand_action": np.array([c[0] for c in planner.candidates], np.int8),
        "cand_v": np.array([c[1] for c in planner.candidates], np.float32),
        "cand_w": np.array([c[2] for c in planner.candidates], np.float32),
        "traj_xy": np.zeros((N, C, S, 2), np.float16),
        "traj_yaw": np.zeros((N, C, S), np.float16),
        "traj_clearance": np.zeros((N, C, S), np.float16),
        "traj_cost": np.zeros((N, C), np.float32),
        "traj_risk": np.zeros((N, C), np.float32),
        "traj_max_step": np.zeros((N, C), np.float32),
        "traj_unknown_frac": np.zeros((N, C), np.float32),
        "traj_mean_conf": np.zeros((N, C), np.float32),
        "traj_feasible": np.zeros((N, C), bool),
        "traj_reject_reason": np.zeros((N, C), REJECT_DT),
        "best_traj_idx": np.full(N, -1, np.int16),
        "policy_action": np.full(N, -1, np.int8),
        "policy_probs": np.zeros((N, A), np.float32),
        "policy_source": np.zeros(N, "<U24"),
        "policy_ckpt": np.zeros(N, "<U96"),
        "decision_kind": np.zeros(N, np.int8),
        "decision_action": np.zeros(N, np.int8),
        "decision_speed": np.zeros(N, np.float32),
        "decision_risk": np.zeros(N, np.float32),
        "decision_conf": np.zeros(N, np.float32),
        "decision_unknown_frac": np.zeros(N, np.float32),
        "decision_rule": np.zeros(N, "<U4"),
        "decision_reason": np.zeros(N, REASON_DT),
        "override": np.zeros(N, bool),
    }

    pol.reset()
    sup.reset()
    wms.reset()
    t_wm = t_plan = t_sup = 0.0
    t0 = time.perf_counter()
    for i in range(N):
        st = states[i]
        a0 = time.perf_counter()
        lat, preds = wms.predict(st)
        a1 = time.perf_counter()
        wmr = [float(p.risk_total) for p in preds]
        trajs = planner.plan(st, wmr)
        a2 = time.perf_counter()

        # policy proposal (obs must match rl_env's layout exactly)
        ctx = state_context(st[None])[0]
        obs = pol.build_obs(lat, ctx, bu.mean_traversability(st), float(np.min(wmr)))
        p_a, p_probs = pol.act(obs)

        ok = bool(odom["tracking_ok"][min(i, len(odom["tracking_ok"]) - 1)]) if odom else True
        tq = float(odom["track_quality"][min(i, len(odom["track_quality"]) - 1)]) if odom else 1.0
        rep = sup.decide(st, trajs, wmr, policy_action=p_a, tracking_ok=ok, track_quality=tq)
        a3 = time.perf_counter()
        t_wm += a1 - a0
        t_plan += a2 - a1
        t_sup += a3 - a2

        out["wm_latent"][i] = lat
        out["wm_latency_ms"][i] = wms.last_latency_ms
        for k, p in enumerate(preds):
            out["wm_occ"][i, p.action] = p.occ_forecast
            out["wm_trav"][i, p.action] = p.trav_forecast
            out["wm_risk"][i, p.action] = p.collision_risk
            out["wm_risk_total"][i, p.action] = p.risk_total
        for j, t in enumerate(trajs):
            out["traj_xy"][i, j] = t.xy
            out["traj_yaw"][i, j] = t.yaw
            out["traj_clearance"][i, j] = t.clearance
            out["traj_cost"][i, j] = t.cost
            out["traj_risk"][i, j] = t.collision_risk
            out["traj_max_step"][i, j] = t.max_step
            out["traj_unknown_frac"][i, j] = t.unknown_frac
            out["traj_mean_conf"][i, j] = t.mean_conf
            out["traj_feasible"][i, j] = t.feasible
            out["traj_reject_reason"][i, j] = t.reject_reason[:96]
        d = rep.decision
        out["best_traj_idx"][i] = rep.best_idx
        out["policy_action"][i] = p_a
        out["policy_probs"][i] = p_probs
        # provenance of the *action*, not of the checkpoint: "rl" when the gate
        # accepted the policy's proposal unchanged, "rl+supervisor" when it kept
        # the action but restricted it, "supervisor" when it replaced the action.
        out["policy_source"][i] = d.policy_source[:24]
        out["policy_ckpt"][i] = pol.source[:96]
        out["decision_kind"][i] = d.kind
        out["decision_action"][i] = d.action
        out["decision_speed"][i] = d.speed_mps
        out["decision_risk"][i] = d.risk
        out["decision_conf"][i] = d.confidence
        out["decision_unknown_frac"][i] = d.unknown_frac
        out["decision_rule"][i] = rep.rule
        out["decision_reason"][i] = d.reason[:200]
        out["override"][i] = rep.overrode
    wall = time.perf_counter() - t0

    # Aliases for prism.pipeline.build_packet, which reads the decision under
    # shorter names.  Same data, written twice; the cache is 4 MB either way.
    out["decision"] = out["decision_kind"]
    out["action"] = out["decision_action"]
    out["speed_cmd"] = out["decision_speed"]
    out["reason"] = out["decision_reason"]
    out["risk"] = out["decision_risk"]
    out["confidence"] = out["decision_conf"]
    out["unknown_frac"] = out["decision_unknown_frac"]

    hist = np.bincount(out["decision_kind"].astype(int), minlength=4)
    rules, rc = np.unique(out["decision_rule"], return_counts=True)
    if verbose:
        print(f"  [{clip_id}] {N} frames in {wall:.1f}s "
              f"({wall/N*1e3:.1f} ms/frame: wm {t_wm/N*1e3:.2f}, "
              f"plan {t_plan/N*1e3:.2f}, gate {t_sup/N*1e3:.2f})")
        print("        decisions " + "  ".join(f"{DECISIONS[k]} {hist[k]}" for k in range(4))
              + f"   overrides {int(out['override'].sum())}/{N}")
        print("        rules     " + "  ".join(f"{r}:{c}" for r, c in zip(rules, rc)))
    out["_stats"] = {"frames": int(N), "wall_s": float(wall),
                     "ms_per_frame": float(wall / N * 1e3),
                     "wm_ms": float(t_wm / N * 1e3), "plan_ms": float(t_plan / N * 1e3),
                     "gate_ms": float(t_sup / N * 1e3),
                     "decisions": {DECISIONS[k]: int(hist[k]) for k in range(4)},
                     "rules": {str(r): int(c) for r, c in zip(rules, rc)},
                     "overrides": int(out["override"].sum()),
                     "had_odom": odom is not None}
    return out


def main(clips=None, device: str | None = None) -> None:
    clips = clips or CLIP_IDS
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    wms = WorldModelStage(device=dev)
    planner = Planner()
    sup = Supervisor(planner)
    pol = PolicyStage(device="cpu", wm=wms.model)
    bench = wms.benchmark(30)

    print("prism prediction + decision inference")
    print(f"  world model : {bench['params']['total']:,} params "
          f"({bench['mb_fp32']:.2f} MB fp32), {bench['full_ms']:.2f} ms per frame "
          f"for {N_ACTIONS} actions x {wms.horizon} steps on {bench['device']}")
    print(f"  checkpoint  : {wms.loaded_from}")
    print(f"  policy      : {pol.source}")
    print(f"  planner     : {len(planner.candidates)} candidate arcs, "
          f"{planner.n_steps} x {planner.dt:.3f} s")

    stats = {}
    for cid in clips:
        out = run_clip(cid, wms, planner, sup, pol)
        if out is None:
            continue
        st = out.pop("_stats")
        meta = {
            "clip": cid, "actions": ACTIONS, "decisions": DECISIONS,
            "wm_dt_s": bu.WM_DT, "wm_horizon": wms.horizon,
            "wm_checkpoint": wms.loaded_from, "wm_params": bench["params"],
            "wm_latency_ms": bench["full_ms"], "wm_device": bench["device"],
            "policy_source": pol.source,
            "planner_candidates": len(planner.candidates),
            "planner_dt_s": planner.dt, "planner_steps": planner.n_steps,
            "bev_channels": bu.BEV_CH, "occ_grid": bu.OCC_G,
            "stats": st,
        }
        out["meta"] = np.array(json.dumps(meta))
        p = save_stage(cid, "plan", **out)
        stats[cid] = st
        print(f"        -> {p}  ({p.stat().st_size/1e6:.1f} MB)")

    if not stats:
        print("\nNo clips had a `bev` cache. Run the mapping stage first, then re-run this.")
        return

    print("\n" + "=" * 78)
    print("`plan` cache written for: " + ", ".join(stats))
    print("keys (N frames, A=6 actions, T=%d wm steps, C=%d candidates, S=%d rollout steps):"
          % (CFG.wm.pred_horizon, len(planner.candidates), planner.n_steps))
    groups = [
        ("world model", ["wm_latent (N,96) f16", "wm_occ (N,A,T,16,16) f16",
                         "wm_trav (N,A,T) f16", "wm_risk (N,A,T) f16",
                         "wm_risk_total (N,A) f32", "wm_latency_ms (N,) f32"]),
        ("candidates", ["cand_action (C,) i8", "cand_v (C,) f32", "cand_w (C,) f32",
                        "traj_xy (N,C,S,2) f16", "traj_yaw (N,C,S) f16",
                        "traj_clearance (N,C,S) f16", "traj_cost (N,C) f32",
                        "traj_risk (N,C) f32", "traj_max_step (N,C) f32",
                        "traj_unknown_frac (N,C) f32", "traj_mean_conf (N,C) f32",
                        "traj_feasible (N,C) bool", "traj_reject_reason (N,C) <U96",
                        "best_traj_idx (N,) i16"]),
        ("policy", ["policy_action (N,) i8", "policy_probs (N,A) f32",
                    'policy_source (N,) <U24  "rl" | "rl+supervisor" | "supervisor"',
                    "policy_ckpt (N,) <U96"]),
        ("decision", ["decision_kind (N,) i8", "decision_action (N,) i8",
                      "decision_speed (N,) f32", "decision_risk (N,) f32",
                      "decision_conf (N,) f32", "decision_unknown_frac (N,) f32",
                      "decision_rule (N,) <U4", "decision_reason (N,) <U200",
                      "override (N,) bool"]),
        ("aliases for prism.pipeline.build_packet (same arrays, shorter names)",
         ["decision == decision_kind", "action == decision_action",
          "speed_cmd == decision_speed", "reason == decision_reason",
          "risk == decision_risk", "confidence == decision_conf",
          "unknown_frac == decision_unknown_frac"]),
        ("meta", ["meta () json string"]),
    ]
    for g, ks in groups:
        print(f"  {g}:")
        for k in ks:
            print(f"    {k}")
    print("\nrenderers: prism.render.r_world.render(packet, state) for 09_world_model,")
    print("           prism.render.r_rl.render(packet, state) for 10_rl_policy.")
    print("  r_world `state` keys: bev_state (8,H,W) float32, "
          "wm_meta={params_total, latency_ms, device}")
    print("  r_rl    `state` keys: bev_state, policy_probs (6,), policy_action int,")
    print("           best_idx int, policy_source str  (falls back to packet fields)")
    print("=" * 78)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    main(a.clips, a.device)
