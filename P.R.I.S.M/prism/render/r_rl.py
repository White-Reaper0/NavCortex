"""Stage 10 renderer - PPO policy trained inside the world model, and the safety gate.

Left    : the BEV map with every candidate arc drawn - rejected arcs greyed out,
          the committed arc highlighted, the blocking reason called out.
Top mid : the policy's action probabilities and the action it proposed.
Top right: the PPO learning curve and the held-out comparison against baselines.
Middle  : the supervisor's verdict, its reason string, and - loudly - whether it
          overrode the policy.
Bottom  : the per-candidate table with each rejection reason.

`render(packet, state) -> 1280x720 BGR`, viz_common styling only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import cv2

from ..config import CFG, ACTIONS, N_ACTIONS, DECISIONS, WORK_DIR
from ..types import Decision, FramePacket, Trajectory
from .. import viz_common as V
from ..nav import bev_utils as bu
from .r_world import ACTION_COLORS

W, H = 1280, 720
GREY = (78, 74, 70)


def _fit_rect(src, rect):
    x, y, w, h = rect
    sh, sw = src.shape[:2]
    s = min(w / sw, h / sh)
    nw, nh = max(1, int(sw * s)), max(1, int(sh * s))
    return (x + (w - nw) // 2, y + (h - nh) // 2, nw, nh)


def _wrap(s: str, width: int, scale: float = 0.40) -> list[str]:
    """Greedy word wrap to a pixel width."""
    out, line = [], ""
    for word in str(s).split():
        t = (line + " " + word).strip()
        if V.text_size(t, scale)[0] > width and line:
            out.append(line)
            line = word
        else:
            line = t
    if line:
        out.append(line)
    return out


def _load_rl_log(state: dict) -> dict:
    if "rl_log" in state:
        return state["rl_log"]
    d = {}
    p = WORK_DIR / "rl_train_log.npz"
    if p.exists():
        try:
            with np.load(p, allow_pickle=True) as z:
                d = {k: z[k] for k in z.files}
                if "meta" in d:
                    d["meta"] = json.loads(str(d["meta"]))
        except Exception:
            d = {}
    state["rl_log"] = d
    return d


def _state_bgr(packet: FramePacket, state: dict) -> Optional[np.ndarray]:
    st = state.get("bev_state")
    if st is None and packet.bev is not None:
        st = bu.pack_state(packet.bev.height, packet.bev.trav_prob, packet.bev.conf,
                           packet.bev.age, getattr(packet.bev, "hits", None))
        state["bev_state"] = st
    return None if st is None else bu.render_state_bgr(st)


#: the arc panel zooms into the nearest few metres so the 2 s rollouts are legible
ZOOM_FORWARD_M = 3.6


def _veh_px(x: float, y: float, rect, n_rows: int):
    """Vehicle-frame metres -> pixel in a BEV panel cropped to the nearest n_rows."""
    b = CFG.bev
    px0, py0, pw, ph = rect
    cx = (x / b.res_m + b.n_lateral + 0.5) / b.W
    row_full = (b.n_forward - 1) + 0.5 - y / b.res_m
    cy = (row_full - (b.H - n_rows)) / n_rows
    return int(px0 + cx * pw), int(py0 + cy * ph)


def _draw_trajs(img, rect, trajs: Sequence[Trajectory], best_idx: int,
                chosen_action: int, n_rows: int):
    """Candidate arcs on the BEV panel: grey = rejected, coloured = feasible."""
    def P(x, y):
        return _veh_px(float(x), float(y), rect, n_rows)

    for i, t in enumerate(trajs):
        if i == best_idx:
            continue
        c = GREY if not t.feasible else tuple(
            int(v * 0.65) for v in ACTION_COLORS[t.action % len(ACTION_COLORS)])
        pts = np.array([P(0.0, 0.0)] + [P(x, y) for x, y in t.xy], np.int32)
        cv2.polylines(img, [pts], False, c, 2 if t.feasible else 1, cv2.LINE_AA)
        if not t.feasible and t.xy.shape[0]:
            k = min(int(np.argmax(np.linalg.norm(t.xy, axis=1))), t.xy.shape[0] - 1)
            px, py = P(t.xy[k, 0], t.xy[k, 1])
            cv2.line(img, (px - 4, py - 4), (px + 4, py + 4), V.BAD, 2, cv2.LINE_AA)
            cv2.line(img, (px - 4, py + 4), (px + 4, py - 4), V.BAD, 2, cv2.LINE_AA)
    if 0 <= best_idx < len(trajs):
        t = trajs[best_idx]
        c = ACTION_COLORS[chosen_action % len(ACTION_COLORS)]
        pts = np.array([P(0.0, 0.0)] + [P(x, y) for x, y in t.xy], np.int32)
        cv2.polylines(img, [pts], False, (12, 12, 12), 6, cv2.LINE_AA)
        cv2.polylines(img, [pts], False, c, 3, cv2.LINE_AA)
        cv2.circle(img, tuple(pts[-1]), 5, c, -1, cv2.LINE_AA)
    ex, ey = P(0.0, 0.0)
    cv2.circle(img, (ex, ey), 5, V.TEXT, -1, cv2.LINE_AA)


def trajs_from_cache(packet: FramePacket) -> list[Trajectory]:
    """Rebuild the candidate arcs from the `plan` cache attached by pipeline."""
    z = getattr(packet, "_plan_cache", None)
    if z is None or "traj_xy" not in z:
        return []
    i = min(int(getattr(packet, "_plan_i", packet.idx)), z["traj_xy"].shape[0] - 1)
    act = z["cand_action"]
    out = []
    for j in range(z["traj_xy"].shape[1]):
        out.append(Trajectory(
            action=int(act[j]),
            xy=np.asarray(z["traj_xy"][i, j], np.float32),
            yaw=np.asarray(z["traj_yaw"][i, j], np.float32),
            clearance=np.asarray(z["traj_clearance"][i, j], np.float32),
            max_step=float(z["traj_max_step"][i, j]),
            unknown_frac=float(z["traj_unknown_frac"][i, j]),
            mean_conf=float(z["traj_mean_conf"][i, j]),
            collision_risk=float(z["traj_risk"][i, j]),
            cost=float(z["traj_cost"][i, j]),
            feasible=bool(z["traj_feasible"][i, j]),
            reject_reason=str(z["traj_reject_reason"][i, j])))
    return out


def _plan_scalar(packet, key, default):
    z = getattr(packet, "_plan_cache", None)
    if z is None or key not in z:
        return default
    a = z[key]
    i = min(int(getattr(packet, "_plan_i", packet.idx)), len(a) - 1)
    return a[i]


def render(packet: FramePacket, state: dict) -> np.ndarray:
    img = V.canvas(W, H)
    trajs: list[Trajectory] = list(packet.trajectories or []) or trajs_from_cache(packet)
    d: Decision = packet.decision or Decision(reason="no decision on this packet")
    probs = state.get("policy_probs", getattr(packet, "_policy_probs", None))
    if probs is None:
        probs = _plan_scalar(packet, "policy_probs", np.zeros(N_ACTIONS, np.float32))
    probs = np.asarray(probs, np.float32).ravel()
    if probs.size != N_ACTIONS:
        probs = np.zeros(N_ACTIONS, np.float32)
    pol_a = int(state.get("policy_action",
                          getattr(packet, "_policy_action",
                                  int(_plan_scalar(packet, "policy_action", -1)))))
    best_idx = int(state.get("best_idx", int(_plan_scalar(packet, "best_traj_idx", -1))))
    if best_idx < 0 and trajs:
        feas = [i for i, t in enumerate(trajs) if t.feasible and t.action == d.action]
        best_idx = feas[0] if feas else -1
    overrode = bool(pol_a >= 0 and pol_a != d.action)
    log = _load_rl_log(state)

    V.header(img, W, "prism", "10 | PPO policy inside the world model + safety supervisor",
             right=f"{packet.clip_id}   frame {packet.idx:03d}")

    # ------------------------------------------------------------ A: BEV + arcs
    ax, ay, aw, ah = 14, 52, 480, 430
    r = V.panel(img, ax, ay, aw, ah, "Candidate arcs",
                f"{len(trajs)} rollouts | {CFG.safety.horizon_s:.0f} s horizon")
    sb = _state_bgr(packet, state)
    if sb is not None:
        n_rows = int(min(CFG.bev.H, round(ZOOM_FORWARD_M / CFG.bev.res_m)))
        crop = sb[CFG.bev.H - n_rows:]
        fr = _fit_rect(crop, (r[0] + 6, r[1] + 6, r[2] - 12, r[3] - 44))
        img[fr[1]:fr[1] + fr[3], fr[0]:fr[0] + fr[2]] = cv2.resize(
            crop, (fr[2], fr[3]), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(img, (fr[0], fr[1]), (fr[0] + fr[2], fr[1] + fr[3]), V.EDGE, 1)
        _draw_trajs(img, fr, trajs, best_idx, d.action, n_rows)
        V.text(img, f"nearest {n_rows*CFG.bev.res_m:.1f} m of the "
                    f"{CFG.bev.range_forward_m:.1f} m map",
               (fr[0] + 6, fr[1] + 14), 0.34, V.TEXT_DIM)
    else:
        V.text(img, "no BEV map on this packet", (r[0] + 12, r[1] + 40), 0.44, V.TEXT_DIM)
    n_rej = sum(1 for t in trajs if not t.feasible)
    V.text(img, f"grey = rejected ({n_rej}/{len(trajs)})   thick = committed arc",
           (r[0] + 8, r[1] + r[3] - 22), 0.35, V.TEXT_DIM)
    if trajs:
        rej = [t for t in trajs if not t.feasible]
        if rej:
            top = min(rej, key=lambda t: t.cost).reject_reason
            V.text(img, top[:70], (r[0] + 8, r[1] + r[3] - 6), 0.33, V.BAD)

    # ------------------------------------------------------------ B: policy probs
    bx, by, bw, bh = 506, 52, 376, 250
    # provenance of the *checkpoint*; the provenance of the executed action is
    # `decision.policy_source` and is shown in the supervisor panel
    src = str(state.get("policy_ckpt", _plan_scalar(packet, "policy_ckpt", "")))
    src = Path(src).name if src and ("\\" in src or "/" in src) else src
    r = V.panel(img, bx, by, bw, bh, "Policy proposal",
                "PPO, trained in the world model")
    yy = r[1] + 22
    for a in range(N_ACTIONS):
        c = ACTION_COLORS[a]
        is_pol = (a == pol_a)
        V.text(img, ACTIONS[a], (r[0] + 10, yy + 10), 0.38,
               V.ACCENT if is_pol else V.TEXT, 1)
        V.bar_meter(img, r[0] + 84, yy, r[2] - 150, 13, float(probs[a]), color=c)
        V.text(img, f"{probs[a]*100:4.1f}%", (r[0] + r[2] - 58, yy + 11), 0.35, V.TEXT_DIM)
        if is_pol:
            cv2.rectangle(img, (r[0] + 4, yy - 4), (r[0] + r[2] - 8, yy + 17), V.ACCENT, 1)
        yy += 24
    lab = f"proposes {ACTIONS[pol_a]}" if pol_a >= 0 else "no policy proposal"
    V.text(img, lab, (r[0] + 10, r[1] + r[3] - 26), 0.46, V.ACCENT, 1, V.FONT_B)
    V.text(img, ("checkpoint: " + src)[:56] if src else "no PPO checkpoint loaded",
           (r[0] + 10, r[1] + r[3] - 8), 0.31, V.TEXT_DIM)

    # ------------------------------------------------------------ D: training curve
    dx, dy, dw, dh = 892, 52, 374, 250
    r = V.panel(img, dx, dy, dw, dh, "PPO training", "held-out evaluation")
    cur = log.get("curve_rew")
    if cur is not None and np.asarray(cur).size > 2:
        V.sparkline(img, r[0] + 10, r[1] + 8, r[2] - 20, 74, np.asarray(cur).ravel(),
                    color=V.ACCENT2)
        V.text(img, "episode return during training", (r[0] + 10, r[1] + 96), 0.32, V.TEXT_DIM)
    else:
        cv2.rectangle(img, (r[0] + 10, r[1] + 8), (r[0] + r[2] - 10, r[1] + 82), (26, 24, 22), -1)
        V.text(img, "no training log yet", (r[0] + 18, r[1] + 50), 0.4, V.TEXT_DIM)
    ev = (log.get("meta") or {}).get("eval", {}) if isinstance(log.get("meta"), dict) else {}
    yy = r[1] + 116
    V.text(img, "policy", (r[0] + 10, yy), 0.33, V.TEXT_DIM)
    V.text(img, "return", (r[0] + 132, yy), 0.33, V.TEXT_DIM)
    V.text(img, "prog m", (r[0] + 208, yy), 0.33, V.TEXT_DIM)
    V.text(img, "coll%", (r[0] + 274, yy), 0.33, V.TEXT_DIM)
    V.text(img, "stop%", (r[0] + 324, yy), 0.33, V.TEXT_DIM)
    yy += 16
    names = [("ppo", "PPO"), ("always_forward", "always FORWARD"),
             ("random", "random"), ("always_stop", "always STOP")]
    if ev:
        for key, nice in names:
            m = ev.get(key)
            if not m:
                continue
            c = V.ACCENT if key == "ppo" else V.TEXT_DIM
            V.text(img, nice, (r[0] + 10, yy), 0.34, c, 1)
            V.text(img, f"{m['return']:+.1f}", (r[0] + 132, yy), 0.34, c, 1)
            V.text(img, f"{m['progress_m']:.2f}", (r[0] + 208, yy), 0.34, c, 1)
            V.text(img, f"{m['collision_rate']*100:.0f}", (r[0] + 274, yy), 0.34, c, 1)
            V.text(img, f"{m['stop_rate']*100:.0f}", (r[0] + 324, yy), 0.34, c, 1)
            yy += 18
        V.text(img, "collisions scored geometrically on held-out scenes",
               (r[0] + 10, r[1] + r[3] - 8), 0.30, V.TEXT_DIM)
    else:
        V.text(img, "no held-out evaluation available", (r[0] + 10, yy + 6), 0.35, V.TEXT_DIM)

    # ------------------------------------------------------------ C: verdict
    cx, cy, cw, ch = 506, 312, 760, 170
    r = V.panel(img, cx, cy, cw, ch, "Safety supervisor", "gate over the policy",
                accent=V.DECISION_COLORS_BGR[d.kind] if hasattr(V, "DECISION_COLORS_BGR")
                else V.ACCENT)
    bw_, bh_ = V.decision_badge(img, r[0] + 12, r[1] + 12, int(d.kind), 0.8)
    V.text(img, f"{ACTIONS[d.action]}", (r[0] + 12, r[1] + bh_ + 34), 0.5, V.TEXT, 1, V.FONT_B)
    V.text(img, f"{d.speed_mps:.2f} m/s", (r[0] + 12, r[1] + bh_ + 54), 0.42, V.TEXT_DIM)
    V.text(img, f"source: {d.policy_source}", (r[0] + 12, r[1] + bh_ + 72), 0.34, V.TEXT_DIM)

    tx = r[0] + 12 + max(bw_, 118) + 18
    tw = r[2] - (tx - r[0]) - 200
    for i, ln in enumerate(_wrap(d.reason, tw, 0.40)[:4]):
        V.text(img, ln, (tx, r[1] + 26 + i * 18), 0.40, V.TEXT, 1)

    mx = r[0] + r[2] - 186
    for i, (lab, val, hi, warn) in enumerate([
            ("collision risk", d.risk, 1.0, CFG.safety.risk_slow),
            ("unknown fraction", d.unknown_frac, 1.0, CFG.safety.unknown_frac_slow),
            ("mean confidence", d.confidence, 1.0, None)]):
        V.bar_meter(img, mx, r[1] + 30 + i * 34, 168, 12, float(val),
                    label=f"{lab}  {val:.2f}", warn_at=warn)

    if overrode:
        ox, oy = tx, r[1] + r[3] - 40
        V.badge(img, ox, oy, f"SUPERVISOR OVERRIDE: {ACTIONS[pol_a]} -> {ACTIONS[d.action]}",
                V.BAD, 0.46)
        V.text(img, "the safety gate outranks the learned policy",
               (ox, oy - 6), 0.32, V.BAD)
    elif pol_a >= 0:
        V.text(img, f"policy proposal {ACTIONS[pol_a]} accepted by the gate",
               (tx, r[1] + r[3] - 16), 0.36, V.OK)

    # ------------------------------------------------------------ E: table
    ex_, ey_, ew_, eh_ = 14, 492, 1252, 128
    r = V.panel(img, ex_, ey_, ew_, eh_, "Candidate evaluation",
                "sampling-based rollout planner (not Nav2 MPPI)")
    cols = [(10, "action"), (86, "cost"), (150, "risk"), (208, "step m"),
            (274, "unknown"), (348, "conf"), (410, "verdict / reason")]
    for cxp, lab in cols:
        V.text(img, lab, (r[0] + cxp, r[1] + 16), 0.33, V.TEXT_DIM)
    ok_i = sorted([i for i, t in enumerate(trajs) if t.feasible],
                  key=lambda i: trajs[i].cost)[:3]
    bad_i = sorted([i for i, t in enumerate(trajs) if not t.feasible],
                   key=lambda i: trajs[i].cost)[:2]
    order = ok_i + bad_i
    yy = r[1] + 34
    for i in order[:5]:
        t = trajs[i]
        c = V.TEXT if t.feasible else GREY
        mark = ">" if i == best_idx else " "
        V.text(img, f"{mark} {ACTIONS[t.action]}", (r[0] + 10, yy), 0.35,
               V.ACCENT if i == best_idx else c, 1)
        V.text(img, f"{t.cost:+.2f}", (r[0] + 86, yy), 0.35, c, 1)
        V.text(img, f"{t.collision_risk:.2f}", (r[0] + 150, yy), 0.35, c, 1)
        V.text(img, f"{t.max_step:.3f}", (r[0] + 208, yy), 0.35, c, 1)
        V.text(img, f"{t.unknown_frac*100:.0f}%", (r[0] + 274, yy), 0.35, c, 1)
        V.text(img, f"{t.mean_conf:.2f}", (r[0] + 348, yy), 0.35, c, 1)
        txt = t.reject_reason if not t.feasible else "feasible"
        V.text(img, txt[:110], (r[0] + 410, yy), 0.35, V.BAD if not t.feasible else V.OK, 1)
        yy += 18
    if not trajs:
        V.text(img, "no candidate trajectories on this packet", (r[0] + 10, yy), 0.4, V.TEXT_DIM)

    # ------------------------------------------------------------ note
    V.rounded_note(img, 14, 628, 1252, [
        "PPO never drives the vehicle. It proposes; the supervisor gates the proposal against terrain step, "
        "confidence, unknown fraction, predicted risk and stopping space,",
        "and every override is shown above with the rule that fired. A policy that simply stopped everywhere "
        "would be safe and useless - the reward penalises idling for exactly that reason.",
    ], title="Learned policy, hard safety gate")

    V.footer(img, W, H,
             left=f"thresholds from CFG.safety | clearance {CFG.ugv.clearance_m*100:.1f} cm | "
                  f"footprint {CFG.ugv.width_m*100:.0f} x {CFG.ugv.length_m*100:.0f} cm "
                  f"+ {CFG.safety.corridor_margin_m*100:.0f} cm margin",
             right="confidence is confidence, not a calibrated collision probability")
    return img


# ---------------------------------------------------------------- self-test

if __name__ == "__main__":
    from ..models.world_model import WorldModelStage
    from ..nav.planner import Planner
    from ..nav.supervisor import Supervisor

    st = bu.synthetic_state("wall", wall_dist_m=1.2, wall_height_m=0.10)
    wms = WorldModelStage(device="cpu")
    _, preds = wms.predict(st)
    pl = Planner()
    trajs = pl.plan(st, [p.risk_total for p in preds])
    sup = Supervisor(pl)
    rep = sup.decide(st, trajs, [p.risk_total for p in preds], policy_action=0)

    p = FramePacket(clip_id="clip_04", idx=87, t=87 / 30.0)
    p.trajectories = trajs
    p.wm_preds = preds
    p.decision = rep.decision
    probs = np.array([0.52, 0.11, 0.09, 0.14, 0.05, 0.09], np.float32)
    s = {"bev_state": st, "policy_probs": probs, "policy_action": 0,
         "best_idx": rep.best_idx, "policy_source": "heuristic self-test"}
    img = render(p, s)
    out = WORK_DIR / "preview_rl.png"
    cv2.imwrite(str(out), img)
    print(f"r_rl.render -> {img.shape} {img.dtype}; wrote {out}")
    print(f"decision: {DECISIONS[rep.decision.kind]} {ACTIONS[rep.decision.action]} "
          f"override={rep.overrode}")
    print(f"reason: {rep.decision.reason}")
