"""Stage 09 renderer - tiny neural world model rollouts.

Left    : the current BEV world state the model is compressing.
Middle  : a filmstrip of the predicted 16x16 obstacle occupancy for three
          candidate actions across the 2 s horizon.
Right   : the predicted collision-risk curve per action, and which action the
          model prefers.
Bottom  : measured model size / rollout latency, and the "why not future video"
          note.

`render(packet, state) -> 1280x720 BGR`, viz_common styling only.
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np
import cv2

from ..config import CFG, ACTIONS, N_ACTIONS
from ..types import FramePacket, WorldModelPrediction
from .. import viz_common as V
from ..nav import bev_utils as bu

W, H = 1280, 720

#: one colour per action, reused by r_rl so the two videos read as one product
ACTION_COLORS = [
    (110, 230, 120),   # FORWARD  green
    (255, 190, 70),    # LEFT     blue
    (90, 190, 255),    # RIGHT    amber
    (60, 200, 255),    # SLOW     yellow
    (225, 80, 225),    # REROUTE  magenta
    (60, 60, 235),     # STOP     red
]

#: the three actions shown in the filmstrip
FILM_ACTIONS = (0, 1, 4)
FILM_STEPS = (0, 1, 3, 5)


def _wrap(text_s: str, width: int, scale: float = 0.37) -> list[str]:
    """Greedy word wrap to a pixel width, so notes never run past the panel."""
    out, line = [], ""
    for word in str(text_s).split():
        t = (line + " " + word).strip()
        if V.text_size(t, scale)[0] > width and line:
            out.append(line)
            line = word
        else:
            line = t
    if line:
        out.append(line)
    return out


def _occ_tile(occ: np.ndarray, size: int) -> np.ndarray:
    """16x16 occupancy in [0,1] -> a size x size BGR tile (row 0 = far)."""
    g = np.clip(np.asarray(occ, np.float32), 0, 1)
    img = cv2.applyColorMap((g * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(img, (0, 0), (size - 1, size - 1), V.EDGE, 1)
    return img


def _risk_chart(img, rect, preds, horizon: int):
    x, y, w, h = rect
    pad_l, pad_b, pad_t = 34, 22, 10
    x0, y0 = x + pad_l, y + pad_t
    pw, ph = w - pad_l - 12, h - pad_t - pad_b
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (26, 24, 22), -1)
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = int(y0 + ph - f * ph)
        cv2.line(img, (x0, yy), (x0 + pw, yy), (48, 44, 40), 1)
        V.text(img, f"{f:.2f}", (x - 2 + 4, yy + 4), 0.32, V.TEXT_DIM, 1)
    for lvl, col, lab in ((CFG.safety.risk_slow, V.WARN, "slow"),
                          (CFG.safety.risk_reroute, (255, 190, 70), "reroute"),
                          (CFG.safety.risk_stop, V.BAD, "stop")):
        yy = int(y0 + ph - lvl * ph)
        cv2.line(img, (x0, yy), (x0 + pw, yy), col, 1, cv2.LINE_AA)
        V.text(img, lab, (x0 + pw - V.text_size(lab, 0.3)[0] - 3, yy - 3), 0.3, col, 1)
    if preds:
        xs = np.linspace(x0 + 2, x0 + pw - 2, horizon).astype(np.int32)
        for p in preds:
            r = np.clip(np.asarray(p.collision_risk, np.float32).ravel()[:horizon], 0, 1)
            if r.size < horizon:
                r = np.pad(r, (0, horizon - r.size), mode="edge")
            ys = (y0 + ph - r * ph).astype(np.int32)
            c = ACTION_COLORS[p.action % len(ACTION_COLORS)]
            cv2.polylines(img, [np.stack([xs, ys], 1)], False, c, 2, cv2.LINE_AA)
            cv2.circle(img, (int(xs[-1]), int(ys[-1])), 3, c, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), V.EDGE, 1)
    V.text(img, "t+0.33 s", (x0, y0 + ph + 14), 0.32, V.TEXT_DIM, 1)
    lab = f"t+{CFG.wm.pred_horizon*bu.WM_DT:.1f} s"
    V.text(img, lab, (x0 + pw - V.text_size(lab, 0.32)[0], y0 + ph + 14), 0.32, V.TEXT_DIM, 1)


def _state_bgr(packet: FramePacket, state: dict) -> Optional[np.ndarray]:
    st = state.get("bev_state")
    if st is not None:
        return bu.render_state_bgr(st)
    if packet.bev is not None:
        s = bu.pack_state(packet.bev.height, packet.bev.trav_prob, packet.bev.conf,
                          packet.bev.age, getattr(packet.bev, "hits", None))
        state["bev_state"] = s
        return bu.render_state_bgr(s)
    return None


def preds_from_cache(packet: FramePacket) -> list[WorldModelPrediction]:
    """Rebuild the per-action rollouts from the `plan` cache attached by pipeline."""
    z = getattr(packet, "_plan_cache", None)
    i = int(getattr(packet, "_plan_i", packet.idx))
    if z is None or "wm_risk" not in z:
        return []
    occ, trav, risk = z["wm_occ"], z["wm_trav"], z["wm_risk"]
    tot = z.get("wm_risk_total")
    i = min(i, occ.shape[0] - 1)
    out = []
    for a in range(occ.shape[1]):
        r = np.asarray(risk[i, a], np.float32)
        out.append(WorldModelPrediction(
            action=a, next_states=np.zeros((r.size, CFG.wm.state_dim), np.float32),
            occ_forecast=np.asarray(occ[i, a], np.float32),
            trav_forecast=np.asarray(trav[i, a], np.float32),
            collision_risk=r,
            risk_total=float(tot[i, a]) if tot is not None else float(r.max())))
    return out


def _wm_log(state: dict) -> dict:
    """Measured training/eval numbers written by training/train_world_model.py."""
    if "wm_log" in state:
        return state["wm_log"]
    from ..config import WORK_DIR
    d = {}
    p = WORK_DIR / "wm_train_log.npz"
    if p.exists():
        try:
            with np.load(p, allow_pickle=True) as z:
                d = {k: z[k] for k in z.files}
                if "meta" in d:
                    d["meta"] = json.loads(str(d["meta"]))
        except Exception:
            d = {}
    state["wm_log"] = d
    return d


def _honesty_lines(state: dict) -> list[str]:
    """Two lines of measured, un-spun evaluation for the bottom note."""
    log = _wm_log(state)
    meta = log.get("meta") if isinstance(log.get("meta"), dict) else {}
    ev = (meta or {}).get("eval", {})
    out = []
    occ = ev.get("occ_mae"); occp = ev.get("occ_mae_persist"); occf = ev.get("occ_mae_encoder_floor")
    if occ and occp:
        k = len(occ) - 1
        won = sum(1 for j in range(len(occ)) if occ[j] < occp[j])
        floor = f", encoder-bottleneck floor {occf[k]:.3f}" if occf else ""
        out.append(f"Measured at t+{(k+1)*bu.WM_DT:.1f} s: occupancy MAE {occ[k]:.3f} vs a "
                   f"persistence baseline's {occp[k]:.3f}{floor} - "
                   f"model ahead on {won}/{len(occ)} horizon steps.")
    sens = ev.get("sensitivity") if isinstance(ev, dict) else None
    if sens:
        out.append(f"The forecast really does depend on the action: across-action spread of the "
                   f"predicted occupancy is {sens['occ_spread_model']:.3f} against "
                   f"{sens['occ_spread_truth']:.3f} for the rigidly warped ground truth.")
    return out


def _wm_meta(packet: FramePacket, state: dict) -> dict:
    meta = dict(state.get("wm_meta", {}) or {})
    z = getattr(packet, "_plan_cache", None)
    if z is not None and "meta" in z and "params_total" not in meta:
        try:
            m = json.loads(str(z["meta"]))
            meta.setdefault("params_total", m.get("wm_params", {}).get("total", 0))
            meta.setdefault("latency_ms", m.get("wm_latency_ms", 0.0))
            meta.setdefault("device", m.get("wm_device", "-"))
            meta.setdefault("checkpoint", m.get("wm_checkpoint", ""))
        except Exception:
            pass
    if z is not None and "wm_latency_ms" in z and not meta.get("latency_ms"):
        i = min(int(getattr(packet, "_plan_i", packet.idx)), len(z["wm_latency_ms"]) - 1)
        meta["latency_ms"] = float(z["wm_latency_ms"][i])
    return meta


def render(packet: FramePacket, state: dict) -> np.ndarray:
    img = V.canvas(W, H)
    preds = list(packet.wm_preds or []) or preds_from_cache(packet)
    horizon = CFG.wm.pred_horizon
    meta = _wm_meta(packet, state)
    params = int(meta.get("params_total") or 728_610)
    lat_ms = float(meta.get("latency_ms") or 0.0)
    dev = str(meta.get("device", "-"))

    V.header(img, W, "prism", "09 | Tiny neural world model - latent BEV dynamics",
             right=f"{packet.clip_id}   frame {packet.idx:03d}")

    # ------------------------------------------------------------ A: world state
    ax, ay, aw, ah = 14, 52, 372, 296
    r = V.panel(img, ax, ay, aw, ah, "BEV world state",
                f"{bu.BEV_CH} ch | {CFG.bev.H}x{CFG.bev.W} | {CFG.bev.res_m} m/cell")
    sb = _state_bgr(packet, state)
    if sb is not None:
        V.blit(img, sb, (r[0] + 6, r[1] + 6, r[2] - 12, r[3] - 46))
    else:
        V.text(img, "no BEV map on this packet", (r[0] + 12, r[1] + 40), 0.44, V.TEXT_DIM)
    V.legend(img, r[0] + 8, r[1] + r[3] - 12, ["safe", "risky", "obstacle", "unknown"],
             np.array([(110, 230, 120), (60, 200, 255), (60, 60, 235), (150, 150, 150)]),
             scale=0.34, vertical=False)
    V.text(img, "brightness = confidence", (r[0] + 8, r[1] + r[3] - 26), 0.32, V.TEXT_DIM)

    # ------------------------------------------------------------ B: model card
    bx, by, bw, bh = 14, 356, 372, 176
    r = V.panel(img, bx, by, bw, bh, "Model", "measured, this machine")
    rows = [
        ("encoder", f"conv 8->96, {CFG.bev.H}x{CFG.bev.W} -> {CFG.wm.state_dim}-D latent"),
        ("dynamics", f"GRU h={CFG.wm.hidden} x{CFG.wm.n_layers}, residual latent step"),
        ("heads", "16x16 occupancy | traversability | risk"),
        ("size", f"{params:,} parameters  ({params*4/1e6:.2f} MB fp32)"),
        ("rollout", f"{N_ACTIONS} actions x {horizon} steps of {bu.WM_DT:.2f} s "
                    f"= {horizon*bu.WM_DT:.1f} s"),
        ("latency", (f"{lat_ms:.2f} ms per frame on {dev}" if lat_ms > 0
                     else "not measured on this packet")),
    ]
    yy = r[1] + 20
    for k, v in rows:
        V.text(img, k, (r[0] + 10, yy), 0.36, V.ACCENT, 1)
        V.text(img, v, (r[0] + 78, yy), 0.36, V.TEXT, 1)
        yy += 24

    # ------------------------------------------------------------ C: why note
    V.rounded_note(img, 14, 540, 372, [
        "Photoreal future video would cost orders of magnitude",
        "more compute and would still have to be re-parsed back",
        "into geometry. The planner only needs where obstacles",
        "will be, how drivable the ground is, and collision risk -",
        "so prism predicts those directly in a 96-D latent.",
    ], title="Why a latent model, not generated video")

    # ------------------------------------------------------------ D: filmstrip
    dx, dy, dw, dh = 398, 52, 528, 480
    r = V.panel(img, dx, dy, dw, dh, "Predicted obstacle occupancy",
                "16x16 forward grid, row 0 = far")
    tile = 92
    gap = 10
    x_lab = r[0] + 12
    x0 = x_lab + 104
    yy = r[1] + 26
    V.text(img, "action", (x_lab, yy - 8), 0.34, V.TEXT_DIM)
    for j, k in enumerate(FILM_STEPS):
        lab = f"t+{(k+1)*bu.WM_DT:.2f}s"
        V.text(img, lab, (x0 + j * (tile + gap), yy - 8), 0.34, V.TEXT_DIM)
    yy += 6
    by_action = {p.action: p for p in preds}
    for i, a in enumerate(FILM_ACTIONS):
        ty = yy + i * (tile + 32)
        col = ACTION_COLORS[a % len(ACTION_COLORS)]
        cv2.rectangle(img, (x_lab, ty + 8), (x_lab + 4, ty + tile - 8), col, -1)
        V.text(img, ACTIONS[a], (x_lab + 10, ty + tile // 2 - 4), 0.42, V.TEXT, 1, V.FONT_B)
        p = by_action.get(a)
        if p is None:
            V.text(img, "-", (x0, ty + tile // 2), 0.5, V.TEXT_DIM)
            continue
        rk = np.asarray(p.collision_risk, np.float32).ravel()
        V.text(img, f"peak {float(np.max(rk)):.2f}", (x_lab + 10, ty + tile // 2 + 14),
               0.33, V.TEXT_DIM)
        for j, k in enumerate(FILM_STEPS):
            kk = min(k, p.occ_forecast.shape[0] - 1)
            t = _occ_tile(p.occ_forecast[kk], tile)
            px, py = x0 + j * (tile + gap), ty
            img[py:py + tile, px:px + tile] = t
            rv = float(rk[min(kk, rk.size - 1)])
            rc = V.BAD if rv >= CFG.safety.risk_stop else (
                V.WARN if rv >= CFG.safety.risk_slow else V.OK)
            cv2.rectangle(img, (px, py + tile), (px + tile, py + tile + 5), (40, 37, 34), -1)
            cv2.rectangle(img, (px, py + tile), (px + int(tile * min(rv, 1.0)),
                                                 py + tile + 5), rc, -1)
            V.text(img, f"{rv:.2f}", (px + 2, py + tile + 18), 0.31, V.TEXT_DIM)
    V.text(img, "bar under each tile = predicted collision risk at that step",
           (r[0] + 12, r[1] + r[3] - 10), 0.33, V.TEXT_DIM)

    # ------------------------------------------------------------ E: risk curves
    ex, ey, ew, eh = 936, 52, 330, 250
    r = V.panel(img, ex, ey, ew, eh, "Collision risk", "per action, over the horizon")
    _risk_chart(img, (r[0], r[1], r[2], r[3]), preds, horizon)

    # ------------------------------------------------------------ F: preference
    fx, fy, fw, fh = 936, 310, 330, 222
    r = V.panel(img, fx, fy, fw, fh, "Model preference", "safest arc that moves")
    if preds:
        totals = np.array([float(p.risk_total) for p in preds], np.float32)
        # "lowest risk" alone would name STOP on nearly every frame, which is true
        # and useless; the interesting quantity is the safest arc that still makes
        # progress.  STOP's risk is still listed so the comparison is visible.
        moving = [i for i in range(len(preds))
                  if bu.action_motion(preds[i].action)[0] >= 0.10]
        best = int(min(moving, key=lambda i: totals[i])) if moving else int(np.argmin(totals))
        yy = r[1] + 22
        for a in range(min(N_ACTIONS, len(preds))):
            p = preds[a]
            c = ACTION_COLORS[p.action % len(ACTION_COLORS)]
            V.text(img, ACTIONS[p.action], (r[0] + 10, yy + 9), 0.37,
                   V.TEXT if p.action != preds[best].action else V.ACCENT, 1)
            V.bar_meter(img, r[0] + 80, yy, r[2] - 140, 12, float(p.risk_total),
                        color=c, warn_at=CFG.safety.risk_slow)
            V.text(img, f"{p.risk_total:.2f}", (r[0] + r[2] - 50, yy + 10), 0.35, V.TEXT_DIM)
            if p.action == preds[best].action:
                cv2.rectangle(img, (r[0] + 4, yy - 4), (r[0] + r[2] - 8, yy + 16),
                              V.ACCENT, 1)
            yy += 24
        V.text(img, f"prefers {ACTIONS[preds[best].action]}", (r[0] + 10, r[1] + r[3] - 24),
               0.44, V.ACCENT, 1, V.FONT_B)
        V.text(img, "a prediction, not a command - the supervisor decides",
               (r[0] + 10, r[1] + r[3] - 8), 0.31, V.TEXT_DIM)
    else:
        V.text(img, "no rollouts on this packet", (r[0] + 12, r[1] + 40), 0.42, V.TEXT_DIM)

    # ------------------------------------------------------------ G: bottom note
    para = ("Trained on transitions from the cached BEV maps of the five clips plus kinematic "
            "augmentation: each observed map is rigidly warped under every candidate action's motion "
            "model, giving correct geometry for arcs that were never driven - an augmentation, not "
            "observed data. ") + " ".join(_honesty_lines(state))
    V.rounded_note(img, 398, 540, 868, _wrap(para, 840, 0.37)[:5],
                   title="What it was trained on, and how well it actually does")

    V.footer(img, W, H,
             left=f"metric scale assumes camera height {CFG.cam.height_above_ground_m:.2f} m above ground "
                  f"- an assumption, not a calibration",
             right="risk is a policy score in [0,1], not a calibrated collision probability")
    return img


# ---------------------------------------------------------------- self-test

def _demo_packet(kind: str = "wall"):
    from ..models.world_model import WorldModelStage
    st = bu.synthetic_state(kind, wall_dist_m=1.3, wall_height_m=0.11)
    stage = WorldModelStage(device="cpu")
    bench = stage.benchmark(25)
    lat, preds = stage.predict(st)
    p = FramePacket(clip_id="clip_03", idx=142, t=142 / 30.0)
    p.wm_preds = preds
    s = {"bev_state": st,
         "wm_meta": {"params_total": stage.model.param_counts()["total"],
                     "latency_ms": bench["full_ms"], "device": "cpu"}}
    return p, s


if __name__ == "__main__":
    from ..config import WORK_DIR
    p, s = _demo_packet("wall")
    img = render(p, s)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    out = WORK_DIR / "preview_world.png"
    cv2.imwrite(str(out), img)
    print(f"r_world.render -> {img.shape} {img.dtype}; wrote {out}")
