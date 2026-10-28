"""Stage 11 renderer: the full synchronized prism dashboard.

Rendered at 1920x1080 rather than the 1280x720 used by the single-stage videos - twelve
live panels do not stay legible at 720p. Every panel degrades gracefully: if a stage's
cache is missing the panel says so instead of inventing data.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import cv2

from ..config import (CFG, TERRAIN_CLASSES, TERRAIN_COLORS_BGR, TRAV_CLASSES,
                      TRAV_COLORS_BGR, DECISIONS, DECISION_COLORS_BGR, ACTIONS,
                      SAFE, RISKY, OBSTACLE, UNKNOWN, GO, SLOW, REROUTE, STOP)
from ..types import FramePacket
from .. import viz_common as V
from ..perception.geometry import (unproject, to_vehicle, GroundFit,
                                   height_slope_roughness, veh_to_bev_px)

W, H = 1920, 1080


# --------------------------------------------------------------------- helpers

def _missing(img, rect, what: str):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (26, 24, 22), -1)
    msg = f"{what} unavailable"
    V.text(img, msg, (x + w // 2 - V.text_size(msg, 0.42)[0] // 2, y + h // 2), 0.42, V.TEXT_DIM)
    V.text(img, "cache not built", (x + w // 2 - V.text_size("cache not built", 0.34)[0] // 2,
                                    y + h // 2 + 18), 0.34, (110, 106, 100))


def _ground_fit(packet: FramePacket) -> Optional[GroundFit]:
    d = packet.depth
    if d is None:
        return None
    n = getattr(d, "_normal", None)
    if n is None:
        n = np.array([0.0, 1.0, 0.0], np.float32)
    return GroundFit(a=d.scale, b=d.shift, normal=np.asarray(n, np.float32),
                     height=CFG.cam.height_above_ground_m)


def _elevation(packet: FramePacket):
    d = packet.depth
    fit = _ground_fit(packet)
    valid = d.valid if d.valid is not None else np.isfinite(d.depth_m)
    pv = to_vehicle(unproject(np.nan_to_num(d.depth_m, nan=0.0)), fit)
    z, _, _ = height_slope_roughness(pv, valid)
    fwd = pv[..., 1]
    in_range = valid & np.isfinite(fwd) & (fwd > 0.05) & (fwd < CFG.bev.range_forward_m)
    return z, in_range


# --------------------------------------------------------------------- BEV hero

def _draw_bev(img, rect, packet: FramePacket, mode: str = "trav"):
    """The 2.5-D coloured-cell local map: colour = risk or height, dy = obstacle height."""
    x, y, w, h = rect
    bev = packet.bev
    cv2.rectangle(img, (x, y), (x + w, y + h), (14, 13, 12), -1)
    if bev is None:
        _missing(img, rect, "local 2.5D map")
        return

    gh, gw = bev.trav.shape
    observed = bev.conf > 0.02
    if mode == "height":
        hh = np.nan_to_num(bev.height, nan=0.0)
        colors = V.colorize_height(hh, -0.35, 0.65)
    else:
        colors = V.colorize_trav(bev.trav)
    # confidence modulates brightness so uncertain cells visibly recede
    conf = np.clip(bev.conf, 0.0, 1.0)[..., None]
    colors = (colors.astype(np.float32) * (0.30 + 0.70 * conf)).astype(np.uint8)

    # range rings and centreline first, so cells draw over them
    for rng in (1, 2, 3, 4, 5, 6, 7):
        if rng > CFG.bev.range_forward_m:
            break
        pts = []
        for ang in np.linspace(-np.pi / 2, np.pi / 2, 60):
            px, py = veh_to_bev_px(rng * np.sin(ang), rng * np.cos(ang), rect)
            pts.append((px, py))
        cv2.polylines(img, [np.array(pts, np.int32)], False, (46, 43, 40), 1, cv2.LINE_AA)
        px, py = veh_to_bev_px(0.0, float(rng), rect)
        V.text(img, f"{rng}m", (px + 5, py - 4), 0.32, (96, 92, 86), 1, shadow=False)
    px0, py0 = veh_to_bev_px(0.0, 0.0, rect)
    px1, py1 = veh_to_bev_px(0.0, CFG.bev.range_forward_m, rect)
    cv2.line(img, (px0, py0), (px1, py1), (46, 43, 40), 1, cv2.LINE_AA)

    elev_px = (h / gh) * 2.6
    V.draw_grid_dots(img, rect, colors, observed,
                     elev=np.nan_to_num(bev.height, nan=0.0), dot=2, elev_px=elev_px)

    # unobserved cells are drawn explicitly - never left blank, never filled in as free
    ys, xs = np.nonzero(~observed)
    cw, ch = w / gw, h / gh
    for gy, gx in zip(ys[::13], xs[::13]):
        cv2.circle(img, (int(x + (gx + 0.5) * cw), int(y + (gy + 0.5) * ch)), 1, (46, 44, 41), -1)

    # candidate trajectories: rejected greyed, chosen highlighted
    chosen = None
    for tr in packet.trajectories or []:
        pts = [veh_to_bev_px(float(a), float(b), rect) for a, b in tr.xy]
        col = (78, 74, 70) if not tr.feasible else (150, 190, 120)
        cv2.polylines(img, [np.array(pts, np.int32)], False, col, 1, cv2.LINE_AA)
        if packet.decision is not None and tr.action == packet.decision.action and tr.feasible:
            chosen = (pts, tr)
    if chosen is not None:
        pts, tr = chosen
        kind = packet.decision.kind if packet.decision else GO
        cv2.polylines(img, [np.array(pts, np.int32)], False, DECISION_COLORS_BGR[kind], 3, cv2.LINE_AA)
        # swept corridor edges
        half = CFG.ugv.width_m / 2 + CFG.safety.corridor_margin_m
        for sgn in (-1, 1):
            e = [veh_to_bev_px(float(a) + sgn * half, float(b), rect) for a, b in tr.xy]
            cv2.polylines(img, [np.array(e, np.int32)], False, (90, 110, 90), 1, cv2.LINE_AA)

    # vehicle footprint drawn to scale
    hw, lf = CFG.ugv.width_m / 2, CFG.ugv.length_m
    quad = [veh_to_bev_px(-hw, 0.0, rect), veh_to_bev_px(hw, 0.0, rect),
            veh_to_bev_px(hw, -lf * 0.0 + 0.0, rect), veh_to_bev_px(-hw, 0.0, rect)]
    p_l = veh_to_bev_px(-hw, 0.0, rect)
    p_r = veh_to_bev_px(hw, 0.0, rect)
    p_f = veh_to_bev_px(0.0, lf * 0.55, rect)
    cv2.fillPoly(img, [np.array([p_l, p_r, p_f], np.int32)], (200, 220, 240))
    cv2.polylines(img, [np.array([p_l, p_r, p_f], np.int32)], True, (30, 30, 30), 1, cv2.LINE_AA)

    stats = [("safe", float((bev.trav == SAFE).mean()), TRAV_COLORS_BGR[SAFE]),
             ("risky", float((bev.trav == RISKY).mean()), TRAV_COLORS_BGR[RISKY]),
             ("obstacle", float((bev.trav == OBSTACLE).mean()), TRAV_COLORS_BGR[OBSTACLE]),
             ("unknown", float((bev.trav == UNKNOWN).mean()), TRAV_COLORS_BGR[UNKNOWN])]
    bx, by = x + 10, y + h - 14
    cv2.rectangle(img, (x, by - 16), (x + w, y + h), (16, 15, 14), -1)
    for i, (nm, fr, col) in enumerate(stats):
        cv2.rectangle(img, (bx + i * 118, by - 9), (bx + i * 118 + 9, by), tuple(int(c) for c in col), -1)
        V.text(img, f"{nm} {fr*100:4.1f}%", (bx + i * 118 + 14, by), 0.34, V.TEXT)
    V.text(img, f"{CFG.bev.res_m*100:.0f} cm cells  |  {CFG.bev.range_forward_m:.1f} m x "
                f"{2*CFG.bev.range_lateral_m:.1f} m", (x + 10, y + 18), 0.32, (124, 120, 114))
    V.text(img, f"scale from the assumed {CFG.cam.height_above_ground_m:.2f} m camera height",
           (x + 10, y + 34), 0.30, (108, 104, 99))


# --------------------------------------------------------------------- trajectory

def _draw_trajectory(img, rect, packet: FramePacket, state: dict):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (18, 17, 16), -1)
    od = packet.odom
    if od is None:
        _missing(img, rect, "visual odometry")
        return
    traj = od.trajectory
    if traj is None or len(traj) < 2:
        traj = np.array([[0.0, 0.0], [od.pose.x, od.pose.y]], np.float32)
    t = np.asarray(traj, np.float32)
    lo = t.min(0) - 0.5
    hi = t.max(0) + 0.5
    span = np.maximum(hi - lo, 1.0)
    s = min((w - 40) / span[0], (h - 60) / span[1])
    cx, cy = x + w / 2, y + h / 2 + 6
    mid = (lo + hi) / 2

    def to_px(p):
        return (int(cx + (p[0] - mid[0]) * s), int(cy - (p[1] - mid[1]) * s))

    for gm in np.arange(np.floor(lo[0]), np.ceil(hi[0]) + 1, 1.0):
        a, b = to_px((gm, lo[1])), to_px((gm, hi[1]))
        cv2.line(img, a, b, (38, 36, 33), 1)
    for gm in np.arange(np.floor(lo[1]), np.ceil(hi[1]) + 1, 1.0):
        a, b = to_px((lo[0], gm)), to_px((hi[0], gm))
        cv2.line(img, a, b, (38, 36, 33), 1)

    pts = np.array([to_px(p) for p in t], np.int32)
    cv2.polylines(img, [pts], False, V.ACCENT, 2, cv2.LINE_AA)
    cur = to_px(t[-1])
    yaw = od.pose.yaw
    tip = (int(cur[0] + 22 * np.sin(yaw)), int(cur[1] - 22 * np.cos(yaw)))
    cv2.arrowedLine(img, cur, tip, (240, 240, 240), 2, cv2.LINE_AA, tipLength=0.4)
    cv2.circle(img, cur, 5, (255, 255, 255), -1)

    ok = od.tracking_ok
    V.badge(img, x + 10, y + 8, "TRACKING OK" if ok else "TRACKING LOST",
            V.OK if ok else V.BAD, 0.36)
    V.text(img, f"{od.speed_mps:4.2f} m/s   yaw {np.degrees(yaw):+6.1f} deg",
           (x + 10, y + h - 30), 0.38, V.TEXT)
    V.text(img, f"inliers {od.n_inliers:3d}/{od.n_matches:3d}   path {float(np.abs(np.diff(t,axis=0)).sum()):4.2f} m",
           (x + 10, y + h - 12), 0.36, V.TEXT_DIM)
    if not ok:
        V.text(img, "goal pursuit suspended", (x + w - 190, y + h - 12), 0.36, V.BAD)


# --------------------------------------------------------------------- world model

def _draw_world_model(img, rect, packet: FramePacket):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (18, 17, 16), -1)
    preds = packet.wm_preds or []
    if not preds:
        _missing(img, rect, "world model")
        return
    V.text(img, "predicted obstacle occupancy, +0.2 s .. +1.2 s", (x + 8, y + 16), 0.34, V.TEXT_DIM)
    show = preds[:3]
    tile = min((w - 24) // 6, 52)
    for r, pr in enumerate(show):
        yy = y + 26 + r * (tile + 26)
        V.text(img, ACTIONS[pr.action], (x + 8, yy + 12), 0.36, V.ACCENT2, 1, V.FONT_B)
        for k in range(min(6, len(pr.occ_forecast))):
            o = np.clip(pr.occ_forecast[k], 0, 1)
            im = cv2.applyColorMap((o * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            im = cv2.resize(im, (tile, tile), interpolation=cv2.INTER_NEAREST)
            ox = x + 70 + k * (tile + 4)
            if ox + tile > x + w - 8:
                break
            img[yy:yy + tile, ox:ox + tile] = im
            cv2.rectangle(img, (ox, yy), (ox + tile, yy + tile), (60, 56, 52), 1)
        rk = float(pr.risk_total)
        V.bar_meter(img, x + 70, yy + tile + 4, w - 90, 7, rk,
                    color=V.OK, warn_at=CFG.safety.risk_slow)
        V.text(img, f"risk {rk:.2f}", (x + w - 78, yy + tile + 12), 0.33, V.TEXT_DIM)


# --------------------------------------------------------------------- decision

def _draw_decision(img, rect, packet: FramePacket):
    x, y, w, h = rect
    dec = packet.decision
    cv2.rectangle(img, (x, y), (x + w, y + h), (22, 20, 19), -1)
    if dec is None:
        _missing(img, rect, "planner / supervisor")
        return
    col = DECISION_COLORS_BGR[dec.kind]
    cv2.rectangle(img, (x, y), (x + w, y + 76), col, -1)
    label = DECISIONS[dec.kind]
    V.text(img, label, (x + w // 2 - V.text_size(label, 1.5, 2, V.FONT_B)[0] // 2, y + 56),
           1.5, (18, 18, 18), 2, V.FONT_B, shadow=False)

    yy = y + 100
    V.text(img, "COMMANDED", (x + 12, yy), 0.34, V.TEXT_DIM)
    V.text(img, f"{ACTIONS[dec.action]}   {dec.speed_mps:4.2f} m/s", (x + 12, yy + 20), 0.46, V.TEXT, 1, V.FONT_B)
    src = dec.policy_source
    V.badge(img, x + w - 108, yy - 12, "RL" if src.startswith("rl") else "SUPERVISOR",
            V.ACCENT2 if src.startswith("rl") else V.WARN, 0.32)
    if "override" in src or (src.startswith("rl") and "supervisor" in src):
        V.text(img, "supervisor overrode the policy", (x + 12, yy + 40), 0.34, V.BAD)

    yy += 58
    # R1 (odometry lost) and R2 (perception unusable) trip the gate before the corridor
    # is ever scored, so the stored risk/confidence/unknown are sentinels. Drawing them
    # as meters would claim confidence 1.00 on a frame stopped *for* low confidence.
    rule = getattr(dec, "rule", "")
    short_circuit = rule in ("R1", "R2")
    if short_circuit:
        cv2.rectangle(img, (x + 12, yy - 4), (x + w - 12, yy + 74), (30, 28, 26), -1)
        cv2.rectangle(img, (x + 12, yy - 4), (x + w - 12, yy + 74), V.EDGE, 1)
        V.text(img, "CORRIDOR NOT SCORED", (x + 22, yy + 16), 0.38, V.WARN, 1, V.FONT_B)
        for i, ln in enumerate([
                "the gate tripped before any candidate",
                "arc was evaluated, so risk, confidence",
                "and unknown are not measured here."]):
            V.text(img, ln, (x + 22, yy + 34 + i * 14), 0.33, V.TEXT_DIM)
        yy += 84
        if packet.unc is not None:
            V.text(img, "measured perception confidence", (x + 12, yy), 0.34, V.TEXT_DIM)
            V.bar_meter(img, x + 12, yy + 6, w - 90, 9, float(packet.unc.mean_conf), color=V.OK)
            V.text(img, f"{packet.unc.mean_conf:.2f}", (x + w - 62, yy + 15), 0.36, V.TEXT)
            yy += 38
    else:
        for nm, val, warn in (("collision risk", dec.risk, CFG.safety.risk_slow),
                              ("unknown ahead", dec.unknown_frac, CFG.safety.unknown_frac_slow)):
            V.text(img, nm, (x + 12, yy), 0.34, V.TEXT_DIM)
            V.bar_meter(img, x + 12, yy + 6, w - 90, 9, float(val), warn_at=warn)
            V.text(img, f"{val:.2f}", (x + w - 62, yy + 15), 0.36, V.TEXT)
            yy += 34
        V.text(img, "confidence", (x + 12, yy), 0.34, V.TEXT_DIM)
        V.bar_meter(img, x + 12, yy + 6, w - 90, 9, float(dec.confidence), color=V.OK)
        V.text(img, f"{dec.confidence:.2f}", (x + w - 62, yy + 15), 0.36, V.TEXT)
        yy += 38

    V.text(img, "REASON", (x + 12, yy), 0.34, V.TEXT_DIM)
    words, line, lines = (dec.reason or "-").split(), "", []
    for wd in words:
        t = (line + " " + wd).strip()
        if V.text_size(t, 0.35)[0] > w - 26:
            lines.append(line)
            line = wd
        else:
            line = t
    lines.append(line)
    for i, ln in enumerate(lines[:5]):
        V.text(img, ln, (x + 12, yy + 18 + i * 15), 0.35, col if i == 0 else V.TEXT)


# --------------------------------------------------------------------- main

def render(packet: FramePacket, state: dict) -> np.ndarray:
    img = V.canvas(W, H)
    clip_scene = state.setdefault("scene", "")
    if not clip_scene:
        from ..io_utils import clips_meta
        for m in clips_meta():
            if m["clip_id"] == packet.clip_id:
                clip_scene = state["scene"] = f"{m['scene']}  ({m['lighting']})"

    V.header(img, W, "prism",
             "CAMERA-ONLY UGV NAVIGATION  |  full pipeline, synchronized",
             right=f"{packet.clip_id}   frame {packet.idx:03d}   t={packet.t:4.1f}s", h=52)

    # ---------------------------------------------------------- row A
    r = V.panel(img, 16, 60, 460, 300, "camera", "the only runtime sensor")
    V.blit(img, packet.rgb, r)
    cv2.rectangle(img, (r[0], r[1] + r[3] - 20), (r[0] + r[2], r[1] + r[3]), (20, 19, 18), -1)
    V.text(img, clip_scene[:70], (r[0] + 8, r[1] + r[3] - 6), 0.32, (156, 152, 146), 1, shadow=False)

    r = V.panel(img, 488, 60, 460, 300, "terrain semantics", "PIDNet-S, prism-7")
    if packet.seg is not None:
        ov = V.overlay(packet.rgb, V.colorize_terrain(packet.seg.label), 0.55)
        V.blit(img, ov, r)
        ly = r[1] + r[3] - 16
        cv2.rectangle(img, (r[0], ly - 14), (r[0] + r[2], r[1] + r[3]), (20, 19, 18), -1)
        for i, nm in enumerate(TERRAIN_CLASSES):
            lx = r[0] + 8 + i * 65
            c = tuple(int(v) for v in TERRAIN_COLORS_BGR[i])
            cv2.rectangle(img, (lx, ly - 9), (lx + 9, ly), c, -1)
            V.text(img, nm, (lx + 12, ly), 0.30, V.TEXT, 1, shadow=False)
    else:
        _missing(img, r, "terrain segmentation")

    r = V.panel(img, 960, 60, 460, 300, "height above ground", "metric, vehicle frame")
    if packet.depth is not None:
        z, in_range = _elevation(packet)
        V.blit(img, V.colorize_height(np.nan_to_num(z, nan=0.0), -0.55, 0.55, invalid=~in_range), r)
        V.colorbar(img, r[0] + 10, r[1] + r[3] - 22, 160, 8, cv2.COLORMAP_JET, "below", "above")
    else:
        _missing(img, r, "depth")

    r = V.panel(img, 1432, 60, 472, 300, "confidence", "low confidence -> slow / reroute")
    if packet.unc is not None:
        V.blit(img, V.colorize_conf(packet.unc.fused_conf), r)
        lowf = float((packet.unc.fused_conf < CFG.safety.conf_unknown).mean())
        V.text(img, f"mean {packet.unc.mean_conf:.2f}    below threshold {lowf*100:4.1f}%",
               (r[0] + 10, r[1] + r[3] - 10), 0.36, V.TEXT)
    else:
        _missing(img, r, "uncertainty")

    # ---------------------------------------------------------- row B
    r = V.panel(img, 16, 372, 620, 470, "local 2.5D map",
                "colour = traversability, height = obstacle elevation")
    _draw_bev(img, r, packet, "trav")
    lx, ly = r[0] + r[2] - 104, r[1] + 12
    cv2.rectangle(img, (lx - 8, ly - 6), (r[0] + r[2] - 4, ly + 4 * 15), (22, 21, 20), -1)
    cv2.rectangle(img, (lx - 8, ly - 6), (r[0] + r[2] - 4, ly + 4 * 15), V.EDGE, 1)
    for i, nm in enumerate(TRAV_CLASSES):
        c = tuple(int(v) for v in TRAV_COLORS_BGR[i])
        cv2.rectangle(img, (lx, ly + i * 15 + 2), (lx + 9, ly + i * 15 + 11), c, -1)
        V.text(img, nm, (lx + 14, ly + i * 15 + 11), 0.31, V.TEXT, 1, shadow=False)

    r = V.panel(img, 648, 372, 430, 470, "pose and trajectory", "GPS-free, depth-anchored VO")
    _draw_trajectory(img, r, packet, state)

    r = V.panel(img, 1090, 372, 440, 470, "world model forecast", "what happens if we take this path")
    _draw_world_model(img, r, packet)

    r = V.panel(img, 1542, 372, 362, 470, "decision", "supervised command")
    _draw_decision(img, r, packet)

    # ---------------------------------------------------------- row C telemetry
    r = V.panel(img, 16, 854, 1888, 180, "telemetry", "rolling history and per-stage cost")
    dec = packet.decision
    hr = state.setdefault("hist_risk", [])
    hc = state.setdefault("hist_conf", [])
    hu = state.setdefault("hist_unk", [])
    hd = state.setdefault("hist_dec", [])
    _sc = dec is not None and getattr(dec, "rule", "") in ("R1", "R2")
    hr.append(float(dec.risk) if (dec and not _sc) else (hr[-1] if hr else 0.0))
    hc.append(float(packet.unc.mean_conf) if packet.unc else
              (float(dec.confidence) if (dec and not _sc) else (hc[-1] if hc else 1.0)))
    hu.append(float(dec.unknown_frac) if (dec and not _sc) else (hu[-1] if hu else 0.0))
    hd.append(int(dec.kind) if dec else 0)

    cx = r[0] + 14
    for title, series, col, hi in (("PREDICTED COLLISION RISK", hr, V.BAD, 1.0),
                                   ("PERCEPTION CONFIDENCE", hc, V.OK, 1.0),
                                   ("UNKNOWN FRACTION AHEAD", hu, V.WARN, 1.0)):
        V.text(img, title, (cx, r[1] + 18), 0.34, V.TEXT_DIM)
        V.sparkline(img, cx, r[1] + 24, 240, 56, series[-260:], col, lo=0, hi=hi)
        V.text(img, f"{series[-1]:.2f}", (cx + 202, r[1] + 96), 0.38, col, 1, V.FONT_B)
        cx += 262

    # decision timeline: one column per elapsed frame
    V.text(img, "DECISION TIMELINE", (cx, r[1] + 18), 0.34, V.TEXT_DIM)
    tw = 240
    cv2.rectangle(img, (cx, r[1] + 24), (cx + tw, r[1] + 80), (26, 24, 22), -1)
    n = len(hd)
    for i, k in enumerate(hd[-260:]):
        px = cx + int(i / max(min(n, 260), 1) * tw)
        cv2.line(img, (px, r[1] + 25), (px, r[1] + 79), DECISION_COLORS_BGR[k], 2)
    cv2.rectangle(img, (cx, r[1] + 24), (cx + tw, r[1] + 80), V.EDGE, 1)
    for i, nm in enumerate(DECISIONS):
        frac = float(np.mean(np.asarray(hd) == i)) if hd else 0.0
        short = ("GO", "SLOW", "RRT", "STOP")[i]
        V.text(img, f"{short} {frac*100:3.0f}%", (cx + i * 60, r[1] + 96), 0.30, DECISION_COLORS_BGR[i])
    cx += 262

    V.text(img, "STAGE COST (ms, RTX 4050 FP16)", (cx, r[1] + 18), 0.34, V.TEXT_DIM)
    order = [("depth", "depth"), ("seg", "terrain"), ("trav", "traversability"),
             ("unc", "uncertainty"), ("odom", "odometry"), ("bev", "mapping"),
             ("plan", "plan+world model")]
    yy = r[1] + 30
    tot = 0.0
    for key, nm in order:
        ms = float(packet.timings_ms.get(key, 0.0))
        tot += ms
        V.text(img, nm, (cx, yy), 0.32, V.TEXT_DIM)
        V.bar_meter(img, cx + 108, yy - 8, 152, 8, min(ms, 120.0), lo=0, hi=120.0, color=V.ACCENT)
        V.text(img, f"{ms:5.1f}", (cx + 268, yy), 0.32, V.TEXT)
        yy += 17
    V.text(img, f"total {tot:5.1f} ms", (cx, yy + 4), 0.36, V.ACCENT, 1, V.FONT_B)

    cx += 322
    V.rounded_note(img, cx, r[1] + 10, r[0] + r[2] - cx - 12, [
        "A detector asks: what object is this?",
        "prism asks: can this vehicle drive here, how high is the terrain,",
        "how confident are we, and what happens if we take this path?",
        "",
        f"Height above ground over {CFG.ugv.clearance_m*100:.1f} cm clearance  ->  non-traversable.",
        f"Confidence under {CFG.safety.conf_unknown:.2f}  ->  cell is unknown, slow down or reroute.",
        f"Predicted collision risk over {CFG.safety.risk_stop:.2f}  ->  trajectory rejected.",
    ], title="WHAT MAKES THIS DIFFERENT FROM A YOLO-STYLE DETECTOR", lh=16)

    V.footer(img, W, H,
             left="single RGB camera at runtime - no GPS, no LiDAR, no stereo, no IMU   |   "
                  "metric scale from an assumed 0.12 m camera height   |   confidence is not a calibrated collision probability",
             right="offline video perception - not physical autonomy")
    return img
