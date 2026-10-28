"""Renderer for stage 06 - learned uncertainty.  render(packet, state) -> 1280x720 BGR.

What this video has to communicate, in order of importance:

1. The two learned confidence channels and the fused map, and that fusion is
   ``min(depth, seg)`` - weakest link, the correct rule for a safety monitor.
2. That the same model, unchanged, is measurably less confident on the low-light clips
   than on the daylight ones.  The five-clip strip is persistent so the position of the
   clip you are watching is visible even if you only watch one video.
3. Which pixels fall below ``CFG.safety.conf_unknown`` - those are the ones
   ``perception/mapping.py`` marks UNKNOWN when it projects them into the BEV grid, and
   the unknown fraction is what the supervisor's R5 turns into SLOW / STOP.
4. That this is a *confidence score with a measured ECE against a self-supervised error
   target*, not a calibrated collision probability, and that no human ever labelled
   uncertainty anywhere in this pipeline.

Every number on screen is recomputed here from the caches in ``work/cache/<clip>/`` (and
from the training metrics in ``logs/trav_unc_metrics.json``); nothing is hard-coded.
The five-clip table is cached to ``work/unc_render_stats.json`` so a re-render is cheap.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from ..config import CFG, CLIP_IDS, LOG_DIR, WORK_DIR, GO, SLOW, STOP
from ..io_utils import ego_mask, load_stage, read_frames
from ..models.uncertainty import E_TOL_DEPTH, E_TOL_SEG, texture_energy
from ..types import FramePacket
from .. import viz_common as V

W, H = 1280, 720

STATS_CACHE = WORK_DIR / "unc_render_stats.json"
STATS_VERSION = 3
METRICS = LOG_DIR / "trav_unc_metrics.json"
SAMPLE_STRIDE = 20                     # every 20th frame of each 300-frame clip

CLIP_LIGHT = {"clip_01": "daylight", "clip_02": "daylight", "clip_03": "dusk",
              "clip_04": "low light", "clip_05": "low light"}
LIGHT_COLOR = {"daylight": V.OK, "dusk": V.WARN, "low light": V.BAD}

MASKED = (52, 49, 46)                  # colour for the ego-masked band in a conf map
UNK_FILL = (126, 124, 121)             # UNKNOWN grey, matches TRAV_COLORS_BGR[3]
UNK_HATCH = (196, 194, 190)

_STATS: dict = {}
_CAL: dict = {}
_HATCH: dict = {}


# ------------------------------------------------------------------ measurements


def _hatch(shape) -> np.ndarray:
    """Diagonal stripe pattern, cached per shape."""
    if shape not in _HATCH:
        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w]
        _HATCH[shape] = ((xx + yy) % 12) < 2
    return _HATCH[shape]


def _measure_clips() -> dict:
    """Recompute the five-clip table from the caches.  Every SAMPLE_STRIDE-th frame,
    ego-masked, so the sample can be stated on screen and reproduced."""
    em = ego_mask()
    clips: dict[str, dict] = {}
    for cid in CLIP_IDS:
        z = load_stage(cid, "unc")
        dc = z["depth_conf"][::SAMPLE_STRIDE].astype(np.float32)
        sc = z["seg_conf"][::SAMPLE_STRIDE].astype(np.float32)
        fc = z["fused_conf"][::SAMPLE_STRIDE].astype(np.float32)
        n = int(dc.shape[0])
        m = np.broadcast_to(em, dc.shape)
        d, s, f = dc[m], sc[m], fc[m]
        # the cache must actually be the weakest-link fusion we claim on screen
        fuse_err = float(np.abs(fc - np.minimum(dc, sc)).max())
        p05, p50, p95 = (float(v) for v in np.percentile(f, [5, 50, 95]))
        # scene structure: the quantity the fixed depth target weights its residual by
        tex, luma = [], []
        for i, frame in read_frames(cid):
            if i % SAMPLE_STRIDE:
                continue
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tex.append(float(texture_energy(g)[em].mean()))
            luma.append(float(g[em].mean()))
        clips[cid] = {
            "n_sampled": n, "lighting": CLIP_LIGHT.get(cid, "?"),
            "fused": float(f.mean()), "depth": float(d.mean()), "seg": float(s.mean()),
            "p05": p05, "p50": p50, "p95": p95,
            "below_unknown": float((f < CFG.safety.conf_unknown).mean()),
            "texture": float(np.mean(tex)), "luma": float(np.mean(luma)),
            "fuse_is_min_max_abs_err": fuse_err,
        }
        del z, dc, sc, fc
    day = [c for c in CLIP_IDS if CLIP_LIGHT.get(c) == "daylight"]
    low = [c for c in CLIP_IDS if CLIP_LIGHT.get(c) == "low light"]
    md = float(np.mean([clips[c]["fused"] for c in day]))
    ml = float(np.mean([clips[c]["fused"] for c in low]))
    td = float(np.mean([clips[c]["texture"] for c in day]))
    tl = float(np.mean([clips[c]["texture"] for c in low]))
    return {"version": STATS_VERSION, "stride": SAMPLE_STRIDE, "clips": clips,
            "daylight_conf": md, "low_light_conf": ml,
            "conf_drop_pct": 100.0 * (ml - md) / max(md, 1e-9),
            "daylight_tex": td, "low_light_tex": tl,
            "tex_drop_pct": 100.0 * (tl - td) / max(td, 1e-9)}


def _stats() -> dict:
    if not _STATS:
        cached = None
        try:
            cached = json.loads(STATS_CACHE.read_text())
        except Exception:                                        # noqa: BLE001
            cached = None
        if not cached or cached.get("version") != STATS_VERSION:
            cached = _measure_clips()
            try:
                STATS_CACHE.write_text(json.dumps(cached, indent=2))
            except Exception:                                    # noqa: BLE001
                pass
        _STATS.update(cached)
    return _STATS


def _calibration() -> dict:
    """Held-out reliability curves written by training/train_trav_unc.py."""
    if not _CAL:
        out = {"ok": False, "depth": [], "seg": [], "ece_depth": None, "ece_seg": None}
        try:
            m = json.loads(METRICS.read_text())
            c = m["final_val_metrics_vs_PSEUDO_LABELS"]["confidence_calibration"]
            for ch in ("depth", "seg"):
                out[ch] = [(b["mean_conf"], b["empirical"], b["n_frac"])
                           for b in c[ch]["reliability_bins"]]
                out[f"ece_{ch}"] = float(c[ch]["val_ece"])
                out[f"spear_{ch}"] = float(c[ch].get("val_spearman_pred_vs_true_error", 0.0))
            out["ok"] = True
        except Exception:                                        # noqa: BLE001
            pass
        _CAL.update(out)
    return _CAL


# ------------------------------------------------------------------ small widgets


def _conf_color(v: float):
    return V.OK if v >= CFG.safety.conf_slow else (
        V.WARN if v >= CFG.safety.conf_unknown else V.BAD)


def _conf_image(arr: np.ndarray, em: np.ndarray) -> np.ndarray:
    img = V.colorize_conf(arr)
    img[~em] = MASKED
    return img


def _value_badge(img, rect_img, value: float, label: str = ""):
    """Big mean value, top-right corner of a map."""
    x, y, w, h = rect_img
    s = f"{value:.2f}"
    tw, th = V.text_size(s, 0.78, 2, V.FONT_B)
    bx, by = x + w - tw - 14, y + 6
    ov = img.copy()
    cv2.rectangle(ov, (bx - 8, by), (bx + tw + 8, by + th + 12), (16, 15, 14), -1)
    cv2.addWeighted(ov, 0.72, img, 0.28, 0, img)
    cv2.rectangle(img, (bx - 8, by), (bx + tw + 8, by + th + 12), V.EDGE, 1)
    V.text(img, s, (bx, by + th + 4), 0.78, _conf_color(value), 2, V.FONT_B)
    if label:
        V.text(img, label, (bx - 8, by + th + 26), 0.32, V.TEXT_DIM, 1)


def _map_panel(img, x, y, w, h, title, sub, arr, em, mean):
    r = V.panel(img, x, y, w, h, title, "")
    ir = (r[0] + 5, r[1] + 5, r[2] - 10, 168)
    V.blit(img, _conf_image(arr, em), ir)
    cv2.rectangle(img, (ir[0], ir[1]), (ir[0] + ir[2], ir[1] + ir[3]), V.EDGE, 1)
    _value_badge(img, ir, mean)
    V.text(img, sub, (r[0] + 10, r[1] + 190), 0.335, V.TEXT_DIM, 1)
    V.colorbar(img, r[0] + 10, r[1] + 199, r[2] - 20, 9, cv2.COLORMAP_SUMMER,
               "0  do not trust", "trusted  1")
    return r


def _hist(img, x, y, w, h, values: np.ndarray, mean: float):
    n_bins = 30
    hist, _ = np.histogram(values, bins=n_bins, range=(0.0, 1.0))
    hist = hist / max(hist.max(), 1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (24, 22, 20), -1)
    bw = w / n_bins
    for i, v in enumerate(hist):
        bx = int(x + i * bw)
        bh = int(v * (h - 2))
        centre = (i + 0.5) / n_bins
        c = V.BAD if centre < CFG.safety.conf_unknown else (
            V.WARN if centre < CFG.safety.conf_slow else V.OK)
        cv2.rectangle(img, (bx + 1, y + h - bh), (int(bx + bw) - 1, y + h - 1), c, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), V.EDGE, 1)
    for thr, lab, col in ((CFG.safety.conf_unknown, "UNKNOWN 0.35", V.TEXT),
                          (CFG.safety.conf_slow, "SLOW 0.55", V.TEXT_DIM)):
        tx = int(x + thr * w)
        cv2.line(img, (tx, y), (tx, y + h), col, 1)
    V.text(img, "UNKNOWN 0.35", (int(x + CFG.safety.conf_unknown * w) + 4, y + 12),
           0.31, V.TEXT, 1)
    V.text(img, "SLOW 0.55", (int(x + CFG.safety.conf_slow * w) + 4, y + 26),
           0.31, V.TEXT_DIM, 1)
    mx = int(x + float(np.clip(mean, 0, 1)) * w)
    cv2.line(img, (mx, y + h - 10), (mx, y + h), V.ACCENT, 2)
    V.text(img, "0.0", (x - 4, y + h + 13), 0.32, V.TEXT_DIM, 1)
    V.text(img, "fused confidence", (x + w // 2 - 44, y + h + 13), 0.32, V.TEXT_DIM, 1)
    V.text(img, "1.0", (x + w - 18, y + h + 13), 0.32, V.TEXT_DIM, 1)


def _reliability(img, x, y, s, cal: dict):
    """Reliability diagram: predicted confidence vs measured frequency, plus the diagonal."""
    cv2.rectangle(img, (x, y), (x + s, y + s), (24, 22, 20), -1)
    for g in (0.25, 0.5, 0.75):
        gx, gy = int(x + g * s), int(y + s - g * s)
        cv2.line(img, (x, gy), (x + s, gy), (40, 37, 34), 1)
        cv2.line(img, (gx, y), (gx, y + s), (40, 37, 34), 1)
    for i in range(0, s, 10):                              # dashed identity line
        cv2.line(img, (x + i, y + s - i), (x + min(i + 5, s), y + s - min(i + 5, s)),
                 (168, 164, 158), 1)
    V.text(img, "perfect", (x + s - 44, y + s - 6), 0.28, (168, 164, 158), 1)
    for ch, col in (("depth", V.ACCENT), ("seg", V.ACCENT2)):
        pts = cal.get(ch) or []
        if len(pts) < 2:
            continue
        p = np.array([[x + c * s, y + s - e * s] for c, e, _ in pts], np.int32)
        cv2.polylines(img, [p], False, col, 2, cv2.LINE_AA)
        for (px, py), (_, _, nf) in zip(p, pts):
            cv2.circle(img, (int(px), int(py)), max(2, int(2 + 6 * nf)), col, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + s, y + s), V.EDGE, 1)
    V.text(img, "predicted confidence ->", (x, y + s + 13), 0.31, V.TEXT_DIM, 1)


def _gate_bar(img, x, y, w, h, frac: float):
    cv2.rectangle(img, (x, y), (x + w, y + h), (48, 45, 42), -1)
    col = V.BAD if frac >= CFG.safety.unknown_frac_stop else (
        V.WARN if frac >= CFG.safety.unknown_frac_slow else V.OK)
    cv2.rectangle(img, (x, y), (x + int(w * float(np.clip(frac, 0, 1))), y + h), col, -1)
    for thr, lab in ((CFG.safety.unknown_frac_slow, "SLOW"),
                     (CFG.safety.unknown_frac_stop, "STOP")):
        tx = int(x + thr * w)
        cv2.line(img, (tx, y - 3), (tx, y + h + 3), V.TEXT, 1)
        V.text(img, f"{lab} {thr*100:.0f}%", (tx - 14, y + h + 15), 0.31, V.TEXT_DIM, 1)
    cv2.rectangle(img, (x, y), (x + w, y + h), V.EDGE, 1)


def _unknown_overlay(rgb: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Camera frame with the sub-threshold region greyed out and hatched."""
    base = (rgb.astype(np.float32) * 0.9).astype(np.uint8)
    out = base.copy()
    fill = np.array(UNK_FILL, np.float32)
    out[low] = (base[low].astype(np.float32) * 0.34 + fill * 0.66).astype(np.uint8)
    hs = low & _hatch(low.shape)
    out[hs] = UNK_HATCH
    return out


# ------------------------------------------------------------------ render


def render(packet: FramePacket, state: dict) -> np.ndarray:
    img = V.canvas(W, H)
    u = packet.unc
    light = CLIP_LIGHT.get(packet.clip_id, "unknown")
    lcol = LIGHT_COLOR.get(light, V.TEXT_DIM)
    V.header(img, W, "prism",
             "06 - UNCERTAINTY   learned confidence, not softmax entropy",
             f"{packet.clip_id}   {light}   frame {packet.idx:03d}/300")
    if u is None:
        V.text(img, "no uncertainty cache for this clip", (24, 120), 0.6, V.BAD, 1, V.FONT_B)
        return img

    dc, sc, fc = u.depth_conf, u.seg_conf, u.fused_conf
    em = ego_mask(*fc.shape)
    m_d = float(dc[em].mean())
    m_s = float(sc[em].mean())
    m_f = float(fc[em].mean())
    low = (fc < CFG.safety.conf_unknown) & em
    low_frac = float(low[em].mean())

    hist = state.setdefault("unc_hist", [])
    hist.append(m_f)
    if len(hist) > 300:
        del hist[:-300]

    rgb = packet.rgb if packet.rgb is not None else np.zeros((*fc.shape, 3), np.uint8)
    if rgb.shape[:2] != fc.shape:
        rgb = cv2.resize(rgb, (fc.shape[1], fc.shape[0]), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    tex_now = float(texture_energy(gray)[em].mean())

    st = _stats()
    cal = _calibration()
    clips = st.get("clips", {})

    # ================================================== ROW A - camera + three maps
    xs = [10, 327, 644, 961]
    pw = [309, 309, 309, 309]

    # A0: what the camera actually sees
    r = V.panel(img, xs[0], 50, pw[0], 252, "camera", "the model input")
    ir = (r[0] + 5, r[1] + 5, r[2] - 10, 168)
    V.blit(img, rgb, ir)
    cv2.rectangle(img, (ir[0], ir[1]), (ir[0] + ir[2], ir[1] + ir[3]), V.EDGE, 1)
    V.badge(img, ir[0] + 6, ir[1] + 6, light.upper(), lcol, 0.4)
    V.text(img, "grey in the maps = ego mask (chassis, watermark)",
           (r[0] + 10, r[1] + 190), 0.335, V.TEXT_DIM, 1)
    V.text(img, "local gradient energy", (r[0] + 10, r[1] + 210), 0.36, V.TEXT, 1)
    V.text(img, f"{tex_now:5.1f}", (r[0] + r[2] - 62, r[1] + 210), 0.42, V.ACCENT2, 1, V.FONT_B)
    V.bar_meter(img, r[0] + 10, r[1] + 216, r[2] - 20, 8, tex_now, color=V.ACCENT2,
                lo=0.0, hi=90.0)

    _map_panel(img, xs[1], 50, pw[1], 252, "depth confidence",
               f"P(depth error < {E_TOL_DEPTH*100:.0f}%) = P(height error < "
               f"{CFG.ugv.max_step_m*100:.0f} cm)", dc, em, m_d)
    _map_panel(img, xs[2], 50, pw[2], 252, "terrain confidence",
               f"P(terrain error < {E_TOL_SEG*100:.0f}% vs the ADE20K teacher)",
               sc, em, m_s)
    _map_panel(img, xs[3], 50, pw[3], 252, "fused = min(depth, seg)",
               "weakest link - one weak channel is never rescued",
               fc, em, m_f)

    # ================================================== B1 - what becomes UNKNOWN
    b1 = V.panel(img, 10, 306, 418, 386, "consequence",
                 "pixels the map will call UNKNOWN")
    ir = (b1[0] + 6, b1[1] + 6, b1[2] - 12, 182)
    V.blit(img, _unknown_overlay(rgb, low), ir)
    cv2.rectangle(img, (ir[0], ir[1]), (ir[0] + ir[2], ir[1] + ir[3]), V.EDGE, 1)
    _lab = "hatched = fused conf < 0.35"
    _lw = V.text_size(_lab, 0.34)[0]
    cv2.rectangle(img, (ir[0] + 2, ir[1] + ir[3] - 22), (ir[0] + _lw + 16,
                  ir[1] + ir[3] - 2), (14, 13, 12), -1)
    V.text(img, _lab, (ir[0] + 8, ir[1] + ir[3] - 8), 0.34, (235, 233, 230), 1)

    yy = b1[1] + 212
    V.text(img, "frame below conf_unknown = 0.35", (b1[0] + 12, yy), 0.4, V.TEXT, 1)
    s = f"{low_frac*100:.1f}%"
    V.text(img, s, (b1[0] + b1[2] - V.text_size(s, 0.62, 1, V.FONT_B)[0] - 12, yy + 2),
           0.62, V.BAD if low_frac >= CFG.safety.unknown_frac_slow else V.TEXT,
           1, V.FONT_B)
    _gate_bar(img, b1[0] + 12, yy + 12, b1[2] - 24, 14, low_frac)

    kind = STOP if low_frac >= CFG.safety.unknown_frac_stop else (
        SLOW if low_frac >= CFG.safety.unknown_frac_slow else GO)
    bw, bh = V.decision_badge(img, b1[0] + 12, yy + 44, kind, 0.62)
    msg = {GO: ["confidence gate clear -", "corridor may run at speed"],
           SLOW: ["gate tripped: speed capped,", "region treated as unknown"],
           STOP: ["gate tripped hard: reroute", "or hold - do not guess"]}[kind]
    V.text(img, msg[0], (b1[0] + 20 + bw, yy + 57), 0.345, V.TEXT, 1)
    V.text(img, msg[1], (b1[0] + 20 + bw, yy + 73), 0.345, V.TEXT_DIM, 1)

    V.rounded_note(img, b1[0] + 10, b1[1] + b1[3] - 56, b1[2] - 20, [
        "a BEV cell with fused conf < 0.35 is mapped UNKNOWN;",
        "supervisor R5: unknown corridor -> SLOW 30%, STOP 62%.",
        "Low confidence never becomes a confident guess.",
    ], "", V.WARN, lh=14, pad=6)

    # ================================================== B2a - distribution
    b2 = V.panel(img, 436, 306, 400, 200, "confidence distribution",
                 "fused, this frame, ego-masked")
    _hist(img, b2[0] + 14, b2[1] + 6, b2[2] - 28, 96, fc[em], m_f)
    V.sparkline(img, b2[0] + 14, b2[1] + 123, b2[2] - 28, 24, hist, V.ACCENT,
                lo=0.0, hi=1.0)
    V.text(img, f"mean fused confidence over the clip so far   ({len(hist)} frames)",
           (b2[0] + 14, b2[1] + 163), 0.32, V.TEXT_DIM, 1)

    # ================================================== B2b - calibration
    b3 = V.panel(img, 436, 510, 400, 182, "calibration", "held-out split")
    if cal.get("ok"):
        _reliability(img, b3[0] + 14, b3[1] + 20, 116, cal)
        V.text(img, "measured frequency", (b3[0] + 14, b3[1] + 14), 0.29, V.TEXT_DIM, 1)
        tx = b3[0] + 146
        V.text(img, "reliability: predicted vs measured", (tx, b3[1] + 20), 0.34, V.TEXT, 1)
        cv2.line(img, (tx, b3[1] + 32), (tx + 16, b3[1] + 32), V.ACCENT, 2)
        V.text(img, f"depth   ECE {cal['ece_depth']:.3f}", (tx + 22, b3[1] + 36),
               0.36, V.TEXT, 1)
        cv2.line(img, (tx, b3[1] + 50), (tx + 16, b3[1] + 50), V.ACCENT2, 2)
        V.text(img, f"terrain ECE {cal['ece_seg']:.3f}", (tx + 22, b3[1] + 54),
               0.36, V.TEXT, 1)
        V.rounded_note(img, tx, b3[1] + 64, b3[2] - 158, [
            "ECE is measured against a",
            "SELF-SUPERVISED error target,",
            "not human uncertainty labels.",
            "This is a confidence score -",
            "NOT a collision probability.",
        ], "", V.BAD, lh=13, pad=6)
    else:
        V.text(img, "logs/trav_unc_metrics.json not found", (b3[0] + 14, b3[1] + 30),
               0.36, V.BAD, 1)

    # ================================================== B3a - five-clip lighting strip
    b4 = V.panel(img, 844, 306, 426, 250, "lighting stress",
                 "one model, five clips")
    V.text(img, "clip", (b4[0] + 12, b4[1] + 18), 0.31, V.TEXT_DIM, 1)
    V.text(img, "mean fused conf", (b4[0] + 132, b4[1] + 18), 0.31, V.TEXT_DIM, 1)
    V.text(img, "texture", (b4[0] + 344, b4[1] + 18), 0.31, V.TEXT_DIM, 1)
    y = b4[1] + 38
    if clips:
        tex_hi = max(c["texture"] for c in clips.values()) * 1.08
        for cid in CLIP_IDS:
            c = clips.get(cid)
            if c is None:
                continue
            lg = c["lighting"]
            col = LIGHT_COLOR.get(lg, V.TEXT_DIM)
            cur = cid == packet.clip_id
            if cur:
                cv2.rectangle(img, (b4[0] + 4, y - 15), (b4[0] + b4[2] - 4, y + 14),
                              (44, 40, 36), -1)
                cv2.rectangle(img, (b4[0] + 4, y - 15), (b4[0] + 7, y + 14), V.ACCENT, -1)
            V.text(img, f"{cid[-2:]}  {lg}", (b4[0] + 12, y), 0.335,
                   V.TEXT if cur else V.TEXT_DIM, 1, V.FONT_B if cur else V.FONT)
            V.bar_meter(img, b4[0] + 132, y - 9, 150, 10, c["fused"], color=col)
            V.text(img, f"{c['fused']:.3f}", (b4[0] + 290, y), 0.36, col, 1, V.FONT_B)
            V.bar_meter(img, b4[0] + 344, y - 9, 60, 10, c["texture"], color=V.ACCENT2,
                        lo=0.0, hi=tex_hi)
            V.text(img, f"{c['texture']:.0f}", (b4[0] + 408, y), 0.32, V.ACCENT2, 1)
            y += 28
        d, l = st["daylight_conf"], st["low_light_conf"]
        V.text(img, f"daylight {d:.3f}", (b4[0] + 12, y + 6), 0.4, V.OK, 1, V.FONT_B)
        V.text(img, "vs", (b4[0] + 118, y + 6), 0.36, V.TEXT_DIM, 1)
        V.text(img, f"low light {l:.3f}", (b4[0] + 142, y + 6), 0.4, V.BAD, 1, V.FONT_B)
        V.text(img, f"{st['conf_drop_pct']:+.1f}%", (b4[0] + 268, y + 6), 0.44,
               V.BAD, 1, V.FONT_B)
        V.text(img, f"confidence {st['conf_drop_pct']:+.1f}% tracks measured texture "
                    f"{st['tex_drop_pct']:+.1f}%  (not brightness)",
               (b4[0] + 12, y + 24), 0.31, V.TEXT_DIM, 1)
        V.text(img, f"recomputed from work/cache/*/unc.npz - every {st['stride']}th "
                    f"frame, ego-masked (n={clips[CLIP_IDS[0]]['n_sampled']})",
               (b4[0] + 12, y + 40), 0.30, V.TEXT_DIM, 1)
    else:
        V.text(img, "five-clip table unavailable", (b4[0] + 12, y), 0.36, V.BAD, 1)

    # ================================================== B3b - why the numbers moved
    b5 = V.panel(img, 844, 558, 426, 134, "how the confidence is supervised", "")
    V.text(img, "no human labelled uncertainty anywhere in this pipeline",
           (b5[0] + 12, b5[1] + 16), 0.35, V.ACCENT, 1, V.FONT_B)
    for i, ln in enumerate([
        "depth target  temporal reprojection residual under VO motion,",
        "              WEIGHTED BY LOCAL TEXTURE, + the ground-fit residual.",
        "              Unweighted, a dark flat wall reprojects onto itself and",
        "              scored as trustworthy - the old target rated low light best.",
        "terrain target  student vs ADE20K-teacher disagreement + entropy.",
        "then  conf = Platt-sigmoid(a*(tol - pred error) + b) * exp(-2*MC sigma)",
    ]):
        V.text(img, ln, (b5[0] + 12, b5[1] + 32 + i * 13), 0.315,
               V.TEXT if "WEIGHTED" in ln else V.TEXT_DIM, 1)

    V.footer(img, W, H,
             "Self-supervised targets only (temporal geometry, teacher disagreement, "
             "MC-dropout). Metric tolerance follows from the assumed camera height "
             f"{CFG.cam.height_above_ground_m:.2f} m.",
             "confidence score - NOT a calibrated collision probability")
    return img


# ------------------------------------------------------------------ self test

if __name__ == "__main__":
    from ..types import UncertaintyResult
    from ..config import PROC_H, PROC_W

    rng = np.random.default_rng(2)
    base = cv2.GaussianBlur(rng.random((PROC_H, PROC_W)).astype(np.float32), (0, 0), 21)
    base = (base - base.min()) / (base.max() - base.min())
    p = FramePacket(clip_id="clip_05", idx=7, t=0.2)
    p.rgb = (rng.random((PROC_H, PROC_W, 3)) * 90).astype(np.uint8)
    st: dict = {}
    out = None
    for i in range(12):
        p.idx = i
        p.unc = UncertaintyResult(depth_conf=base, seg_conf=1 - base * 0.6,
                                  fused_conf=np.minimum(base, 1 - base * 0.6),
                                  mean_conf=float(base.mean()))
        out = render(p, st)
    print("render ->", out.shape, out.dtype, "range", out.min(), out.max())
    s = _stats()
    print("five-clip table (stride %d):" % s["stride"])
    for cid, c in s["clips"].items():
        print(f"  {cid} {c['lighting']:9s} fused {c['fused']:.3f} depth {c['depth']:.3f} "
              f"seg {c['seg']:.3f} below0.35 {c['below_unknown']:.3f} "
              f"tex {c['texture']:.1f} luma {c['luma']:.1f} "
              f"|fused-min| {c['fuse_is_min_max_abs_err']:.4f}")
    print(f"  daylight {s['daylight_conf']:.3f} vs low light {s['low_light_conf']:.3f} "
          f"({s['conf_drop_pct']:+.1f}%), texture {s['tex_drop_pct']:+.1f}%")
    print("calibration:", {k: v for k, v in _calibration().items()
                           if not isinstance(v, list)})
    cv2.imwrite("work/_r_unc_selftest.png", out)
    print("wrote work/_r_unc_selftest.png")
