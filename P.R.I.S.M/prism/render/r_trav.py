"""Renderer for stage 03 - traversability / risk.  render(packet, state) -> 1280x720 BGR.

The panel has one job beyond looking good: make it obvious that the answer is
*vehicle-specific*, and that it is anchored to geometry rather than to a class name.

Everything numeric on this panel is measured here, from the caches, at render time:

* ``height above ground`` is recomputed in this file from ``depth.npz`` (metric depth +
  the per-frame fitted ground normal).  It is not read back from the traversability
  head's output - but it is **not independent evidence** either, and the panel says so:
  a locally-referenced version of the same quantity is input channel 4 of the head and
  seeds the pseudo-labels it was trained on (see ``models/traversability.py``).  What
  the correlation shows is that the head reproduces its geometric supervision coherently
  - including across pixels where the depth fit drops out - not that the supervision is
  correct.  Calling it "validation against ground truth" would be false.
* the ``Spearman rho`` and the per-class height quantiles accumulate over a random
  pixel sample drawn from every frame of *this clip* so far, in the 0.20-6.00 m band.
  The sample size and frame span are printed with them.
* the footprint-corridor step is the p95 height above the fitted ground plane inside the
  patch of ground the chassis would actually drive over on the frame on screen.

Nothing here is a stored constant dressed up as a measurement; if a quantity cannot be
computed for a frame the panel says so instead of substituting a number.
"""
from __future__ import annotations

import numpy as np
import cv2

from ..config import (CFG, TRAV_CLASSES, TRAV_COLORS_BGR,
                      SAFE, RISKY, OBSTACLE, UNKNOWN)
from ..io_utils import ego_mask
from ..perception import geometry as G
from ..types import FramePacket
from .. import viz_common as V

W, H = 1280, 720
CLIP_SCENE = {
    "clip_01": "daylight gravel trail", "clip_02": "daylight gravel path",
    "clip_03": "dusk earth path", "clip_04": "low-light service yard",
    "clip_05": "low-light path, pedestrian",
}

# Footprint corridor: the strip of ground the chassis sweeps driving straight on.
CORR_HALF_W = CFG.ugv.width_m / 2.0 + CFG.safety.corridor_margin_m   # 0.16 m
CORR_Y0, CORR_Y1 = 0.25, 1.50                                        # m ahead

# Band over which monocular metric depth is worth quoting for the cross-check.
BAND_M = (0.20, 6.00)
XC_PER_FRAME = 1200          # pixels sampled per frame into the cross-check pool
XC_CAP = 260_000             # pool cap; older samples are thinned, never re-weighted
XC_EVERY = 10                # frames between statistic recomputes

RULER_MAX_M = 1.50           # x-axis extent of the height ruler, sqrt-scaled
RULER_TICKS = (0.0, 0.01, 0.05, 0.15, 0.50, 1.50)
HEIGHT_SHOW_MAX_M = 10.0     # beyond this the metric height is not worth colouring


# --------------------------------------------------------------------- geometry


def _frame_geometry(packet: FramePacket) -> dict | None:
    """Vehicle-frame coordinates recomputed from the cached metric depth.

    Uses the per-frame ground normal that `perception.geometry.fit_metric_ground`
    stored alongside the depth, so `hag` is "height above the single ground plane the
    depth stage fitted", in metres.  Note this is the *plane*-referenced height, not the
    locally-referenced `local_terrain_fields` height the traversability head consumes;
    the two agree near the vehicle and drift by a few centimetres at 6 m.
    """
    d = packet.depth
    if d is None or d.depth_m is None:
        return None
    depth = np.asarray(d.depth_m, np.float32)
    if depth.ndim != 2:
        return None
    h, w = depth.shape

    n = getattr(d, "_normal", None)
    n = None if n is None else np.asarray(n, np.float32).reshape(-1)
    ok_n = n is not None and n.size == 3 and np.isfinite(n).all() and abs(np.linalg.norm(n) - 1.0) < 0.25
    if not ok_n:
        return None
    n = n / (np.linalg.norm(n) + 1e-9)

    R = G.vehicle_basis(n)
    pv = (G.unproject(depth).reshape(-1, 3) @ R.T)
    pv[:, 2] += CFG.cam.height_above_ground_m
    pv = pv.reshape(h, w, 3)

    valid = np.isfinite(depth) & (depth > 0.15)
    if d.valid is not None:
        dv = np.asarray(d.valid, bool)
        if dv.shape == valid.shape:
            valid &= dv
    valid &= ego_mask(h, w)
    valid &= np.isfinite(pv[..., 2])
    return {"hag": pv[..., 2], "vx": pv[..., 0], "vy": pv[..., 1],
            "depth": depth, "valid": valid, "R": R}


def _corridor_mask(g: dict) -> np.ndarray:
    return (g["valid"] & (np.abs(g["vx"]) <= CORR_HALF_W)
            & (g["vy"] > CORR_Y0) & (g["vy"] < CORR_Y1))


def _footprint_poly(R: np.ndarray, shape: tuple[int, int]) -> np.ndarray | None:
    """Ground-plane footprint corridor projected back into the image, as a polygon."""
    h, w = shape
    K = CFG.cam.K
    hc = CFG.cam.height_above_ground_m
    corners = np.array([[-CORR_HALF_W, CORR_Y0, 0.0], [CORR_HALF_W, CORR_Y0, 0.0],
                        [CORR_HALF_W, CORR_Y1, 0.0], [-CORR_HALF_W, CORR_Y1, 0.0]], np.float64)
    corners[:, 2] -= hc                       # vehicle origin (ground) -> camera origin
    cam = corners @ np.asarray(R, np.float64)  # inverse of  pv = P @ R.T
    if np.any(cam[:, 2] < 0.05):
        return None
    u = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
    v = K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]
    if not (np.isfinite(u).all() and np.isfinite(v).all()):
        return None
    if np.abs(u).max() > 6 * w or np.abs(v).max() > 6 * h:
        return None
    return np.stack([u, v], 1).astype(np.int32)


def _draw_poly(img: np.ndarray, poly: np.ndarray | None, color, label: str = "") -> None:
    if poly is None:
        return
    cv2.polylines(img, [poly], True, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.polylines(img, [poly], True, color, 1, cv2.LINE_AA)
    if label:
        x, y = int(poly[:, 0].mean()), int(poly[:, 1].min()) - 5
        V.text(img, label, (max(4, x - V.text_size(label, 0.34)[0] // 2), max(11, y)),
               0.34, color, 1)


# --------------------------------------------------------------------- cross-check pool


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks with ties - Spearman needs ties handled (risk is float16)."""
    try:
        from scipy.stats import rankdata
        return rankdata(a)
    except Exception:                                     # pragma: no cover
        o = np.argsort(a, kind="stable")
        r = np.empty(a.size, np.float64)
        r[o] = np.arange(1, a.size + 1, dtype=np.float64)
        return r


def _xc_update(state: dict, g: dict, risk: np.ndarray, lab: np.ndarray, idx: int) -> dict | None:
    """Grow the per-clip (height, risk, class) sample and refresh its statistics."""
    pool = state.setdefault("_xc", {"h": [], "r": [], "l": [], "n": 0, "f0": idx, "f1": idx})
    rng = state.setdefault("_xc_rng", np.random.default_rng(1337))

    m = g["valid"] & (g["depth"] > BAND_M[0]) & (g["depth"] < BAND_M[1])
    sel = np.flatnonzero(m.ravel())
    if sel.size:
        if sel.size > XC_PER_FRAME:
            sel = rng.choice(sel, XC_PER_FRAME, replace=False)
        pool["h"].append(g["hag"].ravel()[sel].astype(np.float32))
        pool["r"].append(np.clip(risk.ravel()[sel].astype(np.float32), 0, 1))
        pool["l"].append(lab.ravel()[sel].astype(np.uint8))
        pool["n"] += sel.size
        pool["f1"] = idx

    stats = state.get("_xc_stats")
    if len(pool["h"]) > 1 and (stats is None or (idx - state.get("_xc_at", -99)) >= XC_EVERY):
        h = np.concatenate(pool["h"]); r = np.concatenate(pool["r"]); l = np.concatenate(pool["l"])
        if h.size > XC_CAP:                               # thin uniformly, keep it unbiased
            keep = rng.choice(h.size, XC_CAP, replace=False)
            h, r, l = h[keep], r[keep], l[keep]
        pool["h"], pool["r"], pool["l"] = [h], [r], [l]
        rho = float("nan")
        if h.size >= 2000:
            rh, rr = _rankdata(h.astype(np.float64)), _rankdata(r.astype(np.float64))
            sh, sr = rh.std(), rr.std()
            if sh > 0 and sr > 0:
                rho = float(((rh - rh.mean()) * (rr - rr.mean())).mean() / (sh * sr))
        per = {}
        for c in (SAFE, RISKY, OBSTACLE, UNKNOWN):
            k = l == c
            nk = int(k.sum())
            if nk >= 200:
                hv = h[k]
                per[c] = (nk, float(np.median(hv)), float(np.percentile(hv, 90)),
                          float(np.median(r[k])))
        stats = {"rho": rho, "n": int(h.size), "per": per,
                 "f0": int(pool["f0"]), "f1": int(pool["f1"])}
        state["_xc_stats"] = stats
        state["_xc_at"] = idx
    if stats is not None:
        stats["f1"] = int(pool["f1"])
    return stats


# --------------------------------------------------------------------- small helpers


def _risk_map(risk: np.ndarray) -> np.ndarray:
    r = np.clip(np.nan_to_num(risk), 0, 1)
    img = cv2.applyColorMap((r * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    # The chassis silhouette carries no risk value; painting it 0 would read as "free".
    img[~ego_mask(*risk.shape)] = (55, 52, 48)
    return img


HEIGHT_LO, HEIGHT_HI = -0.05, 0.50


def _height_map(g: dict) -> np.ndarray:
    h = np.nan_to_num(g["hag"], nan=0.0, posinf=0.0, neginf=0.0)
    show = g["valid"] & (g["depth"] < HEIGHT_SHOW_MAX_M)
    return V.colorize_height(h, HEIGHT_LO, HEIGHT_HI, invalid=~show)


def _head_top1(packet: FramePacket, mask: np.ndarray | None) -> float | None:
    """Mean softmax top-1 of the traversability head.

    `trav.npz` stores `prob` at HALF resolution (4, 180, 320) with a `prob_hw` key, and
    `pipeline.build_packet` passes it through unresized - so anything that pairs it with
    the full-resolution `label` / `risk` must upsample first.  This does.
    """
    t = packet.trav
    if t is None or t.prob is None:
        return None
    pr = np.asarray(t.prob, np.float32)
    if pr.ndim != 3 or pr.shape[0] < 2:
        return None
    top = pr.max(0)
    th, tw = t.label.shape
    if top.shape != (th, tw):
        top = cv2.resize(top, (tw, th), interpolation=cv2.INTER_LINEAR)
    if mask is None or mask.shape != top.shape or int(mask.sum()) < 200:
        return None                      # the label says "in corridor" - so only ever that
    return float(np.nanmean(top[mask]))


def _ruler_x(x0: int, x1: int, v_m: float) -> int:
    """sqrt-scaled height axis: keeps millimetre-scale classes separable next to a 0.6 m one."""
    t = float(np.sqrt(np.clip(v_m, 0.0, RULER_MAX_M) / RULER_MAX_M))
    return int(round(x0 + t * (x1 - x0)))


# --------------------------------------------------------------------- render


def render(packet: FramePacket, state: dict) -> np.ndarray:
    img = V.canvas(W, H)
    t = packet.trav
    rgb = packet.rgb if packet.rgb is not None else np.zeros((360, 640, 3), np.uint8)
    if t is None:
        V.header(img, W, "prism", "03 - TRAVERSABILITY", "no result")
        return img

    lab, risk = t.label, np.nan_to_num(np.asarray(t.risk, np.float32))
    scene = CLIP_SCENE.get(packet.clip_id, "")
    V.header(img, W, "prism",
             "03 - TRAVERSABILITY  -  can THIS vehicle drive here?",
             f"{packet.clip_id}  {scene}  -  frame {packet.idx:03d}")

    em = ego_mask(*lab.shape)
    shown = lab != 255
    valid = shown & em
    n = max(int(valid.sum()), 1)
    frac = np.array([float(((lab == c) & valid).sum()) / n for c in range(4)], np.float32)
    mr = float(np.nanmean(risk[valid])) if valid.any() else 0.0

    hist = state.setdefault("trav_safe_hist", [])
    hist.append(float(frac[SAFE]))
    del hist[:-300]

    # ------------------------------------------------------------ geometry (measured)
    g = _frame_geometry(packet)
    corr = _corridor_mask(g) if g is not None else None
    poly = _footprint_poly(g["R"], lab.shape) if g is not None else None
    stats = _xc_update(state, g, risk, lab, packet.idx) if g is not None else None
    conf = _head_top1(packet, corr)

    clr_cm = CFG.ugv.clearance_m * 100.0
    step_p95 = frac_over = None
    if corr is not None and int(corr.sum()) >= 300:
        hv = g["hag"][corr]
        step_p95 = float(np.percentile(hv, 95) * 100.0)
        frac_over = float((hv > CFG.ugv.clearance_m).mean())

    # terrain a semantic detector would read as drivable, near enough to trust the metres
    soft_n = soft_block = 0
    soft_med_cm = None
    if g is not None and packet.seg is not None and packet.seg.label.shape == lab.shape:
        soft = np.isin(packet.seg.label, (1, 2)) & g["valid"] & (g["depth"] < 3.0)
        soft_n = int(soft.sum())
        blk = soft & (g["hag"] > CFG.ugv.clearance_m)
        soft_block = int(blk.sum())
        if soft_block >= 200:
            soft_med_cm = float(np.median(g["hag"][blk]) * 100.0)

    # ---------------------------------------------------------------- A1 overlay
    r1 = V.panel(img, 12, 52, 414, 308, "traversability", "4-class overlay")
    ov = V.overlay(rgb, V.colorize_trav(lab), 0.52, mask=shown)
    _draw_poly(ov, poly, (255, 255, 255), "planned footprint")
    V.blit(img, ov, (r1[0] + 6, r1[1] + 6, r1[2] - 12, r1[3] - 60))
    V.legend(img, r1[0] + 12, r1[1] + r1[3] - 32, TRAV_CLASSES[:2], TRAV_COLORS_BGR[:2],
             scale=0.38, vertical=False)
    V.legend(img, r1[0] + 12, r1[1] + r1[3] - 12, TRAV_CLASSES[2:], TRAV_COLORS_BGR[2:],
             scale=0.38, vertical=False)

    # ---------------------------------------------------------------- A2 risk
    r2 = V.panel(img, 434, 52, 412, 308, "continuous risk", "regressed, not thresholded")
    V.blit(img, _risk_map(risk), (r2[0] + 6, r2[1] + 6, r2[2] - 12, r2[3] - 56))
    V.colorbar(img, r2[0] + 20, r2[1] + r2[3] - 38, r2[2] - 40, 11,
               cv2.COLORMAP_INFERNO, "0.0  clear", "1.0  impassable")

    # ---------------------------------------------------------------- A3 height
    r3 = V.panel(img, 854, 52, 414, 308, "height above ground",
                 "recomputed from depth")
    if g is not None:
        hm = _height_map(g)
        _draw_poly(hm, poly, (255, 255, 255), "")
        V.blit(img, hm, (r3[0] + 6, r3[1] + 6, r3[2] - 12, r3[3] - 56))
        cbx, cby, cbw = r3[0] + 20, r3[1] + r3[3] - 38, r3[2] - 40
        V.colorbar(img, cbx, cby, cbw, 11, cv2.COLORMAP_JET, "-5 cm", "+50 cm")
        tx = cbx + int(cbw * (CFG.ugv.clearance_m - HEIGHT_LO) / (HEIGHT_HI - HEIGHT_LO))
        cv2.line(img, (tx, cby - 4), (tx, cby + 15), V.TEXT, 1)
        V.text(img, f"{clr_cm:.1f} cm clearance", (tx - 34, cby - 7), 0.32, V.TEXT, 1)
    else:
        V.text(img, "no ground-plane fit for this frame", (r3[0] + 16, r3[1] + 40),
               0.42, V.TEXT_DIM, 1)
        V.text(img, "heights not measurable - nothing shown", (r3[0] + 16, r3[1] + 62),
               0.38, V.TEXT_DIM, 1)

    # ---------------------------------------------------------------- B1 class area
    b1 = V.panel(img, 12, 368, 340, 240, "class area", "of visible scene")
    y = b1[1] + 20
    for c in range(4):
        col = tuple(int(v) for v in TRAV_COLORS_BGR[c])
        V.text(img, TRAV_CLASSES[c], (b1[0] + 12, y), 0.41, V.TEXT, 1)
        s = f"{frac[c]*100:5.1f}%"
        V.text(img, s, (b1[0] + b1[2] - 12 - V.text_size(s, 0.42, 1, V.FONT_B)[0], y),
               0.42, col, 1, V.FONT_B)
        V.bar_meter(img, b1[0] + 12, y + 7, b1[2] - 26, 8, float(frac[c]), color=col)
        y += 28
    V.text(img, "mean risk", (b1[0] + 12, y), 0.41, V.TEXT, 1)
    s = f"{mr:5.2f}"
    V.text(img, s, (b1[0] + b1[2] - 12 - V.text_size(s, 0.42, 1, V.FONT_B)[0], y),
           0.42, V.ACCENT, 1, V.FONT_B)
    V.bar_meter(img, b1[0] + 12, y + 7, b1[2] - 26, 8, mr, color=V.ACCENT,
                warn_at=CFG.safety.risk_slow)
    y += 30
    cs = "n/a" if conf is None else f"{conf:.2f}"
    V.text(img, "head top-1 (corridor)", (b1[0] + 12, y), 0.37, V.TEXT, 1)
    V.text(img, cs, (b1[0] + b1[2] - 12 - V.text_size(cs, 0.40, 1, V.FONT_B)[0], y),
           0.40, V.TEXT, 1, V.FONT_B)
    V.text(img, "softmax score, not a calibrated probability",
           (b1[0] + 12, y + 13), 0.30, V.TEXT_DIM, 1)
    V.sparkline(img, b1[0] + 12, y + 18, b1[2] - 26, 28, hist, V.OK, lo=0.0, hi=1.0)
    V.text(img, "safe fraction  -  last 300 frames", (b1[0] + 17, y + 31), 0.30, V.TEXT_DIM, 1)

    # ---------------------------------------------------------------- B2 envelope
    u = CFG.ugv
    b2 = V.panel(img, 360, 368, 330, 240, "vehicle envelope", "thresholds used")
    rows = [
        ("chassis clearance", f"{u.clearance_m*1000:.0f} mm", V.TEXT),
        ("max step (binding)", f"{u.max_step_m*1000:.0f} mm", V.ACCENT),
        ("track width", f"{u.width_m*1000:.0f} mm", V.TEXT),
        ("corridor half-width", f"{CORR_HALF_W*1000:.0f} mm", V.TEXT),
        ("max slope", f"{u.max_slope_deg:.0f} deg", V.TEXT),
        ("brake distance", f"{u.brake_distance_m*1000:.0f} mm @ {u.max_speed_mps:.1f} m/s", V.TEXT),
    ]
    y = b2[1] + 24
    for k, s, col in rows:
        V.text(img, k, (b2[0] + 12, y), 0.40, V.TEXT_DIM, 1)
        V.text(img, s, (b2[0] + b2[2] - 12 - V.text_size(s, 0.42, 1, V.FONT_B)[0], y),
               0.42, col, 1, V.FONT_B)
        y += 24
    y += 6
    V.text(img, f"Every metre on this panel scales with the", (b2[0] + 12, y), 0.34, V.TEXT_DIM, 1)
    V.text(img, f"ASSUMED camera height {CFG.cam.height_above_ground_m:.2f} m.",
           (b2[0] + 12, y + 15), 0.34, V.ACCENT, 1)
    V.text(img, "No calibration file exists for this footage.",
           (b2[0] + 12, y + 30), 0.34, V.TEXT_DIM, 1)

    # ---------------------------------------------------------------- B3 cross-check
    b3 = V.panel(img, 698, 368, 570, 240, "geometry cross-check",
                 "does the head track measured terrain height?")
    if stats is not None and stats["per"]:
        rho = stats["rho"]
        rs = "n/a" if not np.isfinite(rho) else f"{rho:+.3f}"
        V.text(img, "Spearman rho (height above ground, predicted risk)",
               (b3[0] + 14, b3[1] + 22), 0.40, V.TEXT, 1)
        V.text(img, rs, (b3[0] + b3[2] - 14 - V.text_size(rs, 0.62, 1, V.FONT_B)[0],
                         b3[1] + 24), 0.62, V.ACCENT, 1, V.FONT_B)
        V.text(img, f"n = {stats['n']/1000:.0f}k px sampled from frames "
                    f"{stats['f0']:03d}-{stats['f1']:03d} of this clip, {BAND_M[0]:.1f}-"
                    f"{BAND_M[1]:.1f} m band",
               (b3[0] + 14, b3[1] + 38), 0.33, V.TEXT_DIM, 1)
        V.text(img, "Consistency check, not ground truth: terrain height also feeds the head and",
               (b3[0] + 14, b3[1] + 52), 0.32, V.WARN, 1)
        V.text(img, "seeds its pseudo-labels, so this shows coherence, not that the labels are right.",
               (b3[0] + 14, b3[1] + 64), 0.32, V.WARN, 1)

        x0, x1 = b3[0] + 76, b3[0] + b3[2] - 168
        xr = b3[0] + b3[2] - 14                      # right edge for the numeric column
        ytop = b3[1] + 72
        # clearance / max-step guides span the whole chart
        for v_m, col in ((u.max_step_m, V.WARN), (u.clearance_m, V.BAD)):
            gx = _ruler_x(x0, x1, v_m)
            cv2.line(img, (gx, ytop + 2), (gx, ytop + 100), col, 1)

        yy = ytop + 20
        for c in (SAFE, RISKY, OBSTACLE):
            col = tuple(int(v) for v in TRAV_COLORS_BGR[c])
            V.text(img, TRAV_CLASSES[c], (b3[0] + 14, yy + 4), 0.38, col, 1)
            cv2.line(img, (x0, yy), (x1, yy), (52, 48, 44), 1)
            e = stats["per"].get(c)
            if e is None:
                V.text(img, "too few px", (x0 + 4, yy + 4), 0.32, V.TEXT_DIM, 1)
            else:
                nk, med, p90, _ = e
                a, b_ = _ruler_x(x0, x1, med), _ruler_x(x0, x1, p90)
                cv2.line(img, (a, yy), (max(b_, a + 2), yy), col, 5, cv2.LINE_AA)
                cv2.circle(img, (a, yy), 4, (16, 16, 16), -1, cv2.LINE_AA)
                cv2.circle(img, (a, yy), 3, col, -1, cv2.LINE_AA)
                s = f"med {med*100:+5.1f}   p90 {p90*100:+6.1f} cm"
                V.text(img, s, (xr - V.text_size(s, 0.32)[0], yy + 4), 0.32, V.TEXT_DIM, 1)
            yy += 32
        for tick in RULER_TICKS:
            gx = _ruler_x(x0, x1, tick)
            cv2.line(img, (gx, yy - 26), (gx, yy - 21), V.EDGE, 1)
            s = f"{tick*100:.0f}"
            V.text(img, s, (gx - V.text_size(s, 0.30)[0] // 2, yy - 10), 0.30, V.TEXT_DIM, 1)
        V.text(img, "median-to-p90 height above ground, cm  (sqrt axis)",
               (b3[0] + 14, yy + 8), 0.32, V.TEXT_DIM, 1)
        V.text(img, f"{u.max_step_m*100:.1f} max step", (xr - 158, yy + 8), 0.31, V.WARN, 1)
        V.text(img, f"{clr_cm:.1f} clearance", (xr - 76, yy + 8), 0.31, V.BAD, 1)
    else:
        V.text(img, "collecting sample...", (b3[0] + 14, b3[1] + 30), 0.42, V.TEXT_DIM, 1)
        V.text(img, "height / risk statistics need a few frames of valid depth.",
               (b3[0] + 14, b3[1] + 52), 0.34, V.TEXT_DIM, 1)

    # ---------------------------------------------------------------- note band
    lines = ['A detector answers "what is it". prism answers "can THIS chassis clear it" '
             '- every threshold above comes from CFG.ugv, not from a dataset.']
    if step_p95 is None:
        lines.append("Footprint corridor: too few valid depth pixels this frame to measure a step.")
        verdict, vcol = "NO MEASUREMENT", V.TEXT_DIM
    else:
        lines.append(f"Footprint corridor {2*CORR_HALF_W*100:.0f} cm wide, {CORR_Y0:.2f}-"
                     f"{CORR_Y1:.2f} m ahead: worst step {step_p95:+.1f} cm (p95), "
                     f"{frac_over*100:.1f}% of it above the {clr_cm:.1f} cm clearance.")
        blocked = step_p95 > clr_cm
        verdict, vcol = ("STEP > CLEARANCE", V.BAD) if blocked else ("CORRIDOR CLEAR", V.OK)
    if soft_med_cm is not None:
        lines.append(f'Wider scene: {soft_block/max(soft_n,1)*100:.0f}% of the pixels the '
                     f'segmenter calls trail/grass within 3 m stand {soft_med_cm:.0f} cm above '
                     f'local ground - drivable-looking, not drivable for a {clr_cm:.1f} cm chassis.')
    elif soft_n > 0:
        lines.append(f'Wider scene: none of the {soft_n/1000:.0f}k trail/grass pixels within 3 m '
                     f'exceed the {clr_cm:.1f} cm clearance this frame - the soft terrain '
                     f'ahead really is flat enough.')
    else:
        lines.append("Wider scene: no trail/grass pixels with valid depth inside 3 m this frame.")

    V.rounded_note(img, 12, 612, 1256, lines, "why this is not object detection", V.ACCENT, lh=15)
    bw = V.text_size(verdict, 0.38, 1, V.FONT_B)[0] + 2 * 7
    V.badge(img, 1268 - 10 - bw, 617, verdict, vcol, 0.38)

    V.footer(img, W, H,
             "Supervision: geometric pseudo-labels from the vehicle envelope + a distilled "
             "terrain teacher. Not human annotation, not RELLIS-3D / GOOSE / ORFD ground truth.",
             "heights measured here from the depth cache")
    return img


if __name__ == "__main__":
    from ..types import TraversabilityResult, SegResult, DepthResult
    from ..config import PROC_H, PROC_W
    rng = np.random.default_rng(0)
    lab = cv2.medianBlur(rng.integers(0, 4, (PROC_H, PROC_W)).astype(np.uint8), 21)
    yy = np.linspace(1.0, 0.0, PROC_H, dtype=np.float32)[:, None]
    depth = (0.25 / np.maximum(yy, 0.02)).astype(np.float32) * np.ones((1, PROC_W), np.float32)
    p = FramePacket(clip_id="clip_01", idx=0, t=0.0,
                    rgb=rng.integers(0, 255, (PROC_H, PROC_W, 3), dtype=np.uint8))
    p.trav = TraversabilityResult(
        prob=rng.random((4, PROC_H // 2, PROC_W // 2)).astype(np.float32),   # half-res, as cached
        label=lab, risk=rng.random((PROC_H, PROC_W)).astype(np.float32))
    p.seg = SegResult(label=np.ones((PROC_H, PROC_W), np.uint8))
    p.depth = DepthResult(rel_inv=1.0 / depth, depth_m=depth,
                          valid=np.isfinite(depth) & (depth < 25))
    p.depth._normal = np.array([0.0, 1.0, 0.0], np.float32)
    st: dict = {}
    for i in range(40):
        p.idx = i
        out = render(p, st)
    print("render ->", out.shape, out.dtype)
    print("cross-check stats ->", st.get("_xc_stats"))
    cv2.imwrite("work/_r_trav_selftest.png", out)
    print("wrote work/_r_trav_selftest.png")
