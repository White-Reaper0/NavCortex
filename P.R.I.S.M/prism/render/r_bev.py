"""Renderer for stage 08_bev_25d_map - the top-down 2.5D coloured-cell local map.

The hero panel is the coloured-dot map the project is named for: every cell of the
rolling BEV grid is a dot whose **colour** is terrain height (or traversability risk)
and whose **vertical pixel offset** is the measured obstacle height, with a stem down to
the ground plane. That displacement is what makes it read as 2.5D rather than as a flat
occupancy grid - you can see a kerb stand up out of the trail.

Everything drawn here is traceable to `CFG`:
  * cell size and extent  -> CFG.bev
  * vehicle footprint / corridor -> CFG.ugv, CFG.safety.corridor_margin_m
  * decision zone -> CFG.safety.horizon_s x CFG.ugv.max_speed_mps forward
  * metric scale -> CFG.cam.height_above_ground_m, an ASSUMED camera height. It is
    stated on screen, because none of these metres are calibrated ground truth.

UNKNOWN cells are never blank and never green: they are drawn as a dim stippled hatch,
so "we have not seen this" is visually distinct from "this is free space".

Reading the unknown fraction honestly
-------------------------------------
The whole grid is 7.68 m forward by 11.52 m wide. A 92 deg camera sitting 0.12 m off the
ground cannot see most of that: the near lateral corners fall outside the horizontal
field of view entirely, and everything past ~4 m is compressed into roughly ten image
rows by the shallow viewing angle. So a whole-grid "unknown 76%" is mostly *geometry*,
not failure, and quoting it on its own makes a working map look broken.

This renderer therefore reports two regions, both labelled and both drawn on the map:

  * **DECISION ZONE** - 0 .. `CFG.safety.horizon_s * CFG.ugv.max_speed_mps` m ahead
    (2.40 m, the distance the supervisor's 2 s horizon actually covers at top speed)
    by +/- (half the body + corridor margin + two body widths of reroute room)
    (0.60 m). This is the region a decision is made from, and it is 83-95% observed at
    ~0.75-0.88 mean confidence on the cached clips.
  * **WHOLE GRID**, plus how much of the whole grid is even inside the camera's
    horizontal FOV cone, so the viewer can see where the unobserved area comes from.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .. import viz_common as V
from ..config import CFG, TRAV_CLASSES, TRAV_COLORS_BGR, SAFE, RISKY, OBSTACLE, UNKNOWN
from ..perception import geometry as geo
from ..types import FramePacket

W_OUT, H_OUT = 1280, 720

STAGE_TITLE = "08 - TOP-DOWN 2.5D LOCAL MAP"
STAGE_SUB = "rolling BEV grid, ego-motion compensated, observation-age decayed"

#: dot colour ramp for height above local ground. 0 m must land mid-ramp so flat ground
#: reads green ("at grade") and anything standing up reads warm, per viz_common's
#: documented colorize_height semantics.
HEIGHT_LO, HEIGHT_HI = -0.35, 0.35
#: a 1 m obstacle lifts its dot by this fraction of the panel height (0.30 m -> ~19 px)
LIFT_FRAC = 0.14

ZONE_COL = (235, 205, 95)          # BGR cyan-blue: the decision-zone outline
FOV_COL = (96, 96, 108)            # BGR: the camera FOV cone edge

# --------------------------------------------------------------------------- regions
# Both regions below are pure functions of CFG, computed once. `bev_to_veh` gives the
# metric vehicle-frame centre of every cell, so these masks are the same geometry the
# planner sees, not an eyeballed pixel box.
_R, _C = np.mgrid[0:CFG.bev.H, 0:CFG.bev.W]
VX, VY = geo.bev_to_veh(_R.astype(np.float32), _C.astype(np.float32))

#: forward extent the 2 s supervisor horizon covers at top speed
ZONE_FWD = float(CFG.safety.horizon_s * CFG.ugv.max_speed_mps)                # 2.40 m
#: lateral extent: the straight-ahead corridor plus two body widths of reroute room
ZONE_LAT = float(CFG.ugv.width_m * 0.5 + CFG.safety.corridor_margin_m
                 + 2.0 * CFG.ugv.width_m)                                     # 0.60 m
ZONE = (VY > 0.0) & (VY <= ZONE_FWD) & (np.abs(VX) <= ZONE_LAT)
VR = np.hypot(VX, VY).astype(np.float32)          # range of each cell centre, metres

#: cells inside the camera's horizontal FOV cone. The near limit is where the ego mask
#: stops occluding the ground for a `CFG.cam.height_above_ground_m` camera; it is a
#: geometric bound on what could ever be seen in a single frame, not a claim about
#: what was seen (the rolling map does fill in ground the vehicle has already driven past).
_TAN_HALF_FOV = float(np.tan(np.deg2rad(CFG.cam.hfov_deg) * 0.5))
FOV_NEAR_M = 0.24
FOV_CONE = (np.abs(VX) <= VY * _TAN_HALF_FOV) & (VY >= FOV_NEAR_M)
FOV_FRAC = float(FOV_CONE.mean())


def zone_stats(bev) -> dict:
    """Whole-grid and decision-zone occupancy statistics for one BEV map.

    `observed` here means what `mapping._publish` means by it: the cell holds live
    geometry (finite height, not stale) *and* survived the `CFG.safety.conf_unknown`
    confidence floor, i.e. it is not reported as UNKNOWN.
    """
    tr = bev.trav
    live = tr != UNKNOWN
    conf = bev.conf
    geom = np.isfinite(bev.height)

    z = ZONE
    zl = live[z]
    n_z = float(z.sum())
    zc = conf[z][zl] if zl.any() else np.zeros(1, np.float32)
    return dict(
        # --- whole grid -----------------------------------------------------
        grid_observed=float(live.mean()),
        grid_unknown=float((~live).mean()),
        grid_geom=float(geom.mean()),
        grid_conf=float(conf[live].mean()) if live.any() else 0.0,
        cone_observed=float(live[FOV_CONE].mean()),
        cone_frac=FOV_FRAC,
        # --- decision zone --------------------------------------------------
        z_cells=int(n_z),
        z_observed=float(zl.mean()),
        z_conf=float(zc.mean()),
        z_safe=float((tr[z] == SAFE).mean()),
        z_risky=float((tr[z] == RISKY).mean()),
        z_obstacle=float((tr[z] == OBSTACLE).mean()),
        z_unknown=float((tr[z] == UNKNOWN).mean()),
    )


# --------------------------------------------------------------------------- splatting
# NOTE: this used to import `splat_disc` from `perception/lidarize.py`. That module is
# owned by another stage and was being edited concurrently, so the ~30 lines are held
# locally instead. Behaviour is identical (painter's algorithm by last-write-wins).

def _splat_disc(img: np.ndarray, px: np.ndarray, py: np.ndarray, colors: np.ndarray,
                radius: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Vectorised painter's-algorithm disc splatting, far -> near."""
    h, w = img.shape[:2]
    n = px.size
    if n == 0:
        return img
    rad = np.clip(np.rint(radius), 0, 4).astype(np.int32)
    order = np.argsort(-np.asarray(depth, np.float32), kind="stable")
    rank = np.empty(n, np.int64)
    rank[order] = np.arange(n)                      # rank 0 = farthest, drawn first

    pxi = np.rint(px).astype(np.int32)
    pyi = np.rint(py).astype(np.int32)
    rmax = int(rad.max())

    levels: dict[int, list[tuple[int, int]]] = {}
    for dy in range(-rmax, rmax + 1):
        for dx in range(-rmax, rmax + 1):
            need = int(np.ceil(np.sqrt(dx * dx + dy * dy) - 1e-9))
            if need <= rmax:
                levels.setdefault(need, []).append((dx, dy))

    Pf, Cf, Kf = [], [], []
    for need, offs in levels.items():
        sel = rad >= need
        if not sel.any():
            continue
        sx, sy, sc, sk = pxi[sel], pyi[sel], colors[sel], rank[sel]
        for dx, dy in offs:
            xx, yy = sx + dx, sy + dy
            ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
            if not ok.any():
                continue
            Pf.append(yy[ok].astype(np.int64) * w + xx[ok])
            Cf.append(sc[ok])
            Kf.append(sk[ok])
    if not Pf:
        return img
    P = np.concatenate(Pf)
    C = np.concatenate(Cf)
    K = np.concatenate(Kf)
    o = np.argsort(K, kind="stable")
    img.reshape(-1, 3)[P[o]] = C[o]
    return img


# --------------------------------------------------------------------------- dot map

def _fit_rect(rect, gw: int, gh: int):
    """Largest sub-rect of `rect` with the grid's aspect ratio, centred."""
    x, y, w, h = rect
    s = min(w / gw, h / gh)
    nw, nh = int(gw * s), int(gh * s)
    return x + (w - nw) // 2, y + (h - nh) // 2, nw, nh


_STAMP_CACHE: dict[tuple, np.ndarray] = {}


def _tiled_stamp(gh: int, gw: int, cell: int, radius: float) -> np.ndarray:
    """A (gh*cell, gw*cell) uint8 0/255 mask: one filled disc centred in every grid cell."""
    key = (gh, gw, cell, round(float(radius), 2))
    st = _STAMP_CACHE.get(key)
    if st is None:
        yy, xx = np.mgrid[0:cell, 0:cell].astype(np.float32)
        c = (cell - 1) / 2.0
        disc = (((xx - c) ** 2 + (yy - c) ** 2) <= radius * radius).astype(np.uint8) * 255
        st = np.ascontiguousarray(np.tile(disc, (gh, gw)))
        if len(_STAMP_CACHE) > 24:
            _STAMP_CACHE.clear()
        _STAMP_CACHE[key] = st
    return st


def _up_nn(a: np.ndarray, cell: int) -> np.ndarray:
    """Nearest-neighbour upsample by an integer factor, via OpenCV (SIMD, ~10x numpy)."""
    h, w = a.shape[:2]
    return cv2.resize(a, (w * cell, h * cell), interpolation=cv2.INTER_NEAREST)


#: colour of the UNKNOWN stipple. Bright enough to read as a deliberate "not observed"
#: texture at panel scale, dark enough that it can never be mistaken for a live cell.
UNKNOWN_DOT = (86, 82, 77)


def bev_dots(size: tuple[int, int], values_bgr: np.ndarray, valid: np.ndarray,
             elev: Optional[np.ndarray] = None, elev_px: float = 0.0,
             unknown: Optional[np.ndarray] = None, dot: float = 0.38,
             cell: int = 6, bg=(21, 19, 17), stems: bool = True,
             lift_thresh_px: float = 1.0, unknown_dot: float = 0.80) -> np.ndarray:
    """Vectorised coloured-dot BEV renderer with 2.5D vertical displacement.

    Same aesthetic as `viz_common.draw_grid_dots` - one disc per cell, far cells drawn
    first, vertical lift by height - but built differently:

      * cells at ground level are stamped in one shot with a tiled disc mask through
        `cv2.copyTo`; that is the overwhelming majority of a BEV grid and none of them
        move, so no per-cell work is needed;
      * only the cells that actually stand up (obstacles, kerbs, banks) go through the
        per-point painter's splat, together with their stems;
      * UNKNOWN cells are composited into the same pass as a dim half-density stipple;
      * everything is drawn at `cell` px per grid cell and box-filtered down, which is
        what gives the dots their soft anti-aliased edge.

    **Honest timing** (best-of-7 on this box, 128x192 grid into a 698x442 panel, so a
    fair comparison needs two `draw_grid_dots` passes - one for live cells, one for the
    UNKNOWN stipple):

        live cells    draw_grid_dots (2 passes)    bev_dots cell=4 / 5 / 6
        6.3k                  6.7 ms                 4.2 / 6.6 / ~9 ms
        8.6k                  7.2 ms                 4.5 / 6.1 / ~9 ms
        24.6k (full)         11.0 ms                 7.4 / 9.1 / ~11 ms

    So this is **not** a large speed win - it is roughly a wash, and it only pulls ahead
    once the grid fills up, because its cost is O(panel pixels) rather than O(cells).
    It exists for the anti-aliasing and for compositing stipple, stems and dots in one
    correctly-ordered pass, which the per-circle loop cannot do.
    """
    w, h = int(size[0]), int(size[1])
    gh, gw = valid.shape
    cell = max(3, int(cell))
    rad_px = max(1.0, dot * cell)
    big = np.empty((gh * cell, gw * cell, 3), np.uint8)
    big[:] = bg

    dz = np.zeros((gh, gw), np.float32)
    if elev is not None and elev_px:
        dz = (np.clip(np.nan_to_num(elev), -0.6, 1.2) * elev_px).astype(np.float32)
    dz_px = dz * (cell / max(h / gh, 1e-6))          # metres->panel px->big-image px

    lifted = valid & (np.abs(dz_px) >= lift_thresh_px)
    flat = valid & ~lifted

    # ---- one pass for everything that sits at ground level ----
    # ground-level known cells get a full-size dot; UNKNOWN cells get a sparse dim
    # stipple (every other cell) so "not observed" can never be mistaken for "free".
    # Both are stamped through a single masked copy.
    colgrid = values_bgr
    mask = cv2.bitwise_and(_tiled_stamp(gh, gw, cell, rad_px),
                           _up_nn(flat.astype(np.uint8) * 255, cell))
    if unknown is not None and unknown.any():
        colgrid = values_bgr.copy()
        colgrid[unknown] = UNKNOWN_DOT
        hatch = np.zeros((gh, gw), np.uint8)
        hatch[::2, ::2] = 255
        hatch[1::2, 1::2] = 255
        hatch &= unknown.astype(np.uint8) * 255
        cv2.bitwise_or(mask,
                       cv2.bitwise_and(_tiled_stamp(gh, gw, cell,
                                                    max(1.0, rad_px * unknown_dot)),
                                       _up_nn(hatch, cell)), dst=mask)
    cv2.copyTo(_up_nn(colgrid, cell), mask, big)

    # ---- standing cells: painter's splat, far to near, with stems ----
    ys, xs = np.nonzero(lifted)
    if ys.size:
        cx = (xs + 0.5) * cell
        gy = (ys + 0.5) * cell
        d = dz_px[ys, xs]
        cols = values_bgr[ys, xs].astype(np.float32)
        depth = (gh - 1 - ys).astype(np.float32)
        P, Q, C, R, D = [], [], [], [], []
        if stems:
            n_seg = 6
            f = np.linspace(0.10, 0.90, n_seg, dtype=np.float32)[None, :]
            P.append(np.repeat(cx[:, None], n_seg, 1).ravel())
            Q.append((gy[:, None] - d[:, None] * f).ravel())
            C.append(np.repeat(cols * 0.38, n_seg, axis=0))
            R.append(np.full(ys.size * n_seg, max(1.0, rad_px * 0.34)))
            D.append(np.repeat(depth, n_seg))
        P.append(cx); Q.append(gy - d); C.append(cols)
        R.append(np.full(ys.size, rad_px)); D.append(depth)
        _splat_disc(big, np.concatenate(P), np.concatenate(Q),
                    np.clip(np.concatenate(C), 0, 255).astype(np.uint8),
                    np.concatenate(R), np.concatenate(D))

    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- overlays

def _plate(img, s, org, scale=0.34, color=V.TEXT, pad=3, alpha=0.62):
    """Text on a translucent dark plate - map furniture has to stay readable on top of a
    dense field of coloured dots, and a 1 px shadow is not enough there."""
    x, y = int(org[0]), int(org[1])
    tw, th = V.text_size(s, scale, 1)
    x0, y0 = max(0, x - pad), max(0, y - th - pad)
    x1, y1 = min(img.shape[1], x + tw + pad), min(img.shape[0], y + pad)
    if x1 > x0 and y1 > y0:
        roi = img[y0:y1, x0:x1]
        roi[:] = (roi.astype(np.float32) * (1.0 - alpha)
                  + np.float32((14, 13, 12)) * alpha).astype(np.uint8)
    V.text(img, s, (x, y), scale, color, 1, shadow=False)


def _poly(img, rect, pts_xy, color, thick=1, closed=False):
    x0, y0, w, h = rect
    p = np.array([geo.veh_to_bev_px(float(x), float(y), rect) for x, y in pts_xy], np.int32)
    p[:, 0] = np.clip(p[:, 0], x0, x0 + w - 1)
    p[:, 1] = np.clip(p[:, 1], y0, y0 + h - 1)
    cv2.polylines(img, [p], closed, color, thick, cv2.LINE_AA)
    return p


def draw_fov_cone(img, rect, label: bool = True):
    """The camera's horizontal FOV cone. Everything outside it can only ever be memory."""
    b = CFG.bev
    fwd, lat = b.range_forward_m, b.range_lateral_m
    r_max = float(np.hypot(fwd, lat)) * 1.2
    for s in (-1, 1):
        _poly(img, rect, [(0.0, 0.0),
                          (s * _TAN_HALF_FOV * r_max, r_max)], FOV_COL, 1)
    if label:
        # place the caption just inside the right-hand cone edge, mid-range
        y = fwd * 0.62
        px, py = geo.veh_to_bev_px(min(_TAN_HALF_FOV * y, lat - 0.1), y, rect)
        _plate(img, f"{CFG.cam.hfov_deg:.0f} deg camera FOV",
               (px - 130, py), 0.33, (150, 148, 158))


def draw_decision_zone(img, rect, label: bool = True):
    """The region the supervisor actually scores - drawn so the stat panel is checkable."""
    pts = [(-ZONE_LAT, 0.0), (ZONE_LAT, 0.0), (ZONE_LAT, ZONE_FWD), (-ZONE_LAT, ZONE_FWD)]
    _poly(img, rect, pts, ZONE_COL, 1, closed=True)
    if label:
        px, py = geo.veh_to_bev_px(ZONE_LAT, ZONE_FWD * 0.55, rect)
        _plate(img, "decision zone", (px + 7, py), 0.33, ZONE_COL)


def ring_confidence(bev, ring_step: float = 2.0) -> dict:
    """Mean confidence of the live cells in each range ring.

    This is the on-map evidence for the claim that confidence falls off with range: a
    sample's weight in `MappingStage` is scaled by 1/(1+(r/4)^2), so distant cells
    accumulate less evidence and eventually drop below the UNKNOWN floor.
    """
    live = bev.trav != UNKNOWN
    out = {}
    for r in np.arange(ring_step, CFG.bev.range_forward_m + 1e-6, ring_step):
        m = live & (np.abs(VR - r) <= ring_step * 0.5)
        out[float(r)] = float(bev.conf[m].mean()) if m.sum() > 20 else float("nan")
    return out


def draw_bev_furniture(img, rect, ring_step: float = 2.0, labels: bool = True,
                       corridor: bool = True, zone: bool = True, fov: bool = True,
                       ring_conf: Optional[dict] = None):
    """Range rings with metre labels, centreline, corridor, decision zone, ego footprint."""
    b, u = CFG.bev, CFG.ugv
    fwd, lat = b.range_forward_m, b.range_lateral_m

    if fov:
        draw_fov_cone(img, rect, label=labels)

    # range rings. Labels sit out on the arc at -35 deg, clear of the centreline, the
    # corridor lines and the decision-zone box, all of which live near x = 0.
    ang = np.linspace(-np.pi / 2 * 0.98, np.pi / 2 * 0.98, 60)
    la = np.deg2rad(35.0)
    for r in np.arange(ring_step, fwd + 1e-6, ring_step):
        pts = [(r * np.sin(a), r * np.cos(a)) for a in ang]
        pts = [(x, y) for x, y in pts if abs(x) <= lat and 0 <= y <= fwd]
        if len(pts) > 2:
            _poly(img, rect, pts, (74, 68, 62), 1)
            if labels:
                px, py = geo.veh_to_bev_px(-r * np.sin(la), r * np.cos(la), rect)
                _plate(img, f"{r:.0f} m", (px + 4, py - 4), 0.35, (206, 200, 192))
                c = (ring_conf or {}).get(float(r), float("nan"))
                if np.isfinite(c):
                    _plate(img, f"conf {c:.2f}", (px + 4, py + 10), 0.31,
                           V.OK if c >= 0.55 else (V.WARN if c >= CFG.safety.conf_unknown
                                                   else (150, 146, 140)))

    # centreline
    _poly(img, rect, [(0.0, 0.0), (0.0, fwd)], (86, 80, 72), 1)

    # the corridor the vehicle sweeps going straight ahead
    if corridor:
        hw = u.width_m / 2.0 + CFG.safety.corridor_margin_m
        for s in (-1, 1):
            _poly(img, rect, [(s * hw, 0.0), (s * hw, fwd)], (60, 130, 180), 1)
        if labels:
            px0, _ = geo.veh_to_bev_px(hw, fwd, rect)
            _plate(img, "corridor", (px0 + 5, rect[1] + 14), 0.33, (140, 185, 220))

    if zone:
        draw_decision_zone(img, rect, label=labels)

    # vehicle footprint, to scale
    hw, L = u.width_m / 2.0, u.length_m
    foot = [(-hw, -L * 0.62), (hw, -L * 0.62), (hw, L * 0.38), (-hw, L * 0.38)]
    p = _poly(img, rect, foot, V.ACCENT, 2, closed=True)
    cv2.fillPoly(img, [p], (44, 78, 104))
    _poly(img, rect, foot, V.ACCENT, 2, closed=True)
    px, py = geo.veh_to_bev_px(0.0, L * 0.38, rect)
    _plate(img, "UGV", (px - 13, min(py + 13, rect[1] + rect[3] - 4)), 0.32, V.ACCENT)
    return img


# --------------------------------------------------------------------------- panels

def _age_image(bev, max_age: float) -> np.ndarray:
    a = np.clip(bev.age / max(max_age, 1e-6), 0.0, 1.0)
    img = cv2.applyColorMap(((1.0 - a) * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
    never = bev.age > max_age
    img[never] = (30, 27, 25)
    return img


def _height_colors(step: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Colour by height above the LOCAL ground estimate, which is what the label claims.

    Using height above the *fitted plane* instead would paint every rising trail red
    just because it slopes; the step map is the quantity that actually matters to a
    wheeled vehicle.
    """
    return V.colorize_height(np.nan_to_num(step, nan=0.0), lo, hi)


def _kv(img, x, y, w, key, val, kc=V.TEXT_DIM, vc=V.TEXT, scale=0.33):
    V.text(img, key, (x, y), scale, kc, 1)
    V.text(img, val, (x + w - V.text_size(val, scale)[0], y), scale, vc, 1)


def _stat_block(img, x, y, w, st: dict, hist: dict) -> int:
    """The headline numbers: decision zone first, whole grid second and captioned."""
    yy = y

    # ---------------- decision zone ----------------
    V.text(img, "DECISION ZONE", (x, yy + 11), 0.40, ZONE_COL, 1, V.FONT_B)
    yy += 15
    V.text(img, f"{ZONE_FWD:.2f} m ahead  x  +/-{ZONE_LAT:.2f} m   "
                f"({st['z_cells']} cells)", (x, yy + 10), 0.30, (140, 136, 130), 1)
    yy += 20

    obs = st["z_observed"]
    oc = V.OK if obs > 0.8 else V.WARN
    V.text(img, "OBSERVED", (x, yy + 14), 0.40, V.TEXT, 1, V.FONT_B)
    lab = f"{obs * 100:.0f}%"
    V.text(img, lab, (x + w - V.text_size(lab, 0.60, 1, V.FONT_B)[0], yy + 16), 0.60,
           oc, 1, V.FONT_B)
    yy += 21
    V.bar_meter(img, x, yy, w, 7, obs, color=oc)
    yy += 14
    _kv(img, x, yy + 9, w, "mean confidence", f"{st['z_conf']:.2f}")
    V.bar_meter(img, x, yy + 12, w, 5, st["z_conf"], color=V.ACCENT2)
    yy += 24

    rows = (("safe", st["z_safe"], V.OK), ("risky", st["z_risky"], V.WARN),
            ("obstacle", st["z_obstacle"], V.BAD),
            ("unknown", st["z_unknown"], (150, 150, 150)))
    for name, val, c in rows:
        _kv(img, x, yy + 9, w, name, f"{val * 100:.1f}%", scale=0.33)
        V.bar_meter(img, x, yy + 12, w, 5, val, color=c)
        yy += 19
    yy += 3
    V.text(img, "unknown in zone, last 4 s", (x, yy + 9), 0.30, V.TEXT_DIM, 1)
    V.sparkline(img, x, yy + 13, w, 24, hist.get("zunk", [0.0]), color=(150, 150, 150),
                lo=0.0, hi=0.5)
    yy += 42

    # ---------------- whole grid ----------------
    cv2.line(img, (x, yy), (x + w, yy), V.EDGE, 1)
    yy += 5
    V.text(img, f"WHOLE GRID   {CFG.bev.H * CFG.bev.W} cells", (x, yy + 11), 0.35,
           V.TEXT_DIM, 1, V.FONT_B)
    yy += 16
    _kv(img, x, yy + 9, w, "has geometry", f"{st['grid_geom'] * 100:.0f}%")
    yy += 14
    _kv(img, x, yy + 9, w, f"above conf floor {CFG.safety.conf_unknown:.2f}",
        f"{st['grid_observed'] * 100:.0f}%")
    yy += 18
    V.text(img, f"inside the {CFG.cam.hfov_deg:.0f} deg FOV cone "
                f"({st['cone_frac'] * 100:.0f}% of grid)", (x, yy + 9), 0.30,
           (140, 136, 130), 1)
    yy += 14
    _kv(img, x, yy + 9, w, "above conf floor", f"{st['cone_observed'] * 100:.0f}%")
    yy += 14
    return yy


# --------------------------------------------------------------------------- main

def render(packet: FramePacket, state: dict) -> np.ndarray:
    import time
    t_render0 = time.perf_counter()

    img = V.canvas(W_OUT, H_OUT)
    b = CFG.bev
    bev = packet.bev

    V.header(img, W_OUT, "prism", STAGE_TITLE,
             right=f"{packet.clip_id}   frame {packet.idx:03d}   t={packet.t:5.2f}s")

    if bev is None:
        V.panel(img, 12, 60, W_OUT - 24, 560, "no bev map", "MappingStage produced nothing")
        V.text(img, "packet.bev is None - run prism.training.run_map_infer first",
               (40, 340), 0.6, V.TEXT_DIM, 1)
        V.footer(img, W_OUT, H_OUT, "prism", "")
        return img

    st = zone_stats(bev)
    live = bev.trav != UNKNOWN
    hist = state.setdefault("bev_hist", {"zunk": []})
    hist["zunk"].append(st["z_unknown"])
    for k in hist:
        if len(hist[k]) > 120:
            hist[k] = hist[k][-120:]
            state["bev_hist"][k] = hist[k]

    known = live
    unknown = ~known
    hstep = getattr(bev, "height_step", None)
    if hstep is None:
        from ..perception.mapping import height_step_map
        hstep, _ = height_step_map(bev)
    elev = np.nan_to_num(np.asarray(hstep, np.float32), nan=0.0)

    src = str(getattr(bev, "trav_source", "geometric"))
    learned = src == "learned"

    # =================================================================== hero
    HX, HY, HW, HH = 12, 52, 620, 470
    inner = V.panel(img, HX, HY, HW, HH, "2.5D height map",
                    "colour = height | brightness = confidence | lift = obstacle height")
    r = _fit_rect((inner[0] + 6, inner[1] + 6, inner[2] - 12, inner[3] - 12), b.W, b.H)
    # Dot brightness carries the per-cell confidence: a cell that only just cleared the
    # UNKNOWN floor must not look as solid as one the map has seen thirty times. Colour
    # still means height; this only scales its value.
    cshade = (0.38 + 0.62 * np.clip(
        (bev.conf - CFG.safety.conf_unknown) / max(1.0 - CFG.safety.conf_unknown, 1e-6),
        0.0, 1.0)).astype(np.float32)
    hcol = (_height_colors(elev, HEIGHT_LO, HEIGHT_HI).astype(np.float32)
            * cshade[..., None]).astype(np.uint8)
    dots = bev_dots((r[2], r[3]), hcol, known,
                    elev=elev, elev_px=r[3] * LIFT_FRAC, unknown=unknown)
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = dots
    draw_bev_furniture(img, r, ring_conf=ring_confidence(bev))
    cv2.rectangle(img, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), V.EDGE, 1)
    _plate(img, f"{b.res_m * 100:.0f} cm cells   {b.range_forward_m:.2f} m forward   "
                f"+/-{b.range_lateral_m:.2f} m lateral   |   ring 'conf' = mean "
                f"confidence of live cells at that range",
           (r[0] + 10, r[1] + r[3] - 8), 0.32, (156, 150, 144))

    # =================================================================== traversability
    inner = V.panel(img, 640, 52, 330, 236, "traversability",
                    "learned head" if learned else "GEOMETRY ONLY (no learned head)",
                    accent=V.ACCENT if learned else V.WARN)
    r2 = _fit_rect((inner[0] + 4, inner[1] + 4, inner[2] - 8, inner[3] - 26), b.W, b.H)
    dots = bev_dots((r2[2], r2[3]), V.colorize_trav(bev.trav), known,
                    elev=elev, elev_px=r2[3] * LIFT_FRAC, unknown=unknown, cell=5)
    img[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = dots
    draw_bev_furniture(img, r2, ring_step=4.0, labels=False, fov=False)
    cv2.rectangle(img, (r2[0], r2[1]), (r2[0] + r2[2], r2[1] + r2[3]), V.EDGE, 1)
    V.legend(img, inner[0] + 6, inner[1] + inner[3] - 6, TRAV_CLASSES, TRAV_COLORS_BGR,
             0.31, swatch=9, gap=62, vertical=False)

    # =================================================================== observation age
    inner = V.panel(img, 640, 296, 330, 226, "observation age",
                    "bright = fresh, dark = stale")
    ai = _age_image(bev, b.max_age_frames)
    r3 = _fit_rect((inner[0] + 4, inner[1] + 4, inner[2] - 8, inner[3] - 34), b.W, b.H)
    img[r3[1]:r3[1] + r3[3], r3[0]:r3[0] + r3[2]] = cv2.resize(
        ai, (r3[2], r3[3]), interpolation=cv2.INTER_NEAREST)
    draw_bev_furniture(img, r3, ring_step=4.0, labels=False, corridor=False,
                       zone=False, fov=False)
    cv2.rectangle(img, (r3[0], r3[1]), (r3[0] + r3[2], r3[1] + r3[3]), V.EDGE, 1)
    cby = inner[1] + inner[3] - 20
    V.colorbar(img, inner[0] + 8, cby, 150, 9, cv2.COLORMAP_PLASMA,
               lo_lab="0 fr", hi_lab=f"{b.max_age_frames} fr", reverse=True)
    V.text(img, f"retired at {b.max_age_frames} fr", (inner[0] + 172, cby + 2),
           0.31, V.TEXT_DIM, 1)
    V.text(img, f"conf decay {b.decay_per_frame}/frame", (inner[0] + 172, cby + 14),
           0.31, V.TEXT_DIM, 1)

    # =================================================================== camera
    inner = V.panel(img, 978, 52, 290, 130, "camera", "source frame")
    if packet.rgb is not None:
        V.blit(img, packet.rgb, (inner[0] + 2, inner[1] + 2, inner[2] - 4, inner[3] - 4))
    else:
        V.text(img, "no rgb", (inner[0] + 12, inner[1] + 40), 0.4, V.TEXT_DIM, 1)

    # =================================================================== stats
    inner = V.panel(img, 978, 190, 290, 332, "map coverage",
                    "what is decision-relevant")
    _stat_block(img, inner[0] + 12, inner[1] + 2, inner[2] - 24, st, hist)

    # =================================================================== bottom: scale
    inner = V.panel(img, 12, 530, 620, 160, "reading the map",
                    "colours, scale and what is being claimed")
    x0, y0 = inner[0] + 12, inner[1] + 6
    V.colorbar(img, x0, y0 + 12, 224, 11, cv2.COLORMAP_JET,
               lo_lab=f"{HEIGHT_LO:+.2f} m (drop)", hi_lab=f"{HEIGHT_HI:+.2f} m (obstacle)",
               title="dot colour = height above local ground")
    V.text(img, "dot brightness = confidence", (x0, y0 + 48), 0.32, (150, 146, 140), 1)
    V.legend(img, x0, y0 + 62, ["safe", "risky", "obstacle"], TRAV_COLORS_BGR, 0.34,
             swatch=10, gap=76, vertical=False, title="")
    # explicit unknown swatch, drawn as the stipple actually used
    ux, uy = x0, y0 + 74
    cv2.rectangle(img, (ux, uy), (ux + 34, uy + 16), (21, 19, 17), -1)
    for i in range(0, 34, 4):
        for j in range(0, 16, 4):
            cv2.circle(img, (ux + i + 2, uy + j + 2), 1, UNKNOWN_DOT, -1)
    cv2.rectangle(img, (ux, uy), (ux + 34, uy + 16), V.EDGE, 1)
    V.text(img, "UNKNOWN - never seen, stale,", (ux + 42, uy + 6), 0.33,
           (172, 168, 162), 1)
    V.text(img, "or below the confidence floor", (ux + 42, uy + 19), 0.33,
           (172, 168, 162), 1)

    xr = x0 + 290
    V.text(img, "vertical dot lift = measured height, stem to ground",
           (xr, y0 + 14), 0.33, V.TEXT_DIM, 1)
    V.text(img, f"grid {b.H} x {b.W} cells @ {b.res_m * 100:.0f} cm = "
                f"{b.range_forward_m:.2f} x {2 * b.range_lateral_m:.2f} m",
           (xr, y0 + 31), 0.33, V.TEXT_DIM, 1)
    V.text(img, f"vehicle {CFG.ugv.width_m:.2f} x {CFG.ugv.length_m:.2f} m, "
                f"step limit {CFG.ugv.max_step_m * 1000:.0f} mm",
           (xr, y0 + 48), 0.33, V.TEXT_DIM, 1)
    V.text(img, f"corridor = {CFG.ugv.width_m:.2f} m body + "
                f"{CFG.safety.corridor_margin_m * 100:.0f} cm margin",
           (xr, y0 + 65), 0.33, (110, 165, 205), 1)
    V.text(img, f"decision zone = {ZONE_FWD:.2f} m ({CFG.safety.horizon_s:.0f} s at "
                f"{CFG.ugv.max_speed_mps:.1f} m/s) x +/-{ZONE_LAT:.2f} m",
           (xr, y0 + 82), 0.33, ZONE_COL, 1)
    warn = not learned
    V.text(img, "Traversability = " + ("LEARNED head, supervised by geometric pseudo-labels "
                                       "from the vehicle envelope"
                                       if learned else
                                       "GEOMETRY ONLY (step height vs chassis clearance); "
                                       "no learned head cached"),
           (x0, y0 + 110), 0.32, V.WARN if warn else (150, 146, 140), 1)
    V.text(img, "- not human annotation, and not RELLIS-3D / GOOSE ground truth. "
                "UNKNOWN is never promoted to free space.",
           (x0, y0 + 125), 0.32, (150, 146, 140), 1)

    # =================================================================== bottom: note
    V.panel(img, 640, 530, 628, 160, "why this matters", "navigation, not detection")
    V.rounded_note(img, 648, 558, 612, [
        "A detector answers 'what object is that'. This answers 'can the vehicle drive there,",
        "how high is it, when did we last look'. The map rolls forward every frame with the VO",
        f"delta pose, so terrain out of view is still planned against - for {b.max_age_frames} frames, then retired.",
    ], title="", accent=V.ACCENT)
    V.rounded_note(img, 648, 625, 612, [
        f"Most of the {b.range_forward_m:.1f} x {2 * b.range_lateral_m:.1f} m grid is UNKNOWN, and that is correct rather than broken.",
        f"Only {st['cone_frac'] * 100:.0f}% of it is inside the {CFG.cam.hfov_deg:.0f} deg FOV cone at all, and inside the cone "
        f"confidence",
        "falls off with range by design. Judge the map on the DECISION ZONE, not the total.",
    ], title="", accent=ZONE_COL)

    # =================================================================== footer
    ms = state.setdefault("_render_ms", [])
    right = (f"renderer {np.median(ms):.0f} ms/frame (this box, CPU numpy+OpenCV)"
             if len(ms) >= 5 else "")
    V.footer(img, W_OUT, H_OUT,
             f"Metric scale from an ASSUMED camera height of "
             f"{CFG.cam.height_above_ground_m:.2f} m - not calibrated ground truth. "
             f"Confidence is a confidence score, not a collision probability.", right)
    ms.append((time.perf_counter() - t_render0) * 1e3)
    if len(ms) > 60:
        del ms[:-60]
    return img


# =========================================================================== self-test

def _self_test() -> None:
    import time
    from ..config import WORK_DIR
    from ..perception.mapping import MappingStage, synthetic_scene, _packet_from_scene

    print("r_bev self-test - rendering the synthetic scene")
    print(f"  decision zone: {ZONE_FWD:.2f} m forward x +/-{ZONE_LAT:.2f} m lateral "
          f"= {int(ZONE.sum())} of {ZONE.size} cells ({ZONE.mean():.1%})")
    print(f"  {CFG.cam.hfov_deg:.0f} deg FOV cone beyond {FOV_NEAR_M:.2f} m covers "
          f"{FOV_FRAC:.1%} of the grid")

    stage = MappingStage("cpu")
    depth, valid, trav, seg, truth = synthetic_scene(0.0)
    state: dict = {}
    rng = np.random.default_rng(0)
    t_render = []
    pk = None
    for i in range(24):
        adv = i * 0.035
        d, v, t, s, _ = synthetic_scene(adv) if i % 4 == 0 else (depth, valid, trav, seg, None)
        d = (d * (1.0 + 0.02 * rng.standard_normal(d.shape))).astype(np.float32)
        pk = _packet_from_scene(i, d, v, t, s, d_trans=0.035)
        pk.rgb = cv2.cvtColor((np.clip(1.6 / np.nan_to_num(d, nan=8.0), 0, 1)
                               * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        stage(pk)
        t0 = time.perf_counter()
        out = render(pk, state)
        t_render.append((time.perf_counter() - t0) * 1e3)
    print(f"  render: {out.shape} dtype={out.dtype}  "
          f"median {np.median(t_render[3:]):.1f} ms/frame "
          f"(min {np.min(t_render[3:]):.1f}, mean {np.mean(t_render[3:]):.1f}; "
          f"this box is shared, so trust the min)")
    assert out.shape == (H_OUT, W_OUT, 3) and out.dtype == np.uint8
    assert pk.bev.trav_source == "learned", pk.bev.trav_source

    s = zone_stats(pk.bev)
    print("  whole grid : observed {grid_observed:.1%}  unknown {grid_unknown:.1%}  "
          "cone observed {cone_observed:.1%}".format(**s))
    print("  decision zone: observed {z_observed:.1%}  conf {z_conf:.2f}  "
          "safe {z_safe:.1%} risky {z_risky:.1%} obst {z_obstacle:.1%} "
          "unknown {z_unknown:.1%}".format(**s))

    # dot-renderer speed against the reference loop implementation
    vals = V.colorize_height(np.nan_to_num(pk.bev.height_step, nan=0.0))
    ok = pk.bev.trav != UNKNOWN
    unk = pk.bev.trav == UNKNOWN
    step = np.nan_to_num(pk.bev.height_step)

    def best_of(fn, n=5):
        """Best-of-N: this box is shared with other agents, so means are meaningless."""
        best = float("inf")
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            best = min(best, (time.perf_counter() - t0) * 1e3)
        return best

    fast = best_of(lambda: bev_dots((698, 442), vals, ok, elev=step,
                                    elev_px=442 * LIFT_FRAC, unknown=unk))
    scratch = V.canvas(698, 442)

    def reference():
        # a like-for-like comparison needs both passes: live cells, then the stipple
        V.draw_grid_dots(scratch, (0, 0, 698, 442), vals, ok, step, 2, 442 * LIFT_FRAC)
        V.draw_grid_dots(scratch, (0, 0, 698, 442), vals, unk, None, 1, 0.0)

    slow = best_of(reference, 3)
    print(f"  dot renderer (best of N, {int(ok.sum())} live + {int(unk.sum())} unknown "
          f"cells): bev_dots {fast:.1f} ms vs viz_common.draw_grid_dots x2 {slow:.1f} ms "
          f"- comparable; bev_dots is here for the anti-aliasing and one-pass "
          f"stipple/stem compositing, not for speed")

    p = WORK_DIR / "preview_bev.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), out)
    print(f"  wrote {p}")


if __name__ == "__main__":
    _self_test()
