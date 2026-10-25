"""Shared BEV-state helpers for the prism prediction / decision stages.

Owned by the world-model / planner / RL agent.  Nothing here mutates the frozen
contract modules; it only reads them.

The world state the tiny dynamics model sees
--------------------------------------------
`BEVMap` carries seven grids.  The world model consumes a fixed **8-channel
tensor** at the native BEV resolution (128 forward x 192 lateral, 0.06 m/cell):

    ch 0  height_norm   clip(height, z_min, z_max) rescaled to [-1, 1]; 0 where unobserved
    ch 1  observed      1.0 where the cell has ever been written, else 0.0
    ch 2  p_safe        traversability posterior, class 0
    ch 3  p_risky       class 1
    ch 4  p_obstacle    class 2
    ch 5  p_unknown     class 3
    ch 6  conf          fused confidence in [0, 1]
    ch 7  age_norm      clip(age / bev.max_age_frames, 0, 1)

Row 0 is the farthest forward cell, row 127 is at the vehicle, column 96 is the
centreline (see CONTRACT.md).  Everything downstream - encoder, warp
augmentation, footprint sweeps, planner scoring, supervisor rules - reads this
same layout so the numbers on the dashboard are all traceable to one tensor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import cv2

from ..config import CFG, N_TRAV, SAFE, RISKY, OBSTACLE, UNKNOWN, ACTION_CMD, ACTIONS
from ..io_utils import has_stage, load_stage
from ..perception.geometry import bev_to_veh

# ------------------------------------------------------------------ constants

BEV_CH = 8
CH_HEIGHT, CH_OBS, CH_SAFE, CH_RISKY, CH_OBST, CH_UNK, CH_CONF, CH_AGE = range(8)

OCC_G = 16                      # decoder occupancy grid side
#: world-model integration step.  Chosen so that pred_horizon steps span exactly
#: the planner's safety horizon: 6 x 0.3333 s = 2.0 s.
WM_DT = float(CFG.safety.horizon_s) / float(CFG.wm.pred_horizon)
#: frame stride at CLIP_FPS that corresponds to WM_DT (30 fps -> 10 frames)
WM_FRAME_STRIDE = max(1, int(round(WM_DT * 30.0)))


# ------------------------------------------------------------------ state packing

_REF_MASK: Optional[np.ndarray] = None


def _ground_reference(h: np.ndarray, observed: np.ndarray) -> float:
    """Median height of the near-field corridor the vehicle is standing on.

    Robust (median over a fixed window immediately ahead of the vehicle), and
    falls back to the global observed median, then to 0.0.
    """
    global _REF_MASK
    b = CFG.bev
    if _REF_MASK is None:
        rr, cc = np.meshgrid(np.arange(b.H, dtype=np.float32),
                             np.arange(b.W, dtype=np.float32), indexing="ij")
        x, y = bev_to_veh(rr, cc)
        _REF_MASK = (y > 0.10) & (y < 1.40) & (np.abs(x) < 0.60)
    m = _REF_MASK & observed & np.isfinite(h)
    if m.sum() >= 40:
        return float(np.median(h[m]))
    m2 = observed & np.isfinite(h)
    return float(np.median(h[m2])) if m2.sum() >= 40 else 0.0


def pack_state(height: np.ndarray,
               trav_prob: np.ndarray,
               conf: np.ndarray,
               age: np.ndarray,
               hits: Optional[np.ndarray] = None) -> np.ndarray:
    """BEVMap grids -> (8, H, W) float32 world-state tensor (see module docstring)."""
    b = CFG.bev
    H, W = b.H, b.W
    h = np.asarray(height, np.float32)
    finite = np.isfinite(h)
    if hits is not None:
        observed = finite & (np.asarray(hits, np.float32) > 0.0)
    else:
        observed = finite
    obs_f = observed.astype(np.float32)

    # Local ground reference.  `BEVMap.height` is nominally already "above local
    # ground", but it inherits the per-frame bias of the monocular ground-plane
    # fit, and a 4.5 cm chassis clearance is smaller than that bias.  The height
    # *step* the vehicle actually has to climb is relative to the patch it is
    # standing on, so the near-field corridor median is subtracted here and every
    # downstream metre-scale number (max_step, collision risk, R3) is referenced
    # to it.  On synthetic scenes the reference is ~0 and this is a no-op.
    h = h - _ground_reference(h, observed)

    span = max(b.z_max - b.z_min, 1e-6)
    hn = np.zeros((H, W), np.float32)
    hn[observed] = np.clip((h[observed] - b.z_min) / span, 0.0, 1.0) * 2.0 - 1.0

    tp = np.asarray(trav_prob, np.float32)
    if tp.shape != (N_TRAV, H, W):
        tp = np.zeros((N_TRAV, H, W), np.float32)
        tp[UNKNOWN] = 1.0
    tp = np.clip(tp, 0.0, 1.0)
    s = tp.sum(0)
    bad = s < 1e-5
    tp = tp / np.maximum(s, 1e-5)
    tp[:, bad] = 0.0
    tp[UNKNOWN, bad] = 1.0
    # anything never observed is unknown by definition
    tp[:, ~observed] = 0.0
    tp[UNKNOWN, ~observed] = 1.0

    c = np.clip(np.nan_to_num(np.asarray(conf, np.float32)), 0.0, 1.0) * obs_f
    a = np.clip(np.nan_to_num(np.asarray(age, np.float32), posinf=1e4) /
                float(b.max_age_frames), 0.0, 1.0)
    a[~observed] = 1.0

    st = np.empty((BEV_CH, H, W), np.float32)
    st[CH_HEIGHT] = hn
    st[CH_OBS] = obs_f
    st[CH_SAFE:CH_UNK + 1] = tp
    st[CH_CONF] = c
    st[CH_AGE] = a
    return st


def state_height_m(state: np.ndarray) -> np.ndarray:
    """(H, W) height above local ground in metres; NaN where unobserved."""
    b = CFG.bev
    span = b.z_max - b.z_min
    h = (state[CH_HEIGHT] + 1.0) * 0.5 * span + b.z_min
    out = np.where(state[CH_OBS] > 0.5, h, np.nan).astype(np.float32)
    return out


#: side of the morphological window used to estimate "local ground", in cells.
#: 15 cells = 0.90 m, which is wider than the vehicle and wider than any obstacle
#: the planner has to squeeze past, and narrower than the terrain's own slope.
STEP_WIN = 15


def height_step_map(state: np.ndarray) -> np.ndarray:
    """(H, W) positive height *step* over local ground, metres; NaN where unobserved.

    A raw height map cannot be thresholded against a 4.5 cm chassis clearance:
    it carries the per-frame bias of the monocular ground-plane fit and the
    terrain's own slope, both of which are much larger than the clearance.  What
    matters for traversability is the *local discontinuity*, so this is a white
    top-hat: `height - morphological_opening(height)`.  A 0.9 m opening window
    erases any raised structure narrower than the window (a kerb, a wall, a rock)
    and leaves a smooth slope untouched, which is exactly the distinction between
    "steep" and "a step".  A final 3x3 erosion drops isolated single-cell spikes,
    which at this depth accuracy are noise rather than geometry.
    """
    h = state_height_m(state)
    obs = state[CH_OBS] > 0.5
    m = obs.astype(np.float32)
    filled = np.nan_to_num(h, nan=0.0) * m
    k = (STEP_WIN, STEP_WIN)
    num = cv2.boxFilter(filled, -1, k, normalize=False)
    den = cv2.boxFilter(m, -1, k, normalize=False)
    mean = num / np.maximum(den, 1.0)
    filled = np.where(obs, np.nan_to_num(h, nan=0.0), mean).astype(np.float32)
    ker = np.ones((STEP_WIN, STEP_WIN), np.uint8)
    opened = cv2.morphologyEx(filled, cv2.MORPH_OPEN, ker)
    step = np.clip(filled - opened, 0.0, None)
    step = cv2.erode(step, np.ones((3, 3), np.uint8))
    return np.where(obs, step, np.nan).astype(np.float32)


def state_trav_label(state: np.ndarray) -> np.ndarray:
    """(H, W) uint8 argmax traversability class."""
    return np.argmax(state[CH_SAFE:CH_UNK + 1], axis=0).astype(np.uint8)


def occupancy_grid(state: np.ndarray, g: int = OCC_G) -> np.ndarray:
    """(g, g) obstacle-occupancy target in [0,1], area-pooled from p_obstacle.

    **Obstacle posterior only.**  An earlier version folded "unknown with low
    confidence" into this target at half weight.  That was a mistake: on these
    clips two thirds of every map is unknown and the unknown region is an almost
    static camera-frustum wedge, so the target's mean was 0.50 and its spread
    across the six candidate actions was only 0.068 - the loss optimum was to
    predict the marginal, and the head duly collapsed to a flat 0.48 everywhere.
    Measured on the cached maps, an obstacle-only target has mean 0.10 and an
    across-action spread of 0.145, i.e. twice the action-dependent signal in a
    sparse target that cannot be won by predicting a constant.  What is unknown
    is reported by the traversability head and by the planner's `unknown_frac`,
    which is where it belongs.
    """
    occ = np.clip(state[CH_OBST], 0.0, 1.0)
    return cv2.resize(occ, (g, g), interpolation=cv2.INTER_AREA).astype(np.float32)


def mean_traversability(state: np.ndarray) -> float:
    """Scalar in [0,1]: mean p_safe over the forward corridor the vehicle would use."""
    b = CFG.bev
    half = int(round((CFG.ugv.width_m * 0.5 + CFG.safety.corridor_margin_m) / b.res_m)) + 1
    c0, c1 = b.n_lateral - half * 2, b.n_lateral + half * 2
    r0 = b.H - int(round(3.0 / b.res_m))          # nearest 3 m of corridor
    sub = state[:, max(r0, 0):, max(c0, 0):min(c1, b.W)]
    if sub.size == 0:
        return 0.0
    return float(np.mean(sub[CH_SAFE] + 0.4 * sub[CH_RISKY]))


# ------------------------------------------------------------------ motion model

def arc_poses(v: float, w: float, dt: float, n_steps: int,
              x0: float = 0.0, y0: float = 0.0, yaw0: float = 0.0):
    """Unicycle rollout in the vehicle frame (X right, Y forward, yaw CCW from +Y).

    Returns (xy (n,2) float32, yaw (n,) float32) for steps 1..n_steps.
    """
    xs = np.empty(n_steps, np.float32)
    ys = np.empty(n_steps, np.float32)
    yw = np.empty(n_steps, np.float32)
    x, y, th = float(x0), float(y0), float(yaw0)
    for i in range(n_steps):
        if abs(w) < 1e-6:
            x += -np.sin(th) * v * dt
            y += np.cos(th) * v * dt
        else:
            th_n = th + w * dt
            r = v / w
            x += r * (np.cos(th_n) - np.cos(th))
            y += r * (np.sin(th_n) - np.sin(th))
            th = th_n
        xs[i], ys[i], yw[i] = x, y, th
    return np.stack([xs, ys], 1), yw


def action_motion(action: int) -> tuple[float, float]:
    """(linear m/s, angular rad/s) prototype for a discrete action."""
    v, w = ACTION_CMD[int(action)]
    return float(v), float(w)


# ------------------------------------------------------------------ rigid BEV warp

def _remap_maps(v: float, w: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Pixel lookup maps taking *new* BEV cells to *old* BEV cells after (v, w) dt."""
    b = CFG.bev
    # pose of the new vehicle frame expressed in the old vehicle frame
    if abs(w) < 1e-6:
        tx, ty, dyaw = 0.0, v * dt, 0.0
    else:
        dyaw = w * dt
        r = v / w
        tx = r * (np.cos(dyaw) - 1.0)
        ty = r * np.sin(dyaw)

    rr, cc = np.meshgrid(np.arange(b.H, dtype=np.float32),
                         np.arange(b.W, dtype=np.float32), indexing="ij")
    xn, yn = bev_to_veh(rr, cc)                     # new-frame metres
    cs, sn = np.cos(dyaw), np.sin(dyaw)
    xo = tx + cs * xn - sn * yn                     # old-frame metres
    yo = ty + sn * xn + cs * yn

    col = xo / b.res_m + b.n_lateral - 0.5
    row = (b.n_forward - 1) + 0.5 - yo / b.res_m
    return col.astype(np.float32), row.astype(np.float32)


_WARP_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def warp_state(state: np.ndarray, v: float, w: float, dt: float = WM_DT) -> np.ndarray:
    """Rigidly re-express the world state in the vehicle frame after executing (v, w) dt.

    This is a **kinematic augmentation, not observed data**: it moves the map the
    way the geometry says it must move if the commanded arc is executed exactly.
    It cannot invent scene content that was never seen, so cells that scroll in
    from outside the previous field of view are marked unobserved / UNKNOWN with
    zero confidence - which is exactly the honest answer.
    """
    key = (round(v, 4), round(w, 4), round(dt, 4))
    if key not in _WARP_CACHE:
        _WARP_CACHE[key] = _remap_maps(v, w, dt)
    mx, my = _WARP_CACHE[key]

    out = np.empty_like(state)
    for c in range(state.shape[0]):
        out[c] = cv2.remap(state[c], mx, my, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    unseen = out[CH_OBS] < 0.5
    out[CH_OBS] = (~unseen).astype(np.float32)
    out[CH_HEIGHT][unseen] = 0.0
    out[CH_SAFE][unseen] = 0.0
    out[CH_RISKY][unseen] = 0.0
    out[CH_OBST][unseen] = 0.0
    out[CH_UNK][unseen] = 1.0
    out[CH_CONF][unseen] = 0.0
    out[CH_AGE][unseen] = 1.0
    # everything that survived the warp is one step staler
    out[CH_AGE] = np.clip(out[CH_AGE] + (dt * 30.0) / CFG.bev.max_age_frames, 0.0, 1.0)
    s = out[CH_SAFE:CH_UNK + 1].sum(0)
    out[CH_SAFE:CH_UNK + 1] /= np.maximum(s, 1e-5)
    return out


# ------------------------------------------------------------------ footprint sweep

@dataclass
class SweepMetrics:
    """What the vehicle footprint sees along a candidate arc."""
    max_step: float = 0.0            # worst height above local ground under the footprint, m
    max_step_dist: float = 0.0       # arc distance at which that step occurs, m
    obstacle_frac: float = 0.0       # peak p_obstacle under the footprint
    unknown_frac: float = 0.0        # fraction of footprint cells unknown or low-confidence
    mean_conf: float = 1.0
    min_clearance: float = 9.9       # lateral distance to nearest obstacle cell, m
    free_distance: float = 9.9       # distance along the arc before the first blocking cell
    per_step_risk: Optional[np.ndarray] = None   # (T,) geometric risk in [0,1]
    n_cells: int = 0
    off_map_frac: float = 0.0


def _footprint_offsets(margin: float) -> np.ndarray:
    """(K,2) body-frame offsets (x right, y forward) tiling the vehicle rectangle."""
    res = CFG.bev.res_m
    hw = CFG.ugv.width_m * 0.5 + margin
    hl = CFG.ugv.length_m * 0.5
    nx = max(1, int(np.ceil(2 * hw / res)))
    ny = max(1, int(np.ceil(2 * hl / res)))
    xs = np.linspace(-hw, hw, nx + 1, dtype=np.float32)
    ys = np.linspace(-hl, hl, ny + 1, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], 1)


_FP_CACHE: dict[float, np.ndarray] = {}


def footprint_offsets(margin: float = CFG.safety.corridor_margin_m) -> np.ndarray:
    k = round(float(margin), 4)
    if k not in _FP_CACHE:
        _FP_CACHE[k] = _footprint_offsets(k)
    return _FP_CACHE[k]


def sweep(state: np.ndarray, xy: np.ndarray, yaw: np.ndarray,
          margin: float = CFG.safety.corridor_margin_m,
          step_map: Optional[np.ndarray] = None) -> SweepMetrics:
    """Evaluate the swept vehicle footprint of a trajectory against the world state.

    `step_map` is `height_step_map(state)`; pass it in when sweeping many
    candidates against the same state so it is only computed once per frame.
    """
    b = CFG.bev
    off = footprint_offsets(margin)                     # (K,2)
    T = xy.shape[0]
    cs, sn = np.cos(yaw), np.sin(yaw)
    # rotate body offsets into the vehicle frame, then translate
    px = xy[:, 0:1] + cs[:, None] * off[None, :, 0] - sn[:, None] * off[None, :, 1]
    py = xy[:, 1:2] + sn[:, None] * off[None, :, 0] + cs[:, None] * off[None, :, 1]

    col = np.floor(px / b.res_m).astype(np.int32) + b.n_lateral
    row = (b.n_forward - 1) - np.floor(py / b.res_m).astype(np.int32)
    inside = (col >= 0) & (col < b.W) & (row >= 0) & (row < b.H)

    hgt = height_step_map(state) if step_map is None else step_map
    p_obst = state[CH_OBST]
    p_unk = state[CH_UNK]
    conf = state[CH_CONF]
    obs = state[CH_OBS] > 0.5

    m = SweepMetrics(per_step_risk=np.zeros(T, np.float32))
    steps = np.zeros(T, np.float32)
    obstf = np.zeros(T, np.float32)
    unkf = np.zeros(T, np.float32)
    conff = np.ones(T, np.float32)
    blocked_at = -1
    total_cells = 0

    arc_d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])

    for t in range(T):
        ins = inside[t]
        if not ins.any():
            unkf[t] = 1.0
            conff[t] = 0.0
            continue
        r, c = row[t][ins], col[t][ins]
        total_cells += r.size
        h = hgt[r, c]
        hf = np.isfinite(h)
        steps[t] = float(np.nanmax(h)) if hf.any() else 0.0
        obstf[t] = float(np.max(p_obst[r, c]))
        low = (~obs[r, c]) | (p_unk[r, c] > 0.5) | (conf[r, c] < CFG.safety.conf_unknown)
        unkf[t] = float(np.mean(low))
        conff[t] = float(np.mean(conf[r, c]))

        # Contact risk ONLY: obstacle posterior and height step.  Not-knowing is
        # deliberately excluded here and carried instead by `unknown_frac` /
        # `mean_conf`, which the planner costs and the supervisor gates in their
        # own rules (R5, R6).  Folding it in as well double-counted it, and
        # because two thirds of these maps are unknown it pinned the risk score
        # permanently above `risk_slow` so the system could never say GO.
        step_term = np.clip((steps[t] - CFG.ugv.max_step_m) /
                            max(CFG.ugv.clearance_m - CFG.ugv.max_step_m, 1e-3), 0.0, 1.0)
        hard = 1.0 if steps[t] > CFG.ugv.clearance_m else 0.0
        risk = max(obstf[t], hard, 0.75 * step_term)
        m.per_step_risk[t] = float(np.clip(risk, 0.0, 1.0))
        if blocked_at < 0 and (obstf[t] > 0.5 or steps[t] > CFG.ugv.clearance_m):
            blocked_at = t

    k = int(np.argmax(steps))
    m.max_step = float(steps[k])
    m.max_step_dist = float(arc_d[k])
    m.obstacle_frac = float(np.max(obstf))
    m.unknown_frac = float(np.mean(unkf))
    m.mean_conf = float(np.mean(conff))
    m.n_cells = int(total_cells)
    m.off_map_frac = float(1.0 - inside.mean())
    m.free_distance = float(arc_d[blocked_at]) if blocked_at >= 0 else float(arc_d[-1] + 1e-3)

    # lateral clearance: nearest obstacle-ish cell to any point on the path
    hard = (p_obst > 0.5) | (np.isfinite(hgt) & (hgt > CFG.ugv.clearance_m))
    if hard.any():
        rr, cc = np.nonzero(hard)
        ox, oy = bev_to_veh(rr.astype(np.float32), cc.astype(np.float32))
        d = np.sqrt((ox[None, :] - xy[:, 0:1]) ** 2 + (oy[None, :] - xy[:, 1:2]) ** 2)
        m.min_clearance = float(np.min(d))
    return m


def sweep_clearance_series(state: np.ndarray, xy: np.ndarray,
                           step_map: Optional[np.ndarray] = None) -> np.ndarray:
    """(T,) per-step lateral clearance to the nearest blocking cell, metres."""
    hgt = height_step_map(state) if step_map is None else step_map
    hard = (state[CH_OBST] > 0.5) | (np.isfinite(hgt) & (hgt > CFG.ugv.clearance_m))
    T = xy.shape[0]
    if not hard.any():
        return np.full(T, 9.9, np.float32)
    rr, cc = np.nonzero(hard)
    ox, oy = bev_to_veh(rr.astype(np.float32), cc.astype(np.float32))
    d = np.sqrt((ox[None, :] - xy[:, 0:1]) ** 2 + (oy[None, :] - xy[:, 1:2]) ** 2)
    return np.min(d, axis=1).astype(np.float32)


def collision_risk_target(state: np.ndarray, action: int,
                          dt: float = WM_DT, n_steps: int = 4) -> float:
    """Supervised target for the world model's collision-risk head.

    Sweeps the vehicle footprint along the action's arc through the (already
    warped) world state and returns the peak geometric risk in [0,1].
    """
    v, w = action_motion(action)
    if v < 1e-6:
        # STOP: the only way to be at risk standing still is to already be inside
        # something, so score the stationary footprint.
        xy = np.zeros((1, 2), np.float32)
        yaw = np.zeros(1, np.float32)
    else:
        xy, yaw = arc_poses(v, w, dt / n_steps, n_steps)
    return float(np.max(sweep(state, xy, yaw).per_step_risk))


def _footprint_cells(xy: np.ndarray, yaw: np.ndarray,
                     margin: float = CFG.safety.corridor_margin_m):
    """(rows, cols) of the BEV cells covered by the footprint along a path."""
    b = CFG.bev
    off = footprint_offsets(margin)
    cs, sn = np.cos(yaw), np.sin(yaw)
    px = xy[:, 0:1] + cs[:, None] * off[None, :, 0] - sn[:, None] * off[None, :, 1]
    py = xy[:, 1:2] + sn[:, None] * off[None, :, 0] + cs[:, None] * off[None, :, 1]
    col = np.floor(px / b.res_m).astype(np.int32) + b.n_lateral
    row = (b.n_forward - 1) - np.floor(py / b.res_m).astype(np.int32)
    ok = (col >= 0) & (col < b.W) & (row >= 0) & (row < b.H)
    return row[ok], col[ok]


def geometric_collision(state: np.ndarray, action: int, dt: float = WM_DT,
                        n_steps: int = 4,
                        step_map: Optional[np.ndarray] = None) -> bool:
    """Strict geometric contact test used to *score* policies, not to train them.

    A collision is the vehicle **driving into** something: the footprint has to
    enter a cell it was not already occupying, and that new cell has to be an
    OBSTACLE cell or ground whose local height step exceeds the chassis
    clearance.  Two exclusions matter, and both were wrong in an earlier version
    of this function:

    * a stationary vehicle cannot collide, so v ~ 0 always returns False;
    * cells already under the footprint at t = 0 are excluded, so a map that
      paints an obstacle posterior onto the vehicle's own cell (which the
      geometric-fallback mapping does near the ego mask) does not make every
      policy look like it crashed.

    The unknown / low-confidence term is deliberately excluded too: not being
    able to see somewhere is a reason to slow down, not evidence of contact.
    """
    v, w = action_motion(action)
    if v < 1e-6:
        return False
    xy, yaw = arc_poses(v, w, dt / n_steps, n_steps)
    r_new, c_new = _footprint_cells(xy, yaw)
    if r_new.size == 0:
        return False
    r0, c0 = _footprint_cells(np.zeros((1, 2), np.float32), np.zeros(1, np.float32))
    b = CFG.bev
    already = np.zeros((b.H, b.W), bool)
    already[r0, c0] = True
    fresh = ~already[r_new, c_new]
    if not fresh.any():
        return False
    r_new, c_new = r_new[fresh], c_new[fresh]
    sm = height_step_map(state) if step_map is None else step_map
    h = sm[r_new, c_new]
    return bool(np.any(state[CH_OBST][r_new, c_new] > 0.5) or
                np.any(np.isfinite(h) & (h > CFG.ugv.clearance_m)))


# ------------------------------------------------------------------ cache loading

_BEV_KEYS = {
    "height": ("height", "bev_height", "h", "z"),
    "trav_prob": ("trav_prob", "bev_trav_prob", "prob", "trav_probs"),
    "trav": ("trav", "bev_trav", "trav_label", "label"),
    "conf": ("conf", "bev_conf", "confidence"),
    "age": ("age", "bev_age"),
    "hits": ("hits", "bev_hits", "count"),
    "terrain": ("terrain", "bev_terrain"),
}


def _pick(d: dict, names: Sequence[str]):
    for n in names:
        if n in d:
            return d[n]
    return None


def load_bev_states(clip_id: str) -> Optional[np.ndarray]:
    """(N, 8, H, W) float32 world states from the cached `bev` stage, or None.

    Tolerant of the exact key names another agent used, because the mapping
    stage is produced in parallel with this one.
    """
    if not has_stage(clip_id, "bev"):
        return None
    d = load_stage(clip_id, "bev")
    hgt = _pick(d, _BEV_KEYS["height"])
    if hgt is None:
        return None
    hgt = np.asarray(hgt, np.float32)
    if hgt.ndim == 2:
        hgt = hgt[None]
    N = hgt.shape[0]
    b = CFG.bev

    def grab(key, default):
        a = _pick(d, _BEV_KEYS[key])
        if a is None:
            return None
        a = np.asarray(a, np.float32)
        return a if a.ndim >= 3 else a[None]

    tp = grab("trav_prob", None)
    conf = grab("conf", None)
    age = grab("age", None)
    hits = grab("hits", None)
    trav = _pick(d, _BEV_KEYS["trav"])

    out = np.empty((N, BEV_CH, b.H, b.W), np.float32)
    for i in range(N):
        if tp is not None and tp.ndim == 4:
            tpi = tp[i]
        elif trav is not None:
            lab = np.asarray(trav, np.uint8)
            lab = lab[i] if lab.ndim == 3 else lab
            tpi = np.zeros((N_TRAV, b.H, b.W), np.float32)
            for c in range(N_TRAV):
                tpi[c] = (lab == c)
        else:
            tpi = None
        out[i] = pack_state(hgt[i],
                            tpi if tpi is not None else np.zeros((N_TRAV, b.H, b.W), np.float32),
                            conf[i] if conf is not None else np.ones((b.H, b.W), np.float32) * 0.6,
                            age[i] if age is not None else np.zeros((b.H, b.W), np.float32),
                            hits[i] if hits is not None else None)
    return out


_ODOM_KEYS = {
    "speed": ("speed_mps", "speed", "v"),
    "d_yaw": ("d_yaw", "dyaw", "yaw_rate"),
    "d_trans": ("d_trans", "dtrans", "trans"),
    "ok": ("tracking_ok", "ok", "track_ok"),
    "quality": ("track_quality", "quality", "trackq"),
}


def load_odom(clip_id: str) -> Optional[dict]:
    """Per-frame odometry arrays from the cached `odom` stage, or None."""
    if not has_stage(clip_id, "odom"):
        return None
    d = load_stage(clip_id, "odom")
    sp = _pick(d, _ODOM_KEYS["speed"])
    dy = _pick(d, _ODOM_KEYS["d_yaw"])
    if sp is None and dy is None:
        return None
    n = len(np.atleast_1d(sp if sp is not None else dy))
    ok = _pick(d, _ODOM_KEYS["ok"])
    return {
        "speed_mps": np.asarray(sp, np.float32).ravel() if sp is not None else np.zeros(n, np.float32),
        "d_yaw": np.asarray(dy, np.float32).ravel() if dy is not None else np.zeros(n, np.float32),
        "d_trans": np.asarray(_pick(d, _ODOM_KEYS["d_trans"]), np.float32).ravel()
        if _pick(d, _ODOM_KEYS["d_trans"]) is not None else np.zeros(n, np.float32),
        "tracking_ok": np.asarray(ok).ravel().astype(bool) if ok is not None
        else np.ones(n, bool),
        "track_quality": np.asarray(_pick(d, _ODOM_KEYS["quality"]), np.float32).ravel()
        if _pick(d, _ODOM_KEYS["quality"]) is not None else np.ones(n, np.float32),
    }


def label_action(speed_mps: float, yaw_rate: float) -> int:
    """Discrete action whose ACTION_CMD prototype best matches a measured motion.

    Nearest prototype in a normalised (linear, angular) space; the angular term is
    scaled so that a 0.9 rad/s yaw error costs the same as a 1.0 m/s speed error,
    which is the ratio between FORWARD and LEFT/RIGHT in ACTION_CMD.  STOP wins
    whenever the vehicle is essentially stationary.
    """
    v = float(abs(speed_mps))
    w = float(yaw_rate)
    if v < 0.08 and abs(w) < 0.15:
        return ACTIONS.index("STOP")
    proto = ACTION_CMD.astype(np.float64)
    d = (proto[:, 0] - v) ** 2 + ((proto[:, 1] - w) / 0.9) ** 2
    d[ACTIONS.index("STOP")] += 4.0          # never match STOP by distance alone
    return int(np.argmin(d))


def label_action_from_maps(state_t: np.ndarray, state_next: np.ndarray,
                           dt: float = WM_DT) -> tuple[int, np.ndarray]:
    """Fallback action label when no VO track is cached: match the map's own motion.

    Warps `state_t` under each action prototype and picks the action whose warp
    best explains `state_next` (L1 over the obstacle posterior and the height
    step, on the cells observed in both maps).  This is strictly weaker than the
    VO-based label - it can only resolve motion that actually moved the map - so
    the trainer prints which of the two labelled the data.

    Returns (action index, per-action residual).
    """
    both_ref = state_next[CH_OBS] > 0.5
    res = np.zeros(len(ACTIONS), np.float64)
    for a in range(len(ACTIONS)):
        v, w = action_motion(a)
        wp = warp_state(state_t, v, w, dt)
        m = both_ref & (wp[CH_OBS] > 0.5)
        n = int(m.sum())
        if n < 200:
            res[a] = np.inf
            continue
        res[a] = (np.abs(wp[CH_OBST][m] - state_next[CH_OBST][m]).mean()
                  + 0.5 * np.abs(wp[CH_HEIGHT][m] - state_next[CH_HEIGHT][m]).mean()
                  + 0.25 * np.abs(wp[CH_SAFE][m] - state_next[CH_SAFE][m]).mean())
    if not np.isfinite(res).any():
        return ACTIONS.index("FORWARD"), res
    return int(np.nanargmin(np.where(np.isfinite(res), res, np.inf))), res


# ------------------------------------------------------------------ synthetic scenes

def synthetic_state(kind: str = "clear", rng: Optional[np.random.Generator] = None,
                    wall_dist_m: float = 1.0, wall_height_m: float = 0.10,
                    conf_level: float = 0.8) -> np.ndarray:
    """Hand-built stand-in world states used before the real `bev` cache exists.

    kinds: 'clear', 'wall', 'wall_left', 'low_conf', 'kerb', 'blind'
    A synthetic scene with a known obstacle is also the right unit test: we can
    assert the supervisor says STOP for a 10 cm wall 1 m ahead and GO otherwise.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    b = CFG.bev
    H, W = b.H, b.W
    height = np.full((H, W), np.nan, np.float32)
    tp = np.zeros((N_TRAV, H, W), np.float32)
    conf = np.zeros((H, W), np.float32)
    age = np.full((H, W), float(b.max_age_frames), np.float32)
    hits = np.zeros((H, W), np.float32)

    rr, cc = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32),
                         indexing="ij")
    X, Y = bev_to_veh(rr, cc)

    # observable wedge: a 92 deg camera sees a triangle in front of the vehicle
    fov = np.deg2rad(CFG.cam.hfov_deg * 0.5)
    seen = (Y > 0.05) & (np.abs(X) < np.tan(fov) * Y + 0.15) & (Y < 6.5)

    ground = 0.008 * np.sin(Y * 1.7) + 0.004 * np.cos(X * 3.1) + rng.normal(0, 0.0025, (H, W))
    height[seen] = ground[seen]
    conf[seen] = np.clip(conf_level * np.exp(-np.maximum(Y[seen] - 1.0, 0) / 4.0) +
                         rng.normal(0, 0.03, seen.sum()), 0.05, 1.0)
    age[seen] = rng.uniform(0, 6, seen.sum())
    hits[seen] = rng.uniform(1, 8, seen.sum())

    trail = seen & (np.abs(X) < 0.9)
    grass = seen & ~trail
    tp[SAFE][trail] = 0.88
    tp[RISKY][trail] = 0.09
    tp[UNKNOWN][trail] = 0.03
    tp[SAFE][grass] = 0.42
    tp[RISKY][grass] = 0.48
    tp[UNKNOWN][grass] = 0.10

    if kind in ("wall", "wall_left", "kerb"):
        d = wall_dist_m
        hgt = wall_height_m if kind != "kerb" else 0.055
        band = seen & (np.abs(Y - d) < 0.16)
        if kind == "wall_left":
            band &= (X < 0.05)
        elif kind == "kerb":
            band &= (np.abs(X) < 1.4)
        else:
            band &= (np.abs(X) < 0.75)
        height[band] = hgt
        tp[:, band] = 0.0
        tp[OBSTACLE][band] = 0.93
        tp[RISKY][band] = 0.07
        conf[band] = np.clip(conf_level, 0.05, 1.0)
    if kind == "low_conf":
        far = seen & (Y > 1.2)
        conf[far] *= 0.25
        tp[UNKNOWN][far] = 0.75
        tp[SAFE][far] *= 0.25
        tp[RISKY][far] *= 0.25
    if kind == "blind":
        conf[seen] *= 0.15
        tp[:, seen] = 0.0
        tp[UNKNOWN][seen] = 1.0

    s = tp.sum(0)
    tp = tp / np.maximum(s, 1e-5)
    tp[:, s < 1e-5] = 0.0
    tp[UNKNOWN, s < 1e-5] = 1.0
    return pack_state(height, tp, conf, age, hits)


def synthetic_sequence(n: int = 64, kind: str = "mixed",
                       seed: int = 0) -> np.ndarray:
    """(n, 8, H, W) plausible stand-in clip: drive forward through a scene."""
    rng = np.random.default_rng(seed)
    kinds = ["clear", "wall", "low_conf", "kerb", "wall_left"]
    base_kind = rng.choice(kinds) if kind == "mixed" else kind
    d0 = float(rng.uniform(2.2, 4.5))
    st = synthetic_state(base_kind, rng, wall_dist_m=d0,
                         wall_height_m=float(rng.uniform(0.06, 0.16)),
                         conf_level=float(rng.uniform(0.5, 0.9)))
    out = np.empty((n, BEV_CH, CFG.bev.H, CFG.bev.W), np.float32)
    out[0] = st
    for i in range(1, n):
        v = float(rng.uniform(0.3, 1.0))
        w = float(rng.normal(0.0, 0.35))
        st = warp_state(st, v, w, WM_DT)
        out[i] = st
    return out


# ------------------------------------------------------------------ preview render

def render_state_bgr(state: np.ndarray) -> np.ndarray:
    """(H, W, 3) BGR quick-look of a world state: trav colour modulated by confidence."""
    from ..config import TRAV_COLORS_BGR
    lab = state_trav_label(state)
    img = TRAV_COLORS_BGR[lab].astype(np.float32)
    shade = 0.30 + 0.70 * state[CH_CONF][..., None]
    img *= shade
    img[state[CH_OBS] < 0.5] = (34, 31, 28)
    return np.clip(img, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    import time
    print(f"BEV state: {BEV_CH} ch x {CFG.bev.H} x {CFG.bev.W}, "
          f"res {CFG.bev.res_m} m, WM_DT {WM_DT:.4f} s, frame stride {WM_FRAME_STRIDE}")
    for k in ("clear", "wall", "low_conf", "blind"):
        st = synthetic_state(k, wall_dist_m=1.0, wall_height_m=0.10)
        v, w = action_motion(0)
        xy, yaw = arc_poses(v, w, CFG.safety.horizon_s / CFG.safety.n_rollout_steps,
                            CFG.safety.n_rollout_steps)
        m = sweep(st, xy, yaw)
        print(f"{k:9s} max_step={m.max_step:+.3f} m @ {m.max_step_dist:.2f} m  "
              f"obst={m.obstacle_frac:.2f} unk={m.unknown_frac:.2f} conf={m.mean_conf:.2f} "
              f"free={m.free_distance:.2f} m  risk_peak={m.per_step_risk.max():.2f}")
    st = synthetic_state("wall")
    t0 = time.perf_counter()
    for _ in range(20):
        warp_state(st, 1.0, 0.3, WM_DT)
    print(f"warp_state: {(time.perf_counter()-t0)/20*1e3:.2f} ms  "
          f"occ16 sum={occupancy_grid(st).sum():.1f}  mean_trav={mean_traversability(st):.3f}")
    print("action labelling:",
          [(round(v, 2), round(w, 2), ACTIONS[label_action(v, w)])
           for v, w in [(1.0, 0.0), (0.7, 0.9), (0.7, -0.9), (0.3, 0.0), (0.0, 0.0), (0.45, 1.6)]])
