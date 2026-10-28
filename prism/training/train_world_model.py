"""Train the prism tiny latent world model.

Data
----
Two sources, kept explicitly separate and reported separately:

1. **Real transitions** - from the cached `bev` maps of the five clips plus the
   cached `odom` track.  A transition is (map at frame t) -> (maps at frames
   t+s, t+2s, ... t+6s) with s = `bev_utils.WM_FRAME_STRIDE` (10 frames = 0.333 s,
   so six steps span the 2 s planner horizon).  The *action label* of a recorded
   transition is the discrete action of `CFG.ACTIONS` whose `CFG.ACTION_CMD`
   prototype (linear m/s, angular rad/s) is nearest to the motion actually
   measured by visual odometry over that window - see `bev_utils.label_action`.
   The angular residual is normalised by 0.9 rad/s so that a full-scale yaw error
   costs the same as a 1 m/s speed error, and STOP is only selected when the
   vehicle is genuinely stationary.

2. **Synthetic (kinematic) transitions** - five 10 s clips give of order 1.5 k
   real transitions, all of them for whatever action the driver happened to take,
   which means the counterfactual branches the planner cares about ("what if I
   turn left here?") are never observed.  So each anchor map is additionally
   **rigidly warped under every candidate action's motion model**
   (`bev_utils.warp_state`, done batched on GPU here).  This is exact geometry for
   the part of the scene that was already observed and honest UNKNOWN for
   everything that scrolls into view.  **It is a kinematic augmentation, not
   observed data** - it cannot teach the model what is around a corner, only how
   the known world moves under a commanded arc.  Every number reported below says
   which mixture it came from.

Losses
------
* latent consistency  - MSE between the rolled-out latent and the (stop-grad)
  encoding of the true future map;
* occupancy BCE + L1  - 16x16 obstacle-occupancy grid decoded from the rolled-out
  latent vs. the true future map's obstacle posterior.  Obstacle posterior only:
  an earlier target that also counted unknown-with-low-confidence cells was
  nearly constant (mean 0.50, across-action spread 0.068) because two thirds of
  these maps are a static unknown wedge, and the head collapsed to the marginal.
  The L1 term keeps the sparse-positive upweighting from inflating the output;
* traversability MSE  - scalar mean traversability of the forward corridor;
* collision-risk BCE  - target is the peak *contact* score under the vehicle
  footprint swept along that action's arc, read off the CURRENT map through the
  cumulative corridor masks: OBSTACLE posterior and height step above
  `ugv.clearance_m`.  Unknown space is excluded on purpose - the supervisor gates
  it under its own rules, and counting it here both double-counted it and pinned
  the score above `risk_slow` on every frame.
* grounding term      - the same three heads applied to the *encoder* output of
  real maps, which stops the latent-consistency term collapsing to a constant.

Training is multi-step: the rollout is unrolled `CFG.wm.pred_horizon` steps and
gradients flow through the whole unroll.

Run:  python -m prism.training.train_world_model [--steps 4000]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG, CLIP_IDS, CKPT_DIR, WORK_DIR, ACTIONS, N_ACTIONS
from ..io_utils import set_seed
from ..models.world_model import PrismWorldModel, WM_CKPT
from ..nav import bev_utils as bu

LOG_NPZ = WORK_DIR / "wm_train_log.npz"


# ==================================================================== torch ops

def _grid_for(action: int, k_steps: int, device) -> torch.Tensor:
    """(1, H, W, 2) grid_sample lookup taking the *new* map back to the *old* map."""
    b = CFG.bev
    v, w = bu.action_motion(action)
    if v < 1e-9 and abs(w) < 1e-9:
        tx = ty = dyaw = 0.0
    else:
        xy, yaw = bu.arc_poses(v, w, bu.WM_DT, k_steps)
        tx, ty = float(xy[-1, 0]), float(xy[-1, 1])
        dyaw = float(yaw[-1])
    rr, cc = np.meshgrid(np.arange(b.H, dtype=np.float32),
                         np.arange(b.W, dtype=np.float32), indexing="ij")
    from ..perception.geometry import bev_to_veh
    xn, yn = bev_to_veh(rr, cc)
    cs, sn = np.cos(dyaw), np.sin(dyaw)
    xo = tx + cs * xn - sn * yn
    yo = ty + sn * xn + cs * yn
    col = xo / b.res_m + b.n_lateral - 0.5
    row = (b.n_forward - 1) + 0.5 - yo / b.res_m
    gx = 2.0 * col / (b.W - 1) - 1.0
    gy = 2.0 * row / (b.H - 1) - 1.0
    g = np.stack([gx, gy], -1)[None].astype(np.float32)
    return torch.from_numpy(g).to(device)


class WarpBank:
    """Precomputed GPU warps for every (action, horizon step) pair."""

    def __init__(self, device, horizon: int = CFG.wm.pred_horizon):
        self.device = device
        self.horizon = horizon
        self.grids = [[_grid_for(a, k + 1, device) for k in range(horizon)]
                      for a in range(N_ACTIONS)]

    def warp(self, x: torch.Tensor, action: int, k: int) -> torch.Tensor:
        """x (B, 8, H, W) at t -> the same map re-expressed at t + (k+1)*WM_DT."""
        g = self.grids[action][k].expand(x.shape[0], -1, -1, -1)
        y = F.grid_sample(x, g, mode="bilinear", padding_mode="zeros",
                          align_corners=True)
        return _fix_unseen(y, (k + 1) * bu.WM_DT)


def _fix_unseen(y: torch.Tensor, dt_total: float) -> torch.Tensor:
    """Cells that scrolled in from outside the map are unobserved -> UNKNOWN."""
    obs = (y[:, bu.CH_OBS:bu.CH_OBS + 1] > 0.5).float()
    y[:, bu.CH_OBS:bu.CH_OBS + 1] = obs
    m = obs
    y[:, bu.CH_HEIGHT:bu.CH_HEIGHT + 1] *= m
    for c in (bu.CH_SAFE, bu.CH_RISKY, bu.CH_OBST):
        y[:, c:c + 1] *= m
    y[:, bu.CH_UNK:bu.CH_UNK + 1] = y[:, bu.CH_UNK:bu.CH_UNK + 1] * m + (1 - m)
    y[:, bu.CH_CONF:bu.CH_CONF + 1] *= m
    y[:, bu.CH_AGE:bu.CH_AGE + 1] = torch.clamp(
        y[:, bu.CH_AGE:bu.CH_AGE + 1] * m + (1 - m) +
        (dt_total * 30.0) / CFG.bev.max_age_frames * m, 0.0, 1.0)
    p = y[:, bu.CH_SAFE:bu.CH_UNK + 1]
    y[:, bu.CH_SAFE:bu.CH_UNK + 1] = p / p.sum(1, keepdim=True).clamp_min(1e-5)
    return y


def corridor_masks(device, horizon: int = CFG.wm.pred_horizon) -> torch.Tensor:
    """(A, T, H, W) cumulative footprint corridors, in the frame at t = 0.

    `masks[a, k]` covers everything the vehicle footprint sweeps if it executes
    action `a` from now until (k+1) * WM_DT.  The collision-risk target is read
    off the *current* map through these masks rather than off a warped future
    map, and that matters: warping scrolls unseen cells into view with a zero
    obstacle posterior, so a target read from the warped map scored every turn as
    perfectly safe (measured: 0.000 for LEFT / RIGHT / REROUTE) simply because
    the model had warped away the evidence.  Read through the current map, the
    target is "does the arc I am about to drive hit anything I can see right
    now", which is exactly what the planner sweeps and the supervisor thresholds.
    """
    b = CFG.bev
    masks = np.zeros((N_ACTIONS, horizon, b.H, b.W), np.float32)
    off = bu.footprint_offsets()
    sub = 4
    for a in range(N_ACTIONS):
        v, w = bu.action_motion(a)
        n = horizon * sub
        if v < 1e-6:
            xy = np.zeros((n, 2), np.float32)
            yaw = np.zeros(n, np.float32)
        else:
            xy, yaw = bu.arc_poses(v, w, bu.WM_DT / sub, n)
        cs, sn = np.cos(yaw), np.sin(yaw)
        px = xy[:, 0:1] + cs[:, None] * off[None, :, 0] - sn[:, None] * off[None, :, 1]
        py = xy[:, 1:2] + sn[:, None] * off[None, :, 0] + cs[:, None] * off[None, :, 1]
        col = np.floor(px / b.res_m).astype(np.int32) + b.n_lateral
        row = (b.n_forward - 1) - np.floor(py / b.res_m).astype(np.int32)
        ok = (col >= 0) & (col < b.W) & (row >= 0) & (row < b.H)
        # always include the stationary footprint so STOP has a defined corridor
        r0, c0 = bu._footprint_cells(np.zeros((1, 2), np.float32), np.zeros(1, np.float32))
        for k in range(horizon):
            hi = (k + 1) * sub
            m = np.zeros((b.H, b.W), np.float32)
            m[r0, c0] = 1.0
            sel = ok[:hi]
            m[row[:hi][sel], col[:hi][sel]] = 1.0
            masks[a, k] = m
    return torch.from_numpy(masks).to(device)


def _height_m(x: torch.Tensor) -> torch.Tensor:
    b = CFG.bev
    return (x[:, bu.CH_HEIGHT] + 1.0) * 0.5 * (b.z_max - b.z_min) + b.z_min


def _step_map(x: torch.Tensor) -> torch.Tensor:
    """Batched torch twin of `bev_utils.height_step_map` (white top-hat).

    Keeping the two implementations in lock-step matters: the collision-risk head
    is trained against this, and the supervisor thresholds the numpy version.
    """
    k, p = bu.STEP_WIN, bu.STEP_WIN // 2
    h = (_height_m(x) * x[:, bu.CH_OBS])[:, None]
    ero = -F.max_pool2d(-h, k, stride=1, padding=p)
    opened = F.max_pool2d(ero, k, stride=1, padding=p)
    step = (h - opened).clamp_min(0.0)
    step = -F.max_pool2d(-step, 3, stride=1, padding=1)
    return (step * x[:, bu.CH_OBS][:, None]).squeeze(1)


def risk_target(x: torch.Tensor, action: int, masks: torch.Tensor,
                k_step: int = 0) -> torch.Tensor:
    """(B,) geometric collision-risk target in [0,1] for `action` in map `x`.

    Mirrors `bev_utils.sweep`: the peak *contact* term under the swept footprint
    (obstacle posterior, height step above chassis clearance).  Unknown space is
    NOT part of this score - the supervisor gates it separately - so that
    `risk_slow / risk_reroute / risk_stop` threshold a quantity that means "the
    footprint hits something", which is what their names claim.
    """
    m = masks[action][k_step][None]                    # (1,H,W)
    h = _step_map(x)
    step_term = ((h - CFG.ugv.max_step_m) /
                 max(CFG.ugv.clearance_m - CFG.ugv.max_step_m, 1e-3)).clamp(0, 1)
    hard_cell = torch.maximum(x[:, bu.CH_OBST],
                              torch.maximum((h > CFG.ugv.clearance_m).float(),
                                            0.75 * step_term))
    return torch.clamp((hard_cell * m).amax(dim=(1, 2)), 0.0, 1.0)


#: BCE positive weighting for the occupancy head.  The obstacle-only target has a
#: mean around 0.10 on these clips.  A little upweighting keeps the sparse
#: positives from being ignored; too much (4.0 was tried) inflates the predicted
#: probability far above the target mean and wrecks the MAE, so this is kept mild
#: and paired with an explicit L1 calibration term in the loss.
OCC_POS_WEIGHT = 2.5


def occ_target(x: torch.Tensor, g: int = bu.OCC_G) -> torch.Tensor:
    """Torch twin of `bev_utils.occupancy_grid` - obstacle posterior only."""
    occ = x[:, bu.CH_OBST].clamp(0, 1)[:, None]
    return F.avg_pool2d(occ, (CFG.bev.H // g, CFG.bev.W // g)).squeeze(1)


_TRAV_SLICE = None


def trav_target(x: torch.Tensor) -> torch.Tensor:
    global _TRAV_SLICE
    b = CFG.bev
    if _TRAV_SLICE is None:
        half = int(round((CFG.ugv.width_m * 0.5 + CFG.safety.corridor_margin_m) / b.res_m)) + 1
        _TRAV_SLICE = (max(b.H - int(round(3.0 / b.res_m)), 0),
                       max(b.n_lateral - 2 * half, 0), min(b.n_lateral + 2 * half, b.W))
    r0, c0, c1 = _TRAV_SLICE
    sub = x[:, :, r0:, c0:c1]
    return (sub[:, bu.CH_SAFE] + 0.4 * sub[:, bu.CH_RISKY]).mean(dim=(1, 2)).clamp(0, 1)


# ==================================================================== dataset

class TransitionStore:
    """Anchor world states + (where available) the real observed futures."""

    def __init__(self, frame_step: int = 2, max_clips: int = len(CLIP_IDS)):
        self.clips: list[dict] = []
        self._pool: list[tuple[int, int]] = []
        self._vo_clips: list[str] = []
        self._mm_clips: list[str] = []
        self.label_hist = np.zeros(N_ACTIONS, np.int64)
        self.n_real = 0
        self.source = []
        stride = bu.WM_FRAME_STRIDE
        for cid in CLIP_IDS[:max_clips]:
            states = bu.load_bev_states(cid)
            if states is None:
                continue
            odom = bu.load_odom(cid)
            sub = states[::frame_step].astype(np.float16)
            k = max(1, stride // frame_step)
            n = sub.shape[0]
            labels = np.zeros(n, np.int64)
            valid = np.zeros(n, bool)
            H = CFG.wm.pred_horizon
            if odom is not None:
                sp, dy, ok = odom["speed_mps"], odom["d_yaw"], odom["tracking_ok"]
                self._vo_clips.append(cid)
                for i in range(n):
                    f0 = i * frame_step
                    f1 = min(f0 + stride, len(sp))
                    if f1 <= f0:
                        continue
                    v = float(np.mean(sp[f0:f1]))
                    w = float(np.sum(dy[f0:f1]) / max((f1 - f0) / 30.0, 1e-3))
                    labels[i] = bu.label_action(v, w)
                    valid[i] = bool(np.all(ok[f0:f1])) and (i + H * k) < n
            else:
                self._mm_clips.append(cid)
                for i in range(n):
                    if (i + H * k) >= n:
                        continue
                    labels[i] = bu.label_action_from_maps(
                        sub[i].astype(np.float32), sub[i + k].astype(np.float32))[0]
                    valid[i] = True
            self.clips.append({"id": cid, "states": sub, "labels": labels,
                               "valid": valid, "k": k})
            self.n_real += int(valid.sum())
            self.label_hist += np.bincount(labels[valid], minlength=N_ACTIONS)
            self.source.append(cid)
        parts = []
        if self._vo_clips:
            parts.append("visual odometry (speed, yaw rate) -> nearest ACTION_CMD for "
                         + ", ".join(self._vo_clips))
        if self._mm_clips:
            parts.append("BEV map-matching fallback (no `odom` cache; the action whose warp "
                         "best explains the next map) for " + ", ".join(self._mm_clips))
        self.label_source = "; ".join(parts) if parts else "n/a"
        self.have_real = len(self.clips) > 0
        if not self.have_real:
            # stand-in scenes so the pipeline is trainable and testable before the
            # mapping agent's `bev` cache lands.
            seqs = [bu.synthetic_sequence(48, "mixed", seed=s) for s in range(12)]
            self.anchors = np.concatenate(seqs, 0).astype(np.float16)
        else:
            self.anchors = np.concatenate([c["states"] for c in self.clips], 0)

    # ---- sampling
    def sample_real(self, rng, batch: int):
        """-> (x0 (B,8,H,W), actions (B,), futures (B,H,8,h,w)) or None."""
        if not self.have_real:
            return None
        if not self._pool:
            for ci, c in enumerate(self.clips):
                for i in np.nonzero(c["valid"])[0]:
                    self._pool.append((ci, int(i)))
            # Inverse-frequency weights.  Whichever labeller is available, the
            # driver's action distribution is wildly unbalanced (the RC car mostly
            # drives straight), and without reweighting 40% of every batch would
            # be the same action.  This lets the handful of real turns count.
            cnt = np.bincount([self.clips[ci]["labels"][i] for ci, i in self._pool],
                              minlength=N_ACTIONS).astype(np.float64)
            self._pool_w = np.array(
                [1.0 / max(cnt[self.clips[ci]["labels"][i]], 1.0) for ci, i in self._pool])
            self._pool_w /= self._pool_w.sum()
        pool = self._pool
        if not pool:
            return None
        pick = rng.choice(len(pool), size=min(batch, len(pool)),
                          replace=len(pool) < batch, p=self._pool_w)
        H = CFG.wm.pred_horizon
        x0, acts, fut = [], [], []
        for p in np.atleast_1d(pick):
            ci, i = pool[int(p)]
            c = self.clips[ci]
            k = c["k"]
            x0.append(c["states"][i])
            acts.append(c["labels"][i])
            fut.append(np.stack([c["states"][min(i + (j + 1) * k, c["states"].shape[0] - 1)]
                                 for j in range(H)]))
        return (np.stack(x0).astype(np.float32), np.array(acts, np.int64),
                np.stack(fut).astype(np.float32))

    def sample_anchor(self, rng, batch: int) -> np.ndarray:
        idx = rng.integers(0, self.anchors.shape[0], batch)
        return self.anchors[idx].astype(np.float32)

    def n_anchors(self) -> int:
        return int(self.anchors.shape[0])


# ==================================================================== training

@torch.no_grad()
def _risk_targets(targets: torch.Tensor, x0: torch.Tensor, groups, masks):
    """Per-sample collision-risk targets for the executed action.

    Read off the *current* map `x0` through the cumulative corridor of each
    horizon step (see `corridor_masks`).  Returns (per-step target (B,H),
    one-step target (B,) used to ground the encoder's own latent).
    """
    B, H = targets.shape[0], targets.shape[1]
    risk_t = torch.zeros(B, H, device=x0.device)
    for lo, hi, a in groups:
        for k in range(H):
            risk_t[lo:hi, k] = risk_target(x0[lo:hi], a, masks, k)
    return risk_t, risk_t[:, 0].clone()


def _losses(model, x0, targets, groups, masks, action_ids):
    """targets: (B, H, 8, h, w) true future maps. groups: [(lo, hi, action)]."""
    B, H = targets.shape[0], targets.shape[1]
    s0 = model.encode(x0)
    acts = action_ids[:, None].expand(B, H)
    pred_states = model.dynamics.rollout(s0, acts)                    # (B,H,S)

    flat = targets.reshape(B * H, *targets.shape[2:])
    with torch.no_grad():
        s_true = model.encode(flat).view(B, H, -1)
        occ_t = occ_target(flat).view(B, H, bu.OCC_G, bu.OCC_G)
        trav_t = trav_target(flat).view(B, H)
        risk_t, risk_gt = _risk_targets(targets, x0, groups, masks)
        occ_gt = occ_target(x0)
        trav_gt = trav_target(x0)

    occ_l, trav_l, risk_l = model.decode_logits(pred_states)
    l_lat = F.mse_loss(pred_states, s_true)
    pw = torch.tensor(OCC_POS_WEIGHT, device=x0.device)
    l_occ = (F.binary_cross_entropy_with_logits(occ_l, occ_t, pos_weight=pw)
             + 1.0 * F.l1_loss(torch.sigmoid(occ_l), occ_t))
    l_trav = F.mse_loss(torch.sigmoid(trav_l), trav_t)
    l_risk = F.binary_cross_entropy_with_logits(risk_l, risk_t)

    # grounding: heads must also explain the encoder's own latent of a real map
    occ_g, trav_g, risk_g = model.decode_logits(s0)
    l_ground = (F.binary_cross_entropy_with_logits(occ_g, occ_gt, pos_weight=pw) +
                1.0 * F.l1_loss(torch.sigmoid(occ_g), occ_gt) +
                F.mse_loss(torch.sigmoid(trav_g), trav_gt) +
                F.binary_cross_entropy_with_logits(risk_g, risk_gt))

    total = 1.0 * l_lat + 1.0 * l_occ + 0.5 * l_trav + 1.5 * l_risk + 0.5 * l_ground
    return total, {"lat": l_lat.item(), "occ": l_occ.item(), "trav": l_trav.item(),
                   "risk": l_risk.item(), "ground": l_ground.item(),
                   "total": total.item()}


@torch.no_grad()
def evaluate(model, store, bank, masks, device, rng, n_batches: int = 24,
             batch: int = 8, real_frac: float = 0.0) -> dict:
    """Open-loop multi-step error of the model vs. a persistence baseline.

    Persistence = "the world does not change": predict the current map's
    occupancy / traversability / risk for every future step.  It is a strong
    baseline at 0.33 s steps, so beating it is the thing worth reporting.
    """
    model.eval()
    H = CFG.wm.pred_horizon
    occ_m = np.zeros(H); occ_p = np.zeros(H); occ_a = np.zeros(H)
    trv_m = np.zeros(H); trv_p = np.zeros(H)
    rsk_m = np.zeros(H); rsk_p = np.zeros(H)
    n = 0
    for _ in range(n_batches):
        x0, aid, tgt, groups = _make_batch(store, bank, rng, batch, device,
                                           real_frac=real_frac)
        B = x0.shape[0]
        s0 = model.encode(x0)
        acts = aid[:, None].expand(B, H)
        ps = model.dynamics.rollout(s0, acts)
        occ_pr, trav_pr, risk_pr = model.decode(ps)
        flat = tgt.reshape(-1, *tgt.shape[2:])
        occ_t = occ_target(flat).view(B, H, bu.OCC_G, bu.OCC_G)
        trav_t = trav_target(flat).view(B, H)
        risk_t, risk_p0 = _risk_targets(tgt, x0, groups, masks)
        # persistence
        occ_0 = occ_target(x0)[:, None].expand(-1, H, -1, -1)
        trav_0 = trav_target(x0)[:, None].expand(-1, H)
        risk_0 = risk_p0[:, None].expand(-1, H)

        # encoder bottleneck floor: decode the *true* future map's own latent.
        # No dynamics model can beat this, so it separates "bad dynamics" from
        # "96-D latent cannot hold the map".
        occ_ae = model.decode(model.encode(flat))[0].view(B, H, bu.OCC_G, bu.OCC_G)

        occ_m += (occ_pr - occ_t).abs().mean(dim=(0, 2, 3)).cpu().numpy()
        occ_p += (occ_0 - occ_t).abs().mean(dim=(0, 2, 3)).cpu().numpy()
        occ_a += (occ_ae - occ_t).abs().mean(dim=(0, 2, 3)).cpu().numpy()
        trv_m += (trav_pr - trav_t).abs().mean(0).cpu().numpy()
        trv_p += (trav_0 - trav_t).abs().mean(0).cpu().numpy()
        rsk_m += (risk_pr - risk_t).abs().mean(0).cpu().numpy()
        rsk_p += (risk_0 - risk_t).abs().mean(0).cpu().numpy()
        n += 1
    model.train()
    d = lambda a: (a / n).tolist()
    return {"occ_mae": d(occ_m), "occ_mae_persist": d(occ_p),
            "occ_mae_encoder_floor": d(occ_a),
            "trav_mae": d(trv_m), "trav_mae_persist": d(trv_p),
            "risk_mae": d(rsk_m), "risk_mae_persist": d(rsk_p)}


@torch.no_grad()
def action_sensitivity(model, store, bank, masks, device, rng,
                       n_batches: int = 8, batch: int = 6) -> dict:
    """Does the forecast actually depend on which action is taken?

    For the same starting map, rolls out all six actions and measures the
    standard deviation across actions of the predicted occupancy grid, next to
    the same statistic computed on the rigidly warped ground truth.  If the
    ground-truth spread is large and the model's is ~0, the model has collapsed
    to the marginal; if both are ~0, the action set or the timestep is too small
    to matter over this horizon.  Also reports the mean predicted vs. target
    collision risk, which is the calibration the supervisor thresholds.
    """
    model.eval()
    H = CFG.wm.pred_horizon
    m_sp, g_sp, m_mu, g_mu = [], [], [], []
    r_pred, r_tgt = np.zeros(N_ACTIONS), np.zeros(N_ACTIONS)
    n = 0
    for _ in range(n_batches):
        x0 = torch.from_numpy(store.sample_anchor(rng, batch)).to(device)
        s0 = model.encode(x0)
        pm, pg = [], []
        for a in range(N_ACTIONS):
            acts = torch.full((batch, H), a, dtype=torch.long, device=device)
            occ_p, _, risk_p = model.decode(model.dynamics.rollout(s0, acts))
            pm.append(occ_p[:, -1])
            tgt = bank.warp(x0, a, H - 1)
            pg.append(occ_target(tgt))
            r_pred[a] += float(risk_p.mean())
            r_tgt[a] += float(torch.stack([risk_target(x0, a, masks, k)
                                           for k in range(H)], 1).mean())
        pm = torch.stack(pm, 1)                      # (B, A, g, g)
        pg = torch.stack(pg, 1)
        m_sp.append(float(pm.std(1).mean())); g_sp.append(float(pg.std(1).mean()))
        m_mu.append(float(pm.mean())); g_mu.append(float(pg.mean()))
        n += 1
    model.train()
    return {"occ_spread_model": float(np.mean(m_sp)),
            "occ_spread_truth": float(np.mean(g_sp)),
            "occ_mean_model": float(np.mean(m_mu)),
            "occ_mean_truth": float(np.mean(g_mu)),
            "risk_pred_per_action": (r_pred / n).tolist(),
            "risk_target_per_action": (r_tgt / n).tolist()}


def _make_batch(store, bank, rng, batch, device, real_frac: float = 0.4):
    """Mixed batch of real and kinematically-augmented transitions.

    Samples are laid out in **contiguous action groups** so the per-action warp
    and the per-action risk target are plain slices, not boolean scatters.
    Returns (x0, action_ids, targets, groups) with groups = [(lo, hi, action)].
    """
    H = CFG.wm.pred_horizon
    parts_x, parts_a, parts_t, groups = [], [], [], []
    cur = 0

    n_real = int(round(batch * real_frac)) if store.have_real else 0
    if n_real > 0:
        r = store.sample_real(rng, n_real)
        if r is not None:
            x0r, ar, futr = r
            order = np.argsort(ar, kind="stable")
            x0r, ar, futr = x0r[order], ar[order], futr[order]
            parts_x.append(torch.from_numpy(x0r).to(device))
            parts_a.append(torch.from_numpy(ar).to(device))
            parts_t.append(torch.from_numpy(futr).to(device))
            lo = 0
            for i in range(1, len(ar) + 1):
                if i == len(ar) or ar[i] != ar[lo]:
                    groups.append((cur + lo, cur + i, int(ar[lo])))
                    lo = i
            cur += len(ar)
            n_real = len(ar)
        else:
            n_real = 0

    n_syn = batch - n_real
    if n_syn > 0:
        # two action groups per batch keeps action diversity without scatter cost
        n_grp = 2 if n_syn >= 2 else 1
        sizes = [n_syn // n_grp] * n_grp
        sizes[-1] += n_syn - sum(sizes)
        acts = rng.choice(N_ACTIONS, size=n_grp, replace=False)
        for a, nsz in zip(acts, sizes):
            if nsz <= 0:
                continue
            x0 = torch.from_numpy(store.sample_anchor(rng, nsz)).to(device)
            fut = torch.stack([bank.warp(x0, int(a), k) for k in range(H)], 1)
            parts_x.append(x0)
            parts_a.append(torch.full((nsz,), int(a), dtype=torch.long, device=device))
            parts_t.append(fut)
            groups.append((cur, cur + nsz, int(a)))
            cur += nsz

    return (torch.cat(parts_x, 0), torch.cat(parts_a, 0),
            torch.cat(parts_t, 0), groups)


def train(steps: int = 4000, batch: int = 10, lr: float = 3e-4,
          real_frac: float = 0.4, device: str | None = None) -> dict:
    set_seed(CFG.seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.default_rng(CFG.seed)

    t_load = time.perf_counter()
    store = TransitionStore()
    src = ("real cached BEV maps: " + ", ".join(store.source)) if store.have_real \
        else "synthetic stand-in scenes (no `bev` cache present yet)"
    print(f"[data] {src}")
    print(f"[data] anchors={store.n_anchors()}  real multi-step transitions={store.n_real}  "
          f"(load {time.perf_counter()-t_load:.1f}s)")
    if store.have_real:
        print(f"[data] action labels from {store.label_source}")
        print("[data] real action histogram: " +
              "  ".join(f"{ACTIONS[a]}={int(store.label_hist[a])}" for a in range(N_ACTIONS)))
        dom = store.label_hist.max() / max(store.label_hist.sum(), 1)
        if dom > 0.6:
            print(f"[data] WARNING: {ACTIONS[int(np.argmax(store.label_hist))]} is "
                  f"{dom*100:.0f}% of the real labels. Real transitions are sampled with "
                  f"inverse-frequency weights so the rare actions still appear, but the "
                  f"action-conditioned part of the dynamics is learned mostly from the "
                  f"kinematic warp augmentation. Re-run once the `odom` cache exists.")
        print(f"[data] plus kinematic warp augmentation for all {N_ACTIONS} actions "
              f"({(1-real_frac)*100:.0f}% of every batch) - augmentation, not observed data")
    if not store.have_real:
        real_frac = 0.0

    model = PrismWorldModel().to(dev)
    pc = model.param_counts()
    print(f"[model] {pc['total']:,} params ({pc['total']*4/1e6:.2f} MB fp32) on {dev}")

    bank = WarpBank(dev)
    masks = corridor_masks(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.15)
    hist = {"step": [], "total": [], "lat": [], "occ": [], "trav": [], "risk": []}
    t0 = time.perf_counter()
    model.train()
    for it in range(steps):
        x0, aid, tgt, groups = _make_batch(store, bank, rng, batch, dev, real_frac)
        loss, parts = _losses(model, x0, tgt, groups, masks, aid)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 25 == 0:
            for k in hist:
                hist[k].append(it if k == "step" else parts[k])
        if it % 400 == 0 or it == steps - 1:
            print(f"  step {it:5d}/{steps}  total {parts['total']:.4f}  "
                  f"lat {parts['lat']:.4f}  occ {parts['occ']:.4f}  "
                  f"trav {parts['trav']:.4f}  risk {parts['risk']:.4f}  "
                  f"({time.perf_counter()-t0:.0f}s)")
    train_s = time.perf_counter() - t0

    ev = evaluate(model, store, bank, masks, dev, np.random.default_rng(999),
                  real_frac=real_frac)
    print("\n[eval] open-loop multi-step error, model vs persistence baseline "
          "(MAE, lower is better)")
    print("  step | occ: model persist  encfloor | trav: model persist | risk: model persist")
    for k in range(CFG.wm.pred_horizon):
        print(f"   {k+1:2d}  |      {ev['occ_mae'][k]:.4f}  {ev['occ_mae_persist'][k]:.4f}"
              f"    {ev['occ_mae_encoder_floor'][k]:.4f} "
              f"|      {ev['trav_mae'][k]:.4f}  {ev['trav_mae_persist'][k]:.4f} "
              f"|      {ev['risk_mae'][k]:.4f}  {ev['risk_mae_persist'][k]:.4f}")
    for nm, k0 in (("occ", 0), ("trav", 0), ("risk", 1)):
        rng_ = range(k0, CFG.wm.pred_horizon)
        won = sum(1 for k in rng_ if ev[f"{nm}_mae"][k] < ev[f"{nm}_mae_persist"][k])
        n_k = len(list(rng_))
        verdict = "beats" if won > n_k // 2 else "does NOT beat"
        extra = ("  (step 1 excluded: the risk persistence baseline IS the step-1 "
                 "target, so it is exact there by construction)" if nm == "risk" else "")
        print(f"  {nm:5s}: model {verdict} persistence ({won}/{n_k} horizon steps){extra}")
    print("  'encfloor' = decoding the TRUE future map's own latent; no dynamics "
          "model can do better than that, it is the 96-D bottleneck.")

    sens = action_sensitivity(model, store, bank, masks, dev,
                              np.random.default_rng(4242))
    ev["sensitivity"] = sens
    print(f"\n[eval] action sensitivity at t+{CFG.wm.pred_horizon*bu.WM_DT:.1f} s")
    print(f"  occupancy across-action spread: model {sens['occ_spread_model']:.4f}  "
          f"vs rigid-warp ground truth {sens['occ_spread_truth']:.4f}  "
          f"({sens['occ_spread_model']/max(sens['occ_spread_truth'],1e-6)*100:.0f}% of it)")
    print(f"  occupancy mean:                 model {sens['occ_mean_model']:.4f}  "
          f"vs ground truth {sens['occ_mean_truth']:.4f}")
    print("  collision-risk calibration (mean over anchors), predicted vs geometric target:")
    for a in range(N_ACTIONS):
        print(f"    {ACTIONS[a]:8s} pred {sens['risk_pred_per_action'][a]:.3f}   "
              f"target {sens['risk_target_per_action'][a]:.3f}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"params": pc, "steps": steps, "batch": batch, "lr": lr,
            "real_frac": real_frac, "data_source": src,
            "n_real_transitions": store.n_real, "n_anchors": store.n_anchors(),
            "label_source": store.label_source,
            "label_hist": {ACTIONS[a]: int(store.label_hist[a]) for a in range(N_ACTIONS)},
            "wm_dt_s": bu.WM_DT, "horizon": CFG.wm.pred_horizon,
            "train_seconds": train_s, "device": str(dev), "eval": ev}
    torch.save({"model": model.state_dict(), "meta": meta}, WM_CKPT)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(LOG_NPZ,
                        **{k: np.asarray(v, np.float32) for k, v in hist.items()},
                        occ_mae=np.asarray(ev["occ_mae"], np.float32),
                        occ_mae_persist=np.asarray(ev["occ_mae_persist"], np.float32),
                        occ_mae_encoder_floor=np.asarray(ev["occ_mae_encoder_floor"], np.float32),
                        risk_mae=np.asarray(ev["risk_mae"], np.float32),
                        risk_mae_persist=np.asarray(ev["risk_mae_persist"], np.float32),
                        meta=np.array(json.dumps(meta)))
    print(f"\n[save] {WM_CKPT}  ({WM_CKPT.stat().st_size/1e6:.2f} MB)")
    print(f"[save] {LOG_NPZ}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--real-frac", type=float, default=0.4)
    ap.add_argument("--device", type=str, default=None)
    a = ap.parse_args()
    train(a.steps, a.batch, a.lr, a.real_frac, a.device)
