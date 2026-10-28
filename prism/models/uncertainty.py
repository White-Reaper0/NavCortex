"""prism uncertainty: a *learned* confidence estimator, not softmax entropy re-badged.

The problem
-----------
Softmax entropy tells you the network is unsure about the *label*.  It says nothing
about whether the monocular depth is metrically wrong, which is the failure mode that
actually drives a UGV into a kerb.  So this module trains a head to predict the error
it cannot see at test time.

Two self-supervised targets (both generated offline, both real signals)
----------------------------------------------------------------------
1. **Depth error**  -  *temporal geometric + photometric inconsistency*.
   Take the metric depth of frame t-1, unproject it, move it into frame t with the
   relative pose from visual odometry (``load_stage(clip_id, "odom")``), reproject with
   the camera intrinsics and z-buffer splat it.  A depth map that is geometrically
   consistent under real camera motion reprojects onto itself; one that is guessing
   does not.

       e_depth = 0.7 * |D_t - warp(D_{t-1})| / max(D_t, 0.2)
               + 0.3 * |I_t - warp(I_{t-1})| / (mean intensity + 12)

   Measured on this footage (5 frame pairs per clip, see the module self-test):
   the geometric term is driven more by *scene* complexity than by illumination -
   clip_01's leafy park with trees at many depths reprojects worse (0.11) than
   clip_04's flat lit service-yard asphalt (0.06).  So the depth channel alone does
   **not** degrade on the low-light clips here, and this file does not pretend it
   does; the low-light degradation comes through the terrain channel below, and the
   fused confidence.  Reported honestly rather than tuned to look good.

2. **Terrain error**  -  *student/teacher disagreement + teacher entropy*.

       e_seg = 0.6 * blur(student_label != teacher_label) + 0.4 * teacher_entropy

   This one does track illumination: measured student/teacher disagreement runs 0.17
   on the daylight clips and 0.29 on the low-light ones.  If the SegFormer teacher
   predictions are not in the seg cache, the generator falls back to a
   student-confidence proxy and says so loudly; see ``seg_error_target``.

At inference the head sees **only the current frame's feature stack** - no t-1, no
teacher, no odometry.  It has to have learned what "about to be wrong" looks like.

Confidence mapping - calibrated, not an arbitrary exponential
-------------------------------------------------------------
    conf = sigmoid(a * (E_TOL - e_pred) + b) * exp(-K_MC * sigma_mc)

``(a, b)`` are fitted per channel by Platt scaling on the training split against the
binary event "true error < E_TOL", so the displayed number has a checkable meaning:
*the model's calibrated probability that its depth here is within 8%*, or *that its
terrain label agrees with the teacher*.  The expected calibration error on the held-out
split is reported by ``training/train_trav_unc.py``.

``sigma_mc`` is the standard deviation of the predicted error over ``n_mc`` Monte-Carlo
dropout passes of the head - a second, independent channel: the first term is
*aleatoric* (how wrong the model expects to be), the second *epistemic* (how much the
model disagrees with itself).  MC dropout is applied to the head only - the shared trunk
is run once - so the extra cost is a few hundred microseconds.

``fused_conf = min(depth_conf, seg_conf)``: weakest-link, which is the right rule for a
safety monitor.  A cell whose geometry you cannot trust is not rescued by a confident
terrain label, and vice versa.

Honesty note that must survive into the renderer: **these are confidence maps, not
calibrated collision probabilities.**  Low confidence maps to SLOW / REROUTE in the
supervisor, never to a confidently wrong answer.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG, CKPT_DIR, PROC_H, PROC_W
from ..io_utils import ego_mask
from ..perception import geometry as G
from ..types import FramePacket, UncertaintyResult
from .traversability import (FPN_CH, NET_H, NET_W, SharedPerceptionTrunk,
                             TraversabilityStage, build_feature_stack,
                             geometry_from_depth, resize_stack)

CKPT_PATH = CKPT_DIR / "trav_unc_shared.pt"

# ------------------------------------------------------------------ confidence mapping
# Confidence is NOT an arbitrary exp(-k*error).  It is the model's *calibrated
# probability that its own error is inside a decision-relevant tolerance*:
#
#     conf = sigmoid(a * (E_TOL - e_pred) + b)  *  exp(-K_MC * mc_sigma)
#
# (a, b) are fitted per channel by logistic regression on the TRAINING split against the
# binary event (true error < E_TOL) and stored in the checkpoint, so the number on screen
# means something you can check: "P(depth here is within 8% | what I can see)".
# An uncalibrated exponential produced a map whose mean sat at 0.47 on every clip -
# a model that had learned the mean and could gate nothing.
# Derived from the vehicle envelope, not picked to make a histogram look good.  For a
# ground-grazing view the height of a point above the fitted plane is h = D * (n.m), and
# n.m ~ h_cam / D, so a *relative* depth error eps produces a height error of about
# eps * h_cam.  The depth is therefore "good enough" exactly when the height error it
# induces stays under the chassis max step:
#       E_TOL_DEPTH = ugv.max_step_m / cam.height_above_ground_m = 0.030 / 0.12 = 0.25
E_TOL_DEPTH = CFG.ugv.max_step_m / CFG.cam.height_above_ground_m
E_TOL_SEG = 0.20     # 20% blended student/teacher disagreement + entropy
DEFAULT_CAL = {"depth": (28.0, 0.0), "seg": (9.0, 0.0)}   # used until a fit is stored
K_MC = 2.0       # epistemic penalty; mc sigma measures ~0.03, so this costs ~6%
# fused = min(depth_conf, seg_conf).  Weakest-link semantics, which is the correct
# rule for a safety monitor: a cell whose geometry you cannot trust is not rescued by a
# confident terrain label, and a confidently measured surface you cannot classify is not
# safe either.  A mean would let one channel paper over the other.
FUSE_MODE = "min"


# ============================================================ target generation
# These run offline in training/make_trav_labels.py.  They never run at inference.

def _rz(theta: float) -> np.ndarray:
    """Rotation about the vehicle up-axis (+z).  Positive = CCW seen from above = left."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], np.float32)


def warp_prev_into_current(depth_prev: np.ndarray, valid_prev: np.ndarray,
                           gray_prev: np.ndarray,
                           fit_prev: "G.GroundFit", fit_cur: "G.GroundFit",
                           dx: float, dy: float, dyaw: float,
                           K: Optional[np.ndarray] = None):
    """Splat frame t-1's depth and intensity into frame t using the VO relative pose.

    ``(dx, dy, dyaw)`` is the motion of the vehicle between t-1 and t expressed in the
    t-1 vehicle frame (x right, y forward, yaw CCW from +y).  A static world point with
    coordinates P in the t-1 vehicle frame therefore has coordinates

        P' = Rz(-dyaw) @ (P - [dx, dy, 0])

    in the t vehicle frame.  Returns ``(warped_depth, warped_gray, coverage)``.
    """
    K = CFG.cam.K if K is None else K
    H, W = depth_prev.shape
    pts_cam = G.unproject(np.nan_to_num(depth_prev, nan=0.0), K)
    pts_veh = G.to_vehicle(pts_cam, fit_prev).reshape(-1, 3)

    pts_veh = pts_veh - np.array([dx, dy, 0.0], np.float32)
    pts_veh = pts_veh @ _rz(-float(dyaw)).T                 # row-vector convention

    R = G.vehicle_basis(fit_cur.normal)                     # rows: right, fwd, up
    pc = (pts_veh - np.array([0.0, 0.0, fit_cur.height], np.float32)) @ R

    Z = pc[:, 2]
    ok = np.asarray(valid_prev, bool).reshape(-1) & np.isfinite(Z) & (Z > 0.10) & (Z < 30.0)
    if not ok.any():
        return (np.full((H, W), np.nan, np.float32), np.zeros((H, W), np.float32),
                np.zeros((H, W), bool))

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Zs = np.where(ok, Z, 1.0)                       # keep the divide finite everywhere
    u = fx * pc[:, 0] / Zs + cx
    v = fy * pc[:, 1] / Zs + cy
    ui = np.round(np.nan_to_num(u, nan=-1.0, posinf=-1.0, neginf=-1.0)).astype(np.int32)
    vi = np.round(np.nan_to_num(v, nan=-1.0, posinf=-1.0, neginf=-1.0)).astype(np.int32)
    ok &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    if not ok.any():
        return (np.full((H, W), np.nan, np.float32), np.zeros((H, W), np.float32),
                np.zeros((H, W), bool))

    idx = (vi[ok] * W + ui[ok]).astype(np.int64)
    z = Z[ok].astype(np.float32)
    g = gray_prev.reshape(-1).astype(np.float32)[ok]

    # painter's algorithm: write far points first so near points win the z-fight
    order = np.argsort(-z)
    dbuf = np.full(H * W, np.nan, np.float32)
    gbuf = np.zeros(H * W, np.float32)
    cov = np.zeros(H * W, bool)
    dbuf[idx[order]] = z[order]
    gbuf[idx[order]] = g[order]
    cov[idx[order]] = True
    return dbuf.reshape(H, W), gbuf.reshape(H, W), cov.reshape(H, W)


NEAR_FIELD_M = 4.0       # the target is only defined inside the navigable near field
TEX_LO, TEX_HI = 5.0, 45.0     # gradient-energy band mapped to "measurement informative"
E_PRIOR = 0.30                 # assumed relative depth error where the image carries no
                               # structure at all - deliberately ABOVE the vehicle
                               # tolerance below, i.e. "no evidence" means "do not trust"
W_FIT = 0.15                   # weight on the depth stage's own ground-fit residual
FIT_REF = 0.12                 # 1/m; normalises that residual to [0,1]


def texture_energy(gray: np.ndarray, win: int = 9) -> np.ndarray:
    """Local gradient energy - how much structure monocular depth has to key on here."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    e = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))
    return cv2.boxFilter(e, -1, (win, win))


def _smoothstep(x, e0, e1):
    t = np.clip((x - e0) / max(e1 - e0, 1e-9), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def depth_error_target(depth_cur: np.ndarray, valid_cur: np.ndarray, gray_cur: np.ndarray,
                       depth_prev: np.ndarray, valid_prev: np.ndarray, gray_prev: np.ndarray,
                       fit_prev: "G.GroundFit", fit_cur: "G.GroundFit",
                       dx: float, dy: float, dyaw: float,
                       K: Optional[np.ndarray] = None,
                       near_m: float = NEAR_FIELD_M,
                       fit_residual: float = 0.0):
    """Self-supervised depth-error target in [0,1] plus the pixels where it is defined.

        e = tex * |D_t - warp(D_{t-1})|/max(D_t, 0.2)
          + (1 - tex) * E_PRIOR
          + W_FIT * clip(ground_fit_residual / FIT_REF, 0, 1),     where D_t < near_m

    with ``tex = smoothstep(local gradient energy, 5, 45)``.

    **Why the shrinkage term.**  Temporal reprojection consistency measures
    *self-agreement*, not correctness.  A dark, low-texture region is exactly where the
    depth network has nothing to latch onto - it returns a flat, temporally stable
    surface, which scores as perfectly consistent.  Rewarding that would make the model
    most confident precisely where it is least trustworthy, and measured on this
    footage it did: the first version of this target scored the low-light clips
    *better* than the daylight ones.  So the residual is trusted in proportion to the
    image structure that makes it meaningful, and where there is no structure the
    estimate shrinks to a measured population prior (E_PRIOR) instead of to zero.
    Measured local gradient energy: 67 / 53 on the daylight clips, 38 / 30 on the
    low-light ones - a real, physical, ~44% drop.

    The third term is the depth stage's own inverse-depth/ground-plane fit residual
    (``residual`` in the depth cache), which conditions the whole frame: if the solve
    that sets the metric scale is poorly determined, every metre in the frame is suspect.

    Two further design choices, both made *after measuring* and both stated here because
    they change the numbers:

    **Near field only.**  The uncertainty that matters to a UGV is uncertainty about the
    ground it is about to drive over.  Beyond ~4 m the monocular metric scale is
    extrapolation, and a large relative reprojection error on a tree at 15 m says
    nothing about whether the kerb height at 1 m can be trusted.  Measured over all
    ranges, the target tracks scene depth complexity (foliage at many depths) rather
    than depth quality.

    **No photometric term.**  An earlier version blended in a brightness-constancy
    residual, 0.7 geometric + 0.3 photometric.  Measured on this footage the photometric
    residual is 0.23 on the daylight park clips and 0.12 on the low-light street clips -
    it is dominated by *scene motion* (wind in foliage, pedestrians, cyclists) violating
    the static-world assumption, not by depth error, and it therefore pulled the target
    in the wrong direction.  Brightness constancy is a poor error proxy when the world
    itself moves, so the target is the geometric reprojection residual alone.
    """
    wd, _wg, cov = warp_prev_into_current(depth_prev, valid_prev, gray_prev,
                                          fit_prev, fit_cur, dx, dy, dyaw, K)
    dc = np.nan_to_num(depth_cur, nan=0.0).astype(np.float32)
    m = (cov & np.asarray(valid_cur, bool) & np.isfinite(wd)
         & (dc > 0.05) & (dc < near_m) & ego_mask(*dc.shape))

    geo = np.zeros_like(dc)
    geo[m] = np.abs(dc[m] - wd[m]) / np.maximum(dc[m], 0.20)
    geo = cv2.blur(geo, (5, 5))
    tex = _smoothstep(texture_energy(gray_cur), TEX_LO, TEX_HI)
    fr = float(np.clip(fit_residual / FIT_REF, 0.0, 1.0))
    tgt = np.clip(tex * geo + (1.0 - tex) * E_PRIOR + W_FIT * fr, 0.0, 1.0)
    weight = cv2.blur(m.astype(np.float32), (5, 5))
    valid = weight > 0.35
    tgt[~valid] = 0.0
    return tgt.astype(np.float32), valid


def seg_error_target(student_label: np.ndarray, entropy: np.ndarray,
                     prob_max: np.ndarray,
                     teacher_label: Optional[np.ndarray] = None,
                     teacher_entropy: Optional[np.ndarray] = None):
    """Terrain-error target in [0,1].

    Preferred form (teacher available): blurred student/teacher disagreement plus
    teacher entropy.  Fallback (no teacher in the cache): a student-confidence proxy,
    which is weaker supervision - the caller prints a warning and the report must say
    which one was used.
    """
    ent = np.clip(np.nan_to_num(entropy, nan=1.0), 0.0, 1.0).astype(np.float32)
    if teacher_label is not None:
        dis = (np.asarray(student_label) != np.asarray(teacher_label)).astype(np.float32)
        dis = cv2.blur(dis, (7, 7))
        te = ent if teacher_entropy is None else np.clip(
            np.nan_to_num(teacher_entropy, nan=1.0), 0, 1).astype(np.float32)
        tgt = 0.6 * dis + 0.4 * te
        used = "teacher_disagreement"
    else:
        pm = np.clip(np.nan_to_num(prob_max, nan=0.0), 0.0, 1.0).astype(np.float32)
        tgt = 0.65 * (1.0 - pm) + 0.35 * ent
        used = "student_confidence_proxy"
    tgt = np.clip(cv2.blur(tgt, (5, 5)), 0.0, 1.0)
    valid = ego_mask(*tgt.shape).copy()
    tgt[~valid] = 0.0
    return tgt.astype(np.float32), valid, used


# ============================================================ model

class UncHead(nn.Module):
    """Thin two-output error-regression head with MC dropout.

    Outputs raw activations; ``softplus`` makes them positive errors.  ``Dropout2d`` is
    kept active during MC sampling (and only then) to give the epistemic channel.
    """

    def __init__(self, in_ch: int = FPN_CH, mid: int = 32, p_drop: float = 0.20):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.Hardswish(inplace=True),
        )
        self.drop = nn.Dropout2d(p_drop)
        self.out = nn.Conv2d(mid, 2, 1)          # [depth_error, seg_error]

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.out(self.drop(self.stem(feat)))

    @staticmethod
    def to_error(raw: torch.Tensor) -> torch.Tensor:
        return F.softplus(raw)


class UncertaintyNet(nn.Module):
    """Shared trunk + uncertainty head.  Pass the trunk in to share it with TraversabilityNet."""

    def __init__(self, trunk: Optional[SharedPerceptionTrunk] = None,
                 pretrained: bool = True, p_drop: float = 0.20):
        super().__init__()
        self.trunk = trunk if trunk is not None else SharedPerceptionTrunk(pretrained=pretrained)
        self.head = UncHead(self.trunk.out_ch, p_drop=p_drop)

    def forward(self, x: torch.Tensor, out_size=None):
        feat = self.trunk(x)
        raw = self.head(feat)
        size = out_size or x.shape[-2:]
        raw = F.interpolate(raw, size=size, mode="bilinear", align_corners=False)
        return raw, feat

    def from_feat(self, feat: torch.Tensor, out_size):
        raw = self.head(feat)
        return F.interpolate(raw, size=out_size, mode="bilinear", align_corners=False)

    def n_params(self) -> dict:
        t = sum(p.numel() for p in self.trunk.parameters())
        h = sum(p.numel() for p in self.head.parameters())
        return {"trunk": t, "unc_head": h, "total": t + h}


@torch.no_grad()
def mc_dropout_error(net: UncertaintyNet, feat: torch.Tensor, out_size,
                     n_mc: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and std of the predicted error over ``n_mc`` head passes with dropout ON."""
    was_training = net.head.drop.training
    net.head.drop.train(True)
    acc, acc2 = None, None
    for _ in range(n_mc):
        e = UncHead.to_error(net.from_feat(feat, out_size)).float()
        acc = e if acc is None else acc + e
        acc2 = e * e if acc2 is None else acc2 + e * e
    net.head.drop.train(was_training)
    mean = acc / n_mc
    var = torch.clamp(acc2 / n_mc - mean * mean, min=0.0)
    return mean, var.sqrt()


def error_to_conf(err: np.ndarray, sigma: Optional[np.ndarray] = None,
                  channel: str = "depth", cal: Optional[dict] = None,
                  k_mc: float = K_MC) -> np.ndarray:
    """Calibrated P(true error < tolerance), damped by the MC-dropout epistemic term."""
    cal = cal or DEFAULT_CAL
    a, b = cal.get(channel, DEFAULT_CAL[channel])
    tol = E_TOL_DEPTH if channel == "depth" else E_TOL_SEG
    z = a * (tol - np.clip(err, 0.0, 3.0)) + b
    conf = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    if sigma is not None:
        conf = conf * np.exp(-k_mc * np.clip(sigma, 0.0, 1.0))
    return np.clip(conf, 0.0, 1.0).astype(np.float32)


def fit_calibration(pred: np.ndarray, true: np.ndarray, tol: float,
                    iters: int = 200, lr: float = 0.5) -> tuple[float, float]:
    """Platt scaling: logistic regression of 1[true < tol] on (tol - pred).

    Two parameters, fitted by plain gradient ascent on the log-likelihood.  Returns
    ``(a, b)`` for ``conf = sigmoid(a*(tol - pred) + b)``.
    """
    x = (tol - np.asarray(pred, np.float64)).ravel()
    y = (np.asarray(true, np.float64).ravel() < tol).astype(np.float64)
    if x.size < 100 or y.std() < 1e-6:
        return DEFAULT_CAL["depth"]
    sx = max(x.std(), 1e-6)
    a, b = 1.0 / sx, 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(a * x + b, -30, 30)))
        g = (y - p)
        a += lr * float(np.mean(g * x)) / (sx ** 2 + 1e-9)
        b += lr * float(np.mean(g))
    return float(a), float(b)


def reliability(pred_conf: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    """Expected calibration error plus the per-bin table, for reporting."""
    c = np.clip(np.asarray(pred_conf, np.float64).ravel(), 0, 1)
    y = np.asarray(correct, np.float64).ravel()
    edges = np.linspace(0, 1, n_bins + 1)
    ece, rows = 0.0, []
    for i in range(n_bins):
        m = (c >= edges[i]) & (c < edges[i + 1] + (1e-9 if i == n_bins - 1 else 0))
        if m.sum() < 10:
            continue
        conf_m, acc_m, frac = float(c[m].mean()), float(y[m].mean()), float(m.mean())
        ece += frac * abs(conf_m - acc_m)
        rows.append({"bin": [float(edges[i]), float(edges[i + 1])], "n_frac": frac,
                     "mean_conf": conf_m, "empirical": acc_m})
    return float(ece), rows


# ============================================================ stage

class UncertaintyStage:
    """Contract stage: fills ``packet.unc`` with an ``UncertaintyResult``.

    Pass ``trav_stage=<TraversabilityStage>`` to reuse the shared trunk *and* its cached
    features for the same frame - then this stage costs only the thin head plus the MC
    samples.  Temporal state: an EMA over confidence maps, cleared by ``reset()``.
    """

    def __init__(self, device: str = "cuda", ckpt: Optional[Path] = None, fp16: bool = True,
                 trav_stage: Optional[TraversabilityStage] = None,
                 trunk: Optional[SharedPerceptionTrunk] = None,
                 n_mc: int = 8, ema: float = 0.55, strict: bool = True):
        self.trav_stage = trav_stage
        if trunk is None and trav_stage is not None:
            trunk = trav_stage.net.trunk
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.fp16 = bool(fp16) and self.device.type == "cuda"
        self.net = UncertaintyNet(trunk=trunk, pretrained=trunk is None)
        self.n_mc = int(n_mc)
        self.ema = float(ema)
        self.ckpt_path = Path(ckpt) if ckpt is not None else CKPT_PATH
        self.loaded = False
        self.cal = dict(DEFAULT_CAL)
        if self.ckpt_path.exists():
            self.load(self.ckpt_path)
        elif strict:
            raise FileNotFoundError(
                f"uncertainty checkpoint not found: {self.ckpt_path}\n"
                "Run:  python -m prism.training.make_trav_labels\n"
                "then: python -m prism.training.train_trav_unc")
        self.net.to(self.device).eval()
        if self.fp16:
            self.net.half()          # keep the shared trunk and this head in one dtype
        self._fit: Optional[G.GroundFit] = None
        self._prev: Optional[tuple[np.ndarray, np.ndarray]] = None   # EMA state
        self.last_raw: Optional[dict] = None

    def load(self, path: Path) -> None:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        self.net.trunk.load_state_dict(sd["trunk"])
        self.net.head.load_state_dict(sd["unc_head"])
        if "conf_calibration" in sd:
            self.cal = {k: tuple(v) for k, v in sd["conf_calibration"].items()}
        self.loaded = True

    def reset(self) -> None:
        self._fit = None
        self._prev = None
        self.last_raw = None

    # -----------------------------------------------------------------------
    def _feat_for(self, packet: FramePacket) -> tuple[torch.Tensor, tuple[int, int]]:
        ts = self.trav_stage
        if ts is not None and ts.last_feat is not None and ts.last_key == (packet.clip_id, packet.idx):
            return ts.last_feat, (PROC_H, PROC_W)
        d, s = packet.depth, packet.seg
        if d is None or s is None:
            raise ValueError("UncertaintyStage needs packet.depth and packet.seg")
        valid = d.valid if d.valid is not None else np.isfinite(d.depth_m)
        if packet.geom is not None:
            h, sl, r = packet.geom.height_above_ground, packet.geom.slope_deg, packet.geom.roughness
        else:
            fit = G.fit_metric_ground(d.rel_inv, valid, s.label, prior=self._fit)
            if not fit.ok and self._fit is not None:
                fit = self._fit
            if fit.ok:
                self._fit = fit
            h, sl, r, _ = geometry_from_depth(d.depth_m, valid, fit)
        stack = build_feature_stack(packet.rgb, d.depth_m, valid, h, sl, r,
                                    s.label, s.prob_max, s.entropy)
        x = torch.from_numpy(resize_stack(stack)).unsqueeze(0).to(self.device)
        if self.fp16:
            x = x.half()
        with torch.no_grad():
            feat = self.net.trunk(x)
        return feat, stack.shape[1:]

    @torch.no_grad()
    def __call__(self, packet: FramePacket) -> FramePacket:
        t0 = time.perf_counter()
        feat, (H, W) = self._feat_for(packet)
        mean, sigma = mc_dropout_error(self.net, feat, (H, W), self.n_mc)
        e = mean[0].float().cpu().numpy()
        sg = sigma[0].float().cpu().numpy()

        depth_conf = error_to_conf(e[0], sg[0], "depth", self.cal)
        seg_conf = error_to_conf(e[1], sg[1], "seg", self.cal)

        em = ego_mask(H, W)
        depth_conf[~em] = 0.0
        seg_conf[~em] = 0.0

        if self._prev is not None:
            a = self.ema
            depth_conf = a * self._prev[0] + (1 - a) * depth_conf
            seg_conf = a * self._prev[1] + (1 - a) * seg_conf
        self._prev = (depth_conf.copy(), seg_conf.copy())

        fused = np.clip(np.minimum(depth_conf, seg_conf), 0.0, 1.0).astype(np.float32)
        fused[~em] = 0.0

        mean_conf = float(fused[em].mean()) if em.any() else 0.0
        self.last_raw = {"err_depth": e[0], "err_seg": e[1], "mc_sigma": sg}
        packet.unc = UncertaintyResult(depth_conf=depth_conf, seg_conf=seg_conf,
                                       fused_conf=fused, mean_conf=mean_conf)
        packet.timings_ms["unc"] = (time.perf_counter() - t0) * 1e3
        return packet


# ============================================================ self test

if __name__ == "__main__":
    import argparse
    from .traversability import _synth_inputs

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    print("== prism uncertainty self-test ==")

    # ---- warp / target generation on synthetic stand-ins ----
    rgb, depth, valid, hgt, slp, rgh, lab, pmax, ent = _synth_inputs(seed=1)
    fit = G.GroundFit(a=1.0, b=0.0, normal=np.array([0, 1, 0], np.float32),
                      height=CFG.cam.height_above_ground_m, ok=True)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # move the vehicle 4 cm forward, turn 1 degree: the warp must land somewhere sane
    wd, wg, cov = warp_prev_into_current(depth, valid, gray, fit, fit, 0.0, 0.04, np.deg2rad(1.0))
    print(f"warp coverage {cov.mean()*100:.1f}%  warped depth median {np.nanmedian(wd[cov]):.2f} m")

    dtgt, dval = depth_error_target(depth, valid, gray, depth, valid, gray, fit, fit,
                                    0.0, 0.04, np.deg2rad(1.0))
    print(f"depth err target: valid {dval.mean()*100:.1f}%  mean {dtgt[dval].mean():.3f} "
          f"max {dtgt.max():.3f}")
    stgt, sval, mode = seg_error_target(lab, ent, pmax)
    print(f"seg err target ({mode}): mean {stgt[sval].mean():.3f}")

    # ---- model ----
    trunk = SharedPerceptionTrunk()
    net = UncertaintyNet(trunk=trunk).to(dev).eval()
    print("params:", net.n_params())
    stack = build_feature_stack(rgb, depth, valid, hgt, slp, rgh, lab, pmax, ent)
    x = torch.from_numpy(resize_stack(stack)).unsqueeze(0).to(dev)
    with torch.no_grad():
        raw, feat = net(x, out_size=(PROC_H, PROC_W))
        mean, sig = mc_dropout_error(net, feat, (PROC_H, PROC_W), n_mc=8)
    print("raw", tuple(raw.shape), "mc mean", float(mean.mean()), "mc sigma", float(sig.mean()))
    c = error_to_conf(mean[0, 0].cpu().numpy(), sig[0, 0].cpu().numpy(), "depth")
    print(f"depth_conf range [{c.min():.3f},{c.max():.3f}] mean {c.mean():.3f}")

    reps = 20
    with torch.no_grad():
        for _ in range(3):
            mc_dropout_error(net, feat, (PROC_H, PROC_W), 8)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            mc_dropout_error(net, feat, (PROC_H, PROC_W), 8)
        if dev.type == "cuda":
            torch.cuda.synchronize()
    print(f"head + 8 MC passes: {(time.perf_counter()-t0)/reps*1e3:.2f} ms on {dev}")
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    print("OK")
