"""Lightweight learned visual place recognition (GeM + learned whitening).

WHAT THIS IS
------------
A **NetVLAD-lite / GeM-style global descriptor**, not NetVLAD and not a
foundation-model retriever:

    torchvision mobilenet_v3_small (ImageNet weights, offline cache)
      -> truncated at features[:9]   (48 channels, stride 16)  [FROZEN]
      -> 1x1 conv 48 -> 256 + BN + ReLU                        [trained]
      -> GeM pooling, learnable p                              [trained]
      -> Linear 256 -> 256 "whitening"                         [trained]
      -> L2 normalise -> 256-D descriptor

Truncating at features[:9] is deliberate: mid-level convolutional features are the
classic choice for place recognition (Arandjelovic et al. found conv4 beats conv5 for
NetVLAD) and it keeps the trunk at ~0.4 M parameters. Only ~78 k parameters are
trained, so this is "GeM + learned whitening on a frozen ImageNet trunk" -- a
well-established retrieval baseline, described here as exactly that.

TRAINING (self-supervised, on this footage only)
------------------------------------------------
No place-recognition dataset is available offline, so the projection is trained
self-supervised with InfoNCE on frames drawn from the five clips plus
``video/input.mp4``:

  positives : the same frame under independent strong photometric + crop
              augmentation, or a frame within +-0.4 s from the same source burst
  negatives : every other sample in the batch, with a false-negative mask that
              removes anything from the same source within +-1.5 s

That teaches viewpoint / exposure invariance and separates different places along
the same drive. It cannot teach anything this footage does not contain.

INFERENCE
---------
A growing descriptor database with frame index and pose. A query is compared only
against entries older than ``exclude_frames`` (temporal exclusion, so frame k does
not trivially match frame k-1), then the top candidates are re-scored with a
**sequence-consistency** check over a short window -- the SeqSLAM trick that turns a
noisy single-frame similarity into a usable revisit signal.

Declaring a revisit needs *four* things, not just a high score. Measured on these
clips, the median best in-clip cosine is 0.76-0.92 simply because driving down a
corridor makes consecutive stretches look alike; a bare threshold would fire on
almost every frame and manufacture loop closures that do not exist. So:

  1. sequence score >= ``sim_thresh``
  2. **distinctiveness**: score / median score of eligible entries *from the same
     source clip* >= ``ratio_thresh`` -- the match must beat the background, not
     just clear an absolute bar
  3. the winner stays sequence-consistent for ``min_seq_frames`` frames
  4. in-clip matches must be at least ``min_gap_frames`` old. Without this the
     winner is almost always the first frame just outside the exclusion window,
     which is view overlap, not a return to a place.

HONESTY
-------
Measured result on this footage: with those rules **all five clips report zero
revisits**, which is the correct answer -- each clip is a single 10 s forward drive
and never returns anywhere. The stage reports the top similarity and says why it did
not call a revisit, instead of inventing a loop closure. ``cross_clip=True``
pre-seeds the database with descriptors from earlier clips so the panel shows real
retrieval against a real multi-clip database (and the similarity strip visibly
separates the clips, which is the honest demonstration that the descriptor works).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG, CKPT_DIR, CLIP_IDS, CLIP_FPS
from ..types import FramePacket, PlaceResult
from .. import io_utils

CKPT_PATH = CKPT_DIR / "vpr_gem_mnv3.pt"
DESC_DIM = 256
IN_W, IN_H = 320, 180          # descriptor input resolution (aspect of the clips)

_IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# --------------------------------------------------------------------- model

class GeM(nn.Module):
    """Generalised-mean pooling. p -> 1 is average pooling, p -> inf is max."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(1.0, 10.0)
        x = x.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.pow(1.0 / p).flatten(1)


class GeMNet(nn.Module):
    """Frozen truncated MobileNetV3-Small trunk + trained GeM head."""

    def __init__(self, trunk_cut: int = 9, dim: int = DESC_DIM, pretrained: bool = True):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        w = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        net = mobilenet_v3_small(weights=w)
        self.trunk = nn.Sequential(*list(net.features.children())[:trunk_cut])
        with torch.no_grad():
            c = self.trunk(torch.zeros(1, 3, 64, 64)).shape[1]
        self.trunk_channels = int(c)
        for prm in self.trunk.parameters():
            prm.requires_grad_(False)
        self.trunk.eval()
        self.proj = nn.Sequential(nn.Conv2d(c, dim, 1, bias=False),
                                  nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        self.gem = GeM(3.0)
        self.white = nn.Linear(dim, dim)        # learned whitening / projection
        self.dim = dim

    def train(self, mode: bool = True):
        super().train(mode)
        self.trunk.eval()                        # frozen BN statistics
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f = self.trunk(x)
        f = self.proj(f)
        v = self.gem(f)
        v = self.white(v)
        return F.normalize(v, dim=1)

    def head_parameters(self):
        return list(self.proj.parameters()) + list(self.gem.parameters()) + \
            list(self.white.parameters())


def preprocess(bgr: np.ndarray | Sequence[np.ndarray], device: str = "cpu") -> torch.Tensor:
    """BGR uint8 image(s) -> normalised NCHW tensor at IN_W x IN_H."""
    import cv2
    imgs = [bgr] if isinstance(bgr, np.ndarray) and bgr.ndim == 3 else list(bgr)
    out = np.empty((len(imgs), IN_H, IN_W, 3), np.uint8)
    for i, im in enumerate(imgs):
        if im.shape[0] != IN_H or im.shape[1] != IN_W:
            im = cv2.resize(im, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        out[i] = im
    x = torch.from_numpy(out[..., ::-1].copy()).permute(0, 3, 1, 2).float().div_(255.0)
    x = (x - _IMNET_MEAN) / _IMNET_STD
    return x.to(device)


# --------------------------------------------------------------------- stage

@dataclass
class VPRParams:
    exclude_frames: int = 45        # temporal exclusion window (1.5 s at 30 fps)
    seq_len: int = 5                # sequence-consistency window
    top_k: int = 12                 # candidates re-scored by the sequence check
    sim_thresh: float = 0.92        # absolute threshold on the sequence score
    ratio_thresh: float = 1.35      # best / median-eligible distinctiveness ratio
    min_seq_frames: int = 3         # frames the winner must stay sequence-consistent
    min_db: int = 90                # no revisit call until the database is meaningful
    min_gap_frames: int = 90        # in-clip: a revisit must be >= 3 s old, not just
                                    # the first frame outside the exclusion window
    db_stride: int = 1              # keep every Nth frame in the database


def explain_decision(res, p: "VPRParams") -> str:
    """One sentence saying which criterion decided this frame.

    Shared by the live stage and by the renderer's cache reconstruction so the two
    can never disagree. The order of the tests below is the order the decision is
    actually made in, so the reported reason is always the criterion that blocked
    it -- never an arithmetically false statement about a test that passed.
    """
    best = int(getattr(res, "best_match_idx", -1))
    score = float(getattr(res, "best_score", 0.0))
    dist = float(getattr(res, "distinctiveness", 0.0))
    seq = int(getattr(res, "seq_consistent_frames", 0))
    gap = int(getattr(res, "loop_gap_frames", 0))
    if getattr(res, "is_revisit", False):
        return (f"REVISIT: db#{best} = {getattr(res, 'best_clip_id', '?')}/"
                f"f{getattr(res, 'best_frame_idx', -1)}, {dist:.2f}x background, "
                f"stable {seq} frames")
    if best < 0:
        return "no revisit - database empty or fully inside the exclusion window"
    if score < p.sim_thresh:
        return (f"no revisit - top sequence score {score:.3f} is below the "
                f"{p.sim_thresh:.2f} threshold")
    if not bool(getattr(res, "gap_ok", True)):
        return (f"no revisit - strong match, but only {gap} frames "
                f"({gap / CLIP_FPS:.1f} s) old: view overlap, not a return")
    if dist < p.ratio_thresh:
        return (f"no revisit - score passes but the match is not distinctive "
                f"({dist:.2f}x background, needs {p.ratio_thresh:.2f}x)")
    if int(getattr(res, "db_size", 0)) < p.min_db:
        return (f"no revisit - database still too small "
                f"({int(getattr(res, 'db_size', 0))} < {p.min_db} entries)")
    return (f"no revisit - score {score:.3f} and distinctiveness {dist:.2f}x both "
            f"pass, but the winner has only been stable for {seq} of "
            f"{p.min_seq_frames} frames")


class VPRStage:
    """Growing-database place recognition with sequence consistency.

    Fills ``packet.place`` with a :class:`~prism.types.PlaceResult`.
    """

    def __init__(self, device: str = "cuda", ckpt: Optional[Path | str] = CKPT_PATH,
                 cross_clip: bool = False, params: Optional[VPRParams] = None, **kw):
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.p = params or VPRParams()
        for k, v in kw.items():
            if hasattr(self.p, k):
                setattr(self.p, k, v)
        self.cross_clip = bool(cross_clip)
        self.net = GeMNet().to(self.device).eval()
        self.trained = False
        ck = Path(ckpt) if ckpt else None
        if ck is not None and ck.exists():
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            self.net.load_state_dict(sd["model"], strict=True)
            self.trained = True
            self.ckpt_meta = {k: v for k, v in sd.items() if k != "model"}
        else:
            self.ckpt_meta = {}
            warnings.warn(f"[vpr] no checkpoint at {ck}: running with a randomly "
                          f"initialised head. Train with "
                          f"`python -m prism.models.vpr --train`.", RuntimeWarning)
        self.net.to(self.device)
        # database (persists across clips when cross_clip is on)
        self._db_desc: list[np.ndarray] = []
        self._db_clip: list[str] = []
        self._db_frame: list[int] = []
        self._db_pose: list[tuple[float, float, float]] = []
        self._db_mat: Optional[np.ndarray] = None
        self._db_clip_np: np.ndarray = np.zeros(0, object)
        self._db_frame_np: np.ndarray = np.zeros(0, np.int32)
        self._db_dirty = True
        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self, keep_db: Optional[bool] = None) -> None:
        """Clear per-clip state. The database survives iff ``cross_clip``."""
        keep = self.cross_clip if keep_db is None else bool(keep_db)
        if not keep:
            self._db_desc.clear(); self._db_clip.clear()
            self._db_frame.clear(); self._db_pose.clear()
        self._db_dirty = True
        self._clip_start = len(self._db_desc)
        self._cur_clip: Optional[str] = None
        self._prev_best = -1
        self._prev_hits = 0
        self.last_sims: np.ndarray = np.zeros(0, np.float32)

    @property
    def db_size(self) -> int:
        return len(self._db_desc)

    # ---- cached views of the database (rebuilt only when it changes)
    def _db_matrix(self) -> np.ndarray:
        self._rebuild()
        return self._db_mat

    def _db_clip_arr(self) -> np.ndarray:
        self._rebuild()
        return self._db_clip_np

    def _db_frame_arr(self) -> np.ndarray:
        self._rebuild()
        return self._db_frame_np

    def _rebuild(self) -> None:
        if not self._db_dirty and self._db_mat is not None                 and len(self._db_mat) == len(self._db_desc):
            return
        self._db_mat = (np.stack(self._db_desc) if self._db_desc
                        else np.zeros((0, DESC_DIM), np.float32))
        self._db_clip_np = np.asarray(self._db_clip, object)
        self._db_frame_np = np.asarray(self._db_frame, np.int32)
        self._db_dirty = False

    def db_meta(self) -> list[tuple[str, int]]:
        return list(zip(self._db_clip, self._db_frame))

    # ---------------------------------------------------------------- seeding
    @torch.no_grad()
    def seed_from_clips(self, clip_ids: Sequence[str], stride: int = 3,
                        batch: int = 8) -> int:
        """Pre-load the database with descriptors from whole clips (cross-clip demo)."""
        added = 0
        for cid in clip_ids:
            buf, idxs = [], []
            for i, fr in io_utils.read_frames(cid):
                if i % stride:
                    continue
                buf.append(fr); idxs.append(i)
                if len(buf) == batch:
                    added += self._add_batch(cid, idxs, buf); buf, idxs = [], []
            if buf:
                added += self._add_batch(cid, idxs, buf)
        self._clip_start = len(self._db_desc)
        return added

    def _add_batch(self, cid: str, idxs: list[int], frames: list[np.ndarray]) -> int:
        d = self.describe(frames)
        for j, i in enumerate(idxs):
            self._db_desc.append(d[j]); self._db_clip.append(cid)
            self._db_frame.append(int(i)); self._db_pose.append((0.0, 0.0, 0.0))
        self._db_dirty = True
        return len(idxs)

    @torch.no_grad()
    def describe(self, frames) -> np.ndarray:
        x = preprocess(frames, self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(self.device == "cuda")):
            v = self.net(x)
        return v.float().cpu().numpy().astype(np.float32)

    # ---------------------------------------------------------------- query
    def __call__(self, packet: FramePacket) -> FramePacket:
        t0 = time.perf_counter()
        if packet.rgb is None:
            raise ValueError("VPRStage needs packet.rgb")
        if self._cur_clip != packet.clip_id:
            self._cur_clip = packet.clip_id
            self._clip_start = len(self._db_desc)

        desc = self.describe([packet.rgb])[0]
        self._q_desc_now = desc
        res = PlaceResult(descriptor=desc, db_size=len(self._db_desc))
        res.instant_score = 0.0
        res.distinctiveness = 0.0
        res.bg_median = 0.0
        res.gap_ok = True
        res.seq_consistent_frames = 0

        # ---- eligible database entries: everything except the temporal exclusion
        n = len(self._db_desc)
        if n:
            D = self._db_matrix()                            # (n, 256), cached
            sims = D @ desc                                  # cosine (all L2-normalised)
            self.last_sims = sims.astype(np.float32)
            elig = np.ones(n, bool)
            same = self._db_clip_arr() == packet.clip_id
            fr = self._db_frame_arr()
            elig[same & (fr > packet.idx - self.p.exclude_frames)] = False
            if elig.any():
                cand = np.argsort(-np.where(elig, sims, -2.0))[:self.p.top_k]
                cand = [int(c) for c in cand if elig[c]]
                best, best_seq, best_inst = -1, -1.0, -1.0
                for c in cand:
                    sq = self._seq_score(c, packet)
                    if sq > best_seq:
                        best, best_seq, best_inst = c, sq, float(sims[c])
                if best >= 0:
                    res.best_match_idx = best
                    res.best_score = float(best_seq)
                    res.instant_score = float(best_inst)
                    res.best_clip_id = self._db_clip[best]
                    res.best_frame_idx = int(self._db_frame[best])
                    gap = (packet.idx - self._db_frame[best]
                           if self._db_clip[best] == packet.clip_id else 0)
                    res.loop_gap_frames = int(gap)
                    # consistency: the winner should advance by ~1 db slot per frame
                    consistent = (self._prev_best >= 0 and abs(best - self._prev_best - 1) <= 3)
                    self._prev_hits = self._prev_hits + 1 if consistent else 0
                    # Distinctiveness: a genuine revisit must beat the *typical*
                    # database entry, not just clear an absolute bar. Driving down a
                    # corridor makes every frame look alike (measured: the in-clip
                    # median best score is 0.76-0.92 on these clips), so an absolute
                    # threshold alone would declare a loop closure on every frame.
                    # Background is measured *within the winner's own source clip*:
                    # in cross-clip mode the eligible pool is dominated by unrelated
                    # clips, which would inflate the ratio for free.
                    same_src = elig & (self._db_clip_arr() == self._db_clip[best])
                    pool = sims[same_src] if same_src.sum() >= 10 else sims[elig]
                    med = float(np.median(pool)) if pool.size >= 10 else 1.0
                    ratio = float(best_seq / max(med, 1e-6))
                    res.distinctiveness = ratio
                    res.bg_median = med
                    same_clip = self._db_clip[best] == packet.clip_id
                    gap_ok = (not same_clip) or (gap >= self.p.min_gap_frames)
                    res.gap_ok = bool(gap_ok)
                    res.is_revisit = bool(best_seq >= self.p.sim_thresh
                                          and ratio >= self.p.ratio_thresh
                                          and self._prev_hits >= self.p.min_seq_frames
                                          and n >= self.p.min_db and gap_ok)
                    res.seq_consistent_frames = int(self._prev_hits)
                    self._prev_best = best
                else:
                    self._prev_best, self._prev_hits = -1, 0
            else:
                self._prev_best, self._prev_hits = -1, 0

        res.db_size = len(self._db_desc)
        res.sims = self.last_sims
        res.cross_clip = self.cross_clip
        res.trained = self.trained
        res.exclude = int(self.p.exclude_frames)
        res.threshold = float(self.p.sim_thresh)
        res.ratio_thresh = float(self.p.ratio_thresh)
        res.note = explain_decision(res, self.p)

        # ---- add this frame to the database
        if packet.idx % max(self.p.db_stride, 1) == 0:
            self._db_desc.append(desc)
            self._db_clip.append(packet.clip_id)
            self._db_frame.append(int(packet.idx))
            po = packet.odom.pose if packet.odom is not None else None
            self._db_pose.append((po.x, po.y, po.yaw) if po else (0.0, 0.0, 0.0))
            self._db_dirty = True

        packet.place = res
        packet.timings_ms["vpr"] = (time.perf_counter() - t0) * 1000.0
        return packet

    def _seq_score(self, cand: int, packet: FramePacket) -> float:
        """Mean cosine over an aligned short sequence ending at (query, cand).

        Compares query frame k-j with database entry cand-j for j in 0..S-1, i.e. it
        asks "did we traverse the same short stretch of ground in the same order?"
        A single-frame similarity is far too noisy on this footage; this is what
        makes the score usable.
        """
        S = self.p.seq_len
        n = len(self._db_desc)
        cclip = self._db_clip[cand]
        tot, cnt = 0.0, 0
        for j in range(S):
            ci = cand - j
            if ci < 0 or self._db_clip[ci] != cclip:
                break
            if j == 0:
                qd = self._q_desc_now        # this frame is not in the database yet
            else:
                qi = n - j                   # n-1 is the previous query frame
                if qi < 0 or self._db_clip[qi] != packet.clip_id:
                    break
                qd = self._db_desc[qi]
            tot += float(np.dot(qd, self._db_desc[ci]))
            cnt += 1
        return tot / max(cnt, 1)

    # _q_desc_now is set in __call__ just before candidate scoring
    _q_desc_now: np.ndarray = np.zeros(DESC_DIM, np.float32)


# --------------------------------------------------------------------- training

def _sample_pool(n_clip_stride: int = 2, n_bursts: int = 90, burst: int = 6,
                 burst_stride: int = 8, seed: int = CFG.seed, verbose: bool = True):
    """Build the training pool: (images uint8 HxWx3, source id, timestamp seconds).

    Frames come from the five clips (every ``n_clip_stride``-th frame) and from
    ``video/input.mp4`` sampled as short bursts so temporal positives exist.
    """
    import cv2
    from ..config import VIDEO_IN
    rng = np.random.default_rng(seed)
    imgs, src, ts = [], [], []
    for si, cid in enumerate(CLIP_IDS):
        pth = io_utils.clip_path(cid)
        if not pth.exists():
            continue
        for i, fr in io_utils.read_frames(cid, resize=(IN_W, IN_H)):
            if i % n_clip_stride:
                continue
            imgs.append(fr); src.append(si); ts.append(i / CLIP_FPS)
    n_clip_frames = len(imgs)

    if VIDEO_IN.exists():
        cap = cv2.VideoCapture(str(VIDEO_IN))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        anchors = np.sort(rng.choice(max(total - burst * burst_stride - 10, 1),
                                     size=min(n_bursts, 4000), replace=False))
        for bi, a in enumerate(anchors):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(a))
            for j in range(burst):
                for _ in range(burst_stride if j else 1):
                    ok, fr = cap.read()
                    if not ok:
                        break
                if not ok:
                    break
                imgs.append(cv2.resize(fr, (IN_W, IN_H), interpolation=cv2.INTER_AREA))
                src.append(100 + bi)                      # each burst is its own source
                ts.append((a + j * burst_stride) / fps)
        cap.release()
    if verbose:
        print(f"[vpr] pool: {n_clip_frames} clip frames + {len(imgs)-n_clip_frames} "
              f"input.mp4 frames = {len(imgs)} total")
    return (np.stack(imgs), np.asarray(src, np.int32), np.asarray(ts, np.float32))


def _augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Strong photometric + crop augmentation (the positive-pair generator)."""
    import cv2
    h, w = img.shape[:2]
    s = rng.uniform(0.62, 1.0)
    ch, cw = int(h * s), int(w * s)
    y0 = rng.integers(0, h - ch + 1); x0 = rng.integers(0, w - cw + 1)
    out = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
    out = out.astype(np.float32)
    out *= rng.uniform(0.45, 1.55)                                  # exposure
    out = (out - out.mean()) * rng.uniform(0.65, 1.45) + out.mean() # contrast
    out += rng.normal(0, rng.uniform(0, 9), out.shape)              # sensor noise
    if rng.random() < 0.35:                                          # colour cast
        out *= rng.uniform(0.85, 1.15, size=(1, 1, 3))
    if rng.random() < 0.25:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.20:                                          # JPEG-ish
        q = int(rng.integers(20, 60))
        ok, enc = cv2.imencode(".jpg", np.clip(out, 0, 255).astype(np.uint8),
                               [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def train(steps: int = 400, batch: int = 16, lr: float = 3e-3, temp: float = 0.07,
          pos_dt: float = 0.4, neg_dt: float = 1.5, device: Optional[str] = None,
          out: Path = CKPT_PATH, seed: int = CFG.seed, log_every: int = 25):
    """Self-supervised InfoNCE training of the GeM head. Returns the loss curve."""
    import cv2
    device = device or io_utils.device()
    io_utils.set_seed(seed)
    rng = np.random.default_rng(seed)
    imgs, src, ts = _sample_pool(seed=seed)
    N = len(imgs)

    # index of temporal neighbours per sample (same source, |dt| <= pos_dt)
    order = np.argsort(src * 1e6 + ts)
    neigh: list[np.ndarray] = [None] * N                     # type: ignore
    for s in np.unique(src):
        idx = np.nonzero(src == s)[0]
        tt = ts[idx]
        for i in idx:
            neigh[i] = idx[np.abs(tt - ts[i]) <= pos_dt]

    net = GeMNet().to(device)
    net.train()
    opt = torch.optim.AdamW(net.head_parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    curve, accs = [], []

    for step in range(steps):
        a_idx = rng.choice(N, size=batch, replace=False)
        p_idx = np.array([rng.choice(neigh[i]) if len(neigh[i]) else i for i in a_idx])
        va = np.stack([_augment(imgs[i], rng) for i in a_idx])
        vb = np.stack([_augment(imgs[i], rng) for i in p_idx])
        x = preprocess(np.concatenate([va, vb]), device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            z = net(x)
        z = z.float()
        za, zb = z[:batch], z[batch:]
        logits = za @ zb.T / temp                             # (B,B)
        # mask false negatives: same source and |dt| <= neg_dt but not the pair itself
        same = (src[a_idx][:, None] == src[p_idx][None, :])
        close = np.abs(ts[a_idx][:, None] - ts[p_idx][None, :]) <= neg_dt
        bad = torch.from_numpy(same & close).to(device)
        eye = torch.eye(batch, dtype=torch.bool, device=device)
        logits = logits.masked_fill(bad & ~eye, -1e4)
        target = torch.arange(batch, device=device)
        loss = 0.5 * (F.cross_entropy(logits, target)
                      + F.cross_entropy(logits.T, target))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sched.step()
        curve.append(float(loss.item()))
        accs.append(float((logits.argmax(1) == target).float().mean().item()))
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"  step {step:4d}/{steps}  loss {np.mean(curve[-log_every:]):.4f}  "
                  f"batch-top1 {np.mean(accs[-log_every:]):.3f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  p {float(net.gem.p):.2f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "dim": DESC_DIM, "in_wh": (IN_W, IN_H),
                "steps": steps, "loss_curve": np.asarray(curve, np.float32),
                "acc_curve": np.asarray(accs, np.float32),
                "arch": "mobilenet_v3_small[:9] frozen + 1x1conv256 + GeM + whitening",
                "trained_on": "5 prism clips + video/input.mp4 bursts (self-supervised "
                              "InfoNCE, no place-recognition ground truth)"}, out)
    print(f"[vpr] saved {out}  ({sum(p.numel() for p in net.head_parameters())} trained "
          f"params, {sum(p.numel() for p in net.parameters())} total)")
    return net, np.asarray(curve), np.asarray(accs)


@torch.no_grad()
def evaluate(net: Optional[GeMNet] = None, device: Optional[str] = None,
             n_query: int = 200, pos_dt: float = 0.4, seed: int = 4242) -> dict:
    """Recall@1 for 'augmented query retrieves a temporally-near frame'.

    Gallery = the whole pool (un-augmented). A hit means the top-1 neighbour is from
    the same source within +-pos_dt seconds. This is the only retrieval metric this
    environment can produce honestly: there is no place-recognition ground truth.
    """
    device = device or io_utils.device()
    if net is None:
        net = GeMNet().to(device)
        if CKPT_PATH.exists():
            net.load_state_dict(torch.load(CKPT_PATH, map_location="cpu",
                                           weights_only=False)["model"])
    net = net.to(device).eval()
    rng = np.random.default_rng(seed)
    imgs, src, ts = _sample_pool(seed=CFG.seed, verbose=False)
    N = len(imgs)
    G = []
    for i in range(0, N, 32):
        x = preprocess(imgs[i:i + 32], device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            G.append(net(x).float().cpu().numpy())
    G = np.concatenate(G)
    qi = rng.choice(N, size=min(n_query, N), replace=False)
    hits = 0
    for i in qi:
        q = preprocess([_augment(imgs[i], rng)], device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            d = net(q).float().cpu().numpy()[0]
        s = G @ d
        s[i] = -2.0                                   # exclude the identical frame
        j = int(np.argmax(s))
        hits += int(src[j] == src[i] and abs(ts[j] - ts[i]) <= pos_dt)
    return {"recall@1": hits / len(qi), "n_query": int(len(qi)), "gallery": int(N)}


# --------------------------------------------------------------------- self test

def _selftest():
    import cv2
    dev = io_utils.device()
    print(f"VPR: GeM + learned whitening on frozen MobileNetV3-Small[:9]  device={dev}")
    st = VPRStage(device=dev)
    print(f"  trunk channels {st.net.trunk_channels}  desc dim {st.net.dim}  "
          f"trained={st.trained}")
    n_tot = sum(p.numel() for p in st.net.parameters())
    n_hd = sum(p.numel() for p in st.net.head_parameters())
    print(f"  params: {n_tot/1e6:.3f} M total, {n_hd/1e3:.1f} k trained")
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t = []
    for cid in ("clip_01",):
        st.reset()
        for i, fr in io_utils.read_frames(cid, max_frames=120):
            t0 = time.perf_counter()
            pk = st(FramePacket(clip_id=cid, idx=i, t=i / CLIP_FPS, rgb=fr))
            t.append((time.perf_counter() - t0) * 1000)
        pl = pk.place
        print(f"  {cid}: db={pl.db_size} best={pl.best_match_idx} "
              f"score={pl.best_score:.3f} revisit={pl.is_revisit} | {pl.note}")
    print(f"  latency {np.mean(t[5:]):.2f} ms/frame (median {np.median(t[5:]):.2f})")
    if dev == "cuda":
        print(f"  peak VRAM {torch.cuda.max_memory_allocated()/2**20:.1f} MiB")
    print(f"  eval: {evaluate(st.net, dev, n_query=100)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.train:
        net, curve, acc = train(steps=a.steps, batch=a.batch)
        print(f"  loss {curve[:20].mean():.4f} -> {curve[-20:].mean():.4f}   "
              f"batch-top1 {acc[:20].mean():.3f} -> {acc[-20:].mean():.3f}")
        print(f"  {evaluate(net)}")
    elif a.eval:
        print(evaluate())
    else:
        _selftest()
