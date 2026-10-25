"""Distil SegFormer-B0/ADE20K -> PIDNet-S on the prism-7 taxonomy.

The student never sees human labels. It sees, per frame:
  * the teacher's **soft** 7-class probabilities  -> temperature KL (the distillation term)
  * the teacher's **argmax** label, 255 ignored    -> OHEM cross-entropy (PIDNet's own loss)
  * Canny boundaries of that label                 -> weighted BCE on the D branch
  * a boundary-aware CE that only scores pixels the D branch calls a boundary
plus PIDNet's auxiliary P-branch head at weight 0.4.

Everything is measured against the teacher, so the number reported at the end is
**teacher-agreement mIoU**, not ground-truth mIoU - there is no off-road ground truth in
this environment. It is labelled that way in the checkpoint, the logs and the renderer.

    python -m prism.training.distill_seg --epochs 32
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..config import (CFG, CKPT_DIR, DATA_DIR, IGNORE_INDEX, LOG_DIR, N_TERRAIN,
                      SEG_INPUT, TERRAIN_CLASSES)
from .. import io_utils
from ..models.seg_pidnet import PIDNetS, count_params

SEG_DIR = DATA_DIR / "seg"
CKPT_PATH = CKPT_DIR / "pidnet_s_drishti7.pt"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ------------------------------------------------------------------------------- data
class SegDistillDataset(Dataset):
    """Frames + teacher targets, with photometric and geometric augmentation.

    Geometric augmentation is a *relative crop box* applied identically to the image,
    the soft target grid and the hard label, so nothing has to be padded and the three
    stay pixel-aligned. Zoom-in only: the camera FOV is fixed on a real vehicle, so
    zooming out would synthesise a view the deployed system can never see.
    """

    def __init__(self, split: str, augment: bool, out_hw: tuple[int, int]):
        meta = json.loads((SEG_DIR / "manifest.json").read_text())
        self.samples = [s for s in meta["samples"] if s["split"] == split]
        self.iw, self.ih = meta["input_size"]
        self.lw, self.lh = meta["label_size"]
        self.oh, self.ow = out_hw            # student output stride grid (36, 64)
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    # -------------------------------------------------------------- augmentation bits
    @staticmethod
    def _crop_box(rng: np.random.Generator):
        if rng.random() < 0.35:
            return None                                   # keep the deployed FOV
        r = float(rng.uniform(0.65, 0.98))                # scale jitter
        x0 = float(rng.uniform(0.0, 1.0 - r))
        y0 = float(rng.uniform(0.0, 1.0 - r))
        return x0, y0, x0 + r, y0 + r

    @staticmethod
    def _apply_box(arr: np.ndarray, box, size, interp):
        h, w = arr.shape[:2]
        x0, y0, x1, y1 = box
        c = arr[int(y0 * h):max(int(y1 * h), int(y0 * h) + 1),
                int(x0 * w):max(int(x1 * w), int(x0 * w) + 1)]
        return cv2.resize(c, size, interpolation=interp)

    @staticmethod
    def _color_jitter(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        img = bgr.astype(np.float32)
        img *= rng.uniform(0.7, 1.35)                          # brightness
        mean = img.mean()
        img = (img - mean) * rng.uniform(0.75, 1.3) + mean      # contrast
        img *= rng.uniform(0.92, 1.08, size=(1, 1, 3))          # channel gain / white balance
        img = np.clip(img, 0, 255)
        if rng.random() < 0.3:                                  # gamma
            g = float(rng.uniform(0.7, 1.4))
            img = 255.0 * np.power(img / 255.0, g)
        if rng.random() < 0.2:                                  # sensor noise (low light)
            img += rng.normal(0, rng.uniform(2, 7), img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    # -------------------------------------------------------------- item
    def __getitem__(self, i):
        s = self.samples[i]
        bgr = cv2.imread(str(SEG_DIR / "frames" / f"{s['name']}.jpg"), cv2.IMREAD_COLOR)
        z = np.load(SEG_DIR / "targets" / f"{s['name']}.npz")
        soft = z["soft"].astype(np.float32)                 # (7, sh, sw)
        label = z["label"]                                  # (lh, lw) uint8

        rng = np.random.default_rng()
        if self.augment:
            box = self._crop_box(rng)
            if box is not None:
                bgr = self._apply_box(bgr, box, (self.iw, self.ih), cv2.INTER_LINEAR)
                label = self._apply_box(label, box, (self.lw, self.lh), cv2.INTER_NEAREST)
                soft = np.stack([self._apply_box(soft[c], box, (self.ow, self.oh),
                                                 cv2.INTER_LINEAR)
                                 for c in range(N_TERRAIN)])
            else:
                soft = np.stack([cv2.resize(soft[c], (self.ow, self.oh),
                                            interpolation=cv2.INTER_AREA)
                                 for c in range(N_TERRAIN)])
            if rng.random() < 0.5:
                bgr = bgr[:, ::-1]
                label = label[:, ::-1]
                soft = soft[:, :, ::-1]
            bgr = self._color_jitter(bgr, rng)
        else:
            soft = np.stack([cv2.resize(soft[c], (self.ow, self.oh),
                                        interpolation=cv2.INTER_AREA)
                             for c in range(N_TERRAIN)])

        soft = np.ascontiguousarray(soft)
        soft /= np.clip(soft.sum(0, keepdims=True), 1e-6, None)
        label = np.ascontiguousarray(label)

        # boundary target: Canny on the teacher label, dilated to ~3 px, ignore-safe
        lab_vis = np.where(label == IGNORE_INDEX, 0, label.astype(np.int32) + 1)
        edge = cv2.Canny((lab_vis * 28).astype(np.uint8), 40, 120)
        edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=1)
        bd = (edge > 0).astype(np.float32)
        bd[label == IGNORE_INDEX] = 0.0

        rgb = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return (torch.from_numpy(rgb.transpose(2, 0, 1).copy()),
                torch.from_numpy(soft),
                torch.from_numpy(label.astype(np.int64)),
                torch.from_numpy(bd))


# ------------------------------------------------------------------------------- losses
class OhemCrossEntropy(nn.Module):
    """PIDNet's OHEM CE: keep only pixels the model is unsure about."""

    def __init__(self, ignore_index=IGNORE_INDEX, thresh=0.9, min_kept_frac=1 / 16,
                 weight=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.thresh = -math.log(thresh)
        self.min_kept_frac = min_kept_frac
        self.register_buffer("cls_weight", weight if weight is not None
                             else torch.ones(N_TERRAIN))

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.cls_weight.to(logits.dtype),
                             ignore_index=self.ignore_index, reduction="none")
        mask = target != self.ignore_index
        if mask.sum() == 0:
            return logits.sum() * 0.0
        losses = ce[mask]
        min_kept = max(1, int(target.numel() * self.min_kept_frac))
        if losses.numel() > min_kept:
            with torch.no_grad():
                sorted_loss, _ = torch.sort(losses.detach(), descending=True)
                thr = max(float(sorted_loss[min_kept - 1]), self.thresh)
                keep = losses.detach() >= thr
            if keep.sum() > 0:
                losses = losses[keep]
        return losses.mean()


def weighted_bce(bd_logits, bd_target):
    """PIDNet's boundary loss: BCE reweighted by the pos/neg imbalance of the batch."""
    logit = bd_logits.reshape(-1)
    tgt = bd_target.reshape(-1)
    pos = tgt > 0.5
    n_pos = int(pos.sum())
    n_neg = int(tgt.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return F.binary_cross_entropy_with_logits(logit, tgt)
    w = torch.empty_like(tgt)
    w[pos] = n_neg / (n_pos + n_neg)
    w[~pos] = n_pos / (n_pos + n_neg)
    return F.binary_cross_entropy_with_logits(logit, tgt, weight=w)


def kd_kl(student_logits, teacher_prob, valid, T=2.0):
    """Temperature-softened KL(teacher || student) over the 7 prism classes."""
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    # re-soften the teacher's probabilities at the same temperature
    log_q = torch.log(teacher_prob.clamp_min(1e-6)) / T
    p_t = F.softmax(log_q, dim=1)
    kl = (p_t * (torch.log(p_t.clamp_min(1e-8)) - log_p_s)).sum(1)
    v = valid.to(kl.dtype)
    denom = v.sum().clamp_min(1.0)
    return (kl * v).sum() / denom * (T * T)


# ------------------------------------------------------------------------------- metrics
class ConfMat:
    def __init__(self, n=N_TERRAIN):
        self.n = n
        self.m = np.zeros((n, n), np.int64)

    def update(self, pred, gt):
        k = gt != IGNORE_INDEX
        self.m += np.bincount(gt[k].astype(np.int64) * self.n + pred[k].astype(np.int64),
                              minlength=self.n ** 2).reshape(self.n, self.n)

    def iou(self):
        inter = np.diag(self.m).astype(np.float64)
        union = self.m.sum(1) + self.m.sum(0) - inter
        with np.errstate(invalid="ignore", divide="ignore"):
            iou = inter / union
        return iou, self.m.sum(1)

    def pixel_acc(self):
        return float(np.diag(self.m).sum() / max(self.m.sum(), 1))


# ------------------------------------------------------------------------------- train
def evaluate(model, loader, device, label_hw):
    model.eval()
    cm = ConfMat()
    with torch.no_grad():
        for x, soft, label, bd in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                out = model(x)
            out = F.interpolate(out.float(), size=label_hw, mode="bilinear",
                                align_corners=False)
            cm.update(out.argmax(1).cpu().numpy().ravel(), label.numpy().ravel())
    model.train()
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--T", type=float, default=2.0, help="distillation temperature")
    ap.add_argument("--w-kd", type=float, default=2.0)
    ap.add_argument("--w-ohem", type=float, default=1.0)
    ap.add_argument("--w-aux", type=float, default=0.4)
    ap.add_argument("--w-bd", type=float, default=20.0)
    ap.add_argument("--w-bdce", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    CFG.ensure_dirs()
    io_utils.set_seed(CFG.seed)
    device = args.device or io_utils.device()
    W, H = SEG_INPUT
    out_hw = (H // 8, W // 8)

    meta = json.loads((SEG_DIR / "manifest.json").read_text())
    label_hw = (meta["label_size"][1], meta["label_size"][0])

    tr = SegDistillDataset("train", augment=True, out_hw=out_hw)
    va = SegDistillDataset("val", augment=False, out_hw=out_hw)
    print(f"[distill] {len(tr)} train / {len(va)} val frames  input {W}x{H}  "
          f"student out {out_hw[1]}x{out_hw[0]}  loss res {label_hw[1]}x{label_hw[0]}")

    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=(device == "cuda"), drop_last=True,
                       persistent_workers=args.workers > 0)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=0,
                       pin_memory=(device == "cuda"))

    # mild inverse-sqrt-frequency class weights, capped so the 0.04% water class cannot
    # dominate the gradient.
    bal = meta["class_balance"]
    freq = np.array([max(bal[c], 1e-4) for c in TERRAIN_CLASSES], np.float32)
    cw = 1.0 / np.sqrt(freq)
    cw = np.clip(cw / np.median(cw), 0.5, 3.0).astype(np.float32)
    print("[distill] class weights: " +
          ", ".join(f"{n}={w:.2f}" for n, w in zip(TERRAIN_CLASSES, cw)))

    model = PIDNetS(num_classes=N_TERRAIN, augment=True).to(device)
    print(f"[distill] PIDNet-S params {count_params(model):,}")
    ohem = OhemCrossEntropy(weight=torch.from_numpy(cw)).to(device)
    ce_plain = nn.CrossEntropyLoss(weight=torch.from_numpy(cw).to(device),
                                   ignore_index=IGNORE_INDEX)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, len(dl_tr))
    total_steps = args.epochs * steps_per_epoch
    warm_steps = args.warmup * steps_per_epoch

    def lr_at(step):
        if step < warm_steps:
            return (step + 1) / max(warm_steps, 1)
        p = (step - warm_steps) / max(total_steps - warm_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * p)) * (1 - 1e-3) + 1e-3

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    history = []
    best_miou, step = -1.0, 0
    t_start = time.time()
    model.train()
    for ep in range(args.epochs):
        agg = np.zeros(6, np.float64)
        n_b = 0
        t_ep = time.time()
        for x, soft, label, bd in dl_tr:
            x = x.to(device, non_blocking=True)
            soft = soft.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            bd = bd.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                out_p, out, out_d = model(x)

            out_f = out.float()
            out_p_f = out_p.float()
            out_d_f = out_d.float()

            # --- distillation KL at the student's native output grid
            valid_lo = F.interpolate((label != IGNORE_INDEX).float().unsqueeze(1),
                                     size=out_hw, mode="nearest")[:, 0] > 0.5
            l_kd = kd_kl(out_f, soft, valid_lo, T=args.T)

            # --- hard losses at the stored label resolution
            up = F.interpolate(out_f, size=label_hw, mode="bilinear", align_corners=False)
            up_p = F.interpolate(out_p_f, size=label_hw, mode="bilinear", align_corners=False)
            up_d = F.interpolate(out_d_f, size=label_hw, mode="bilinear", align_corners=False)

            l_ohem = ohem(up, label)
            l_aux = ohem(up_p, label)
            l_bd = weighted_bce(up_d, bd.unsqueeze(1))

            # boundary-aware CE: score only where the D branch says "boundary"
            with torch.no_grad():
                bd_mask = torch.sigmoid(up_d[:, 0]) > 0.8
                bd_label = torch.where(bd_mask, label,
                                       torch.full_like(label, IGNORE_INDEX))
            l_bdce = ce_plain(up, bd_label) if (bd_label != IGNORE_INDEX).any() \
                else up.sum() * 0.0

            loss = (args.w_kd * l_kd + args.w_ohem * l_ohem + args.w_aux * l_aux
                    + args.w_bd * l_bd + args.w_bdce * l_bdce)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            agg += [float(loss), float(l_kd), float(l_ohem), float(l_aux),
                    float(l_bd), float(l_bdce)]
            n_b += 1

        agg /= max(n_b, 1)
        cm = evaluate(model, dl_va, device, label_hw)
        iou, sup = cm.iou()
        miou = float(np.nanmean(iou[sup > 0]))
        rec = dict(epoch=ep + 1, lr=sched.get_last_lr()[0], loss=agg[0], kd=agg[1],
                   ohem=agg[2], aux=agg[3], bd=agg[4], bdce=agg[5],
                   val_agree_miou=miou, val_pixel_acc=cm.pixel_acc(),
                   secs=time.time() - t_ep)
        history.append(rec)
        print(f"ep {ep+1:3d}/{args.epochs}  loss {agg[0]:6.3f} "
              f"(kd {agg[1]:.3f} ohem {agg[2]:.3f} aux {agg[3]:.3f} "
              f"bd {agg[4]:.4f} bdce {agg[5]:.3f})  "
              f"val agree-mIoU {miou*100:5.2f}  acc {cm.pixel_acc()*100:5.2f}  "
              f"lr {sched.get_last_lr()[0]:.2e}  {rec['secs']:.0f}s")

        if miou > best_miou:
            best_miou = miou
            torch.save(dict(
                state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                arch="PIDNet-S",
                model_cfg=model.cfg,
                num_classes=N_TERRAIN,
                classes=TERRAIN_CLASSES,
                input_size=list(SEG_INPUT),
                output_stride=8,
                normalize=dict(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist(),
                               order="RGB", scale=1 / 255.0),
                teacher=meta["teacher"],
                metric_name="teacher-agreement mIoU (SegFormer-B0/ADE20K pseudo-labels, "
                            "NOT ground-truth mIoU)",
                val_agree_miou=miou,
                val_per_class_agree_iou={TERRAIN_CLASSES[c]: (float(iou[c])
                                                              if sup[c] > 0 else None)
                                         for c in range(N_TERRAIN)},
                val_pixel_acc=cm.pixel_acc(),
                epoch=ep + 1,
                train_args=vars(args),
                dataset=dict(n_train=len(tr), n_val=len(va),
                             class_balance=meta["class_balance"]),
            ), CKPT_PATH)

    if device == "cuda":
        print(f"[distill] peak VRAM {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
        torch.cuda.empty_cache()

    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    print(f"\n[distill] best epoch {ck['epoch']}  saved -> {CKPT_PATH}")
    print(f"[distill] FINAL teacher-agreement mIoU (held-out {len(va)} frames): "
          f"{ck['val_agree_miou']*100:.2f}%   pixel agreement {ck['val_pixel_acc']*100:.2f}%")
    print("[distill] per-class agreement IoU (student vs SegFormer teacher label):")
    for c, name in enumerate(TERRAIN_CLASSES):
        v = ck["val_per_class_agree_iou"][name]
        print(f"   {name:<10s} " + ("  n/a (absent in val)" if v is None else f"{v*100:6.2f}%"))
    print(f"[distill] total wall time {(time.time()-t_start)/60:.1f} min on {device}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "distill_seg_history.json").write_text(json.dumps(history, indent=1))
    print(f"[distill] history -> {LOG_DIR / 'distill_seg_history.json'}")


if __name__ == "__main__":
    main()
