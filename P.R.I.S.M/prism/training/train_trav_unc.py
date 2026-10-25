r"""Joint training of the shared trunk + traversability head + uncertainty head.

    python -m prism.training.train_trav_unc --epochs 30 --batch 8

One MobileNetV3-Small trunk, two thin heads, one optimiser.  That is the whole point:
the contract asks for shared backbones for CPU efficiency, so the two heads must be
trained together or the sharing is a lie.

Losses
------
traversability : class-weighted cross-entropy (ignore_index=255) + soft Dice
risk           : 0.5 * L1 + 0.5 * BCE on sigmoid(risk logit) vs the continuous
                 pseudo-label risk target, masked to non-ignored pixels
uncertainty    : SmoothL1 on softplus(raw) vs the self-supervised depth / terrain error
                 targets, weighted by the mask of pixels where each target is defined

AMP FP16, AdamW with a lower LR on the pretrained trunk, cosine schedule, batch <= 8.

What the reported numbers mean
------------------------------
Per-class IoU is **agreement with the geometric pseudo-labels**, on a held-out temporal
block of the same clips.  It is not accuracy against ground truth - none exists here -
and because held-out frames come from the same five clips it is optimistic about
generalisation.  Say it that way anywhere it is quoted.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG, CKPT_DIR, DATA_DIR, LOG_DIR, N_TRAV
from ..io_utils import set_seed, device as pick_device
from ..models.traversability import (IN_CH, NET_H, NET_W, SharedPerceptionTrunk,
                                     TravHead, TraversabilityNet)
from ..models.uncertainty import (E_TOL_DEPTH, E_TOL_SEG, UncHead, UncertaintyNet,
                                  error_to_conf, fit_calibration, reliability)

DATA = DATA_DIR / "trav"
CKPT = CKPT_DIR / "trav_unc_shared.pt"
TRAV_CLASS_NAMES = ["safe", "risky", "obstacle", "unknown"]


# ---------------------------------------------------------------- data

class TravDataset(torch.utils.data.Dataset):
    """Memory-mapped pseudo-label dataset written by ``make_trav_labels.py``."""

    def __init__(self, idx: np.ndarray, train: bool):
        self.feats = np.load(DATA / "feats.npy", mmap_mode="r")
        self.trav = np.load(DATA / "trav.npy", mmap_mode="r")
        self.risk = np.load(DATA / "risk.npy", mmap_mode="r")
        self.unc = np.load(DATA / "unc.npy", mmap_mode="r")
        self.uncw = np.load(DATA / "uncw.npy", mmap_mode="r")
        self.idx = np.asarray(idx, np.int64)
        self.train = train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        x = np.asarray(self.feats[j], np.float32)
        t = np.asarray(self.trav[j], np.int64)
        r = np.asarray(self.risk[j], np.float32)
        u = np.asarray(self.unc[j], np.float32)
        w = np.asarray(self.uncw[j], np.float32)
        if self.train and np.random.rand() < 0.5:
            # horizontal flip: nothing in the 17-channel stack is chirality dependent
            x, t, r, u, w = (np.ascontiguousarray(a[..., ::-1]) for a in (x, t, r, u, w))
        return (torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(r),
                torch.from_numpy(u), torch.from_numpy(w))


def temporal_block_split(meta: dict, val_frac: float = 0.22):
    """Hold out the LAST ``val_frac`` of each source's frames.

    A random split would put frames 3/30 s apart on both sides and leak almost
    everything.  A trailing temporal block per clip is the honest cheap option; it is
    still the same five clips, so treat the numbers as in-domain agreement.
    """
    samples = meta["samples"]
    by_clip: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        by_clip.setdefault(s["clip_id"], []).append(i)
    tr, va = [], []
    for cid, ids in by_clip.items():
        ids = sorted(ids, key=lambda i: samples[i]["idx"])
        cut = int(round(len(ids) * (1.0 - val_frac)))
        tr += ids[:cut]
        va += ids[cut:]
    return np.array(sorted(tr)), np.array(sorted(va))


# ---------------------------------------------------------------- losses

def soft_dice(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
              eps: float = 1.0) -> torch.Tensor:
    p = torch.softmax(logits, 1)
    t = F.one_hot(target.clamp(0, N_TRAV - 1), N_TRAV).permute(0, 3, 1, 2).float()
    v = valid.unsqueeze(1).float()
    inter = (p * t * v).sum((0, 2, 3))
    denom = (p * v).sum((0, 2, 3)) + (t * v).sum((0, 2, 3))
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


def class_weights_from(trav_mm, sample_idx, cap: float = 6.0) -> torch.Tensor:
    """Median-frequency balancing, capped so 'unknown' cannot dominate the gradient."""
    hist = np.zeros(N_TRAV, np.float64)
    for j in sample_idx[:: max(1, len(sample_idx) // 120)]:
        a = np.asarray(trav_mm[int(j)]).ravel()
        a = a[a != 255]
        hist += np.bincount(a, minlength=N_TRAV)[:N_TRAV]
    freq = hist / max(hist.sum(), 1.0)
    med = np.median(freq[freq > 0]) if (freq > 0).any() else 1.0
    w = np.where(freq > 0, med / np.maximum(freq, 1e-6), 1.0)
    return torch.tensor(np.clip(w, 0.2, cap), dtype=torch.float32)


# ---------------------------------------------------------------- metrics

@torch.no_grad()
def evaluate(trunk, trav_head, unc_head, loader, device, n_corr_px: int = 60000):
    trunk.eval(); trav_head.eval(); unc_head.eval()
    inter = np.zeros(N_TRAV, np.float64)
    union = np.zeros(N_TRAV, np.float64)
    correct = total = 0
    risk_ae, risk_n = 0.0, 0
    pd_all, td_all, ps_all, ts_all = [], [], [], []
    for x, t, r, u, w in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            f = trunk(x)
            lg, rk = trav_head(f)
            raw = unc_head(f)
        size = t.shape[-2:]
        lg = F.interpolate(lg.float(), size=size, mode="bilinear", align_corners=False)
        rk = F.interpolate(rk.float(), size=size, mode="bilinear", align_corners=False)
        er = F.softplus(F.interpolate(raw.float(), size=size, mode="bilinear", align_corners=False))
        pred = lg.argmax(1).cpu().numpy()
        tt = t.numpy()
        m = tt != 255
        correct += int((pred[m] == tt[m]).sum())
        total += int(m.sum())
        for c in range(N_TRAV):
            pc, tc = (pred == c) & m, (tt == c) & m
            inter[c] += float((pc & tc).sum())
            union[c] += float((pc | tc).sum())
        rp = torch.sigmoid(rk)[:, 0].cpu().numpy()
        risk_ae += float(np.abs(rp[m] - r.numpy()[m]).sum())
        risk_n += int(m.sum())
        ep = er.cpu().numpy()
        un, wn = u.numpy(), w.numpy()
        for ch, (pl, tl) in enumerate(((pd_all, td_all), (ps_all, ts_all))):
            sel = wn[:, ch] > 0.5
            if sel.any():
                pl.append(ep[:, ch][sel])
                tl.append(un[:, ch][sel])

    iou = inter / np.maximum(union, 1.0)
    out = {"pixel_acc_vs_pseudo": correct / max(total, 1),
           "mIoU_vs_pseudo": float(np.nanmean(iou)),
           "risk_mae": risk_ae / max(risk_n, 1)}
    for c, n in enumerate(TRAV_CLASS_NAMES):
        out[f"IoU_{n}"] = float(iou[c])

    try:
        from scipy.stats import spearmanr
        rng = np.random.default_rng(0)
        for name, pl, tl in (("depth", pd_all, td_all), ("seg", ps_all, ts_all)):
            if not pl:
                out[f"spearman_{name}_err"] = float("nan")
                continue
            p = np.concatenate(pl)
            t_ = np.concatenate(tl)
            if len(p) > n_corr_px:
                s = rng.choice(len(p), n_corr_px, replace=False)
                p, t_ = p[s], t_[s]
            out[f"spearman_{name}_err"] = float(spearmanr(p, t_).statistic)
            out[f"mae_{name}_err"] = float(np.abs(p - t_).mean())
    except Exception as exc:                       # noqa: BLE001
        out["spearman_error"] = f"{type(exc).__name__}: {exc}"
    return out


@torch.no_grad()
def _collect_err(trunk, unc_head, loader, device, cap: int = 600_000):
    """(predicted, true) uncertainty pairs for both channels, over a whole loader."""
    trunk.eval(); unc_head.eval()
    out = [[[], []], [[], []]]
    for x, t, r, u, w in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            raw = unc_head(trunk(x))
        e = F.softplus(F.interpolate(raw.float(), size=t.shape[-2:], mode="bilinear",
                                     align_corners=False)).cpu().numpy()
        un, wn = u.numpy(), w.numpy()
        for c in range(2):
            m = wn[:, c] > 0.5
            if m.any():
                out[c][0].append(e[:, c][m])
                out[c][1].append(un[:, c][m])
    rng = np.random.default_rng(0)
    res = []
    for c in range(2):
        if not out[c][0]:
            res.append((np.zeros(0), np.zeros(0)))
            continue
        pr = np.concatenate(out[c][0])
        tr = np.concatenate(out[c][1])
        if len(pr) > cap:
            s = rng.choice(len(pr), cap, replace=False)
            pr, tr = pr[s], tr[s]
        res.append((pr, tr))
    return res


def calibrate(trunk, unc_head, dl_tr, dl_va, device):
    """Platt-scale each channel on TRAIN, report reliability + dynamic range on VAL."""
    tol = {"depth": E_TOL_DEPTH, "seg": E_TOL_SEG}
    tr = _collect_err(trunk, unc_head, dl_tr, device)
    va = _collect_err(trunk, unc_head, dl_va, device)
    cal, rep = {}, {}
    for c, name in enumerate(("depth", "seg")):
        p_tr, t_tr = tr[c]
        p_va, t_va = va[c]
        if p_tr.size < 100:
            cal[name] = (28.0, 0.0)
            continue
        a, b = fit_calibration(p_tr, t_tr, tol[name])
        cal[name] = (a, b)
        conf = error_to_conf(p_va, None, name, {name: (a, b)})
        ece, rows = reliability(conf, t_va < tol[name])
        try:
            from scipy.stats import spearmanr
            rho = float(spearmanr(p_va, t_va).statistic)
        except Exception:                          # noqa: BLE001
            rho = float("nan")
        rep[name] = {
            "tolerance": tol[name], "platt_a": a, "platt_b": b,
            "val_ece": ece,
            "val_base_rate_error_below_tol": float((t_va < tol[name]).mean()),
            "val_conf_mean": float(conf.mean()),
            "val_conf_p05": float(np.percentile(conf, 5)),
            "val_conf_p50": float(np.percentile(conf, 50)),
            "val_conf_p95": float(np.percentile(conf, 95)),
            "val_spearman_pred_vs_true_error": rho,
            "reliability_bins": rows,
        }
        print(f"[calib] {name}: a={a:.2f} b={b:.2f}  val ECE {ece:.4f}  "
              f"conf p05/p50/p95 {rep[name]['val_conf_p05']:.3f}/"
              f"{rep[name]['val_conf_p50']:.3f}/{rep[name]['val_conf_p95']:.3f}  "
              f"base rate {rep[name]['val_base_rate_error_below_tol']:.3f}")
    return cal, rep


# ---------------------------------------------------------------- train

def train(epochs: int = 30, batch: int = 8, lr: float = 3e-4, trunk_lr_mult: float = 0.25,
          wd: float = 1e-4, val_frac: float = 0.22, device: str = None,
          w_dice: float = 0.5, w_risk: float = 0.6, w_unc: float = 1.0,
          workers: int = 0, unc_only: bool = False) -> dict:
    """``unc_only=True`` reloads the existing checkpoint, FREEZES the shared trunk and the
    traversability head (trunk kept in eval() so its BatchNorm statistics cannot move),
    and retrains only the uncertainty head.  The traversability branch is then provably
    bit-identical to the checkpoint it started from - useful when only the uncertainty
    supervision has changed and the trav head has already been validated."""
    set_seed(CFG.seed)
    meta_p = DATA / "meta.json"
    if not meta_p.exists():
        raise FileNotFoundError(
            f"{meta_p} not found. Run: python -m prism.training.make_trav_labels")
    meta = json.loads(meta_p.read_text())
    dev = torch.device(device or pick_device())
    print(f"device={dev}  samples={meta['n_samples']}  net={NET_H}x{NET_W}  in_ch={IN_CH}")

    tr_idx, va_idx = temporal_block_split(meta, val_frac)
    print(f"train {len(tr_idx)}  val {len(va_idx)} (trailing temporal block per clip)")

    ds_tr = TravDataset(tr_idx, True)
    ds_va = TravDataset(va_idx, False)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=batch, shuffle=True,
                                        num_workers=workers, drop_last=len(ds_tr) > batch)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=batch, shuffle=False,
                                        num_workers=workers)

    # The two uncertainty targets live on different scales (depth error ~0.11, terrain
    # error ~0.24).  Without per-channel normalisation the terrain term dominates the
    # gradient and the depth head collapses to its own mean.
    _u = np.load(DATA / "unc.npy", mmap_mode="r")
    _w = np.load(DATA / "uncw.npy", mmap_mode="r")
    sub = tr_idx[:: max(1, len(tr_idx) // 60)]
    unc_std = []
    for c in range(2):
        a = np.asarray(_u[sub, c], np.float32)[np.asarray(_w[sub, c]) > 0.5]
        unc_std.append(float(a.std()) if a.size else 1.0)
    unc_scale = torch.tensor([1.0 / max(v, 1e-3) for v in unc_std],
                             dtype=torch.float32, device=dev).view(1, 2, 1, 1)
    print(f"unc target std: depth {unc_std[0]:.4f}  seg {unc_std[1]:.4f}")

    cw = class_weights_from(ds_tr.trav, tr_idx).to(dev)
    print("class weights (safe/risky/obstacle/unknown):",
          " ".join(f"{v:.2f}" for v in cw.tolist()))

    trunk = SharedPerceptionTrunk(IN_CH, pretrained=not unc_only).to(dev)
    trav_head = TravHead(trunk.out_ch).to(dev)
    unc_head = UncHead(trunk.out_ch).to(dev)
    if unc_only:
        if not CKPT.exists():
            raise FileNotFoundError(f"--unc-only needs an existing {CKPT}")
        prev = torch.load(CKPT, map_location="cpu", weights_only=False)
        trunk.load_state_dict(prev["trunk"])
        trav_head.load_state_dict(prev["trav_head"])
        unc_head.load_state_dict(prev["unc_head"])
        for m in (trunk, trav_head):
            for p_ in m.parameters():
                p_.requires_grad_(False)
        print("[unc-only] trunk + trav head frozen; traversability output is unchanged")
    n_trunk = sum(p.numel() for p in trunk.parameters())
    n_tr = sum(p.numel() for p in trav_head.parameters())
    n_un = sum(p.numel() for p in unc_head.parameters())
    print(f"params: trunk {n_trunk:,}  trav_head {n_tr:,}  unc_head {n_un:,}  "
          f"total {n_trunk+n_tr+n_un:,}")

    if unc_only:
        opt = torch.optim.AdamW(unc_head.parameters(), lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.AdamW([
            {"params": trunk.parameters(), "lr": lr * trunk_lr_mult},
            {"params": list(trav_head.parameters()) + list(unc_head.parameters()), "lr": lr},
        ], weight_decay=wd)
    steps = max(1, len(dl_tr)) * epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    ce = nn.CrossEntropyLoss(weight=cw, ignore_index=255)

    hist = []
    t_start = time.time()
    for ep in range(epochs):
        if unc_only:
            trunk.eval(); trav_head.eval(); unc_head.train()
        else:
            trunk.train(); trav_head.train(); unc_head.train()
        agg = np.zeros(5)
        nb = 0
        for x, t, r, u, w in dl_tr:
            x = x.to(dev, non_blocking=True)
            t = t.to(dev, non_blocking=True)
            r = r.to(dev, non_blocking=True)
            u = u.to(dev, non_blocking=True)
            w = w.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.float16, enabled=dev.type == "cuda"):
                f = trunk(x)
                lg, rk = trav_head(f)
                raw = unc_head(f)
                size = t.shape[-2:]
                lg = F.interpolate(lg, size=size, mode="bilinear", align_corners=False)
                rk = F.interpolate(rk, size=size, mode="bilinear", align_corners=False)[:, 0]
                raw = F.interpolate(raw, size=size, mode="bilinear", align_corners=False)

                valid = t != 255
                l_ce = ce(lg, t)
                l_dice = soft_dice(lg, torch.where(valid, t, torch.zeros_like(t)), valid)

                vm = valid.float()
                rk32 = rk.float()
                rp = torch.sigmoid(rk32)
                l_l1 = ((rp - r).abs() * vm).sum() / vm.sum().clamp_min(1.0)
                # BCE *with logits*: plain binary_cross_entropy is unsafe under autocast
                l_bce = (F.binary_cross_entropy_with_logits(rk32, r.clamp(0, 1),
                                                            reduction="none") * vm
                         ).sum() / vm.sum().clamp_min(1.0)
                l_risk = 0.5 * l_l1 + 0.5 * l_bce

                err = F.softplus(raw.float())
                # beta small => essentially L1: median-seeking, so the prediction keeps
                # the spread of the target instead of shrinking to its mean.
                l_unc = (F.smooth_l1_loss(err, u, beta=0.02, reduction="none")
                         * unc_scale * w).sum() / w.sum().clamp_min(1.0)

                loss = (w_unc * l_unc if unc_only
                        else l_ce + w_dice * l_dice + w_risk * l_risk + w_unc * l_unc)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            trainable = (list(unc_head.parameters()) if unc_only else
                         list(trunk.parameters()) + list(trav_head.parameters())
                         + list(unc_head.parameters()))
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            agg += np.array([float(loss), float(l_ce), float(l_dice), float(l_risk), float(l_unc)])
            nb += 1
        agg /= max(nb, 1)
        line = (f"ep {ep+1:3d}/{epochs}  loss {agg[0]:.4f}  ce {agg[1]:.4f}  "
                f"dice {agg[2]:.4f}  risk {agg[3]:.4f}  unc {agg[4]:.4f}  "
                f"lr {sched.get_last_lr()[-1]:.2e}")
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            m = evaluate(trunk, trav_head, unc_head, dl_va, dev)
            line += (f"  | val mIoU {m['mIoU_vs_pseudo']:.3f}  acc {m['pixel_acc_vs_pseudo']:.3f}"
                     f"  rho_d {m.get('spearman_depth_err', float('nan')):.3f}"
                     f"  rho_s {m.get('spearman_seg_err', float('nan')):.3f}")
            hist.append({"epoch": ep + 1, **{k: v for k, v in m.items()}})
        print(line, flush=True)

    # ---------------- calibrate the confidence mapping ----------------
    cal, cal_report = calibrate(trunk, unc_head, dl_tr, dl_va, dev)
    final = evaluate(trunk, trav_head, unc_head, dl_va, dev)
    final["confidence_calibration"] = cal_report
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == "cuda" else 0.0
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "trunk": trunk.state_dict(),
        "trav_head": trav_head.state_dict(),
        "unc_head": unc_head.state_dict(),
        "in_ch": IN_CH, "net_hw": [NET_H, NET_W], "fpn_ch": trunk.out_ch,
        "class_weights": cw.cpu().tolist(),
        "conf_calibration": {k: list(v) for k, v in cal.items()},
        "metrics_vs_pseudo_labels": final,
        "supervision": meta.get("supervision"),
        "thresholds": meta.get("thresholds"),
        "n_params": {"trunk": n_trunk, "trav_head": n_tr, "unc_head": n_un,
                     "total": n_trunk + n_tr + n_un},
        "unc_only_retrain": bool(unc_only),
    }, CKPT)

    report = {
        "checkpoint": str(CKPT),
        "epochs": epochs, "batch": batch, "device": str(dev),
        "train_samples": int(len(tr_idx)), "val_samples": int(len(va_idx)),
        "train_seconds": round(time.time() - t_start, 1),
        "peak_vram_mb": round(peak, 1),
        "n_params": {"trunk": n_trunk, "trav_head": n_tr, "unc_head": n_un,
                     "total": n_trunk + n_tr + n_un},
        "final_val_metrics_vs_PSEUDO_LABELS": final,
        "history": hist,
        "caveat": "IoU / accuracy are agreement with the geometric pseudo-labels on a "
                  "held-out temporal block of the SAME five clips. Not ground truth, "
                  "and optimistic about cross-scene generalisation.",
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "trav_unc_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.22)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--unc-only", action="store_true",
                    help="freeze the shared trunk + traversability head and retrain only "
                         "the uncertainty head (traversability output stays identical)")
    a = ap.parse_args()
    train(epochs=a.epochs, batch=a.batch, lr=a.lr, val_frac=a.val_frac,
          device=a.device, workers=a.workers, unc_only=a.unc_only)
