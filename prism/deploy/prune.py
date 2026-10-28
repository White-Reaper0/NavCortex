"""Structured (channel) pruning of the prism student networks, with a sparsity sweep.

What "structured" means here, and why it is the only honest option
-----------------------------------------------------------------
Unstructured magnitude pruning produces a sparse weight tensor that is *smaller on
paper and exactly as slow in practice*, because neither PyTorch's CPU kernels nor ONNX
Runtime's exploit unstructured sparsity.  Claiming a speedup from it would be dishonest.
So this module prunes whole output channels and physically rebuilds the convolutions at
the smaller width: the resulting model has fewer parameters, fewer FLOPs, and a real
(if sometimes disappointing) change in wall-clock latency.

Where channels can be removed safely
------------------------------------
A channel can be dropped only if every consumer of that tensor can be narrowed with it.
Residual adds, concatenations, PagFM's elementwise similarity product and PAPPM's
grouped convolution all couple channel counts across branches, so they are left alone.
What *is* free is the **interior** of a block - a convolution whose output is consumed
by exactly one normalisation and one following convolution:

    BasicBlock      conv1 -> bn1 -> relu -> conv2                 (conv1 out-channels)
    Bottleneck      conv1 -> bn1 -> relu -> conv2                 (conv1 out-channels)
                    conv2 -> bn2 -> relu -> conv3                 (conv2 out-channels)
    segmenthead     conv1 -> bn2 -> relu -> conv2                 (interplanes)
    TravHead/UncHead  stem[0] -> stem[1] -> act -> cls & risk     (mid channels)
    world model     stem block i -> GroupNorm -> SiLU -> block i+1

Criterion: L1 norm of each output filter (Li et al., "Pruning Filters for Efficient
ConvNets", ICLR 2017), which for a BN-followed conv is scaled by the BN gamma so a
filter that the normalisation has already switched off is correctly ranked last.

Fine-tuning after pruning
-------------------------
There is no ground-truth terrain annotation in this repo, so "recover accuracy" is
defined precisely as **recover agreement with the unpruned student**: the pruned model
is fine-tuned by self-distillation (KL on the teacher student's soft logits, on real
clip frames).  Every accuracy number below is agreement with the unpruned FP32 model,
never accuracy against ground truth.

Reporting
---------
The sweep writes ``output/pruning.json`` with, per sparsity level, the parameter count,
file size, FLOPs, agreement before and after fine-tuning, and measured CPU latency.  If
pruning does not pay off at these model sizes, that is what the file says.

Usage
-----
    python -m prism.deploy.prune                       # every available student
    python -m prism.deploy.prune --only pidnet
    python -m prism.deploy.prune --sparsity 0.2 0.4 --steps 300
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG, CKPT_DIR, CLIP_IDS, N_TERRAIN, OUT_DIR, SEG_INPUT
from .export_onnx import hardware_info, sampled_frames, seg_input

REPORT = OUT_DIR / "pruning.json"
PRUNE_DIR = CKPT_DIR / "pruned"
DEFAULT_SPARSITY = (0.10, 0.20, 0.30, 0.50)


# ============================================================ channel surgery

def _filter_scores(conv: nn.Conv2d, norm: Optional[nn.Module]) -> torch.Tensor:
    """L1 norm per output filter, scaled by the following norm layer's gain."""
    w = conv.weight.detach()
    s = w.abs().sum(dim=(1, 2, 3))
    if norm is not None and getattr(norm, "weight", None) is not None:
        s = s * norm.weight.detach().abs()
    return s


def _slice_conv_out(conv: nn.Conv2d, keep: torch.Tensor) -> nn.Conv2d:
    new = nn.Conv2d(conv.in_channels, len(keep), conv.kernel_size, conv.stride,
                    conv.padding, conv.dilation, conv.groups,
                    bias=conv.bias is not None)
    new.weight.data = conv.weight.data[keep].clone()
    if conv.bias is not None:
        new.bias.data = conv.bias.data[keep].clone()
    return new


def _slice_conv_in(conv: nn.Conv2d, keep: torch.Tensor) -> nn.Conv2d:
    if conv.groups != 1:
        raise ValueError("grouped conv input slicing is not safe here")
    new = nn.Conv2d(len(keep), conv.out_channels, conv.kernel_size, conv.stride,
                    conv.padding, conv.dilation, 1, bias=conv.bias is not None)
    new.weight.data = conv.weight.data[:, keep].clone()
    if conv.bias is not None:
        new.bias.data = conv.bias.data.clone()
    return new


def _slice_norm(norm: nn.Module, keep: torch.Tensor) -> nn.Module:
    if isinstance(norm, nn.BatchNorm2d):
        new = nn.BatchNorm2d(len(keep), eps=norm.eps, momentum=norm.momentum,
                             affine=norm.affine, track_running_stats=norm.track_running_stats)
        if norm.affine:
            new.weight.data = norm.weight.data[keep].clone()
            new.bias.data = norm.bias.data[keep].clone()
        if norm.track_running_stats:
            new.running_mean.data = norm.running_mean.data[keep].clone()
            new.running_var.data = norm.running_var.data[keep].clone()
            new.num_batches_tracked.data = norm.num_batches_tracked.data.clone()
        return new
    if isinstance(norm, nn.GroupNorm):
        g = norm.num_groups
        while len(keep) % g != 0 and g > 1:
            g -= 1
        new = nn.GroupNorm(g, len(keep), eps=norm.eps, affine=norm.affine)
        if norm.affine:
            new.weight.data = norm.weight.data[keep].clone()
            new.bias.data = norm.bias.data[keep].clone()
        return new
    raise TypeError(f"cannot slice norm of type {type(norm)}")


class _Triple:
    """One prunable interior: producer conv -> norm -> (one or more) consumer convs."""

    def __init__(self, owner: nn.Module, a_name: str, n_name: str, b_names: list[str]):
        self.owner, self.a_name, self.n_name, self.b_names = owner, a_name, n_name, b_names

    @property
    def conv_a(self) -> nn.Conv2d:
        return _get(self.owner, self.a_name)

    @property
    def norm(self) -> nn.Module:
        return _get(self.owner, self.n_name)

    def width(self) -> int:
        return self.conv_a.out_channels

    def prune(self, ratio: float, min_keep: int = 4) -> int:
        c = self.conv_a.out_channels
        k = max(min_keep, int(round(c * (1.0 - ratio))))
        # GroupNorm needs the width to stay divisible by (a divisor of) its groups
        if isinstance(self.norm, nn.GroupNorm):
            g = self.norm.num_groups
            k = max(min_keep, (k // g) * g) if k >= g else g
        if k >= c:
            return 0
        keep = torch.argsort(_filter_scores(self.conv_a, self.norm), descending=True)[:k]
        keep, _ = torch.sort(keep)
        _set(self.owner, self.a_name, _slice_conv_out(self.conv_a, keep))
        _set(self.owner, self.n_name, _slice_norm(self.norm, keep))
        for bn in self.b_names:
            _set(self.owner, bn, _slice_conv_in(_get(self.owner, bn), keep))
        return c - k


def _get(root: nn.Module, path: str):
    o = root
    for p in path.split("."):
        o = o[int(p)] if p.isdigit() and isinstance(o, (nn.Sequential, nn.ModuleList)) else getattr(o, p)
    return o


def _set(root: nn.Module, path: str, value):
    parts = path.split(".")
    o = root
    for p in parts[:-1]:
        o = o[int(p)] if p.isdigit() and isinstance(o, (nn.Sequential, nn.ModuleList)) else getattr(o, p)
    last = parts[-1]
    if last.isdigit() and isinstance(o, (nn.Sequential, nn.ModuleList)):
        o[int(last)] = value
    else:
        setattr(o, last, value)


# ---- architecture-specific interior maps -----------------------------------------

def pidnet_triples(net: nn.Module) -> list[_Triple]:
    from ..models.seg_pidnet import BasicBlock, Bottleneck, segmenthead
    out = []
    for m in net.modules():
        if isinstance(m, BasicBlock):
            out.append(_Triple(m, "conv1", "bn1", ["conv2"]))
        elif isinstance(m, Bottleneck):
            out.append(_Triple(m, "conv1", "bn1", ["conv2"]))
            out.append(_Triple(m, "conv2", "bn2", ["conv3"]))
        elif isinstance(m, segmenthead):
            out.append(_Triple(m, "conv1", "bn2", ["conv2"]))
    return out


def world_model_triples(net: nn.Module) -> list[_Triple]:
    """The encoder's strided conv chain; the last block feeds an AdaptiveAvgPool+Linear."""
    # the export wrapper holds the real model in `.m`
    core = getattr(net, "m", net)
    stem = core.encoder.stem
    out = []
    for i in range(len(stem) - 1):
        out.append(_Triple(stem, f"{i}.0", f"{i}.1", [f"{i+1}.0"]))
    return out


def head_triples(net: nn.Module) -> list[_Triple]:
    """Traversability / uncertainty thin heads (the shared MobileNet trunk is residual)."""
    out = []
    for name in ("trav_head", "unc_head", "head"):
        h = getattr(net, name, None)
        if h is None:
            continue
        if hasattr(h, "cls") and hasattr(h, "risk"):
            out.append(_Triple(h, "stem.0", "stem.1", ["cls", "risk"]))
        elif hasattr(h, "out"):
            out.append(_Triple(h, "stem.0", "stem.1", ["out"]))
    return out


# ============================================================ measurement helpers

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def count_flops(model: nn.Module, shape, device="cpu") -> int:
    total = [0]

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            oh, ow = out.shape[-2:]
            k = mod.kernel_size[0] * mod.kernel_size[1]
            total[0] += 2 * oh * ow * mod.out_channels * (mod.in_channels // mod.groups) * k
        elif isinstance(mod, nn.Linear):
            total[0] += 2 * mod.in_features * mod.out_features

    hs = [m.register_forward_hook(hook) for m in model.modules()
          if isinstance(m, (nn.Conv2d, nn.Linear))]
    was = model.training
    model.eval().to(device)
    model(torch.zeros(*shape, device=device))
    for h in hs:
        h.remove()
    model.train(was)
    return total[0]


@torch.no_grad()
def cpu_latency_ms(model: nn.Module, x: torch.Tensor, warmup: int = 5,
                   iters: int = 30, threads: int = 4) -> dict:
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    model = model.eval().cpu()
    x = x.cpu()
    try:
        for _ in range(warmup):
            model(x)
        ts = []
        for _ in range(iters):
            t = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t) * 1e3)
    finally:
        torch.set_num_threads(old)
    a = np.array(ts)
    return {"median_ms": round(float(np.median(a)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3),
            "iters": iters, "torch_threads": threads}


def _miou(cm: np.ndarray) -> float:
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    p = union > 0
    return float(np.mean(inter[p] / union[p])) if p.any() else float("nan")


# ============================================================ tasks

class PruneTask:
    """One prunable model plus everything needed to prune, retrain and score it."""

    def __init__(self, name: str, build: Callable[[], nn.Module],
                 triples: Callable[[nn.Module], list[_Triple]],
                 input_shape: tuple, batch_builder: Callable[[int], torch.Tensor],
                 agreement: Callable[[nn.Module, nn.Module, list[torch.Tensor]], dict],
                 distill_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 ckpt: Optional[Path], provenance: str):
        self.name, self.build, self.triples = name, build, triples
        self.input_shape, self.batch_builder = input_shape, batch_builder
        self.agreement, self.distill_loss = agreement, distill_loss
        self.ckpt, self.provenance = ckpt, provenance

    def available(self) -> bool:
        return self.ckpt is None or self.ckpt.exists()


def _seg_agreement(teacher: nn.Module, student: nn.Module, batches: list[torch.Tensor]) -> dict:
    n = N_TERRAIN
    cm = np.zeros((n, n), np.float64)
    teacher.eval().cpu()
    student.eval().cpu()
    with torch.no_grad():
        for x in batches:
            a = teacher(x).argmax(1).numpy().ravel()
            b = student(x).argmax(1).numpy().ravel()
            cm += np.bincount(a * n + b, minlength=n * n).reshape(n, n)
    return {"mIoU_vs_unpruned": round(_miou(cm), 5),
            "pixel_agreement": round(float(np.diag(cm).sum() / max(cm.sum(), 1)), 5)}


def _wm_agreement(teacher: nn.Module, student: nn.Module, batches: list[torch.Tensor]) -> dict:
    teacher.eval().cpu()
    student.eval().cpu()
    errs, best = [], []
    with torch.no_grad():
        for x in batches:
            a = teacher(x)
            b = student(x)
            errs.append(float((a[2] - b[2]).abs().mean()))
            best.append(int(a[2].mean(-1).argmin() == b[2].mean(-1).argmin()))
    return {"collision_risk_mae_vs_unpruned": round(float(np.mean(errs)), 6),
            "safest_action_agreement": round(float(np.mean(best)), 4)}


def _kl_logits(student_out, teacher_out, T: float = 2.0) -> torch.Tensor:
    s = student_out if torch.is_tensor(student_out) else student_out[0]
    t = teacher_out if torch.is_tensor(teacher_out) else teacher_out[0]
    return F.kl_div(F.log_softmax(s / T, 1), F.softmax(t / T, 1),
                    reduction="batchmean") * (T * T)


def _wm_loss(student_out, teacher_out) -> torch.Tensor:
    return sum(F.mse_loss(s, t) for s, t in zip(student_out, teacher_out))


def _seg_batches(n: int, bs: int = 2) -> list[torch.Tensor]:
    per_clip = max(1, int(round(n / len(CLIP_IDS))))
    xs = [torch.from_numpy(seg_input(bgr)) for _, _, bgr in sampled_frames(per_clip)]
    return [torch.cat(xs[i:i + bs]) for i in range(0, len(xs) - bs + 1, bs)]


def _wm_batches(n: int, bs: int = 1) -> list[torch.Tensor]:
    # batch 1 only: the exported wrapper rolls out all six actions from a single latent
    # (`s0.expand(N_ACTIONS, -1)`), which is exactly what the pipeline does per frame.
    from ..nav import bev_utils as bu
    kinds = ["clear", "wall", "wall_left", "low_conf", "kerb", "blind"]
    xs = [torch.from_numpy(bu.synthetic_state(kinds[i % len(kinds)],
                                              rng=np.random.default_rng(i))[None].astype(np.float32))
          for i in range(n)]
    return [torch.cat(xs[i:i + bs]) for i in range(0, len(xs) - bs + 1, bs)]


def build_tasks() -> list[PruneTask]:
    from .export_onnx import _build_pidnet, _build_world_model

    tasks = [
        PruneTask(
            name="pidnet_s_terrain",
            build=_build_pidnet,
            triples=pidnet_triples,
            input_shape=(1, 3, SEG_INPUT[1], SEG_INPUT[0]),
            batch_builder=_seg_batches,
            agreement=_seg_agreement,
            distill_loss=_kl_logits,
            ckpt=CKPT_DIR / "pidnet_s_prism7.pt",
            provenance="distilled student trained in this repo",
        ),
        PruneTask(
            name="world_model",
            build=_build_world_model,
            triples=world_model_triples,
            input_shape=(1, 8, CFG.bev.H, CFG.bev.W),
            batch_builder=_wm_batches,
            agreement=_wm_agreement,
            distill_loss=_wm_loss,
            ckpt=CKPT_DIR / "world_model.pt",
            provenance="trained in this repo on BEV states from this footage",
        ),
    ]

    # The traversability/uncertainty pair only becomes prunable once its checkpoint
    # exists; the shared MobileNet trunk is residual throughout, so only the two thin
    # heads are structurally prunable - which is itself the finding for that model.
    tu = CKPT_DIR / "trav_unc_shared.pt"
    if tu.exists():
        from .export_onnx import _build_trav_unc, trav_input

        def _tu_batches(n: int, bs: int = 2) -> list[torch.Tensor]:
            per_clip = max(1, int(round(n / len(CLIP_IDS))))
            xs = [torch.from_numpy(trav_input(bgr, clip_id=c, idx=i))
                  for c, i, bgr in sampled_frames(per_clip)]
            return [torch.cat(xs[j:j + bs]) for j in range(0, len(xs) - bs + 1, bs)]

        def _tu_triples(net):
            return [_Triple(net.trav_head, "stem.0", "stem.1", ["cls", "risk"]),
                    _Triple(net.unc_head, "stem.0", "stem.1", ["out"])]

        def _tu_agreement(teacher, student, batches):
            teacher.eval().cpu(); student.eval().cpu()
            cm = np.zeros((4, 4), np.float64)
            risk = []
            with torch.no_grad():
                for x in batches:
                    a, b = teacher(x), student(x)
                    la = a[0].argmax(1).numpy().ravel()
                    lb = b[0].argmax(1).numpy().ravel()
                    cm += np.bincount(la * 4 + lb, minlength=16).reshape(4, 4)
                    risk.append(float((torch.sigmoid(a[1]) - torch.sigmoid(b[1])).abs().mean()))
            return {"mIoU_vs_unpruned": round(_miou(cm), 5),
                    "risk_mae_vs_unpruned": round(float(np.mean(risk)), 6)}

        tasks.append(PruneTask(
            name="trav_unc_shared", build=_build_trav_unc, triples=_tu_triples,
            input_shape=(1, 17, 180, 320), batch_builder=_tu_batches,
            agreement=_tu_agreement,
            distill_loss=lambda s, t: _kl_logits(s[0], t[0]) + F.mse_loss(s[1], t[1]) + F.mse_loss(s[2], t[2]),
            ckpt=tu,
            provenance="trained in this repo on geometric+semantic pseudo-labels",
        ))
    return tasks


# ============================================================ sweep

def finetune(student: nn.Module, teacher: nn.Module, batches: list[torch.Tensor],
             loss_fn, steps: int, device: str, lr: float = 2e-4,
             vram_budget_gb: float = 1.0) -> dict:
    """Short self-distillation pass: match the unpruned model's outputs."""
    dev = torch.device(device)
    student.to(dev).train()
    teacher.to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    losses = []
    t0 = time.perf_counter()
    i = 0
    while i < steps:
        for x in batches:
            if i >= steps:
                break
            x = x.to(dev)
            with torch.no_grad():
                t = teacher(x)
            s = student(x)
            loss = loss_fn(s, t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
            i += 1
    peak = (torch.cuda.max_memory_allocated() / 2**30) if dev.type == "cuda" else 0.0
    student.eval().cpu()
    teacher.eval().cpu()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {"steps": steps, "device": str(dev), "seconds": round(time.perf_counter() - t0, 1),
            "loss_first": round(float(np.mean(losses[:10])), 5) if losses else None,
            "loss_last": round(float(np.mean(losses[-10:])), 5) if losses else None,
            "peak_vram_gb": round(peak, 3),
            "vram_budget_gb": vram_budget_gb,
            "objective": "self-distillation against the unpruned student (no ground truth exists)"}


def sweep_task(task: PruneTask, sparsities, steps: int, n_frames: int,
               device: str, verbose: bool = True) -> dict:
    rec = {"name": task.name, "provenance": task.provenance,
           "checkpoint": str(task.ckpt) if task.ckpt else None,
           "input_shape": list(task.input_shape), "levels": []}
    if not task.available():
        rec["status"] = "skipped"
        rec["reason"] = f"checkpoint not found: {task.ckpt}"
        if verbose:
            print(f"  [SKIP] {task.name}: {rec['reason']}")
        return rec
    rec["status"] = "ok"

    base = task.build().eval()
    x = torch.zeros(*task.input_shape)
    base_params = count_params(base)
    base_flops = count_flops(base, task.input_shape)
    base_lat = cpu_latency_ms(base, x)
    batches = task.batch_builder(n_frames)
    eval_batches = batches[: max(2, len(batches) // 3)]
    rec["baseline"] = {"params": base_params, "mb_fp32": round(base_params * 4 / 1e6, 3),
                       "gflops": round(base_flops / 1e9, 3),
                       "cpu_latency": base_lat,
                       "n_prunable_interiors": len(task.triples(task.build())),
                       "eval_batches": len(eval_batches),
                       "train_batches": len(batches)}
    if verbose:
        print(f"  {task.name}: baseline {base_params/1e6:.2f} M params, "
              f"{base_flops/1e9:.2f} GFLOPs, CPU {base_lat['median_ms']:.1f} ms "
              f"({rec['baseline']['n_prunable_interiors']} prunable interiors)")

    PRUNE_DIR.mkdir(parents=True, exist_ok=True)
    for sp in sparsities:
        lvl = {"sparsity": sp}
        try:
            pruned = copy.deepcopy(base)
            removed = sum(t.prune(sp) for t in task.triples(pruned))
            pruned.eval()
            with torch.no_grad():                       # shape sanity
                pruned(x)
            p = count_params(pruned)
            fl = count_flops(pruned, task.input_shape)
            lvl.update({
                "channels_removed": removed,
                "params": p,
                "params_pct_of_baseline": round(100 * p / base_params, 2),
                "mb_fp32": round(p * 4 / 1e6, 3),
                "gflops": round(fl / 1e9, 3),
                "gflops_pct_of_baseline": round(100 * fl / base_flops, 2),
                "agreement_before_finetune": task.agreement(base, pruned, eval_batches),
            })
            if steps > 0:
                lvl["finetune"] = finetune(pruned, base, batches, task.distill_loss,
                                           steps, device)
                lvl["agreement_after_finetune"] = task.agreement(base, pruned, eval_batches)
            lvl["cpu_latency"] = cpu_latency_ms(pruned, x)
            lvl["cpu_speedup_vs_baseline"] = round(
                base_lat["median_ms"] / max(lvl["cpu_latency"]["median_ms"], 1e-6), 3)
            out = PRUNE_DIR / f"{task.name}.prune{int(sp*100):02d}.pt"
            torch.save({"model": pruned.state_dict(), "sparsity": sp,
                        "note": "channel-pruned; the architecture is rebuilt at the "
                                "smaller widths, so load into a model built by "
                                "prism.deploy.prune, not the stock constructor."}, out)
            lvl["checkpoint"] = str(out)
            lvl["status"] = "ok"
            if verbose:
                a0 = lvl["agreement_before_finetune"]
                a1 = lvl.get("agreement_after_finetune", {})
                key = next(iter(a0))
                print(f"    sparsity {sp:.0%}: {p/1e6:5.2f} M "
                      f"({lvl['params_pct_of_baseline']:5.1f}%)  "
                      f"{lvl['gflops']:.2f} GF ({lvl['gflops_pct_of_baseline']:5.1f}%)  "
                      f"CPU {lvl['cpu_latency']['median_ms']:6.1f} ms "
                      f"(x{lvl['cpu_speedup_vs_baseline']:.2f})  "
                      f"{key} {a0[key]:.4f} -> {a1.get(key, float('nan')):.4f}")
        except Exception as e:
            lvl["status"] = "failed"
            lvl["error"] = f"{type(e).__name__}: {str(e)[:250]}"
            if verbose:
                print(f"    sparsity {sp:.0%}: FAILED {lvl['error']}")
        rec["levels"].append(lvl)
    return rec


def run(only: Optional[list[str]] = None, sparsities=DEFAULT_SPARSITY, steps: int = 250,
        n_frames: int = 120, device: Optional[str] = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tasks = build_tasks()
    if only:
        tasks = [t for t in tasks if any(o in t.name for o in only)]
    print(f"structured channel pruning; sparsity sweep {list(sparsities)}, "
          f"{steps} self-distillation steps on {device}\n")
    recs = [sweep_task(t, sparsities, steps, n_frames, device) for t in tasks]
    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": hardware_info(),
        "method": {
            "type": "structured output-channel pruning of block interiors",
            "criterion": "L1 filter norm scaled by the following norm layer's gain "
                         "(Li et al., ICLR 2017)",
            "not_pruned": "residual adds, concatenations, PagFM similarity products, "
                          "PAPPM grouped convolutions, and the MobileNetV3 trunk - "
                          "their channel counts are coupled across branches",
            "finetune": "self-distillation against the unpruned student; there is no "
                        "ground-truth annotation in this repo, so 'accuracy' here means "
                        "agreement with the unpruned FP32 model",
            "latency": "PyTorch CPU, fixed thread count, median of 30 timed iterations "
                       "after warm-up",
        },
        "models": recs,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {REPORT}")
    return report


def load_report() -> dict:
    return json.loads(REPORT.read_text()) if REPORT.exists() else {"models": []}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--sparsity", nargs="*", type=float, default=list(DEFAULT_SPARSITY))
    ap.add_argument("--steps", type=int, default=250, help="self-distillation steps (0 = skip)")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    hw = hardware_info()
    print(f"host: {hw['cpu']} ({hw['cpu_physical_cores']}P/{hw['cpu_logical_processors']}T), "
          f"GPU {hw['gpu']}\n")
    run(a.only, a.sparsity, a.steps, a.frames, a.device)
