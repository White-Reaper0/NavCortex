"""SegStage - prism-7 terrain semantics for the perception pipeline.

Runs the distilled PIDNet-S student at `CFG.SEG_INPUT` (512x288) in FP16 on GPU, then
upsamples the 1/8-stride logits to PROC_W x PROC_H and fills `packet.seg` with a
`SegResult` (label / prob_max / normalised entropy).

Two operating modes:
  * ``use_teacher=False`` (default) - the 7.7 M-parameter distilled student.
  * ``use_teacher=True``            - SegFormer-B0/ADE20K through `SegTeacher`, the same
    model the student was distilled from. This is the fallback when
    `checkpoints/pidnet_s_drishti7.pt` is missing, so the pipeline never breaks; it is
    ~3x slower and is reported as a different model name in the renderer.

Temporal smoothing: the input is video, so an EMA over the class probability maps
(`ema=0.6` by default, `reset()` clears it) removes per-frame flicker in the overlay.
It is a display/stability aid, not a tracker - no motion compensation is applied, so it
is deliberately light and can be switched off with ``smooth=False``.

Ego mask: `io_utils.ego_mask()` pixels (RC chassis + channel watermark) get label 255,
prob_max 0 and entropy 1 - they carry no scene semantics and must never be treated as
terrain.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..config import (CFG, CKPT_DIR, IGNORE_INDEX, N_TERRAIN, PROC_H, PROC_W, SEG_INPUT,
                      TERRAIN_CLASSES, TERRAIN_DRIVE_PRIOR)
from .. import io_utils
from ..types import FramePacket, SegResult
from .seg_pidnet import PIDNetS, count_params

CKPT_PATH = CKPT_DIR / "pidnet_s_drishti7.pt"
_LOG_N = float(np.log(N_TERRAIN))


class SegStage:
    """Terrain semantic segmentation stage (see module docstring)."""

    def __init__(self, device: str = "cuda", ckpt: Optional[Path] = None,
                 use_teacher: bool = False, fp16: bool = True, smooth: bool = True,
                 ema: float = 0.6, keep_logits: bool = False):
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.fp16 = bool(fp16 and self.device == "cuda")
        self.smooth = bool(smooth)
        self.ema = float(ema)
        self.keep_logits = bool(keep_logits)
        self.ckpt_path = Path(ckpt) if ckpt is not None else CKPT_PATH

        self._prev: Optional[torch.Tensor] = None
        self._keep_t: Optional[torch.Tensor] = None
        self.teacher = None
        self.model = None
        self.n_params = 0
        self.last_ms = 0.0
        self.agree_miou: Optional[float] = None

        want_teacher = use_teacher or not self.ckpt_path.exists()
        if want_teacher:
            if not use_teacher:
                print(f"[seg] {self.ckpt_path} missing -> falling back to the SegFormer "
                      f"teacher (slower, same taxonomy)")
            from .seg_teacher import SegTeacher
            self.teacher = SegTeacher(device=self.device, fp16=self.fp16,
                                      out_size=(PROC_W, PROC_H))
            self.backend = "teacher"
            self.model_name = "SegFormer-B0 (ADE20K) -> prism-7"
            self.n_params = sum(p.numel() for p in self.teacher.model.parameters())
        else:
            ck = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            cfg = ck.get("model_cfg", {})
            self.model = PIDNetS(num_classes=cfg.get("num_classes", N_TERRAIN),
                                 augment=True)
            self.model.load_state_dict(ck["state_dict"])
            self.model.eval().to(self.device)
            if self.fp16:
                self.model.half()
            norm = ck.get("normalize", {})
            self._mean = np.array(norm.get("mean", [0.485, 0.456, 0.406]), np.float32)
            self._std = np.array(norm.get("std", [0.229, 0.224, 0.225]), np.float32)
            self.backend = "student"
            self.model_name = "PIDNet-S (distilled)"
            self.n_params = count_params(self.model)
            self.agree_miou = ck.get("val_agree_miou")
            self.ckpt_meta = {k: ck.get(k) for k in
                              ("arch", "epoch", "val_agree_miou", "val_pixel_acc",
                               "metric_name", "val_per_class_agree_iou")}

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Clear the temporal EMA. Mandatory between clips."""
        self._prev = None

    def close(self) -> None:
        if self.teacher is not None:
            self.teacher.close()
        self.model = None
        self.teacher = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def _student_prob(self, bgr: np.ndarray) -> torch.Tensor:
        w, h = SEG_INPUT
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if (rgb.shape[1], rgb.shape[0]) != (w, h):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        x = (rgb.astype(np.float32) / 255.0 - self._mean) / self._std
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0)
        t = t.half() if self.fp16 else t.float()
        logits = self.model(t.to(self.device, non_blocking=True))
        logits = F.interpolate(logits.float(), size=(PROC_H, PROC_W),
                               mode="bilinear", align_corners=False)
        return logits.softmax(1)[0]

    def _probs_t(self, bgr: np.ndarray) -> torch.Tensor:
        """(7, PROC_H, PROC_W) float32 on `self.device`, temporally smoothed."""
        if self.backend == "teacher":
            p, _ = self.teacher.probs(bgr, out_size=(PROC_W, PROC_H))
            p = torch.from_numpy(p).to(self.device)
        else:
            p = self._student_prob(bgr)
        if self.smooth:
            self._prev = p if self._prev is None else \
                self._prev.mul_(self.ema).add_(p, alpha=1.0 - self.ema)
            p = self._prev
        return p

    def probs(self, bgr: np.ndarray) -> np.ndarray:
        """(7, PROC_H, PROC_W) float32 class probabilities, temporally smoothed."""
        return self._probs_t(bgr).cpu().numpy()

    def _keep_mask_t(self) -> torch.Tensor:
        if self._keep_t is None:
            self._keep_t = torch.from_numpy(
                io_utils.ego_mask(PROC_H, PROC_W).copy()).to(self.device)
        return self._keep_t

    @torch.no_grad()
    def result(self, bgr: np.ndarray) -> SegResult:
        t0 = time.perf_counter()
        p = self._probs_t(bgr)
        # argmax / max / entropy stay on the device: only three small maps cross the bus
        pmax_t, label_t = p.max(0)
        ent_t = (-(p * p.clamp_min(1e-8).log()).sum(0) / _LOG_N).clamp_(0.0, 1.0)
        label_t = label_t.to(torch.uint8)

        keep = self._keep_mask_t()
        label_t = torch.where(keep, label_t, torch.full_like(label_t, IGNORE_INDEX))
        pmax_t = torch.where(keep, pmax_t, torch.zeros_like(pmax_t))
        ent_t = torch.where(keep, ent_t, torch.ones_like(ent_t))

        label = label_t.cpu().numpy()
        prob_max = pmax_t.cpu().numpy().astype(np.float32)
        ent = ent_t.cpu().numpy().astype(np.float32)
        logits = p.cpu().numpy().astype(np.float16) if self.keep_logits else None

        if self.device == "cuda":
            torch.cuda.synchronize()
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return SegResult(logits=logits, label=label, prob_max=prob_max, entropy=ent)

    def __call__(self, packet: FramePacket) -> FramePacket:
        if packet.rgb is None:
            raise ValueError("SegStage needs packet.rgb")
        packet.seg = self.result(packet.rgb)
        packet.timings_ms["seg"] = self.last_ms
        return packet

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def drivable_fraction(label: np.ndarray) -> float:
        """Fraction of *scene* pixels whose class carries a non-trivial drive prior."""
        m = label != IGNORE_INDEX
        n = int(m.sum())
        if n == 0:
            return 0.0
        prior = TERRAIN_DRIVE_PRIOR[np.clip(label, 0, N_TERRAIN - 1)]
        return float((prior[m] >= 0.5).sum() / n)

    @staticmethod
    def class_shares(label: np.ndarray) -> np.ndarray:
        """(7,) fraction of scene pixels per class (ignores the ego mask)."""
        m = label != IGNORE_INDEX
        n = max(int(m.sum()), 1)
        return np.array([(label[m] == c).sum() / n for c in range(N_TERRAIN)], np.float32)


# ------------------------------------------------------------------------------ self-test
if __name__ == "__main__":
    from ..config import WORK_DIR
    from ..viz_common import colorize_terrain, overlay

    dev = io_utils.device()
    stage = SegStage(device=dev)
    print(f"[seg] backend={stage.backend}  model={stage.model_name}  "
          f"params={stage.n_params:,}  fp16={stage.fp16}  device={stage.device}")
    if stage.agree_miou is not None:
        print(f"[seg] checkpoint teacher-agreement mIoU {stage.agree_miou*100:.2f}% "
              f"(agreement with the SegFormer teacher, not ground truth)")

    frames = [f for _, f in io_utils.read_frames("clip_01", max_frames=40)]
    stage.reset()
    r = stage.result(frames[0])
    print(f"[seg] label {r.label.shape} {r.label.dtype} uniq={np.unique(r.label)}")
    print(f"[seg] prob_max [{r.prob_max.min():.3f}, {r.prob_max.max():.3f}]  "
          f"entropy [{r.entropy.min():.3f}, {r.entropy.max():.3f}]")
    print("[seg] class shares: " + ", ".join(
        f"{n}={v*100:.1f}%" for n, v in zip(TERRAIN_CLASSES, stage.class_shares(r.label))))
    print(f"[seg] drivable fraction {stage.drivable_fraction(r.label)*100:.1f}%")

    for f in frames[:10]:
        stage.result(f)
    t0 = time.perf_counter()
    for f in frames:
        stage.result(f)
    gpu_ms = (time.perf_counter() - t0) / len(frames) * 1000
    print(f"[seg] {stage.device} latency {gpu_ms:.1f} ms/frame (end-to-end, "
          f"incl. pre/post-processing at {PROC_W}x{PROC_H})")
    if stage.device == "cuda":
        print(f"[seg] peak VRAM {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")

    rows = []
    for cid in ["clip_01", "clip_03", "clip_05"]:
        stage.reset()
        fs = [f for _, f in io_utils.read_frames(cid, max_frames=160)]
        for f in fs[-8:]:
            rr = stage.result(f)
        rows.append(np.hstack([fs[-1], overlay(fs[-1], colorize_terrain(rr.label), 0.6)]))
    cv2.imwrite(str(WORK_DIR / "_seg_stage_check.png"), np.vstack(rows))
    print(f"[seg] wrote {WORK_DIR / '_seg_stage_check.png'}")
    stage.close()

    # CPU reference latency for the deployment story
    if dev == "cuda":
        torch.cuda.empty_cache()
        cpu_stage = SegStage(device="cpu", fp16=False, smooth=False)
        for f in frames[:3]:
            cpu_stage.result(f)
        t0 = time.perf_counter()
        for f in frames[:10]:
            cpu_stage.result(f)
        print(f"[seg] cpu latency {(time.perf_counter()-t0)/10*1000:.0f} ms/frame "
              f"(fp32, PyTorch eager, no ONNX/INT8)")
        cpu_stage.close()
