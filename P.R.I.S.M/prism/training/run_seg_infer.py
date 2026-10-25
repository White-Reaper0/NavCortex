"""Run SegStage over all five clips once and cache the result.

Downstream stages (traversability, uncertainty, BEV mapping, the dashboard) read the
terrain semantics from `work/cache/<clip_id>/seg.npz` instead of re-running the network:

    python -m prism.training.run_seg_infer

Arrays written per clip, stacked along axis 0 over the 300 frames at PROC_H x PROC_W:

    label      (300, 360, 640) uint8    prism-7 class, 255 = ignore (ego mask)
    prob_max   (300, 360, 640) float16  max class probability
    entropy    (300, 360, 640) float16  Shannon entropy / log(7), 0..1
    shares     (300, 7)        float32  per-frame class pixel share (scene pixels only)
    drivable   (300,)          float32  trail+grass share of scene pixels
    ms         (300,)          float32  per-frame stage latency, milliseconds
    meta       (1,)            object   json string: model, params, latency, taxonomy

`label` is uint8 and the two confidence maps are float16, which keeps a clip at ~15 MB
compressed. Full 7-channel probabilities are deliberately *not* cached (they would be
~1.9 GB per clip); a consumer that needs them can re-run the stage with
`keep_logits=True`.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from ..config import CFG, CLIP_FPS, CLIP_IDS, N_TERRAIN, PROC_H, PROC_W, TERRAIN_CLASSES
from .. import io_utils
from ..models.segmentation import SegStage
from ..types import FramePacket


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=CLIP_IDS)
    ap.add_argument("--device", default=None)
    ap.add_argument("--use-teacher", action="store_true",
                    help="ignore the distilled student and cache the teacher instead")
    ap.add_argument("--no-smooth", dest="smooth", action="store_false", default=True)
    ap.add_argument("--overwrite", action="store_true", default=True)
    args = ap.parse_args()

    CFG.ensure_dirs()
    dev = args.device or io_utils.device()
    stage = SegStage(device=dev, use_teacher=args.use_teacher, smooth=args.smooth)
    print(f"[seg-infer] backend={stage.backend} model={stage.model_name} "
          f"params={stage.n_params:,} device={stage.device} fp16={stage.fp16} "
          f"smooth={stage.smooth}(ema={stage.ema})")
    if stage.agree_miou is not None:
        print(f"[seg-infer] checkpoint teacher-agreement mIoU "
              f"{stage.agree_miou*100:.2f}% (agreement with the SegFormer teacher, "
              f"NOT ground-truth mIoU)")

    grand = np.zeros(N_TERRAIN + 1, np.int64)
    for cid in args.clips:
        if not args.overwrite and io_utils.has_stage(cid, "seg"):
            print(f"[seg-infer] {cid}: cached, skipping")
            continue
        stage.reset()
        labels, pmax, ents, shares, driv, ms = [], [], [], [], [], []
        t0 = time.perf_counter()
        n = 0
        for i, frame in io_utils.read_frames(cid):
            pk = stage(FramePacket(clip_id=cid, idx=i, t=i / CLIP_FPS, rgb=frame))
            r = pk.seg
            ms.append(pk.timings_ms.get("seg", 0.0))
            labels.append(r.label)
            pmax.append(r.prob_max.astype(np.float16))
            ents.append(r.entropy.astype(np.float16))
            sh = SegStage.class_shares(r.label)
            shares.append(sh)
            driv.append(SegStage.drivable_fraction(r.label))
            for c in range(N_TERRAIN):
                grand[c] += int((r.label == c).sum())
            grand[-1] += int((r.label == 255).sum())
            n += 1
        dt = time.perf_counter() - t0

        meta = dict(model=stage.model_name, backend=stage.backend,
                    params=int(stage.n_params), device=stage.device, fp16=stage.fp16,
                    smooth=stage.smooth, ema=stage.ema,
                    ms_per_frame=1000.0 * dt / max(n, 1),
                    classes=TERRAIN_CLASSES, ignore_index=255,
                    proc_size=[PROC_W, PROC_H],
                    metric_note=("teacher-agreement mIoU vs SegFormer-B0/ADE20K "
                                 "pseudo-labels; not ground truth"),
                    agree_miou=stage.agree_miou)
        p = io_utils.save_stage(
            cid, "seg",
            label=np.stack(labels).astype(np.uint8),
            prob_max=np.stack(pmax),
            entropy=np.stack(ents),
            shares=np.stack(shares).astype(np.float32),
            drivable=np.asarray(driv, np.float32),
            ms=np.asarray(ms, np.float32),
            meta=np.array([json.dumps(meta)]),
        )
        mb = p.stat().st_size / 2 ** 20
        print(f"[seg-infer] {cid}: {n} frames  {1000*dt/max(n,1):.1f} ms/frame  "
              f"drivable {100*np.mean(driv):.1f}%  -> {p.name} ({mb:.1f} MB)")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tot = max(int(grand.sum()), 1)
    print("\n[seg-infer] cached class balance over all clips:")
    for c in range(N_TERRAIN):
        print(f"   {c} {TERRAIN_CLASSES[c]:<10s} {100*grand[c]/tot:6.2f}%")
    print(f"   - {'ignore':<10s} {100*grand[-1]/tot:6.2f}%")
    if torch.cuda.is_available():
        print(f"[seg-infer] peak VRAM {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
    stage.close()
    print("[seg-infer] downstream stages: io_utils.load_stage(clip_id, 'seg')"
          " -> label / prob_max / entropy / shares / drivable / meta")


if __name__ == "__main__":
    main()
