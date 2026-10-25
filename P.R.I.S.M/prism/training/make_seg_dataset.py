"""Build the terrain-distillation dataset: frames + SegFormer soft targets.

Sources
-------
* all five 10 s clips (`clips/clip_01..05.mp4`), every `--clip-stride` frame;
* extra frames seeked out of the full source video `video/input.mp4` for domain
  variety - a daylight range and a low-light range - so the student sees more than the
  50 seconds the clips cover.  cv2 seeking is used, the file is never decoded in full.

Outputs (under `work/dataset/seg/`)
-----------------------------------
    frames/<name>.jpg      512x288 BGR, the student's input resolution
    targets/<name>.npz     soft  : (7, 72, 128) float16  teacher probabilities
                           label : (144, 256)  uint8     argmax, 255 = ignore
    manifest.json          per-sample source + train/val split + class histogram

The soft targets are stored at stride 4 rather than the student's stride 8: the extra
resolution costs ~190 MB total and lets the training-time scale-jitter crop resample
cleanly down to 64x36 without smearing the boundaries the D branch has to learn.

Ignored pixels (255) come from two places: ADE classes the mapping abstains on
(see `models/seg_teacher.py`) and `io_utils.ego_mask()` - the RC chassis and the channel
watermark, which must never be learned as scene content.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ..config import (CFG, CLIP_IDS, DATA_DIR, IGNORE_INDEX, N_TERRAIN, SEG_INPUT,
                      TERRAIN_CLASSES, VIDEO_IN)
from .. import io_utils
from ..models.seg_teacher import SegTeacher

SEG_DIR = DATA_DIR / "seg"
FRAME_DIR = SEG_DIR / "frames"
TARGET_DIR = SEG_DIR / "targets"

SOFT_DIV = 4          # soft targets stored at input/4
LABEL_DIV = 2         # hard labels stored at input/2

# extra sampling windows in the source video, seconds
EXTRA_RANGES = [("day", 60.0, 340.0), ("lowlight", 420.0, 900.0)]


def _sizes():
    w, h = SEG_INPUT
    return (w, h), (w // SOFT_DIV, h // SOFT_DIV), (w // LABEL_DIV, h // LABEL_DIV)


def _write_sample(name: str, bgr: np.ndarray, teacher: SegTeacher) -> np.ndarray:
    """Run the teacher on one already-resized frame and persist it. Returns the label."""
    (iw, ih), (sw, sh), (lw, lh) = _sizes()
    prob, label = teacher.probs(bgr, out_size=(lw, lh))          # teacher at label res

    # ego mask -> ignore (chassis + watermark carry no scene semantics)
    keep = io_utils.ego_mask(lh, lw)
    label = label.copy()
    label[~keep] = IGNORE_INDEX

    soft = np.stack([cv2.resize(prob[c], (sw, sh), interpolation=cv2.INTER_AREA)
                     for c in range(N_TERRAIN)])
    soft /= np.clip(soft.sum(0, keepdims=True), 1e-6, None)

    cv2.imwrite(str(FRAME_DIR / f"{name}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 94])
    np.savez_compressed(TARGET_DIR / f"{name}.npz",
                        soft=soft.astype(np.float16), label=label.astype(np.uint8))
    return label


def _iter_clip_frames(clip_stride: int):
    (iw, ih), _, _ = _sizes()
    for cid in CLIP_IDS:
        for i, f in io_utils.read_frames(cid, resize=(iw, ih)):
            if i % clip_stride == 0:
                yield f"{cid}_f{i:04d}", cid, f


def _iter_extra_frames(n_per_range: int):
    """Seek `n_per_range` evenly spaced frames out of each source-video range."""
    (iw, ih), _, _ = _sizes()
    if not VIDEO_IN.exists():
        print(f"[dataset] {VIDEO_IN} missing - skipping extra frames")
        return
    cap = cv2.VideoCapture(str(VIDEO_IN))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = n_total / fps
    for tag, t0, t1 in EXTRA_RANGES:
        t1 = min(t1, dur - 1.0)
        if t1 <= t0:
            continue
        times = np.linspace(t0, t1, n_per_range)
        for t in times:
            fi = int(round(t * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, f = cap.read()
            if not ok:
                continue
            if (f.shape[1], f.shape[0]) != (iw, ih):
                f = cv2.resize(f, (iw, ih), interpolation=cv2.INTER_AREA)
            yield f"src_{tag}_{fi:07d}", f"source:{tag}", f
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-stride", type=int, default=2,
                    help="take every Nth frame of every clip (2 -> 150/clip -> 750)")
    ap.add_argument("--extra-per-range", type=int, default=300,
                    help="frames seeked from each source-video range")
    ap.add_argument("--val-every", type=int, default=8,
                    help="every Nth sample is held out for teacher-agreement scoring")
    ap.add_argument("--flip-tta", action="store_true", default=True)
    ap.add_argument("--no-flip-tta", dest="flip_tta", action="store_false")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    CFG.ensure_dirs()
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    dev = args.device or io_utils.device()
    teacher = SegTeacher(device=dev, flip_tta=args.flip_tta)
    print(f"[dataset] teacher on {dev} fp16={teacher.fp16} flip_tta={args.flip_tta} "
          f"input_mode={teacher.input_mode}")
    (iw, ih), (sw, sh), (lw, lh) = _sizes()
    print(f"[dataset] frame {iw}x{ih}  soft {N_TERRAIN}x{sh}x{sw} fp16  label {lh}x{lw} u8")

    counts = np.zeros(N_TERRAIN + 1, np.int64)
    manifest = []
    t0 = time.time()

    gen = list(_iter_clip_frames(args.clip_stride))
    n_clip = len(gen)
    print(f"[dataset] {n_clip} clip frames; seeking extras from {VIDEO_IN.name} ...")
    gen += list(_iter_extra_frames(args.extra_per_range))
    print(f"[dataset] {len(gen)} frames total ({len(gen)-n_clip} extra), running teacher ...")

    for k, (name, src, frame) in enumerate(gen):
        label = _write_sample(name, frame, teacher)
        for c in range(N_TERRAIN):
            counts[c] += int((label == c).sum())
        counts[-1] += int((label == IGNORE_INDEX).sum())
        manifest.append(dict(name=name, source=src,
                             split="val" if (k % args.val_every == 0) else "train"))
        if (k + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {k+1}/{len(gen)}  {el:.0f}s  ({1000*el/(k+1):.0f} ms/frame)")

    teacher.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tot = int(counts.sum())
    hist = {TERRAIN_CLASSES[c]: float(counts[c] / tot) for c in range(N_TERRAIN)}
    hist["ignore"] = float(counts[-1] / tot)
    n_val = sum(1 for m in manifest if m["split"] == "val")
    meta = dict(n_samples=len(manifest), n_train=len(manifest) - n_val, n_val=n_val,
                input_size=[iw, ih], soft_size=[sw, sh], label_size=[lw, lh],
                teacher=dict(model=teacher.__class__.__name__,
                             checkpoint="nvidia/segformer-b0-finetuned-ade-512-512",
                             input_mode="short512", flip_tta=bool(args.flip_tta)),
                class_balance=hist, samples=manifest)
    (SEG_DIR / "manifest.json").write_text(json.dumps(meta, indent=1))

    print(f"\n[dataset] wrote {len(manifest)} samples to {SEG_DIR}")
    print(f"[dataset] split: {meta['n_train']} train / {n_val} val "
          f"(every {args.val_every}th sample held out)")
    print(f"[dataset] elapsed {time.time()-t0:.0f}s")
    print("[dataset] pixel class balance (teacher pseudo-labels, ego-masked):")
    for c in range(N_TERRAIN):
        print(f"   {c} {TERRAIN_CLASSES[c]:<10s} {100*counts[c]/tot:6.2f}%")
    print(f"   - {'ignore':<10s} {100*counts[-1]/tot:6.2f}%")
    sz = sum(p.stat().st_size for p in SEG_DIR.rglob("*") if p.is_file())
    print(f"[dataset] on disk {sz/2**20:.0f} MiB")


if __name__ == "__main__":
    main()
