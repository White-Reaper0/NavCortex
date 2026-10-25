"""Run the depth stage over every clip and write the `depth` cache.

Cache keys (per clip, stacked on axis 0 over 300 frames):
    q        (N,H,W) float16  relative inverse depth from the network
    depth    (N,H,W) float16  metric-aligned optical-axis depth, metres (NaN = invalid)
    valid    (N,H,W) uint8    1 where depth may be used
    scale    (N,)   float32   a in 1/D = a*q + b
    shift    (N,)   float32   b (0.0 under the scale-only model)
    residual (N,)   float32   RMS of the ground-plane fit, 1/m
    inliers  (N,)   int32
    normal   (N,3)  float32   ground normal in the camera frame
    ms       (N,)   float32   per-frame latency

If a `seg` cache already exists, its labels are fed to the metric fit so the ground
reference uses only trail/grass pixels, which tightens the alignment. Re-run this script
after segmentation lands to get the better fit.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from prism.config import CFG, CLIP_IDS, PROC_H, PROC_W
from prism.types import FramePacket
from prism.io_utils import read_frames, save_stage, has_stage, load_stage
from prism.models.depth import DepthStage
from prism.perception.geometry import fit_metric_ground


def run_clip(stage: DepthStage, clip_id: str, seg_labels=None) -> dict:
    stage.reset()
    q_l, d_l, v_l, sc, sh, rs, il, nl, ms = [], [], [], [], [], [], [], [], []
    prev_fit = None
    for i, frame in read_frames(clip_id):
        sl = seg_labels[i] if seg_labels is not None and i < len(seg_labels) else None
        p = stage(FramePacket(clip_id=clip_id, idx=i, t=i / 30.0, rgb=frame), seg_label=sl)
        d = p.depth
        q_l.append(d.rel_inv.astype(np.float16))
        d_l.append(d.depth_m.astype(np.float16))
        v_l.append(d.valid.astype(np.uint8))
        sc.append(d.scale); sh.append(d.shift)
        rs.append(d.align_residual); il.append(d.align_inliers)
        nl.append(stage._prev_fit.normal if stage._prev_fit is not None else np.array([0, 1, 0], np.float32))
        ms.append(p.timings_ms["depth"])
    return dict(q=np.stack(q_l), depth=np.stack(d_l), valid=np.stack(v_l),
                scale=np.array(sc, np.float32), shift=np.array(sh, np.float32),
                residual=np.array(rs, np.float32), inliers=np.array(il, np.int32),
                normal=np.stack(nl).astype(np.float32), ms=np.array(ms, np.float32))


def main() -> None:
    CFG.ensure_dirs()
    stage = DepthStage()
    print(f"Depth Anything V2-Small  {stage.n_params/1e6:.2f} M params  device={stage.device} fp16={stage.fp16}")
    for cid in CLIP_IDS:
        seg = None
        if has_stage(cid, "seg"):
            try:
                seg = load_stage(cid, "seg")["label"]
                print(f"  {cid}: using cached segmentation to constrain the ground reference")
            except Exception as e:
                print(f"  {cid}: seg cache unreadable ({e}); falling back to the image-band prior")
        t0 = time.perf_counter()
        out = run_clip(stage, cid, seg)
        p = save_stage(cid, "depth", **out)
        d, v = out["depth"].astype(np.float32), out["valid"].astype(bool)
        med = float(np.nanmedian(d[v])) if v.any() else float("nan")
        print(f"{cid}: {out['q'].shape[0]} frames in {time.perf_counter()-t0:5.1f}s "
              f"({out['ms'].mean():5.1f} ms/frame) | a={out['scale'].mean():.2f} "
              f"resid={out['residual'].mean():.3f} inl={out['inliers'].mean():.0f} "
              f"valid={v.mean()*100:.0f}% med_depth={med:.2f} m | {p.stat().st_size/1e6:.0f} MB")


if __name__ == "__main__":
    main()
