r"""Run the traversability + uncertainty stages over all five clips and cache the result.

    python -m prism.training.run_trav_infer

Writes, per clip:
    save_stage(clip_id, "trav", prob=(N,4,H,W) f16, label=(N,H,W) u8, risk=(N,H,W) f16)
    save_stage(clip_id, "unc",  depth_conf=, seg_conf=, fused_conf=  each (N,H,W) f16)
plus ``work/unc_summary.json`` (per-clip mean confidence, used by ``render/r_unc.py`` to
show the daylight vs low-light degradation) and the two preview PNGs.

Both stages share **one** ``SharedPerceptionTrunk`` and one backbone forward per frame:
``UncertaintyStage`` is handed the ``TraversabilityStage`` and reuses its cached
features.  The measured split between the two is printed at the end.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ..config import CFG, CLIP_FPS, CLIP_IDS, PROC_H, PROC_W, WORK_DIR
from ..io_utils import (ego_mask, has_stage, load_stage, read_frames, save_stage,
                        cache_dir, device as pick_device)
from ..models.traversability import TraversabilityStage, geometry_from_depth
from ..models.uncertainty import UncertaintyStage
from ..perception import geometry as G
from ..types import DepthResult, FramePacket, GeometryResult, SegResult
from ..render import r_trav, r_unc

SUMMARY = WORK_DIR / "unc_summary.json"
CLIP_LIGHT = {"clip_01": "daylight", "clip_02": "daylight", "clip_03": "dusk",
              "clip_04": "low light", "clip_05": "low light"}


def fit_from_cache(dz: dict, i: int) -> "G.GroundFit":
    """Rebuild the depth stage's own GroundFit instead of re-solving it.

    The depth cache stores the inverse-depth scale/shift and the ground normal it used,
    so re-fitting here would only risk disagreeing with the depth map we were given.
    Falls back to a fresh solve when those keys are absent.
    """
    if all(k in dz for k in ("scale", "shift", "normal")):
        n = np.asarray(dz["normal"][i], np.float32)
        if np.isfinite(n).all() and np.linalg.norm(n) > 1e-6:
            return G.GroundFit(a=float(dz["scale"][i]), b=float(dz["shift"][i]),
                               normal=n / np.linalg.norm(n),
                               height=CFG.cam.height_above_ground_m,
                               residual=float(dz["residual"][i]) if "residual" in dz else 0.0,
                               inliers=int(dz["inliers"][i]) if "inliers" in dz else 0,
                               ok=True)
    q = np.asarray(dz["q"][i], np.float32)
    v = np.asarray(dz["valid"][i], bool)
    return G.fit_metric_ground(q, v)


def run_clip(clip_id: str, trav: TraversabilityStage, unc: UncertaintyStage,
             save: bool = True, preview_at: int | None = None,
             preview_dir: Path = WORK_DIR, prob_half: bool = False) -> dict:
    if not (has_stage(clip_id, "depth") and has_stage(clip_id, "seg")):
        raise FileNotFoundError(
            f"{clip_id}: need both work/cache/{clip_id}/depth.npz and seg.npz "
            "(produced by the depth and segmentation agents) before this can run.")
    dz = load_stage(clip_id, "depth")
    sz = load_stage(clip_id, "seg")
    n = len(dz["depth"])
    trav.reset()
    unc.reset()
    st_t, st_u = {}, {}

    prob = np.zeros((n, 4, PROC_H, PROC_W), np.float16)
    label = np.zeros((n, PROC_H, PROC_W), np.uint8)
    risk = np.zeros((n, PROC_H, PROC_W), np.float16)
    dconf = np.zeros((n, PROC_H, PROC_W), np.float16)
    sconf = np.zeros((n, PROC_H, PROC_W), np.float16)
    fconf = np.zeros((n, PROC_H, PROC_W), np.float16)

    em = ego_mask(PROC_H, PROC_W)
    t_trav = t_unc = t_geo = 0.0
    t0 = time.perf_counter()
    for i, frame in read_frames(clip_id):
        if i >= n:
            break
        tg = time.perf_counter()
        depth = np.asarray(dz["depth"][i], np.float32)
        valid = np.asarray(dz["valid"][i], bool)
        q = np.asarray(dz["q"][i], np.float32)
        fit = fit_from_cache(dz, i)
        h, s, r, pts = geometry_from_depth(depth, valid, fit)
        t_geo += time.perf_counter() - tg

        pk = FramePacket(clip_id=clip_id, idx=i, t=i / float(CLIP_FPS), rgb=frame)
        pk.depth = DepthResult(rel_inv=q, depth_m=depth, scale=fit.a, shift=fit.b,
                               align_residual=fit.residual, align_inliers=fit.inliers,
                               valid=valid)
        pk.seg = SegResult(label=np.asarray(sz["label"][i], np.uint8),
                           prob_max=np.asarray(sz["prob_max"][i], np.float32),
                           entropy=np.asarray(sz["entropy"][i], np.float32))
        pk.geom = GeometryResult(points_cam=np.empty((0,), np.float32), points_veh=pts,
                                 height_above_ground=h, slope_deg=s, roughness=r,
                                 ground_normal=fit.normal, ground_d=fit.height,
                                 ground_inlier_frac=0.0)
        pk = trav(pk)
        pk = unc(pk)
        t_trav += pk.timings_ms.get("trav", 0.0)
        t_unc += pk.timings_ms.get("unc", 0.0)

        prob[i] = pk.trav.prob.astype(np.float16)
        label[i] = pk.trav.label
        risk[i] = pk.trav.risk.astype(np.float16)
        dconf[i] = pk.unc.depth_conf.astype(np.float16)
        sconf[i] = pk.unc.seg_conf.astype(np.float16)
        fconf[i] = pk.unc.fused_conf.astype(np.float16)

        if preview_at is not None and i <= preview_at:
            a = r_trav.render(pk, st_t)
            b = r_unc.render(pk, st_u)
            if i == preview_at:
                cv2.imwrite(str(preview_dir / "preview_trav.png"), a)
                cv2.imwrite(str(preview_dir / "preview_unc.png"), b)
    wall = time.perf_counter() - t0

    if save:
        if prob_half:
            # 300 x 4 x 360 x 640 float16 is 553 MB raw per clip; halving the spatial
            # resolution of the class posterior (label and risk stay full-res) keeps a
            # clip's cache small.  prob_hw records what consumers must upsample from.
            ph = np.stack([np.stack([cv2.resize(prob[i, c].astype(np.float32),
                                                (PROC_W // 2, PROC_H // 2),
                                                interpolation=cv2.INTER_AREA)
                                     for c in range(4)]) for i in range(n)]).astype(np.float16)
            save_stage(clip_id, "trav", prob=ph, label=label, risk=risk,
                       prob_hw=np.array([PROC_H // 2, PROC_W // 2], np.int32))
        else:
            save_stage(clip_id, "trav", prob=prob, label=label, risk=risk,
                       prob_hw=np.array([PROC_H, PROC_W], np.int32))
        save_stage(clip_id, "unc", depth_conf=dconf, seg_conf=sconf, fused_conf=fconf)

    fc = fconf.astype(np.float32)[:, em]
    dc = dconf.astype(np.float32)[:, em]
    sc = sconf.astype(np.float32)[:, em]
    lab_v = label[:, em]
    stats = dict(
        n_frames=int(n),
        lighting=CLIP_LIGHT.get(clip_id, "?"),
        mean_fused_conf=float(fc.mean()),
        mean_depth_conf=float(dc.mean()),
        mean_seg_conf=float(sc.mean()),
        p10_fused_conf=float(np.percentile(fc, 10)),
        p90_fused_conf=float(np.percentile(fc, 90)),
        frac_below_unknown_thr=float((fc < CFG.safety.conf_unknown).mean()),
        frac_above_slow_thr=float((fc >= CFG.safety.conf_slow).mean()),
        mean_conf_on_safe_near=float(fc[lab_v == 0].mean()) if (lab_v == 0).any() else 0.0,
        trav_fraction={k: float((lab_v == c).mean())
                       for c, k in enumerate(["safe", "risky", "obstacle", "unknown"])},
        mean_risk=float(risk.astype(np.float32)[:, em].mean()),
        ms_per_frame_total=1000.0 * wall / max(n, 1),
        ms_trav=t_trav / max(n, 1),
        ms_unc=t_unc / max(n, 1),
        ms_geometry=1000.0 * t_geo / max(n, 1),
        trav_npz_mb=round((cache_dir(clip_id) / "trav.npz").stat().st_size / 2 ** 20, 1) if save else 0,
        unc_npz_mb=round((cache_dir(clip_id) / "unc.npz").stat().st_size / 2 ** 20, 1) if save else 0,
    )
    print(f"[{clip_id}] {n} frames  {stats['ms_per_frame_total']:.1f} ms/frame  "
          f"(trav {stats['ms_trav']:.1f} + unc {stats['ms_unc']:.1f} + geom "
          f"{stats['ms_geometry']:.1f})  mean conf {stats['mean_fused_conf']:.3f}  "
          f"safe {stats['trav_fraction']['safe']*100:.1f}%  "
          f"npz {stats['trav_npz_mb']}+{stats['unc_npz_mb']} MB", flush=True)
    return stats


def stats_from_cache(clip_id: str) -> dict:
    """Per-clip summary recomputed from an existing trav/unc cache, without inference."""
    tz = load_stage(clip_id, "trav")
    uz = load_stage(clip_id, "unc")
    em = ego_mask(PROC_H, PROC_W)
    fc = np.asarray(uz["fused_conf"], np.float32)[:, em]
    dc = np.asarray(uz["depth_conf"], np.float32)[:, em]
    sc = np.asarray(uz["seg_conf"], np.float32)[:, em]
    lab = np.asarray(tz["label"])[:, em]
    risk = np.asarray(tz["risk"], np.float32)[:, em]
    return dict(
        n_frames=int(len(tz["label"])), lighting=CLIP_LIGHT.get(clip_id, "?"),
        mean_fused_conf=float(fc.mean()), mean_depth_conf=float(dc.mean()),
        mean_seg_conf=float(sc.mean()), p10_fused_conf=float(np.percentile(fc, 10)),
        p90_fused_conf=float(np.percentile(fc, 90)),
        frac_below_unknown_thr=float((fc < CFG.safety.conf_unknown).mean()),
        frac_above_slow_thr=float((fc >= CFG.safety.conf_slow).mean()),
        mean_conf_on_safe_near=float(fc[(lab == 0)].mean()) if (lab == 0).any() else 0.0,
        trav_fraction={k: float((lab == c).mean())
                       for c, k in enumerate(["safe", "risky", "obstacle", "unknown"])},
        mean_risk=float(risk.mean()),
    )


def summarise(clip_ids=None, out=None) -> dict:
    """Rebuild work/unc_summary.json from the caches (no inference)."""
    clip_ids = list(clip_ids or CLIP_IDS)
    out = out or {"clips": {}}
    out.setdefault("clips", {})
    for cid in clip_ids:
        if has_stage(cid, "trav") and has_stage(cid, "unc"):
            out["clips"][cid] = stats_from_cache(cid)
    _add_lighting_check(out)
    SUMMARY.write_text(json.dumps(out, indent=2))
    return out


def _add_lighting_check(out: dict) -> dict:
    """The contract's sanity check: mean confidence must be lower on the low-light clips."""
    day = [c for c in out["clips"] if out["clips"][c]["lighting"] == "daylight"]
    low = [c for c in out["clips"] if out["clips"][c]["lighting"] == "low light"]
    if day and low:
        md = float(np.mean([out["clips"][c]["mean_fused_conf"] for c in day]))
        ml = float(np.mean([out["clips"][c]["mean_fused_conf"] for c in low]))
        out["daylight_vs_lowlight"] = {
            "daylight_clips": day, "low_light_clips": low,
            "mean_fused_conf_daylight": md, "mean_fused_conf_low_light": ml,
            "delta": md - ml, "relative_drop_pct": 100.0 * (md - ml) / max(md, 1e-6),
            "sanity_check_passes": bool(ml < md),
        }
        print(f"\nSANITY CHECK  daylight {md:.4f}  vs  low light {ml:.4f}  "
              f"-> {'PASS' if ml < md else 'FAIL'} "
              f"({100.0*(md-ml)/max(md,1e-6):+.1f}% relative)")
    return out


def main(clip_ids=None, device: str | None = None, save: bool = True,
         preview_clip: str = "clip_01", preview_frame: int = 120,
         prob_half: bool = False) -> dict:
    clip_ids = list(clip_ids or CLIP_IDS)
    dev = device or pick_device()
    trav = TraversabilityStage(device=dev)
    unc = UncertaintyStage(device=dev, trav_stage=trav)
    print(f"device={dev}  checkpoint={trav.ckpt_path}  "
          f"trunk shared={trav.net.trunk is unc.net.trunk}")

    out = {"device": dev, "checkpoint": str(trav.ckpt_path), "clips": {}}
    if SUMMARY.exists():                 # keep entries for clips we are not re-running
        try:
            out["clips"] = json.loads(SUMMARY.read_text()).get("clips", {})
        except Exception:                # noqa: BLE001
            pass
    for cid in clip_ids:
        pv = preview_frame if cid == preview_clip else None
        out["clips"][cid] = run_clip(cid, trav, unc, save=save, preview_at=pv,
                                     prob_half=prob_half)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    day = [c for c in out["clips"] if out["clips"][c]["lighting"] == "daylight"]
    low = [c for c in out["clips"] if out["clips"][c]["lighting"] == "low light"]
    if day and low:
        md = float(np.mean([out["clips"][c]["mean_fused_conf"] for c in day]))
        ml = float(np.mean([out["clips"][c]["mean_fused_conf"] for c in low]))
        out["daylight_vs_lowlight"] = {
            "daylight_clips": day, "low_light_clips": low,
            "mean_fused_conf_daylight": md, "mean_fused_conf_low_light": ml,
            "delta": md - ml, "relative_drop_pct": 100.0 * (md - ml) / max(md, 1e-6),
            "sanity_check_passes": bool(ml < md),
        }
        print(f"\nSANITY CHECK  daylight {md:.4f}  vs  low light {ml:.4f}  "
              f"-> {'PASS' if ml < md else 'FAIL'} "
              f"({100.0*(md-ml)/max(md,1e-6):+.1f}% relative)")
    if torch.cuda.is_available():
        out["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    SUMMARY.write_text(json.dumps(out, indent=2))
    print(f"wrote {SUMMARY}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--preview-clip", default="clip_01")
    ap.add_argument("--preview-frame", type=int, default=120)
    ap.add_argument("--summary-only", action="store_true",
                    help="rebuild work/unc_summary.json from existing caches, no inference")
    ap.add_argument("--prob-half", action="store_true",
                    help="store the 4-class posterior at half resolution to keep the "
                         "per-clip npz small (label and risk stay full resolution)")
    a = ap.parse_args()
    if a.summary_only:
        summarise(a.clips)
        raise SystemExit(0)
    main(a.clips, a.device, save=not a.no_save,
         preview_clip=a.preview_clip, preview_frame=a.preview_frame,
         prob_half=a.prob_half)
