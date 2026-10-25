"""Run `MappingStage` over every clip and cache the rolling 2.5D BEV map.

    python -m prism.training.run_map_infer                # all 5 clips
    python -m prism.training.run_map_infer --clips clip_02 --preview
    python -m prism.training.run_map_infer --allow-missing # run on whatever exists yet

Consumes the per-clip caches `depth` (required), `seg`, `trav`, `unc` and `odom`, and
writes `work/cache/<clip>/bev.npz` with one stacked array per BEVMap field, float16
where precision allows:

    height     (N,H,W)  f16   metres above the fitted ground plane, NaN = unobserved
    trav       (N,H,W)  u8    argmax class, 3 = UNKNOWN
    trav_prob  (N,4,H,W) f16  per-cell traversability distribution
    conf       (N,H,W)  f16   fused confidence 0..1
    age        (N,H,W)  f16   frames since the cell was last observed
    terrain    (N,H,W)  u8    dominant prism-7 terrain class
    hits       (N,H,W)  f16   accumulated (decayed) observation count
    height_step(N,H,W)  f16   height above the LOCAL ground estimate  [derived, extra]
    inflated   (N,H,W)  u8    obstacles dilated by the vehicle half-width [derived, extra]

The run is idempotent: re-running overwrites cleanly, and `--skip-existing` makes it
resumable. Missing upstream caches are reported by name rather than crashing.

Memory note: `io_utils.load_stage` materialises every key in an npz. The traversability
cache alone is ~550 MB per clip that way, and the GPU box is shared, so this module
opens the npz directly and pulls only the keys it needs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from typing import Optional

import cv2
import numpy as np

from ..config import CFG, CLIP_FPS, CLIP_IDS, N_TRAV, PROC_H, PROC_W, WORK_DIR
from ..io_utils import cache_dir, has_stage, save_stage, read_frames
from ..perception.mapping import MappingStage, cell_stats
from ..types import (BEVMap, DepthResult, FramePacket, OdometryResult, SegResult,
                     TraversabilityResult, UncertaintyResult)

#: `depth` is the only hard requirement - without metric depth there is no geometry.
REQUIRED = ("depth",)
OPTIONAL = ("seg", "trav", "unc", "odom")

#: cache-key aliases, so a small naming difference upstream is not a hard failure
ALIASES = {
    "depth.depth": ("depth", "depth_m", "metric_depth"),
    "depth.valid": ("valid", "depth_valid", "mask"),
    "depth.normal": ("normal", "ground_normal", "plane_normal"),
    "depth.scale": ("scale", "a"),
    "depth.shift": ("shift", "b"),
    "seg.label": ("label", "seg_label", "pred"),
    "trav.prob": ("prob", "trav_prob", "probs"),
    "trav.label": ("label", "trav_label", "pred"),
    "trav.risk": ("risk", "risk_map"),
    "unc.fused": ("fused_conf", "fused", "conf", "confidence"),
    "odom.d_trans": ("d_trans", "dtrans", "delta_trans", "trans"),
    "odom.d_yaw": ("d_yaw", "dyaw", "delta_yaw", "yaw_rate"),
    "odom.ok": ("tracking_ok", "ok", "track_ok"),
    "odom.quality": ("track_quality", "quality"),
    "odom.pose": ("pose", "poses", "trajectory"),
}


class MissingCache(RuntimeError):
    pass


# --------------------------------------------------------------------------- loading

def _npz(clip_id: str, stage: str):
    p = cache_dir(clip_id) / f"{stage}.npz"
    if not p.exists():
        raise MissingCache(f"missing cache stage '{stage}' for {clip_id}: {p}")
    return np.load(p, allow_pickle=True)


def _pick(z, key: str, required: bool = True) -> Optional[np.ndarray]:
    """Fetch one array by its contract name, tolerating a few upstream spellings."""
    names = ALIASES.get(key, (key.split(".")[-1],))
    for n in names:
        if n in z.files:
            return z[n]
    if required:
        raise MissingCache(f"cache is missing any of {names}; it has {sorted(z.files)}")
    return None


def _load_clip(clip_id: str, allow_missing: bool, max_array_mb: float,
               verbose: bool = True) -> tuple[dict, list[str]]:
    """Load exactly the arrays MappingStage consumes. Returns (data, missing_stages)."""
    data: dict = {}
    missing: list[str] = []

    for st in REQUIRED:
        if not has_stage(clip_id, st):
            raise MissingCache(
                f"{clip_id}: required upstream cache '{st}' not found at "
                f"{cache_dir(clip_id) / (st + '.npz')}.\n"
                f"  -> run the {st} stage first (python -m prism.training.run_{st}_infer "
                f"or the module that owns it).")

    with _npz(clip_id, "depth") as z:
        data["depth"] = _pick(z, "depth.depth")
        v = _pick(z, "depth.valid", required=False)
        data["depth_valid"] = None if v is None else v
        data["normal"] = _pick(z, "depth.normal", required=False)
        data["scale"] = _pick(z, "depth.scale", required=False)
        data["shift"] = _pick(z, "depth.shift", required=False)
    n = int(data["depth"].shape[0])
    data["n"] = n

    for st in OPTIONAL:
        if not has_stage(clip_id, st):
            missing.append(st)
            continue
        try:
            zf = _npz(clip_id, st)
        except zipfile.BadZipFile:
            # another agent is writing this cache right now: treat it as not-yet-ready
            # rather than crashing the whole run
            if verbose:
                print(f"    {st}.npz is incomplete (being written?) - treating as missing")
            missing.append(st)
            continue
        with zf as z:
            if st == "seg":
                data["seg_label"] = _pick(z, "seg.label")
            elif st == "trav":
                prob = None
                if any(k in z.files for k in ALIASES["trav.prob"]):
                    # estimate the decompressed size without materialising it: an npz
                    # gives no shape metadata until you actually read the member
                    mb = n * N_TRAV * PROC_H * PROC_W * 2 / 1e6
                    if mb <= max_array_mb:
                        try:
                            prob = _pick(z, "trav.prob")
                        except MemoryError:
                            prob = None
                            if verbose:
                                print("    out of memory reading trav.prob; "
                                      "falling back to trav labels")
                    elif verbose:
                        print(f"    trav.prob is ~{mb:.0f} MB "
                              f"(> --max-array-mb {max_array_mb:.0f}); using labels instead")
                data["trav_prob"] = prob
                data["trav_label"] = _pick(z, "trav.label", required=prob is None)
            elif st == "unc":
                data["unc_conf"] = _pick(z, "unc.fused")
            elif st == "odom":
                data["d_trans"] = _pick(z, "odom.d_trans")
                data["d_yaw"] = _pick(z, "odom.d_yaw")
                data["odom_ok"] = _pick(z, "odom.ok", required=False)
                data["odom_q"] = _pick(z, "odom.quality", required=False)

    if missing and not allow_missing:
        raise MissingCache(
            f"{clip_id}: upstream caches not ready: {', '.join(missing)}.\n"
            f"  Those stages are owned by other modules and have not written "
            f"{', '.join(f'work/cache/{clip_id}/{m}.npz' for m in missing)} yet.\n"
            f"  Re-run once they exist, or pass --allow-missing to build a partial map "
            f"now (the map degrades honestly: no odom = no ego-motion compensation, "
            f"no trav = every cell reported UNKNOWN).")
    return data, missing


# --------------------------------------------------------------------------- packets

def _packet(clip_id: str, i: int, d: dict) -> FramePacket:
    pk = FramePacket(clip_id=clip_id, idx=i, t=i / float(CLIP_FPS))
    depth = np.asarray(d["depth"][i], np.float32)
    valid = (None if d.get("depth_valid") is None
             else np.asarray(d["depth_valid"][i]).astype(bool))
    dr = DepthResult(rel_inv=np.zeros((1, 1), np.float32), depth_m=depth, valid=valid,
                     scale=float(d["scale"][i]) if d.get("scale") is not None else 1.0,
                     shift=float(d["shift"][i]) if d.get("shift") is not None else 0.0)
    if d.get("normal") is not None:
        # the ground normal the depth stage actually fitted: use it verbatim so every
        # stage shares one vehicle frame instead of each re-deriving its own
        dr.ground_normal = np.asarray(d["normal"][i], np.float32)
    pk.depth = dr

    if d.get("seg_label") is not None:
        pk.seg = SegResult(label=np.asarray(d["seg_label"][i], np.uint8))
    if d.get("trav_prob") is not None or d.get("trav_label") is not None:
        prob = (np.asarray(d["trav_prob"][i], np.float32)
                if d.get("trav_prob") is not None else None)
        lab = (np.asarray(d["trav_label"][i], np.uint8)
               if d.get("trav_label") is not None else
               np.argmax(prob, axis=0).astype(np.uint8))
        pk.trav = TraversabilityResult(
            prob=prob if prob is not None else np.zeros((N_TRAV, *lab.shape), np.float32),
            label=lab, risk=np.zeros(lab.shape, np.float32))
        if prob is None:
            pk.trav.prob = None
    if d.get("unc_conf") is not None:
        c = np.asarray(d["unc_conf"][i], np.float32)
        pk.unc = UncertaintyResult(depth_conf=c, seg_conf=c, fused_conf=c,
                                   mean_conf=float(np.nanmean(c)))
    dt = float(d["d_trans"][i]) if d.get("d_trans") is not None else 0.0
    dy = float(d["d_yaw"][i]) if d.get("d_yaw") is not None else 0.0
    ok = bool(d["odom_ok"][i]) if d.get("odom_ok") is not None else True
    pk.odom = OdometryResult(d_trans=dt, d_yaw=dy, tracking_ok=ok,
                             speed_mps=dt * 30.0,
                             track_quality=float(d["odom_q"][i])
                             if d.get("odom_q") is not None else 1.0)
    return pk


# --------------------------------------------------------------------------- per clip

def run_clip(clip_id: str, allow_missing: bool = False, max_frames: Optional[int] = None,
             max_array_mb: float = 600.0, preview: bool = False,
             pixel_stride: int = 1, verbose: bool = True) -> dict:
    t_load = time.perf_counter()
    data, missing = _load_clip(clip_id, allow_missing, max_array_mb, verbose)
    t_load = time.perf_counter() - t_load
    n = data["n"] if max_frames is None else min(data["n"], int(max_frames))
    if verbose:
        note = f"  (missing, degraded: {', '.join(missing)})" if missing else ""
        print(f"  loaded {n} frames of {data['depth'].shape[1:]} depth in {t_load:.1f} s{note}")

    H, W = CFG.bev.H, CFG.bev.W
    out = {
        "height": np.empty((n, H, W), np.float16),
        "trav": np.empty((n, H, W), np.uint8),
        "trav_prob": np.empty((n, N_TRAV, H, W), np.float16),
        "conf": np.empty((n, H, W), np.float16),
        "age": np.empty((n, H, W), np.float16),
        "terrain": np.empty((n, H, W), np.uint8),
        "hits": np.empty((n, H, W), np.float16),
        "height_step": np.empty((n, H, W), np.float16),
        "inflated": np.empty((n, H, W), np.uint8),
    }

    stage = MappingStage(device="cpu", pixel_stride=pixel_stride)
    stage.reset()
    times, stats = [], []
    for i in range(n):
        pk = _packet(clip_id, i, data)
        t0 = time.perf_counter()
        stage(pk)
        times.append((time.perf_counter() - t0) * 1e3)
        b: BEVMap = pk.bev
        out["height"][i] = b.height.astype(np.float16)
        out["trav"][i] = b.trav
        out["trav_prob"][i] = b.trav_prob.astype(np.float16)
        out["conf"][i] = b.conf.astype(np.float16)
        # age is capped for storage: 9999 does not fit float16's useful range anyway
        out["age"][i] = np.minimum(b.age, 1000.0).astype(np.float16)
        out["terrain"][i] = b.terrain
        out["hits"][i] = np.minimum(b.hits, 4000.0).astype(np.float16)
        out["height_step"][i] = getattr(b, "height_step",
                                        np.full((H, W), np.nan, np.float32)).astype(np.float16)
        out["inflated"][i] = getattr(b, "inflated",
                                     np.zeros((H, W), bool)).astype(np.uint8)
        stats.append(cell_stats(b))
        if verbose and (i + 1) % 100 == 0:
            print(f"    {i + 1}/{n}  {np.mean(times[-100:]):.1f} ms/frame")

    # Record whether traversability came from the learned head or the geometric
    # fallback. Without this the renderer cannot tell, and defaulting to "learned"
    # would present a degraded map as a learned one.
    out["trav_source"] = np.array(getattr(stage, "trav_source", "geometric"))
    p = save_stage(clip_id, "bev", **out)
    mb = p.stat().st_size / 1e6

    agg = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
    summary = dict(clip_id=clip_id, frames=n, missing=missing,
                   ms_per_frame=float(np.mean(times)),
                   ms_median=float(np.median(times)), cache_mb=mb, **agg)
    if verbose:
        print(f"  {np.mean(times):5.1f} ms/frame (median {np.median(times):.1f})  "
              f"-> {p.name} {mb:.0f} MB")
        print(f"  cells: safe {agg['frac_safe']:.1%}  risky {agg['frac_risky']:.1%}  "
              f"obstacle {agg['frac_obstacle']:.1%}  unknown {agg['frac_unknown']:.1%}  "
              f"conf {agg['mean_conf']:.2f}")

    if preview:
        _write_preview(clip_id, pk, n)
    return summary


def _write_preview(clip_id: str, pk: FramePacket, n: int) -> None:
    """Render the last frame through both renderers as a visual smoke test."""
    from ..render import r_bev, r_lidar
    from ..perception.lidarize import LidarizeStage
    try:
        for idx, frame in read_frames(clip_id, max_frames=pk.idx + 1):
            if idx == pk.idx:
                pk.rgb = frame
                break
    except Exception:
        pass
    cv2.imwrite(str(WORK_DIR / "preview_bev.png"), r_bev.render(pk, {}))
    LidarizeStage("cpu")(pk)
    cv2.imwrite(str(WORK_DIR / "preview_lidar.png"), r_lidar.render(pk, {}))
    print(f"  wrote {WORK_DIR / 'preview_bev.png'} and "
          f"{WORK_DIR / 'preview_lidar.png'} (frame {pk.idx})")


# --------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clips", nargs="*", default=CLIP_IDS)
    ap.add_argument("--allow-missing", action="store_true",
                    help="run with whatever upstream caches exist (degrades honestly)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave clips that already have a bev cache alone")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--pixel-stride", type=int, default=1,
                    help="subsample depth pixels before scattering (speed knob)")
    ap.add_argument("--max-array-mb", type=float, default=600.0,
                    help="fall back to trav labels if trav.prob exceeds this")
    ap.add_argument("--preview", action="store_true",
                    help="also write work/preview_bev.png and work/preview_lidar.png")
    a = ap.parse_args(argv)

    print("=" * 74)
    print("prism - rolling 2.5D BEV mapping")
    b = CFG.bev
    print(f"grid {b.H}x{b.W} @ {b.res_m} m  ({b.range_forward_m:.2f} m forward, "
          f"+/-{b.range_lateral_m:.2f} m lateral)   decay {b.decay_per_frame}/frame, "
          f"retire at {b.max_age_frames} frames")
    print(f"metric scale from an ASSUMED camera height of "
          f"{CFG.cam.height_above_ground_m:.2f} m - not calibrated")
    print("=" * 74)

    summaries, failed = [], []
    for clip_id in a.clips:
        print(f"\n[{clip_id}]")
        if a.skip_existing and has_stage(clip_id, "bev"):
            print("  bev cache already present, skipping (--skip-existing)")
            continue
        try:
            summaries.append(run_clip(clip_id, allow_missing=a.allow_missing,
                                      max_frames=a.max_frames,
                                      max_array_mb=a.max_array_mb,
                                      preview=a.preview and clip_id == a.clips[0],
                                      pixel_stride=a.pixel_stride))
        except MissingCache as e:
            print(f"  SKIPPED - {e}")
            failed.append(clip_id)
        except MemoryError:
            print(f"  SKIPPED - out of memory loading {clip_id}; retry with "
                  f"--max-array-mb 100 (uses trav labels instead of probabilities)")
            failed.append(clip_id)

    print("\n" + "=" * 74)
    if summaries:
        ms = float(np.mean([s["ms_per_frame"] for s in summaries]))
        tot = int(sum(s["frames"] for s in summaries))
        print(f"wrote bev cache for {len(summaries)}/{len(a.clips)} clips, "
              f"{tot} frames, {ms:.1f} ms/frame mean "
              f"(CPU, numpy+OpenCV; no GPU used by this stage)")
        print(f"{'clip':10s} {'frames':>6s} {'ms/fr':>6s} {'safe':>6s} {'risky':>6s} "
              f"{'obst':>6s} {'unk':>6s} {'conf':>5s} {'MB':>6s}")
        for s in summaries:
            print(f"{s['clip_id']:10s} {s['frames']:6d} {s['ms_per_frame']:6.1f} "
                  f"{s['frac_safe'] * 100:5.1f}% {s['frac_risky'] * 100:5.1f}% "
                  f"{s['frac_obstacle'] * 100:5.1f}% {s['frac_unknown'] * 100:5.1f}% "
                  f"{s['mean_conf']:5.2f} {s['cache_mb']:6.0f}")
        (WORK_DIR / "bev_infer_summary.json").write_text(json.dumps(summaries, indent=2))
        print(f"summary -> {WORK_DIR / 'bev_infer_summary.json'}")
    if failed:
        print(f"\nNOT built: {', '.join(failed)} (see the messages above)")
    print("=" * 74)
    return 0 if summaries else 1


if __name__ == "__main__":
    sys.exit(main())
