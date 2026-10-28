"""Run the localisation stages (VO + VPR) over all five clips and write the caches.

    python -m prism.training.run_odom_infer                 # both stages, all clips
    python -m prism.training.run_odom_infer --only odom
    python -m prism.training.run_odom_infer --clips clip_01 clip_04 --preview

Writes, per clip:

  work/cache/<clip>/odom.npz
      pose (N,3) x,y,yaw | d_trans (N,) | d_yaw (N,) | speed (N,) |
      n_inliers (N,) | n_matches (N,) | tracking_ok (N,) | track_quality (N,) |
      trajectory (N,2) | plus diagnostics: scale, scale_source, track_state,
      parallax_deg, sampson_px, ms
  work/cache/<clip>/vpr.npz
      desc (N,256) float16 | best_idx (N,) | best_score (N,) | is_revisit (N,) |
      plus: instant_score, distinctiveness, loop_gap, seq_frames, gap_ok,
      best_clip (index into CLIP_IDS, -1 = none), best_frame, db_size,
      db_clip / db_frame (the database layout, so a renderer can resolve
      best_idx to an actual frame), ms

The script is idempotent and re-runnable. **If the depth cache does not exist yet it
still runs**, with a constant-baseline fallback for the monocular scale and a loud
warning; re-run it once `work/cache/<clip>/depth.npz` exists to get depth-anchored
metric scale.
"""
from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import cv2

from ..config import CFG, CLIP_IDS, CLIP_FPS, WORK_DIR
from ..types import FramePacket
from .. import io_utils
from ..perception.odometry import (OdometryStage, SCALE_SOURCE_NAMES, TRACK_STATE_NAMES,
                                   SCALE_CONST)
from ..models.vpr import VPRStage, CKPT_PATH


MAX_KP = 1500       # == the detector's cap, so keypoints are never subsampled
MAX_FLOW = 400      # inlier flow vectors kept per frame (display only)


def _sub(a, cap: int) -> np.ndarray:
    """Stride-subsample to at most `cap` rows (stride keeps the grid spread)."""
    a = np.zeros((0, 2), np.float32) if a is None else np.asarray(a, np.float32)
    if len(a) > cap:
        a = a[:: int(np.ceil(len(a) / cap))][:cap]
    return a


def _pad(rows, cap: int, dim: int):
    out = np.zeros((len(rows), cap, dim), np.float32)
    cnt = np.zeros(len(rows), np.int16)
    for i, r in enumerate(rows):
        k = min(len(r), cap)
        if k:
            out[i, :k] = r[:k]
        cnt[i] = k
    return out, cnt


def _banner(msg: str, ch: str = "=") -> None:
    print(f"\n{ch * 78}\n{msg}\n{ch * 78}", flush=True)


# --------------------------------------------------------------------- odometry

def run_odometry(clip_ids, preview: bool = False, verbose: bool = True) -> dict:
    stage = OdometryStage(verbose=False)
    missing = [c for c in clip_ids if not io_utils.has_stage(c, "depth")]
    if missing:
        print("!" * 78)
        print("!! WARNING: no depth cache for " + ", ".join(missing))
        print("!! Monocular VO has NO metric scale of its own. Running these clips with a")
        print(f"!! CONSTANT baseline of {stage.p.fallback_step_m:.3f} m/frame "
              f"(~{stage.p.fallback_step_m * CLIP_FPS:.2f} m/s), which is an ASSUMPTION,")
        print("!! not a measurement. Every distance/speed for those clips is a placeholder.")
        print("!! Re-run this script after the depth stage has written its cache.")
        print("!" * 78, flush=True)

    summary = {}
    from ..render import r_vo
    for cid in clip_ids:
        stage.reset()
        rows = {k: [] for k in ("pose", "d_trans", "d_yaw", "speed", "n_inliers",
                                "n_matches", "tracking_ok", "track_quality", "scale",
                                "scale_source", "track_state", "parallax_deg",
                                "sampson_px", "ms")}
        traj = None
        rstate: dict = {}
        prev_img = None
        kps_l, flow_l = [], []
        t0 = time.perf_counter()
        for i, frame in io_utils.read_frames(cid):
            pk = FramePacket(clip_id=cid, idx=i, t=i / CLIP_FPS, rgb=frame)
            pk = stage(pk)
            o = pk.odom
            rows["pose"].append(o.pose.as_array())
            rows["d_trans"].append(o.d_trans)
            rows["d_yaw"].append(o.d_yaw)
            rows["speed"].append(o.speed_mps)
            rows["n_inliers"].append(o.n_inliers)
            rows["n_matches"].append(o.n_matches)
            rows["tracking_ok"].append(o.tracking_ok)
            rows["track_quality"].append(o.track_quality)
            rows["scale"].append(getattr(o, "scale_m", 0.0))
            rows["scale_source"].append(getattr(o, "scale_source", 3))
            rows["track_state"].append(getattr(o, "state", 2))
            rows["parallax_deg"].append(getattr(o, "parallax_deg", 0.0))
            rows["sampson_px"].append(getattr(o, "sampson_px", 0.0))
            rows["ms"].append(pk.timings_ms.get("odom", 0.0))
            kps_l.append(_sub(o.keypoints, MAX_KP))
            flow_l.append(_sub(o.flow, MAX_FLOW))
            traj = o.trajectory
            if preview:
                prev_img = r_vo.render(pk, rstate)
        el = time.perf_counter() - t0

        arr = {
            "pose": np.asarray(rows["pose"], np.float32),
            "d_trans": np.asarray(rows["d_trans"], np.float32),
            "d_yaw": np.asarray(rows["d_yaw"], np.float32),
            "speed": np.asarray(rows["speed"], np.float32),
            "n_inliers": np.asarray(rows["n_inliers"], np.int32),
            "n_matches": np.asarray(rows["n_matches"], np.int32),
            "tracking_ok": np.asarray(rows["tracking_ok"], bool),
            "track_quality": np.asarray(rows["track_quality"], np.float32),
            "trajectory": np.asarray(traj, np.float32),
            "scale": np.asarray(rows["scale"], np.float32),
            "scale_source": np.asarray(rows["scale_source"], np.int8),
            "track_state": np.asarray(rows["track_state"], np.int8),
            "parallax_deg": np.asarray(rows["parallax_deg"], np.float32),
            "sampson_px": np.asarray(rows["sampson_px"], np.float32),
            "ms": np.asarray(rows["ms"], np.float32),
        }
        # keypoints / inlier flow, subsampled and zero-padded so the renderer can
        # redraw the camera panel straight from the cache (no re-running the stage)
        arr["keypoints"], arr["n_keypoints"] = _pad(kps_l, MAX_KP, 2)
        arr["flow"], arr["n_flow"] = _pad(flow_l, MAX_FLOW, 4)
        io_utils.save_stage(cid, "odom", **arr)
        if preview and prev_img is not None:
            WORK_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(WORK_DIR / "preview_vo.png"), prev_img)

        st = arr["track_state"]
        srcs = arr["scale_source"]
        n = len(st)
        summary[cid] = dict(
            frames=n,
            ms_per_frame=el * 1000 / n,
            matches=float(arr["n_matches"][1:].mean()),
            inliers=float(arr["n_inliers"][1:].mean()),
            inlier_ratio=float((arr["n_inliers"][1:] /
                                np.maximum(arr["n_matches"][1:], 1)).mean()),
            lost_pct=100.0 * float((st == 2).mean()),
            coast_pct=100.0 * float((st == 1).mean()),
            quality=float(arr["track_quality"][1:].mean()),
            speed=float(arr["speed"][1:].mean()),
            path_m=float(np.linalg.norm(np.diff(arr["trajectory"], axis=0), axis=1).sum()),
            depth_anchored_pct=100.0 * float((srcs == 0).mean()),
            const_fallback=bool((srcs == SCALE_CONST).any()),
        )
        if verbose:
            s = summary[cid]
            print(f"{cid}: {n} frames  {s['ms_per_frame']:6.1f} ms/f | "
                  f"match {s['matches']:6.1f} inl {s['inliers']:6.1f} "
                  f"({s['inlier_ratio']:.2f}) | LOST {s['lost_pct']:4.1f}% "
                  f"COAST {s['coast_pct']:4.1f}% | q {s['quality']:.2f} | "
                  f"{s['speed']:.2f} m/s  path {s['path_m']:5.2f} m | "
                  f"depth-anchored {s['depth_anchored_pct']:5.1f}%", flush=True)
    return summary


# --------------------------------------------------------------------- vpr

def run_vpr(clip_ids, cross_clip: bool = True, preview: bool = False,
            verbose: bool = True) -> dict:
    if not CKPT_PATH.exists():
        print("!" * 78)
        print(f"!! WARNING: {CKPT_PATH} missing - the VPR head is UNTRAINED (random).")
        print("!! Train it first:  python -m prism.models.vpr --train")
        print("!" * 78, flush=True)
    stage = VPRStage(cross_clip=cross_clip)
    from ..render import r_vpr

    summary = {}
    for ci, cid in enumerate(clip_ids):
        # cross-clip: the database keeps everything seen so far, so clip_02 onwards
        # queries against real earlier places instead of only its own history.
        stage.reset(keep_db=cross_clip)
        clip_start = stage.db_size
        rows = {k: [] for k in ("desc", "best_idx", "best_score", "is_revisit",
                                "instant_score", "distinctiveness", "loop_gap",
                                "best_clip", "best_frame", "seq_frames", "gap_ok",
                                "ms")}
        rstate: dict = {}
        prev_img = None
        t0 = time.perf_counter()
        for i, frame in io_utils.read_frames(cid):
            pk = FramePacket(clip_id=cid, idx=i, t=i / CLIP_FPS, rgb=frame)
            pk = stage(pk)
            pl = pk.place
            rows["desc"].append(pl.descriptor)
            rows["best_idx"].append(pl.best_match_idx)
            rows["best_score"].append(pl.best_score)
            rows["is_revisit"].append(pl.is_revisit)
            rows["instant_score"].append(getattr(pl, "instant_score", 0.0))
            rows["distinctiveness"].append(getattr(pl, "distinctiveness", 0.0))
            rows["loop_gap"].append(pl.loop_gap_frames)
            bc = getattr(pl, "best_clip_id", None)
            rows["best_clip"].append(CLIP_IDS.index(bc) if bc in CLIP_IDS else -1)
            rows["best_frame"].append(int(getattr(pl, "best_frame_idx", -1)))
            rows["seq_frames"].append(int(getattr(pl, "seq_consistent_frames", 0)))
            rows["gap_ok"].append(bool(getattr(pl, "gap_ok", True)))
            rows["ms"].append(pk.timings_ms.get("vpr", 0.0))
            if preview:
                rstate["_vpr_clip_bounds"] = _bounds(stage)
                prev_img = r_vpr.render(pk, rstate)
        el = time.perf_counter() - t0

        db_clip = np.asarray([CLIP_IDS.index(c) if c in CLIP_IDS else -1
                              for c in stage._db_clip], np.int16)
        arr = {
            "desc": np.asarray(rows["desc"], np.float16),
            "best_idx": np.asarray(rows["best_idx"], np.int32),
            "best_score": np.asarray(rows["best_score"], np.float32),
            "is_revisit": np.asarray(rows["is_revisit"], bool),
            "instant_score": np.asarray(rows["instant_score"], np.float32),
            "distinctiveness": np.asarray(rows["distinctiveness"], np.float32),
            "loop_gap": np.asarray(rows["loop_gap"], np.int32),
            "best_clip": np.asarray(rows["best_clip"], np.int16),
            "best_frame": np.asarray(rows["best_frame"], np.int32),
            "seq_frames": np.asarray(rows["seq_frames"], np.int16),
            "gap_ok": np.asarray(rows["gap_ok"], bool),
            "db_size": np.int32(stage.db_size),
            "db_clip": db_clip,
            "db_frame": np.asarray(stage._db_frame, np.int32),
            "clip_db_start": np.int32(clip_start),
            # the decision thresholds actually used, so a renderer reading this
            # cache explains the decision with the same numbers the stage used
            "params": np.array([stage.p.exclude_frames, stage.p.seq_len,
                                stage.p.top_k, stage.p.sim_thresh,
                                stage.p.ratio_thresh, stage.p.min_seq_frames,
                                stage.p.min_db, stage.p.min_gap_frames],
                               np.float32),
            "ms": np.asarray(rows["ms"], np.float32),
        }
        io_utils.save_stage(cid, "vpr", **arr)
        if preview and prev_img is not None:
            WORK_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(WORK_DIR / "preview_vpr.png"), prev_img)

        sc = arr["best_score"]
        valid = arr["best_idx"] >= 0
        xclip = int(((arr["best_clip"] >= 0) &
                     (arr["best_clip"] != CLIP_IDS.index(cid))).sum())
        summary[cid] = dict(
            frames=len(sc), ms_per_frame=el * 1000 / len(sc),
            db_size=int(stage.db_size),
            revisits=int(arr["is_revisit"].sum()),
            top_score=float(sc[valid].max()) if valid.any() else 0.0,
            med_score=float(np.median(sc[valid])) if valid.any() else 0.0,
            cross_clip_wins=xclip,
        )
        if verbose:
            s = summary[cid]
            print(f"{cid}: {s['frames']} frames  {s['ms_per_frame']:5.1f} ms/f | "
                  f"db {s['db_size']:4d} | revisits {s['revisits']:3d} | "
                  f"top score {s['top_score']:.3f} med {s['med_score']:.3f} | "
                  f"cross-clip winners {s['cross_clip_wins']:3d}", flush=True)
    return summary


def _bounds(stage) -> list:
    out, cur, start = [], None, 0
    for i, c in enumerate(stage._db_clip):
        if c != cur:
            if cur is not None:
                out.append((cur, start, i))
            cur, start = c, i
    if cur is not None:
        out.append((cur, start, len(stage._db_clip)))
    return out


# --------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="*", default=CLIP_IDS)
    ap.add_argument("--only", choices=["odom", "vpr", "both"], default="both")
    ap.add_argument("--no-cross-clip", action="store_true",
                    help="VPR database is cleared between clips (in-clip only)")
    ap.add_argument("--preview", action="store_true",
                    help="also write work/preview_vo.png and work/preview_vpr.png")
    a = ap.parse_args(argv)
    CFG.ensure_dirs()
    clips = [c for c in a.clips if io_utils.clip_path(c).exists()]
    if not clips:
        print("no clips found", file=sys.stderr)
        return 2

    t_all = time.perf_counter()
    if a.only in ("odom", "both"):
        _banner("VISUAL ODOMETRY - ORB-SLAM3-style monocular front end (Python/OpenCV)")
        run_odometry(clips, preview=a.preview)
    if a.only in ("vpr", "both"):
        _banner("PLACE RECOGNITION - GeM + learned whitening on MobileNetV3-Small")
        run_vpr(clips, cross_clip=not a.no_cross_clip, preview=a.preview)
    print(f"\ntotal {time.perf_counter() - t_all:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
