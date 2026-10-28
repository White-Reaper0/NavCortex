"""Assemble cached perception stages into FramePackets and drive the stage renderers.

Perception runs once (the `run_*_infer` scripts write per-clip caches); rendering then
reads those caches, so a visual tweak costs a re-render, not a re-inference.
"""
from __future__ import annotations
import importlib
import time
from typing import Callable, Iterator, Optional, Sequence
import numpy as np
import cv2

from .config import CFG, CLIP_IDS, OUT_DIR, OUTPUT_STAGES, N_TRAV
from .types import (FramePacket, DepthResult, SegResult, TraversabilityResult,
                    UncertaintyResult, OdometryResult, PlaceResult, BEVMap, Pose, Decision,
                    Trajectory, WorldModelPrediction)
from .io_utils import read_frames, load_stage, has_stage, VideoWriter, clips_meta

ALL_STAGES = ("depth", "seg", "trav", "unc", "odom", "vpr", "bev", "wm", "plan")


def _has_keys(cache: dict, *keys: str) -> bool:
    """A cache written by a partial or older run may be missing arrays we need."""
    return all(cache.get(k) is not None for k in keys)


def _get(cache: dict, key: str, i: int, default=None):
    a = cache.get(key)
    if a is None:
        return default
    if a.ndim == 0:
        return a.item()
    return a[i] if i < len(a) else default


class ClipCache:
    """Lazily loads whichever stage caches exist for a clip."""

    def __init__(self, clip_id: str, stages: Sequence[str] = ALL_STAGES, quiet: bool = False):
        self.clip_id = clip_id
        self.data: dict[str, dict] = {}
        self.missing: list[str] = []
        for s in stages:
            if has_stage(clip_id, s):
                try:
                    self.data[s] = load_stage(clip_id, s)
                except Exception as e:
                    self.missing.append(s)
                    if not quiet:
                        print(f"  ! {clip_id}/{s} unreadable: {e}")
            else:
                self.missing.append(s)
        if self.missing and not quiet:
            print(f"  {clip_id}: missing caches -> {', '.join(self.missing)}")

    def has(self, s: str) -> bool:
        return s in self.data

    def __getitem__(self, s: str) -> dict:
        return self.data[s]


def build_packet(clip_id: str, i: int, frame: np.ndarray, c: ClipCache) -> FramePacket:
    p = FramePacket(clip_id=clip_id, idx=i, t=i / 30.0, rgb=frame)

    if c.has("depth") and _has_keys(c["depth"], "depth", "valid", "q"):
        z = c["depth"]
        dep = np.asarray(_get(z, "depth", i), np.float32)
        val = np.asarray(_get(z, "valid", i), bool)
        p.depth = DepthResult(
            rel_inv=np.asarray(_get(z, "q", i), np.float32),
            depth_m=dep, valid=val,
            scale=float(_get(z, "scale", i, 1.0)), shift=float(_get(z, "shift", i, 0.0)),
            align_residual=float(_get(z, "residual", i, 0.0)),
            align_inliers=int(_get(z, "inliers", i, 0)))
        nrm = _get(z, "normal", i)
        if nrm is not None:
            p.depth._normal = np.asarray(nrm, np.float32)
        p.timings_ms["depth"] = float(_get(z, "ms", i, 0.0))

    if c.has("seg") and _has_keys(c["seg"], "label"):
        z = c["seg"]
        lab = np.asarray(_get(z, "label", i), np.uint8)
        pm = _get(z, "prob_max", i)
        en = _get(z, "entropy", i)
        p.seg = SegResult(label=lab,
                          prob_max=np.asarray(pm, np.float32) if pm is not None else np.ones_like(lab, np.float32),
                          entropy=np.asarray(en, np.float32) if en is not None else np.zeros_like(lab, np.float32))
        p.timings_ms["seg"] = float(_get(z, "ms", i, 0.0))

    if c.has("trav") and _has_keys(c["trav"], "prob", "label", "risk"):
        z = c["trav"]
        lab = np.asarray(_get(z, "label", i), np.uint8)
        prob = np.asarray(_get(z, "prob", i), np.float32)
        # trav.npz stores `prob` at half resolution (N,4,180,320) with a `prob_hw` key,
        # while label/risk are full (N,360,640). Upsample here so a consumer can pair
        # prob with label without a silent 2x shape mismatch.
        if prob.ndim == 3 and prob.shape[1:] != lab.shape:
            prob = np.stack([cv2.resize(c, (lab.shape[1], lab.shape[0]),
                                        interpolation=cv2.INTER_LINEAR) for c in prob])
        p.trav = TraversabilityResult(prob=prob, label=lab,
                                      risk=np.asarray(_get(z, "risk", i), np.float32))
        p.timings_ms["trav"] = float(_get(z, "ms", i, 0.0))

    if c.has("unc") and _has_keys(c["unc"], "depth_conf", "seg_conf"):
        z = c["unc"]
        dc = np.asarray(_get(z, "depth_conf", i), np.float32)
        sc = np.asarray(_get(z, "seg_conf", i), np.float32)
        fc = _get(z, "fused_conf", i)
        fc = np.asarray(fc, np.float32) if fc is not None else np.sqrt(np.clip(dc * sc, 0, 1))
        p.unc = UncertaintyResult(depth_conf=dc, seg_conf=sc, fused_conf=fc,
                                  mean_conf=float(np.nanmean(fc)))
        p.timings_ms["unc"] = float(_get(z, "ms", i, 0.0))

    if c.has("odom") and _has_keys(c["odom"], "pose"):
        z = c["odom"]
        pose = np.asarray(_get(z, "pose", i, np.zeros(3, np.float32)), np.float32)
        traj = z.get("trajectory")
        p.odom = OdometryResult(
            pose=Pose(float(pose[0]), float(pose[1]), float(pose[2])),
            d_trans=float(_get(z, "d_trans", i, 0.0)), d_yaw=float(_get(z, "d_yaw", i, 0.0)),
            speed_mps=float(_get(z, "speed", i, 0.0)),
            n_matches=int(_get(z, "n_matches", i, 0)), n_inliers=int(_get(z, "n_inliers", i, 0)),
            tracking_ok=bool(_get(z, "tracking_ok", i, True)),
            track_quality=float(_get(z, "track_quality", i, 1.0)),
            keypoints=_get(z, "keypoints", i), flow=_get(z, "flow", i),
            trajectory=np.asarray(traj[:i + 1], np.float32) if traj is not None else None)
        p.timings_ms["odom"] = float(_get(z, "ms", i, 0.0))

    if c.has("vpr") and _has_keys(c["vpr"], "desc"):
        z = c["vpr"]
        p.place = PlaceResult(descriptor=np.asarray(_get(z, "desc", i, np.zeros(1, np.float32)), np.float32),
                              best_match_idx=int(_get(z, "best_idx", i, -1)),
                              best_score=float(_get(z, "best_score", i, 0.0)),
                              is_revisit=bool(_get(z, "is_revisit", i, False)),
                              db_size=int(_get(z, "db_size", i, 0)))
        p.timings_ms["vpr"] = float(_get(z, "ms", i, 0.0))

    if c.has("bev") and _has_keys(c["bev"], "height", "trav", "conf", "age"):
        z = c["bev"]
        tp = _get(z, "trav_prob", i)
        H, Wd = CFG.bev.H, CFG.bev.W
        p.bev = BEVMap(
            height=np.asarray(_get(z, "height", i), np.float32),
            trav_prob=np.asarray(tp, np.float32) if tp is not None else np.zeros((N_TRAV, H, Wd), np.float32),
            trav=np.asarray(_get(z, "trav", i), np.uint8),
            conf=np.asarray(_get(z, "conf", i), np.float32),
            age=np.asarray(_get(z, "age", i), np.float32),
            hits=np.asarray(_get(z, "hits", i, np.zeros((H, Wd), np.float32)), np.float32),
            terrain=np.asarray(_get(z, "terrain", i, np.zeros((H, Wd), np.uint8)), np.uint8))
        ts = z.get("trav_source")
        # fail safe: an unlabelled cache is assumed geometric, never "learned"
        p.bev.trav_source = str(ts) if ts is not None else "geometric"
        for extra in ("height_step", "inflated", "observed", "local_ground"):
            v = _get(z, extra, i)
            if v is not None:
                setattr(p.bev, extra, np.asarray(v))
        # `observed` is not persisted by run_map_infer; derive it rather than leaving the
        # attribute missing, because consumers getattr it and would silently disagree.
        if not hasattr(p.bev, "observed"):
            p.bev.observed = np.isfinite(p.bev.height)
        p.timings_ms["bev"] = float(_get(z, "ms", i, 0.0))

    if c.has("plan") and _has_keys(c["plan"], "decision_kind"):
        z = c["plan"]
        src = str(_get(z, "policy_source", i, "supervisor"))
        override = bool(_get(z, "override", i, False))
        # the nav stage writes the policy checkpoint path here; normalise it to the
        # provenance string the renderers expect
        if ("\\" in src) or ("/" in src) or src.endswith(".zip"):
            src = "rl+supervisor (override)" if override else "rl"
        elif override and "supervisor" not in src:
            src = f"{src}+supervisor (override)"
        p.decision = Decision(kind=int(_get(z, "decision_kind", i, 0)),
                              action=int(_get(z, "decision_action", i, 0)),
                              speed_mps=float(_get(z, "decision_speed", i, 0.0)),
                              reason=str(_get(z, "decision_reason", i, "")),
                              risk=float(_get(z, "decision_risk", i, 0.0)),
                              confidence=float(_get(z, "decision_conf", i, 1.0)),
                              unknown_frac=float(_get(z, "decision_unknown_frac", i, 0.0)),
                              policy_source=src)
        # which supervisor rule fired; R1/R2 short-circuit before the corridor is
        # scored, so the risk/confidence/unknown fields carry sentinels, not measurements
        p.decision.rule = str(_get(z, "decision_rule", i, ""))
        p.timings_ms["plan"] = float(_get(z, "wm_latency_ms", i, 0.0))

        # candidate trajectories evaluated by the planner this frame
        xy = _get(z, "traj_xy", i)
        if xy is not None:
            act = z.get("cand_action")
            yaw = _get(z, "traj_yaw", i)
            clr = _get(z, "traj_clearance", i)
            best = int(_get(z, "best_traj_idx", i, -1))
            for k in range(len(xy)):
                p.trajectories.append(Trajectory(
                    action=int(act[k]) if act is not None and k < len(act) else 0,
                    xy=np.asarray(xy[k], np.float32),
                    yaw=np.asarray(yaw[k], np.float32) if yaw is not None else np.zeros(len(xy[k]), np.float32),
                    clearance=np.asarray(clr[k], np.float32) if clr is not None else np.zeros(len(xy[k]), np.float32),
                    max_step=float(_get(z, "traj_max_step", i, np.zeros(len(xy)))[k]),
                    unknown_frac=float(_get(z, "traj_unknown_frac", i, np.zeros(len(xy)))[k]),
                    mean_conf=float(_get(z, "traj_mean_conf", i, np.ones(len(xy)))[k]),
                    collision_risk=float(_get(z, "traj_risk", i, np.zeros(len(xy)))[k]),
                    cost=float(_get(z, "traj_cost", i, np.zeros(len(xy)))[k]),
                    feasible=bool(_get(z, "traj_feasible", i, np.ones(len(xy), bool))[k]),
                    reject_reason=str(_get(z, "traj_reject_reason", i, [""] * len(xy))[k])))
            p._best_traj_idx = best

        # one world-model rollout per action
        occ = _get(z, "wm_occ", i)
        if occ is not None:
            trav = _get(z, "wm_trav", i)
            risk = _get(z, "wm_risk", i)
            rtot = _get(z, "wm_risk_total", i)
            lat = _get(z, "wm_latent", i)
            for a in range(len(occ)):
                p.wm_preds.append(WorldModelPrediction(
                    action=a,
                    next_states=np.asarray(lat, np.float32)[None] if lat is not None else np.zeros((1, 1), np.float32),
                    occ_forecast=np.asarray(occ[a], np.float32),
                    trav_forecast=np.asarray(trav[a], np.float32) if trav is not None else np.zeros(len(occ[a]), np.float32),
                    collision_risk=np.asarray(risk[a], np.float32) if risk is not None else np.zeros(len(occ[a]), np.float32),
                    risk_total=float(rtot[a]) if rtot is not None else 0.0))
        p._plan_cache = z
        p._plan_i = i
    return p


def iter_packets(clip_id: str, stages: Sequence[str] = ALL_STAGES,
                 max_frames: Optional[int] = None, quiet: bool = False) -> Iterator[FramePacket]:
    c = ClipCache(clip_id, stages, quiet=quiet)
    for i, frame in read_frames(clip_id, max_frames=max_frames):
        yield build_packet(clip_id, i, frame, c)


# --------------------------------------------------------------------- rendering

RENDERERS = {
    "01_depth_anything_v2": ("prism.render.r_depth", ("depth",)),
    "02_terrain_segmentation": ("prism.render.r_seg", ("depth", "seg")),
    "03_traversability": ("prism.render.r_trav", ("depth", "seg", "trav")),
    "04_visual_odometry": ("prism.render.r_vo", ("depth", "odom")),
    "05_place_recognition": ("prism.render.r_vpr", ("vpr", "odom")),
    "06_uncertainty": ("prism.render.r_unc", ("depth", "seg", "unc")),
    "07_lidar_like_pointcloud": ("prism.render.r_lidar", ("depth", "seg", "trav")),
    "07b_lidar_3d_view": ("prism.render.r_lidar3d", ("depth", "seg", "trav", "odom")),
    "08_bev_25d_map": ("prism.render.r_bev", ("depth", "trav", "unc", "odom", "bev")),
    "09_world_model": ("prism.render.r_world", ("bev", "plan")),
    "10_rl_policy": ("prism.render.r_rl", ("bev", "odom", "plan")),
    "11_final_dashboard": ("prism.render.r_final", ALL_STAGES),
}


def _packet_source(mod, clip_id: str, stages, state: dict, max_frames, quiet: bool):
    """Renderers may supply their own generator when the generic cache -> packet path
    cannot express what they need. `r_vpr` does: its similarity strip requires the
    descriptors of the whole cross-clip database, which live in the other clips' caches.
    """
    fn = getattr(mod, "packets_from_cache", None)
    if callable(fn):
        # some take (clip_id), others (clip_id, state) when they need cross-clip context
        import inspect
        try:
            n_args = len([p for p in inspect.signature(fn).parameters.values()
                          if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
        except (TypeError, ValueError):
            n_args = 1
        gen = fn(clip_id, state) if n_args >= 2 else fn(clip_id)
        if max_frames is None:
            return gen
        return (p for i, p in enumerate(gen) if i < max_frames)
    return iter_packets(clip_id, stages, max_frames=max_frames, quiet=quiet)


def render_stage(stage_dir: str, clip_ids: Sequence[str] = CLIP_IDS,
                 max_frames: Optional[int] = None) -> list:
    mod_name, stages = RENDERERS[stage_dir]
    mod = importlib.import_module(mod_name)
    out = []
    for cid in clip_ids:
        dst = OUT_DIR / stage_dir / f"{cid}.mp4"
        st: dict = {"clip_id": cid, "stage_dir": stage_dir}
        t0 = time.perf_counter()
        n = 0
        writer = None
        try:
            for p in _packet_source(mod, cid, stages, st, max_frames, cid != clip_ids[0]):
                frame = mod.render(p, st)
                if writer is None:
                    # renderers declare their own canvas size; the final dashboard is 1080p
                    writer = VideoWriter(dst, (frame.shape[1], frame.shape[0]))
                writer.write(frame)
                n += 1
        finally:
            if writer is not None:
                writer.close()
        dt = time.perf_counter() - t0
        print(f"  {stage_dir}/{cid}.mp4  {n} frames  {dt:5.1f}s  ({n/max(dt,1e-6):4.1f} fps render)"
              f"  {dst.stat().st_size/1e6:5.1f} MB")
        out.append(dst)
    return out


def preview_stage(stage_dir: str, clip_id: str = "clip_01", frame: int = 120):
    """Render one frame to a PNG for fast iteration."""
    import cv2
    mod_name, stages = RENDERERS[stage_dir]
    mod = importlib.import_module(mod_name)
    st: dict = {"clip_id": clip_id, "stage_dir": stage_dir}
    img = None
    for p in _packet_source(mod, clip_id, stages, st, frame + 1, False):
        img = mod.render(p, st)
    path = CFG and (OUT_DIR.parent / "work" / f"preview_{stage_dir}.png")
    cv2.imwrite(str(path), img)
    print(f"wrote {path}")
    return path
