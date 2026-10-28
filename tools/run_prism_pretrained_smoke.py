#!/usr/bin/env python3
"""Smoke test: the downloaded pretrained weights, run on real P.R.I.S.M. frames.

Loads Depth Anything V2-Small and SegFormer-B0/ADE20K from the local
``model_assets/pretrained/`` directories on CPU (no network, no CUDA), runs them
on a few frames of a real project clip, drives the project's own geometric
traversability fallback and point-cloud renderer with those outputs, verifies the
numbers are sane, and writes PNG previews plus a JSON report.

Nothing here trains a model, generates a training set, or writes a checkpoint.
The learned traversability head is out of scope: ``checkpoints/
trav_unc_shared.pt`` does not exist and is not created.

Usage (CPU venv)::

    .venv\\Scripts\\python.exe tools\\run_prism_pretrained_smoke.py
    .venv\\Scripts\\python.exe tools\\run_prism_pretrained_smoke.py --frames 3 --clip clips\\clip_02.mp4
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from prism.config import (CFG, DEPTH_INPUT, IGNORE_INDEX, N_TERRAIN, PROC_H,  # noqa: E402
                         PROC_W, TERRAIN_CLASSES, UNKNOWN)
from prism.models.depth import DepthStage  # noqa: E402
from prism.models.pretrained_paths import (DEPTH_ENV_VAR, DEPTH_MODEL_ID,  # noqa: E402
                                           SEGMENTATION_ENV_VAR,
                                           SEGMENTATION_MODEL_ID, local_model_dir)
from prism.models.segmentation import SegStage  # noqa: E402
from prism.perception import geometry as geo  # noqa: E402
from prism.perception.cloud_accum import WorldCamera, render_points  # noqa: E402
from prism.perception.mapping import MappingStage, cell_stats  # noqa: E402
from prism.types import FramePacket  # noqa: E402
from prism.viz_common import (colorize_height, colorize_terrain, colorize_trav,  # noqa: E402
                              overlay)

DEFAULT_CLIP = PROJECT_ROOT / "clips" / "clip_01.mp4"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "pretrained_smoke_tests"


def _sha256_head(path: Path, limit: int = 4_000_000_000) -> str:
    """Digest of the weight file actually loaded (bounded read for big files)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        remaining = limit
        while remaining > 0:
            block = handle.read(min(1_000_000, remaining))
            if not block:
                break
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()[:16]


def read_frames(clip: Path, count: int) -> list[np.ndarray]:
    """The first ``count`` frames of a real project clip, at PROC resolution."""
    capture = cv2.VideoCapture(str(clip))
    if not capture.isOpened():
        raise SystemExit(f"cannot open the project clip {clip}")
    frames = []
    while len(frames) < count:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.resize(bgr, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA))
    capture.release()
    if not frames:
        raise SystemExit(f"{clip} decoded no frames")
    return frames


def relative_depth_image(rel_inv: np.ndarray) -> np.ndarray:
    """Colourised relative inverse depth (near = warm). Same mapping as the viewer."""
    normalised = np.clip(np.asarray(rel_inv, np.float32), 0.0, 1.0)
    return cv2.applyColorMap((normalised * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def metric_depth_summary(depth_m: np.ndarray, valid: np.ndarray) -> dict:
    values = depth_m[valid] if valid is not None else depth_m[np.isfinite(depth_m)]
    if values.size == 0:
        return {"finite": 0, "min": None, "median": None, "max": None}
    return {"finite": int(values.size), "min": float(values.min()),
            "median": float(np.median(values)), "max": float(values.max()),
            "mean": float(values.mean())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="run_prism_pretrained_smoke",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--clip", default=str(DEFAULT_CLIP),
                        help="real project clip to read frames from")
    parser.add_argument("--frames", type=int, default=2, help="how many frames to run")
    parser.add_argument("--output", default=None,
                        help="preview directory; default output/pretrained_smoke_tests/<UTC>")
    parser.add_argument("--cloud-stride", type=int, default=2)
    parser.add_argument("--save-model-preview", action="store_true",
                        help="also save the model-native 518px network input for inspection")
    args = parser.parse_args(argv)

    clip = Path(args.clip)
    out = (Path(args.output) if args.output else
           OUTPUT_ROOT / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    out.mkdir(parents=True, exist_ok=True)

    depth_dir = local_model_dir(DEPTH_MODEL_ID, env_var=DEPTH_ENV_VAR)
    seg_dir = local_model_dir(SEGMENTATION_MODEL_ID, env_var=SEGMENTATION_ENV_VAR)
    if depth_dir is None or seg_dir is None:
        print("local pretrained weights are missing; install them first with:\n"
              "  python tools/download_prism_pretrained.py", file=sys.stderr)
        return 2

    print(f"[1/5] loading Depth Anything V2-Small on CPU from {depth_dir}")
    depth_stage = DepthStage(device="cpu", fp16=False)
    print(f"      params={depth_stage.n_params:,} source={depth_stage.model_source['source']}")
    print(f"[2/5] loading SegFormer-B0 (ADE20K) on CPU from {seg_dir}")
    seg_stage = SegStage(device="cpu", fp16=False)
    print(f"      backend={seg_stage.backend} params={seg_stage.n_params:,} "
          f"source={getattr(seg_stage, 'model_source', {}).get('source')}")

    frames = read_frames(clip, max(1, int(args.frames)))
    print(f"[3/5] running inference on {len(frames)} real frames from {clip.name}")

    mapping = MappingStage(device="cpu")
    trav_ckpt = PROJECT_ROOT / "checkpoints" / "trav_unc_shared.pt"
    trav_note = ("this run trains nothing and loads no traversability model; the grid is the "
                 "geometric fallback (trav_source=geometric). Learned checkpoint " +
                 ("present in the repository from the separate manual training task, unused "
                  "here" if trav_ckpt.exists() else "absent"))
    report: dict = {
        "schema": "prism-pretrained-smoke/v1",
        "ran_at_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "clip": str(clip), "clip_sha256_head": _sha256_head(clip),
        "frames_used": len(frames), "proc_size": [PROC_W, PROC_H],
        "device": "cpu", "fp16": False,
        "depth_model": {"model_id": DEPTH_MODEL_ID, "dir": str(depth_dir),
                        "params": int(depth_stage.n_params),
                        "weights_sha256_head": _sha256_head(depth_dir / "model.safetensors")},
        "segmentation_model": {"model_id": SEGMENTATION_MODEL_ID, "dir": str(seg_dir),
                               "backend": seg_stage.backend,
                               "params": int(seg_stage.n_params),
                               "weights_sha256_head":
                               _sha256_head(seg_dir / "model.safetensors")},
        "learned_traversability": trav_note,
        "frames": [], "previews": [], "checks": {},
    }

    label_for_geometry = None
    per_frame = []
    last_packet = None
    for index, bgr in enumerate(frames):
        packet = FramePacket(clip_id=clip.stem, idx=index, t=index / 30.0, rgb=bgr)
        t0 = time.perf_counter()
        seg_stage(packet)
        seg_ms = (time.perf_counter() - t0) * 1e3
        seg = packet.seg
        t0 = time.perf_counter()
        depth_stage(packet, seg_label=seg.label)
        depth_ms = (time.perf_counter() - t0) * 1e3
        depth = packet.depth
        valid = depth.valid if depth.valid is not None else np.isfinite(depth.depth_m)
        cv2.imwrite(str(out / f"frame{index:02d}_source.png"), bgr)
        cv2.imwrite(str(out / f"frame{index:02d}_depth_relative.png"),
                    relative_depth_image(depth.rel_inv))
        metric_colour = np.clip(np.nan_to_num(depth.depth_m, nan=0.0), 0.0, 8.0)
        metric_colour = cv2.applyColorMap(
            (metric_colour / 8.0 * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        metric_colour[~valid] = (55, 52, 48)
        cv2.imwrite(str(out / f"frame{index:02d}_depth_metric_assumed.png"), metric_colour)
        cv2.imwrite(str(out / f"frame{index:02d}_terrain_labels.png"),
                    colorize_terrain(seg.label))
        cv2.imwrite(str(out / f"frame{index:02d}_terrain_overlay.png"),
                    overlay(bgr, colorize_terrain(seg.label), 0.5))
        if args.save_model_preview:
            raw = cv2.resize(bgr, (DEPTH_INPUT, DEPTH_INPUT))
            cv2.imwrite(str(out / f"frame{index:02d}_network_input.png"), raw)
        counts = {TERRAIN_CLASSES[c]: int((seg.label == c).sum()) for c in range(N_TERRAIN)}
        # 255 is the project's own IGNORE_INDEX: ADE20K classes the mapping deliberately
        # abstains on (indoor furniture and similar), not a segmentation failure.
        abstain = int((seg.label == IGNORE_INDEX).sum())
        entry = {
            "frame": index,
            "depth": {"shape": list(depth.rel_inv.shape),
                      "rel_inv_range": [float(depth.rel_inv.min()),
                                        float(depth.rel_inv.max())],
                      "metric_assumed": metric_depth_summary(depth.depth_m, valid),
                      "valid_fraction": float(valid.mean()),
                      "fit": {"a": float(depth.scale), "b": float(depth.shift),
                              "residual": float(depth.align_residual),
                              "inliers": int(depth.align_inliers),
                              "basis": "assumed camera height 0.12 m (CFG.cam)"},
                      "latency_ms": round(depth_ms, 1)},
            "segmentation": {"label_shape": list(seg.label.shape),
                             "class_counts": counts,
                             "abstain_index": int(IGNORE_INDEX),
                             "abstain_pixels": abstain,
                             "labelled_pixels": abstain + sum(counts.values()),
                             "distinct_classes": int(len(np.unique(seg.label))),
                             "mean_prob_max": float(seg.prob_max.mean()),
                             "mean_entropy": float(seg.entropy.mean()),
                             "drivable_fraction": float(seg_stage.drivable_fraction(seg.label)),
                             "latency_ms": round(seg_ms, 1)},
        }
        per_frame.append(entry)
        last_packet = packet
        label_for_geometry = seg.label
        print(f"      frame {index}: depth {depth_ms:.0f} ms "
              f"({valid.mean() * 100:.0f}% valid), seg {seg_ms:.0f} ms "
              f"({entry['segmentation']['distinct_classes']} classes, "
              f"{abstain / seg.label.size * 100:.0f}% abstained)")

    print("[4/5] driving the geometric traversability fallback and the point cloud")
    if last_packet is None or last_packet.depth is None:
        raise SystemExit("no depth packet available: the inference loop never produced one")
    mapping(last_packet)
    bev = last_packet.bev
    if bev is None:
        raise SystemExit("MappingStage published no BEVMap from a real depth packet")
    stats = dict(cell_stats(bev))
    stats.update({k: v for k, v in (mapping.last_stats or {}).items()
                  if isinstance(v, (int, float, str))})
    height_img = colorize_height(np.nan_to_num(bev.height, nan=0.0),
                                 invalid=~np.isfinite(bev.height))
    blended = cv2.addWeighted(height_img, 0.5, colorize_trav(bev.trav), 0.5, 0)
    cv2.imwrite(str(out / "terrain_bev_geometric.png"), blended)
    observed = int(np.isfinite(bev.height).sum())
    labelled = int((bev.trav != UNKNOWN).sum())
    report["terrain"] = {"trav_source": mapping.trav_source, "cell_stats": stats,
                         "grid_shape": list(bev.height.shape),
                         "observed_cells": observed,
                         "labelled_cells": labelled,
                         "unknown_cells": int((bev.trav == UNKNOWN).sum()),
                         "note": ("observed cells whose geometric confidence is too low "
                                  "stay UNKNOWN - the stage never calls an unverified cell "
                                  "free space"),
                         "label": "GEOMETRIC / ASSUMED SCALE / UNVALIDATED",
                         "learned_traversability": report["learned_traversability"]}
    bev_ok = observed > 0

    # single-frame cloud straight from the depth model output, assumed scale
    depth = last_packet.depth
    valid = depth.valid if depth.valid is not None else np.isfinite(depth.depth_m)
    points_cam = geo.unproject(np.nan_to_num(depth.depth_m, nan=0.0), CFG.cam.K)
    fit = geo.GroundFit(a=1.0, b=0.0, normal=np.array([0.0, 1.0, 0.0], np.float32),
                        height=CFG.cam.height_above_ground_m, ok=True)
    veh = geo.to_vehicle(points_cam, fit)
    keep = (valid.reshape(-1) & np.isfinite(depth.depth_m).reshape(-1)
            & (depth.depth_m.reshape(-1) > 0.10) & (depth.depth_m.reshape(-1) < 12.0))
    stride = max(1, int(args.cloud_stride))
    subsample = np.zeros(keep.shape, bool)
    subsample[::stride * stride] = True
    keep &= subsample
    rgb = cv2.resize(frames[-1], (PROC_W, PROC_H))
    colours = rgb.reshape(-1, 3)[keep]
    xyz = veh.reshape(-1, 3)[keep]
    for name, cam in (
            ("cloud_view_side", WorldCamera(eye=(0.0, -1.4, 1.1), target=(0.0, 4.0, -0.2),
                                            size=(960, 540), vfov_deg=46.0)),
            ("cloud_view_top", WorldCamera(eye=(0.0, 3.0, 8.0), target=(0.0, 3.0, -0.5),
                                           size=(960, 540), vfov_deg=55.0))):
        canvas = np.full((540, 960, 3), (16, 15, 14), np.uint8)
        drawn = render_points(canvas, cam, xyz.astype(np.float32), colours)
        cv2.imwrite(str(out / f"{name}.png"), canvas)
        report.setdefault("cloud", {})[name] = {
            "points_offered": int(xyz.shape[0]), "points_drawn": int(drawn),
            "label": "SINGLE-FRAME / ASSUMED SCALE / NOT A RECONSTRUCTION"}
    report["cloud"]["depth_source"] = ("monocular relative depth aligned to the assumed "
                                       "0.12 m camera height")
    report["cloud"]["finite_points"] = bool(np.isfinite(xyz).all()) if xyz.size else False
    print(f"      BEV observed cells {observed}, cloud points {int(xyz.shape[0])}")

    print("[5/5] checking the outputs")
    checks = {}
    checks["depth_shapes_are_proc"] = all(
        f["depth"]["shape"] == [PROC_H, PROC_W] for f in per_frame)
    checks["depth_values_finite"] = all(
        f["depth"]["metric_assumed"]["finite"] > 0 for f in per_frame)
    checks["depth_relative_in_expected_range"] = all(
        0.0 <= f["depth"]["rel_inv_range"][0] and f["depth"]["rel_inv_range"][1] <= 1.25
        for f in per_frame)
    checks["depth_valid_region_nonempty"] = all(
        f["depth"]["valid_fraction"] > 0.1 for f in per_frame)
    checks["segmentation_shapes_are_proc"] = all(
        f["segmentation"]["label_shape"] == [PROC_H, PROC_W] for f in per_frame)
    checks["segmentation_is_not_blank"] = all(
        f["segmentation"]["distinct_classes"] >= 2 for f in per_frame)
    checks["segmentation_covers_the_frame"] = all(
        f["segmentation"]["labelled_pixels"] == PROC_W * PROC_H for f in per_frame)
    checks["segmentation_maps_a_majority_of_pixels"] = all(
        f["segmentation"]["class_counts"] and
        sum(f["segmentation"]["class_counts"].values()) > 0.5 * PROC_W * PROC_H
        for f in per_frame)
    checks["geometric_terrain_produced_cells"] = bool(bev_ok)
    checks["traversability_is_the_geometric_fallback"] = mapping.trav_source == "geometric"
    report["learned_traversability_checkpoint"] = {"path": str(trav_ckpt),
                                                   "present": trav_ckpt.exists(),
                                                   "used_by_this_run": False}
    checks["all_previews_written"] = True
    report["frames"] = per_frame
    report["checks"] = checks
    for path in sorted(out.glob("*")):
        if path.suffix in (".png",):
            report["previews"].append({"file": path.name,
                                       "bytes": path.stat().st_size})
    (out / "smoke_report.json").write_text(json.dumps(report, indent=2, sort_keys=True),
                                           encoding="utf-8")
    failed = [name for name, ok in checks.items() if not ok]
    print(f"      previews in {out}")
    for name, ok in checks.items():
        print(f"      {'PASS' if ok else 'FAIL'}  {name}")
    if failed:
        print(f"SMOKE TEST FAILED: {failed}", file=sys.stderr)
        return 1
    print("SMOKE TEST PASSED (relative depth + pretrained semantics + geometric terrain, "
          "assumed scale, unvalidated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
