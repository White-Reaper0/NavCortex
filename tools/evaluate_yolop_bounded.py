"""Bounded CPU evaluation of the optional YOLOP provider (offline, no accuracy claim).

Runs the YOLOP ONNX detection branch and the existing LibreYOLO baseline on the
same recorded frames, and records latency, sampled process memory, detection
counts/scores, and YOLOP segmentation-output statistics.

The clip has no reviewed object annotations, so detection *accuracy* is not
measurable here and is reported as ``unmeasured``.  Segmentation statistics
describe BDD100K road-scene heads only; they are not terrain labels and not
traversability.  Nothing is written to a map, ledger, or planner.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

try:
    import psutil
except Exception:  # pragma: no cover - platform dependent
    psutil = None

from prism.belief_ledger import EvidenceRelation
from prism.integrations.yolop_provider import (
    YOLOP_320_SHA256,
    YOLOPDetectionProvider,
    YOLOPProviderError,
    segmentation_summary,
)
from prism.video_observation_replay import DecodedFrame

CAPTURE_START_NS = 1_000_000_000


def _percentiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    at = lambda q: ordered[round((len(ordered) - 1) * q)]
    return {"count": len(ordered), "p50": round(at(0.5), 2),
            "p95": round(at(0.95), 2), "max": round(ordered[-1], 2)}


def _detection_stats(per_frame: list[list[Any]]) -> dict[str, Any]:
    counts = [len(items) for items in per_frame]
    scores = [float(item.score) for items in per_frame for item in items]
    return {
        "frames": len(per_frame),
        "frames_with_detections": sum(1 for count in counts if count),
        "detections_total": sum(counts),
        "detections_per_frame": {"min": min(counts) if counts else 0,
                                 "median": float(np.median(counts)) if counts else 0.0,
                                 "max": max(counts) if counts else 0},
        "score": ({"min": float(min(scores)), "median": float(np.median(scores)),
                   "max": float(max(scores))} if scores else None),
        "score_semantics": "per-detector score; not calibrated confidence",
    }


def _run_yolop(frames: list[DecodedFrame], provider: YOLOPDetectionProvider) -> dict[str, Any]:
    latencies: list[float] = []
    detections: list[list[Any]] = []
    segmentation: list[dict[str, Any]] = []
    segmentation_errors: list[str] = []
    for frame in frames:
        tick = perf_counter()
        supplied, outputs = provider.run_detailed(frame)
        latencies.append((perf_counter() - tick) * 1000.0)
        evidence = supplied.observation.evidence.object_detections
        if evidence is None:
            raise YOLOPProviderError("YOLOP provider returned no detection evidence")
        detections.append(list(evidence.detections))
        try:
            segmentation.append(segmentation_summary(outputs))
        except YOLOPProviderError as error:
            segmentation_errors.append(f"frame {frame.frame_index}: {error}")
    stats = _detection_stats(detections)
    stats["latency_ms"] = _percentiles(latencies)
    stats["segmentation_outputs"] = {
        "frames_measured": len(segmentation),
        "errors": segmentation_errors,
        "aggregate": ({
            "drivable_class_1_fraction_mean": float(np.mean(
                [item["drivable_class_1_fraction"] for item in segmentation])),
            "lane_class_1_fraction_mean": float(np.mean(
                [item["lane_class_1_fraction"] for item in segmentation])),
            "per_frame_drivable_fraction": [
                round(item["drivable_class_1_fraction"], 4) for item in segmentation],
        } if segmentation else None),
        "semantics": (
            "BDD100K road-scene drivable-area and lane-line heads. These are not "
            "terrain labels, not off-road traversability, and no segmentation "
            "accuracy is claimed."),
    }
    return stats


def _run_baseline(frames: list[DecodedFrame], weights: Path) -> dict[str, Any]:
    from prism.libreyolo_provider import LibreYOLODetectionProvider

    provider = LibreYOLODetectionProvider.from_checkpoint(
        weights, source="yolop-bounded-evaluation", hypothesis_id="object-detection-evidence",
        relation=EvidenceRelation.SUPPORTS, expires_after_ns=1_000_000_000)
    latencies: list[float] = []
    detections: list[list[Any]] = []
    for frame in frames:
        tick = perf_counter()
        supplied = provider(frame)
        latencies.append((perf_counter() - tick) * 1000.0)
        evidence = supplied.observation.evidence.object_detections
        detections.append([] if evidence is None else list(evidence.detections))
    stats = _detection_stats(detections)
    stats["latency_ms"] = _percentiles(latencies)
    stats["model_identifier"] = provider.config.model_identifier
    stats["weights_sha256"] = provider.config.weights_sha256
    return stats


def evaluate(clip: Path, checkpoint: Path, *, max_frames: int, expected_sha256: str,
             confidence_threshold: float, baseline_weights: Path | None) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise SystemExit(f"could not open clip: {clip}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 1.0
    frames: list[DecodedFrame] = []
    try:
        while len(frames) < max_frames:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            index = len(frames)
            frames.append(DecodedFrame(index, CAPTURE_START_NS + round(index * 1e9 / fps),
                                       int(bgr.shape[1]), int(bgr.shape[0]), bgr))
    finally:
        cap.release()
    if not frames:
        raise SystemExit("clip yielded no decodable frames")

    process = psutil.Process() if psutil is not None else None
    rss_before = int(process.memory_info().rss) if process is not None else None
    provider = YOLOPDetectionProvider.from_onnx(
        checkpoint, expected_sha256=expected_sha256, source=f"recording:{clip.name}",
        hypothesis_id="object-detection-evidence", relation=EvidenceRelation.SUPPORTS,
        expires_after_ns=1_000_000_000, confidence_threshold=confidence_threshold)
    started = perf_counter()
    yolop = _run_yolop(frames, provider)
    yolop_elapsed = perf_counter() - started
    rss_after_yolop = int(process.memory_info().rss) if process is not None else None

    baseline = None
    baseline_error = None
    rss_after_baseline = None
    if baseline_weights is not None and baseline_weights.is_file():
        try:
            baseline = _run_baseline(frames, baseline_weights)
            rss_after_baseline = int(process.memory_info().rss) if process is not None else None
        except Exception as error:
            baseline_error = f"{type(error).__name__}: {error}"

    return {
        "format": "prism-yolop-bounded-evaluation/v1",
        "clip": str(clip),
        "frames_evaluated": len(frames),
        "frame_indices": [frame.frame_index for frame in frames],
        "input_dimensions": [frames[0].width, frames[0].height],
        "confidence_threshold": confidence_threshold,
        "yolop": {
            "model": {
                "identifier": provider.config.model_identifier,
                "version": provider.config.model_version,
                "source_url": provider.config.model_source_url,
                "weights_sha256": provider.config.weights_sha256,
                "backend": provider.config.backend,
                "input_size": provider.config.input_size,
                "preprocessing": provider.config.preprocessing,
            },
            "elapsed_s": yolop_elapsed,
            **yolop,
        },
        "baseline_libreyolo": None if baseline is None else {
            "model_identifier": baseline.get("model_identifier"),
            "weights_sha256": baseline.get("weights_sha256"),
            **{key: value for key, value in baseline.items()
               if key not in ("model_identifier", "weights_sha256")},
        },
        "baseline_error": baseline_error,
        "memory": {
            "measurement": "psutil Process.memory_info().rss sampled after each stage; process-wide, not per-model",
            "rss_before_bytes": rss_before,
            "rss_after_yolop_bytes": rss_after_yolop,
            "rss_after_baseline_bytes": rss_after_baseline,
        },
        "comparison_note": (
            "YOLOP and LibreYOLO use different taxonomies (BDD100K vehicle vs COCO), "
            "so detection counts are not directly comparable and no accuracy "
            "comparison is possible without reviewed labels."),
        "accuracy": "unmeasured",
        "accuracy_unmeasured_reason": (
            "clips/clip_01.mp4 has no reviewed object annotations; only latency, "
            "counts, and score distributions are measured"),
        "prediction": "not used; this tool does not run tracking or prediction",
        "disposition": "EVALUATE: optional provider, not the default detector",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", default="clips/clip_01.mp4")
    parser.add_argument("--checkpoint", default="tools/external/yolop/yolop-320-320.onnx")
    parser.add_argument("--expected-sha256", default=YOLOP_320_SHA256)
    parser.add_argument("--max-frames", type=int, default=27)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--baseline-weights", default="model_assets/libreyolo/LibreYOLO9t.pt")
    parser.add_argument("--output", default="output/yolop_bounded_evaluation.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")
    output = Path(args.output)
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing report: {output} (use --force)")
    baseline = Path(args.baseline_weights)
    report = evaluate(Path(args.clip), Path(args.checkpoint), max_frames=args.max_frames,
                      expected_sha256=args.expected_sha256,
                      confidence_threshold=args.confidence,
                      baseline_weights=baseline if baseline.is_file() else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    import json
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "frames": report["frames_evaluated"],
        "yolop_p50_ms": report["yolop"]["latency_ms"]["p50"],
        "yolop_detections": report["yolop"]["detections_total"],
        "baseline_p50_ms": (None if report["baseline_libreyolo"] is None
                            else report["baseline_libreyolo"]["latency_ms"]["p50"]),
        "accuracy": report["accuracy"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
