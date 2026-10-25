"""Evaluate reviewed annotations with CPU-only LibreYOLO9t; no control interfaces."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prism.belief_ledger import EvidenceRelation
from prism.detection_evaluation import evaluate_detections, load_ground_truth
from prism.libreyolo_provider import LibreYOLODetectionProvider
from prism.video_observation_replay import DecodedFrame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="evaluation/clip_01_libreyolo9t/annotations.json")
    parser.add_argument("--weights", default="model_assets/libreyolo/LibreYOLO9t.pt")
    parser.add_argument("--confidence", type=float, action="append", default=None,
                        help="score threshold; repeat to set a documented sweep (default: 0.10, 0.25, 0.50)")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--report", default="evaluation/clip_01_libreyolo9t/evaluation_report.json")
    args = parser.parse_args()
    thresholds = args.confidence if args.confidence is not None else [0.10, 0.25, 0.50]
    if len(set(thresholds)) != len(thresholds):
        raise SystemExit("confidence thresholds must be unique")
    ground_truth = load_ground_truth(args.annotations)
    reviewed = tuple(frame for frame in ground_truth.frames if frame.evaluable)
    predictions = {}
    timings = []
    if reviewed:
        provider = LibreYOLODetectionProvider.from_checkpoint(
            args.weights, source="evaluation", hypothesis_id="objects-present",
            relation=EvidenceRelation.SUPPORTS, expires_after_ns=1_000_000_000,
            taxonomy=ground_truth.taxonomy, confidence_threshold=0.0)
        root = Path(args.annotations).parent
        for frame in reviewed:
            image = cv2.imread(str(root / frame.image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise SystemExit(f"could not read annotated image for {frame.frame_id}")
            start = perf_counter()
            supplied = provider(DecodedFrame(frame.frame_index, frame.capture_at_ns,
                                             frame.width, frame.height, image))
            timings.append((perf_counter() - start) * 1_000)
            predictions[frame.frame_id] = supplied.observation.evidence.object_detections.detections
    sweep = [evaluate_detections(ground_truth, predictions, confidence_threshold=threshold,
                                 iou_threshold=args.iou) for threshold in thresholds]
    report = {"annotations": str(args.annotations), "weights": str(args.weights),
              "reviewed_inference_ran": bool(reviewed), "latency_ms": timings,
              "iou_threshold": args.iou, "confidence_thresholds": thresholds,
              "threshold_sweep": [asdict(metrics) for metrics in sweep],
              "limitation": "Metrics quantify only human-reviewed frames; predictions were never used as ground truth."}
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
