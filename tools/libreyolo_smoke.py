"""Bounded CPU-only LibreYOLO9t smoke replay; no vehicle interfaces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

# Permit direct ``python tools/libreyolo_smoke.py`` invocation from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prism.belief_ledger import EvidenceRelation
from prism.libreyolo_provider import LibreYOLODetectionProvider
from prism.video_observation_replay import VideoObservationReplayHarness


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="clips/clip_01.mp4")
    parser.add_argument("--weights", default="model_assets/libreyolo/LibreYOLO9t.pt")
    parser.add_argument("--max-frames", type=int, default=3)
    args = parser.parse_args()
    if args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")
    provider = LibreYOLODetectionProvider.from_checkpoint(
        args.weights, source=f"recording:{Path(args.video).name}",
        hypothesis_id="objects-present", relation=EvidenceRelation.SUPPORTS,
        expires_after_ns=1_000_000_000,
    )
    timings: list[float] = []
    def timed(frame):
        start = perf_counter()
        supplied = provider(frame)
        timings.append((perf_counter() - start) * 1_000)
        return supplied
    start = perf_counter()
    report = VideoObservationReplayHarness().replay(
        args.video, timed, capture_start_ns=1_000_000_000,
        assessment_at_ns=1_200_000_000, max_frames=args.max_frames,
    )
    counts = [len(record.input.observation.evidence.object_detections.detections)
              for record in report.pipeline_result.records] if report.pipeline_result else []
    print(json.dumps({
        "opened": report.opened, "decoded_frames": report.decoded_frames,
        "frame_dimensions": [report.width, report.height], "fps": report.fps,
        "detection_counts": counts, "provider_latency_ms": timings,
        "total_elapsed_ms": (perf_counter() - start) * 1_000,
        "frame_statuses": [record.status.value for record in report.records],
        "weights_sha256": provider.config.weights_sha256,
        "backend": provider.config.backend,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
