"""Sequential batch-1 CPU benchmark over prepared frames; no ground-truth claims."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prism.belief_ledger import EvidenceRelation
from prism.libreyolo_provider import LibreYOLODetectionProvider
from prism.video_observation_replay import DecodedFrame


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t)]


def peak_working_set_bytes() -> int | None:
    """Read this process's Windows peak working set without adding psutil."""
    try:
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters), wintypes.DWORD)
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)
    except (AttributeError, OSError):
        return None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="evaluation/clip_01_libreyolo9t/manifest.json")
    parser.add_argument("--weights", default="model_assets/libreyolo/LibreYOLO9t.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--report", default="evaluation/clip_01_libreyolo9t/benchmark_report.json")
    args = parser.parse_args()
    if args.warmup_frames < 0:
        raise SystemExit("--warmup-frames must not be negative")
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["frames"]
    if len(rows) <= args.warmup_frames:
        raise SystemExit("manifest must have more frames than --warmup-frames")
    provider = LibreYOLODetectionProvider.from_checkpoint(
        args.weights, source="benchmark", hypothesis_id="objects-present",
        relation=EvidenceRelation.SUPPORTS, expires_after_ns=1_000_000_000,
        taxonomy=manifest["taxonomy"], confidence_threshold=args.confidence)
    warmup, steady, detections = [], [], []
    for position, row in enumerate(rows):
        image = cv2.imread(str(manifest_path.parent / row["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"could not read {row['image_path']}")
        start = perf_counter()
        supplied = provider(DecodedFrame(row["frame_index"], row["capture_at_ns"],
                                         row["width"], row["height"], image))
        elapsed = (perf_counter() - start) * 1_000
        (warmup if position < args.warmup_frames else steady).append(elapsed)
        detections.append(len(supplied.observation.evidence.object_detections.detections))
    report = {"frames": len(rows), "batch_size": 1, "backend": provider.config.backend,
              "confidence_threshold": args.confidence, "warmup_latency_ms": warmup,
              "steady_latency_ms": steady, "steady_p50_ms": percentile(steady, 0.50),
              "steady_p95_ms": percentile(steady, 0.95), "detection_counts": detections,
              "peak_working_set_bytes": peak_working_set_bytes(),
              "note": "Sequential CPU-only measurement on this machine; not a real-time or GPU estimate."}
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
