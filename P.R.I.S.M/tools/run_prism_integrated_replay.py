"""Run P.R.I.S.M.'s bounded, offline, advisory-only recorded-video prototype.

No live camera, network request, planner invocation, or vehicle command is
performed. Missing detector, calibration, tracking, geometry, and terrain
inputs are reported as unavailable rather than fabricated.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any
from collections.abc import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

try:  # Optional; available in the current .venv through Open3D's dependency set.
    import psutil
except Exception:  # pragma: no cover - platform/package dependent
    psutil = None

from prism.belief_ledger import BeliefLedger, EvidenceRelation
from prism.doubt_engine import DoubtEngine
from prism.integrations.geometry import sparse_geometry_report
from prism.integrations.localization import localization_backend_report
from prism.integrations.mapping_diagnostic import mapping_diagnostic_status
from prism.integrations.terrain import terrain_status
from prism.integrations.tracking import create_image_tracker, norfair_availability
from prism.libreyolo_provider import LibreYOLODetectionProvider
from prism.observations import CoordinateFrame, EvidencePayload, Observation, Validity
from prism.runtime_adapter import ObservationAdapterError, RuntimeAdapter
from prism.video_observation_replay import DecodedFrame
from prism.visual_odometry_replay import VideoOdometryReplayHarness

CAPTURE_START_NS = 1_000_000_000


def _timestamp(frame_index: int, fps: float) -> int:
    return CAPTURE_START_NS + round(frame_index * 1_000_000_000 / fps)


def _json(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: _json(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _draw(frame, tracks, status: str):
    for track in tracks:
        x0, y0, x1, y1 = (round(value) for value in track.bbox_xyxy)
        color = (0, 210, 0) if track.observed else (0, 170, 255)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        tag = f"T{track.track_id} {track.class_id} {track.score:.2f}"
        if track.predicted:
            tag += " predicted"
        cv2.putText(frame, tag, (x0, max(16, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)
    cv2.putText(frame, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 255), 2, cv2.LINE_AA)
    return frame


def _unavailable_terrain_observation(at_ns: int) -> Observation:
    return Observation(
        observation_id="integrated-replay:terrain-spatial-evidence-unavailable",
        source="integrated-replay", captured_at_ns=at_ns,
        coordinate_frame=CoordinateFrame.LOCAL_MAP, validity=Validity.UNAVAILABLE,
        provenance={"reason": "no validated metric spatial terrain input exists in this recorded replay"},
        evidence=EvidencePayload(),
    )


def run(clip: Path, output: Path, *, max_frames: int | None, weights: Path,
        model_dir: Path, manifest: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    if not clip.is_file():
        raise ValueError(f"clip does not exist: {clip}")
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise ValueError(f"could not open clip: {clip}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        cap.release(); raise ValueError("clip reports no positive FPS")
    writer = cv2.VideoWriter(str(output / "annotated_replay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    provider = None; detector_error = None
    try:
        provider = LibreYOLODetectionProvider.from_checkpoint(
            weights, source=f"recording:{clip.name}", hypothesis_id="object-detection-evidence",
            relation=EvidenceRelation.SUPPORTS, expires_after_ns=1_000_000_000)
    except Exception as error:
        detector_error = f"{type(error).__name__}: {error}"
    tracker = create_image_tracker()
    ledger = BeliefLedger(); adapter = RuntimeAdapter()
    detections: list[dict] = []; tracks: list[dict] = []; events: list[dict] = []
    latencies: list[float] = []; frame_index = 0
    process = psutil.Process() if psutil is not None else None
    peak_rss = process.memory_info().rss if process is not None else None
    started = perf_counter()
    try:
        while max_frames is None or frame_index < max_frames:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            capture_at_ns = _timestamp(frame_index, fps)
            event: dict[str, Any] = {"frame_index": frame_index, "capture_at_ns": capture_at_ns,
                                     "width": int(bgr.shape[1]), "height": int(bgr.shape[0])}
            observation = None; frame_detections = ()
            if provider is None:
                event.update({"detector_status": "UNAVAILABLE", "detector_error": detector_error})
            else:
                tick = perf_counter()
                try:
                    supplied = provider(DecodedFrame(frame_index, capture_at_ns, int(bgr.shape[1]), int(bgr.shape[0]), bgr))
                    latencies.append((perf_counter() - tick) * 1000.0)
                    observation = supplied.observation
                    frame_detections = observation.evidence.object_detections.detections  # provider validates this
                    detections.append({"frame_index": frame_index, "capture_at_ns": capture_at_ns,
                                       "observation_id": observation.observation_id,
                                       "detections": [{"class_id": item.class_id, "taxonomy": item.taxonomy,
                                                       "score": item.score, "bbox_xyxy": list(item.bbox_xyxy)}
                                                      for item in frame_detections],
                                       "provenance": _json(observation.provenance)})
                    try:
                        adapter.to_packet(observation, packet_index=frame_index, clip_id=clip.name)
                        event["runtime_adapter"] = "ADAPTED"
                    except ObservationAdapterError as error:
                        # 2-D boxes are not geometry; this is an expected conservative rejection.
                        event["runtime_adapter"] = "REJECTED"
                        event["runtime_adapter_reason"] = str(error)
                    ledger.record("object-detection-evidence", observation, EvidenceRelation.SUPPORTS,
                                  expires_at_ns=capture_at_ns + 1_000_000_000)
                    event.update({"detector_status": "AVAILABLE", "detection_count": len(frame_detections)})
                except Exception as error:
                    latencies.append((perf_counter() - tick) * 1000.0)
                    event.update({"detector_status": "PROVIDER_ERROR", "detector_error": f"{type(error).__name__}: {error}"})
            frame_tracks = tracker.update(frame_detections, frame_index=frame_index, capture_at_ns=capture_at_ns)
            tracks.extend(_json(item) for item in frame_tracks)
            event["tracks"] = [{"track_id": item.track_id, "observed": item.observed,
                                "predicted": item.predicted} for item in frame_tracks]
            writer.write(_draw(bgr.copy(), frame_tracks, event["detector_status"]))
            events.append(event); frame_index += 1
            if process is not None:
                peak_rss = max(int(peak_rss or 0), int(process.memory_info().rss))
    finally:
        cap.release(); writer.release()
    ply_export = output / "sparse_points_open3d_native.ply"
    geometry = sparse_geometry_report(
        model_dir, manifest, (item["frame_index"] for item in events),
        export_open3d_ply=ply_export,
    )
    terrain = terrain_status(geometry_report=geometry)
    localization = localization_backend_report(clip_has_rgb_only=True)
    terrain_observation = _unavailable_terrain_observation(_timestamp(max(0, frame_index - 1), fps))
    ledger.record("terrain-spatial-evidence", terrain_observation, EvidenceRelation.SUPPORTS,
                  expires_at_ns=terrain_observation.captured_at_ns + 1)
    assessment_at_ns = _timestamp(max(0, frame_index - 1), fps) + 1
    ledger.advance_to(assessment_at_ns)
    assessment = DoubtEngine().assess(ledger, at_ns=assessment_at_ns,
                                      hypothesis_ids=("object-detection-evidence", "terrain-spatial-evidence"))
    advisory = {"disposition": "CAUTIOUS_STOP" if assessment.recommendations else "NO_ACTION",
                "reason": "advisory only; UNKNOWN/UNAVAILABLE terrain-spatial evidence is never safe",
                "assessment": _json(assessment)}
    # VO is a separate bounded decode pass; its default missing-calibration mode is tracking-only.
    vo = VideoOdometryReplayHarness().replay(clip, max_frames=frame_index, capture_start_ns=CAPTURE_START_NS)
    mapping = mapping_diagnostic_status(geometry_report=geometry, visual_motion_summary=vo.summary())
    _write_json(output / "detections.json", {"detector_error": detector_error, "frames": detections})
    _write_json(output / "tracks.csv.json", {"coordinate_frame": "image", "records": tracks,
                                               "prediction_note": "predictions are not observations or world trajectories"})
    with (output / "tracks.csv").open("w", encoding="utf-8") as file:
        file.write("track_id,frame_index,capture_at_ns,class_id,taxonomy,score,x_min,y_min,x_max,y_max,observed,predicted,age_frames,prediction_note\n")
        for item in tracks:
            box = item["bbox_xyxy"]
            file.write(",".join(str(value) for value in (item["track_id"], item["frame_index"], item["capture_at_ns"], item["class_id"], item["taxonomy"], item["score"], *box, item["observed"], item["predicted"], item["age_frames"], item["prediction_note"] or "")) + "\n")
    with (output / "frame_events.jsonl").open("w", encoding="utf-8") as file:
        for item in events: file.write(json.dumps(_json(item), sort_keys=True) + "\n")
    with (output / "camera_motion.csv").open("w", encoding="utf-8") as file:
        file.write("frame_index,capture_order_ns,status,n_features,n_matches,n_inliers,inlier_ratio,latency_ms,reason\n")
        for item in vo.records:
            file.write(",".join(str(value) for value in (item.frame_index, item.capture_order_ns, item.status.value,
                       item.n_features, item.n_matches, item.n_inliers, item.inlier_ratio, item.latency_ms,
                       item.reason.replace(",", ";"))) + "\n")
    _write_json(output / "sparse_geometry_report.json", geometry)
    _write_json(output / "terrain_report.json", terrain)
    _write_json(output / "mapping_diagnostic.json", mapping)
    _write_json(output / "localization_backends.json", localization)
    _write_json(output / "advisory_decisions.json", advisory)
    run_report = {
        "format": "prism-integrated-offline-replay/v1", "clip": str(clip), "frame_count": frame_index,
        "fps": fps, "dimensions": [width, height], "runtime_s": perf_counter() - started,
        "process_memory": (None if peak_rss is None else {"peak_working_set_bytes": peak_rss,
                           "measurement": "psutil Process.memory_info().rss sampled in-process during replay"}),
        "detector": {"status": "AVAILABLE" if provider else "UNAVAILABLE", "error": detector_error,
                     "latency_ms": _latency(latencies), "model": None if provider is None else _json(provider.config)},
        "tracking": {"backend": tracker.backend, "norfair": norfair_availability(), "records": len(tracks),
                     "coordinate_frame": "image; no metric/world trajectory"},
        "visual_motion": vo.summary(), "geometry": {"status": geometry["status"],
                     "exact_pose_matches": len(geometry["exact_pose_matches"]),
                     "unmatched_source_frames": len(geometry["unmatched_source_frame_indices"])},
        "terrain": terrain, "mapping": mapping, "localization": localization, "advisory": advisory,
        "limits": ["No live camera or vehicle control was used.",
                   "2-D detections do not establish depth, clearance, terrain, or safety.",
                   "COLMAP scale remains unverified and coordinates remain relative."]}
    _write_json(output / "run_report.json", run_report)
    readme = (
        "# P.R.I.S.M. offline replay output\n\n"
        "1. Open `run_report.json` for component status, latency, and memory.\n"
        "2. Watch `annotated_replay.mp4` for 2-D boxes, track IDs, and detector status.\n"
        "3. Inspect `detections.json`, `tracks.csv`, `frame_events.jsonl`, and `camera_motion.csv`.\n"
        "4. Read `sparse_geometry_report.json` and optional `sparse_points_open3d_native.ply` (COLMAP native units, not metres).\n"
        "5. `terrain_report.json` and `mapping_diagnostic.json` explain unavailable metric map/terrain outputs.\n"
        "6. `localization_backends.json` documents blocked SLAM candidates and the selected OpenCV/COLMAP path.\n"
        "7. `advisory_decisions.json` is conservative guidance only; it does not command the vehicle.\n"
    )
    (output / "README.md").write_text(readme, encoding="utf-8")
    return run_report


def _latency(values: list[float]) -> dict:
    if not values: return {"count": 0, "p50_ms": None, "p95_ms": None}
    values = sorted(values); at = lambda q: values[round((len(values) - 1) * q)]
    return {"count": len(values), "p50_ms": at(.5), "p95_ms": at(.95), "max_ms": values[-1]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", default="clips/clip_01.mp4")
    parser.add_argument("--output", default="output/prism_integrated_test")
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--full-clip", action="store_true")
    parser.add_argument("--weights", default="model_assets/libreyolo/LibreYOLO9t.pt")
    parser.add_argument("--model-dir", default="output/09_colmap_sparse/clip_01_repro/sparse/0")
    parser.add_argument("--manifest", default="output/09_colmap_sparse/clip_01_repro/frames_manifest.json")
    args = parser.parse_args()
    if args.max_frames <= 0: raise SystemExit("--max-frames must be positive")
    report = run(Path(args.clip), Path(args.output), max_frames=None if args.full_clip else args.max_frames,
                 weights=Path(args.weights), model_dir=Path(args.model_dir), manifest=Path(args.manifest))
    print(json.dumps({"output": args.output, "frames": report["frame_count"], "runtime_s": report["runtime_s"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
