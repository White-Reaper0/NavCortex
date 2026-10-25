"""Image-space multi-object tracking with an optional Norfair boundary.

Tracks are image observations and short-horizon image-space extrapolations;
they are never world trajectories, depth estimates, or safety claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..observations import ObjectDetection


@dataclass(frozen=True, slots=True)
class TrackRecord:
    track_id: int
    frame_index: int
    capture_at_ns: int
    class_id: str | int
    taxonomy: str
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    observed: bool
    predicted: bool
    age_frames: int
    prediction_note: str | None = None


def iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    overlap = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = ((left[2] - left[0]) * (left[3] - left[1])
             + (right[2] - right[0]) * (right[3] - right[1]) - overlap)
    return overlap / union if union > 0 else 0.0


class ImageSpaceIoUTracker:
    """Deterministic dependency-free fallback when Norfair is unavailable."""

    backend = "iou-fallback"
    coordinate_frame = "image"

    def __init__(self, *, min_iou: float = 0.3, max_missed_frames: int = 3) -> None:
        self._min_iou = float(min_iou)
        self._max_missed = int(max_missed_frames)
        self._next_id = 1
        self._active: dict[int, TrackRecord] = {}

    def update(self, detections: Iterable[ObjectDetection], *, frame_index: int,
               capture_at_ns: int) -> tuple[TrackRecord, ...]:
        detections = tuple(detections)
        assigned: set[int] = set()
        output: list[TrackRecord] = []
        for detection in detections:
            candidates = [track for track in self._active.values()
                          if track.track_id not in assigned and track.class_id == detection.class_id
                          and track.taxonomy == detection.taxonomy]
            matched = max(candidates, key=lambda track: iou(track.bbox_xyxy, detection.bbox_xyxy), default=None)
            if matched is not None and iou(matched.bbox_xyxy, detection.bbox_xyxy) >= self._min_iou:
                track_id, age = matched.track_id, matched.age_frames + 1
            else:
                track_id, age = self._next_id, 1
                self._next_id += 1
            item = TrackRecord(track_id, frame_index, capture_at_ns, detection.class_id,
                               detection.taxonomy, detection.bbox_xyxy, detection.score,
                               True, False, age)
            self._active[track_id] = item
            assigned.add(track_id)
            output.append(item)
        for track_id, track in tuple(self._active.items()):
            if track_id in assigned:
                continue
            gap = frame_index - track.frame_index
            if gap > self._max_missed:
                del self._active[track_id]
                continue
            # Retaining the last image-space box is an explicitly labelled,
            # bounded prediction, not a new detector observation.
            predicted = TrackRecord(track.track_id, frame_index, capture_at_ns,
                                    track.class_id, track.taxonomy, track.bbox_xyxy,
                                    track.score, False, True, track.age_frames,
                                    "last-observed image box retained during a missed detection; not a new detector observation")
            self._active[track_id] = predicted
            output.append(predicted)
        return tuple(sorted(output, key=lambda item: item.track_id))


class NorfairImageTracker:
    """Thin Norfair adapter retaining only image-space detector information."""

    backend = "norfair-2d-centroid"
    coordinate_frame = "image"

    def __init__(self) -> None:
        from norfair import Detection, Tracker  # type: ignore
        self._Detection = Detection

        def distance(detection, tracked_object):
            left = detection.data["class_id"]
            right = tracked_object.last_detection.data["class_id"]
            if left != right or detection.data["taxonomy"] != tracked_object.last_detection.data["taxonomy"]:
                return float("inf")
            return float(np.linalg.norm(detection.points - tracked_object.estimate))
        self._tracker = Tracker(distance_function=distance, distance_threshold=80,
                                initialization_delay=0, hit_counter_max=3)

    def update(self, detections: Iterable[ObjectDetection], *, frame_index: int,
               capture_at_ns: int) -> tuple[TrackRecord, ...]:
        packed = []
        for item in detections:
            x0, y0, x1, y1 = item.bbox_xyxy
            packed.append(self._Detection(points=np.asarray([[(x0 + x1) / 2, (y0 + y1) / 2]], dtype=float),
                                          data={"class_id": item.class_id, "taxonomy": item.taxonomy,
                                                "bbox": item.bbox_xyxy, "score": item.score}))
        output = []
        for item in self._tracker.update(detections=packed):
            data = item.last_detection.data
            output.append(TrackRecord(int(item.id), frame_index, capture_at_ns,
                                      data["class_id"], data["taxonomy"], tuple(data["bbox"]),
                                      float(data["score"]), True, False, int(item.age)))
        return tuple(sorted(output, key=lambda item: item.track_id))


def create_image_tracker():
    """Prefer the installed Norfair backend, with a deterministic local fallback."""
    try:
        return NorfairImageTracker()
    except Exception:
        return ImageSpaceIoUTracker()


def norfair_availability() -> dict:
    try:
        import norfair  # type: ignore
        return {"status": "AVAILABLE", "version": getattr(norfair, "__version__", "unknown")}
    except Exception as error:
        return {"status": "UNAVAILABLE", "reason": f"Norfair import failed: {type(error).__name__}: {error}"}
