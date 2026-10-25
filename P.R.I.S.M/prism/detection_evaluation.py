"""Human-ground-truth validation and deterministic 2-D detection metrics.

This module deliberately separates annotation review from detector output.  It
does not generate labels from predictions, and it has no mapping, planning, or
vehicle-control dependency.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .observations import ImageCoordinateConvention, ObjectDetection


ANNOTATION_SCHEMA = "prism-detection-ground-truth/v1"
COCO_2017_80_TAXONOMY = "coco-2017@80"
COCO_2017_80_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)


class ReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    REVIEWED = "reviewed"
    REVIEWED_EMPTY = "reviewed_empty"
    SKIP = "skip"


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_box(value: Sequence[Any], name: str) -> tuple[float, float, float, float]:
    if len(value) != 4:
        raise ValueError(f"{name} must have four coordinates")
    box = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in box) or not box[0] < box[2] or not box[1] < box[3]:
        raise ValueError(f"{name} must be finite with strictly ordered xyxy edges")
    return box


def supported_class_ids(taxonomy: str) -> frozenset[int]:
    if taxonomy != COCO_2017_80_TAXONOMY:
        raise ValueError(f"unsupported taxonomy: {taxonomy}")
    return frozenset(range(len(COCO_2017_80_CLASSES)))


@dataclass(frozen=True, slots=True)
class GroundTruthObject:
    class_id: int
    bbox_xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        _integer(self.class_id, "class_id")
        object.__setattr__(self, "bbox_xyxy", _finite_box(self.bbox_xyxy, "bbox_xyxy"))


@dataclass(frozen=True, slots=True)
class AnnotationFrame:
    frame_id: str
    source_video: str
    frame_index: int
    capture_at_ns: int
    width: int
    height: int
    review_status: ReviewStatus
    objects: tuple[GroundTruthObject, ...] = ()
    image_path: str | None = None
    reviewer_note: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.frame_id, "frame_id")
        _nonempty(self.source_video, "source_video")
        _integer(self.frame_index, "frame_index")
        _integer(self.capture_at_ns, "capture_at_ns")
        _integer(self.width, "width", minimum=1)
        _integer(self.height, "height", minimum=1)
        if not isinstance(self.review_status, ReviewStatus):
            raise ValueError("review_status must be a ReviewStatus")
        objects = tuple(self.objects)
        for item in objects:
            if not isinstance(item, GroundTruthObject):
                raise ValueError("objects must contain GroundTruthObject values")
            x1, y1, x2, y2 = item.bbox_xyxy
            if not (0 <= x1 < x2 <= self.width and 0 <= y1 < y2 <= self.height):
                raise ValueError("ground-truth box must lie inside image bounds")
        if self.review_status in (ReviewStatus.UNREVIEWED, ReviewStatus.SKIP) and objects:
            raise ValueError("unreviewed or skipped frames cannot contain reviewed objects")
        if self.review_status is ReviewStatus.REVIEWED_EMPTY and objects:
            raise ValueError("reviewed_empty frames must contain zero objects")
        object.__setattr__(self, "objects", objects)

    @property
    def evaluable(self) -> bool:
        return self.review_status in (ReviewStatus.REVIEWED, ReviewStatus.REVIEWED_EMPTY)


@dataclass(frozen=True, slots=True)
class GroundTruthSet:
    taxonomy: str
    frames: tuple[AnnotationFrame, ...]
    coordinate_convention: ImageCoordinateConvention = ImageCoordinateConvention.PIXEL_XYXY_EXCLUSIVE

    def __post_init__(self) -> None:
        supported = supported_class_ids(self.taxonomy)
        if self.coordinate_convention is not ImageCoordinateConvention.PIXEL_XYXY_EXCLUSIVE:
            raise ValueError("unsupported annotation coordinate convention")
        frames = tuple(self.frames)
        ids = set()
        for frame in frames:
            if not isinstance(frame, AnnotationFrame):
                raise ValueError("frames must contain AnnotationFrame values")
            if frame.frame_id in ids:
                raise ValueError("frame_id values must be unique")
            ids.add(frame.frame_id)
            for item in frame.objects:
                if item.class_id not in supported:
                    raise ValueError("ground-truth class_id is not supported by taxonomy")
        object.__setattr__(self, "frames", frames)


@dataclass(frozen=True, slots=True)
class DetectionMetrics:
    frames_annotated: int
    frames_evaluated: int
    frames_skipped: int
    ground_truth_count: int
    predicted_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    confidence_threshold: float
    iou_threshold: float
    ap: None = None
    map: None = None
    ap_reason: str = "not computed: a small single-threshold review set is not an AP/mAP protocol"
    by_class: tuple[tuple[int, "DetectionMetrics"], ...] = ()


def iou_xyxy(left: Sequence[float], right: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = _finite_box(left, "left box")
    bx1, by1, bx2, by2 = _finite_box(right, "right box")
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union else 0.0


def evaluate_detections(ground_truth: GroundTruthSet,
                        predictions_by_frame: Mapping[str, Sequence[ObjectDetection]], *,
                        confidence_threshold: float, iou_threshold: float,
                        min_examples_per_class: int = 5) -> DetectionMetrics:
    """Greedily match score-sorted same-class predictions at one IoU threshold."""
    if not isinstance(ground_truth, GroundTruthSet):
        raise ValueError("ground_truth must be a GroundTruthSet")
    if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be finite in [0, 1]")
    if not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be finite in (0, 1]")
    _integer(min_examples_per_class, "min_examples_per_class", minimum=1)
    known_ids = supported_class_ids(ground_truth.taxonomy)
    totals: dict[int | None, list[int]] = {None: [0, 0, 0, 0]}  # gt, pred, tp, fp
    annotated = evaluated = skipped = 0
    for frame in ground_truth.frames:
        if frame.review_status is not ReviewStatus.UNREVIEWED:
            annotated += 1
        if not frame.evaluable:
            skipped += 1
            continue
        evaluated += 1
        predictions = tuple(predictions_by_frame.get(frame.frame_id, ()))
        for prediction in predictions:
            if not isinstance(prediction, ObjectDetection):
                raise ValueError("predictions must contain ObjectDetection values")
            if prediction.taxonomy != ground_truth.taxonomy or prediction.class_id not in known_ids:
                raise ValueError("prediction taxonomy or class_id is not supported")
        candidates = [item for item in predictions if item.score >= confidence_threshold]
        gt_by_class: dict[int, list[GroundTruthObject]] = {}
        for item in frame.objects:
            gt_by_class.setdefault(item.class_id, []).append(item)
        pred_by_class: dict[int, list[ObjectDetection]] = {}
        for item in candidates:
            pred_by_class.setdefault(int(item.class_id), []).append(item)
        for class_id in sorted(set(gt_by_class) | set(pred_by_class)):
            gts = gt_by_class.get(class_id, [])
            preds = sorted(pred_by_class.get(class_id, []), key=lambda item: (-item.score, item.bbox_xyxy))
            matched: set[int] = set()
            tp = 0
            for prediction in preds:
                options = [(iou_xyxy(prediction.bbox_xyxy, target.bbox_xyxy), index)
                           for index, target in enumerate(gts) if index not in matched]
                best_iou, best_index = max(options, default=(0.0, -1))
                if best_iou >= iou_threshold:
                    matched.add(best_index)
                    tp += 1
            fp = len(preds) - tp
            for key in (None, class_id):
                totals.setdefault(key, [0, 0, 0, 0])
                totals[key][0] += len(gts); totals[key][1] += len(preds)
                totals[key][2] += tp; totals[key][3] += fp
    overall = _metrics_from_counts(totals[None], annotated, evaluated, skipped,
                                   confidence_threshold, iou_threshold)
    classes = []
    for class_id in sorted(key for key in totals if key is not None):
        counts = totals[class_id]
        if counts[0] >= min_examples_per_class:
            classes.append((class_id, _metrics_from_counts(counts, annotated, evaluated, skipped,
                                                            confidence_threshold, iou_threshold)))
    return DetectionMetrics(**{**asdict(overall), "by_class": tuple(classes)})


def _metrics_from_counts(counts: Sequence[int], annotated: int, evaluated: int, skipped: int,
                         confidence: float, iou: float) -> DetectionMetrics:
    ground_truth_count, predicted_count, tp, fp = counts
    fn = ground_truth_count - tp
    return DetectionMetrics(annotated, evaluated, skipped, ground_truth_count, predicted_count,
                            tp, fp, fn, tp / (tp + fp) if tp + fp else None,
                            tp / (tp + fn) if tp + fn else None, confidence, iou)


def load_ground_truth(path: str | Path) -> GroundTruthSet:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != ANNOTATION_SCHEMA:
        raise ValueError("unsupported annotation schema_version")
    frames = []
    for entry in raw.get("frames", []):
        frames.append(AnnotationFrame(
            frame_id=entry["frame_id"], source_video=entry["source_video"],
            frame_index=entry["frame_index"], capture_at_ns=entry["capture_at_ns"],
            width=entry["width"], height=entry["height"], image_path=entry.get("image_path"),
            reviewer_note=entry.get("reviewer_note"),
            review_status=ReviewStatus(entry["review_status"]),
            objects=tuple(GroundTruthObject(int(item["class_id"]), tuple(item["bbox_xyxy"]))
                          for item in entry.get("objects", []))))
    return GroundTruthSet(raw["taxonomy"], tuple(frames),
                          ImageCoordinateConvention(raw["coordinate_convention"]))


def write_annotation_template(path: str | Path, *, source_video: str, taxonomy: str,
                              frames: Iterable[AnnotationFrame]) -> None:
    """Write an explicitly unreviewed human-label template, never predictions."""
    supported_class_ids(taxonomy)
    frame_data = []
    for frame in frames:
        if frame.review_status is not ReviewStatus.UNREVIEWED:
            raise ValueError("new annotation template frames must be unreviewed")
        frame_data.append({"frame_id": frame.frame_id, "source_video": frame.source_video,
                           "frame_index": frame.frame_index, "capture_at_ns": frame.capture_at_ns,
                           "width": frame.width, "height": frame.height, "image_path": frame.image_path,
                           "review_status": frame.review_status.value, "objects": [],
                           "reviewer_note": "Human review required. For a reviewed zero-object frame, set review_status to reviewed_empty."})
    data = {"schema_version": ANNOTATION_SCHEMA, "source_video": source_video,
            "taxonomy": taxonomy,
            "coordinate_convention": ImageCoordinateConvention.PIXEL_XYXY_EXCLUSIVE.value,
            "class_catalog": {str(index): name for index, name in enumerate(COCO_2017_80_CLASSES)},
            "frames": frame_data}
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
