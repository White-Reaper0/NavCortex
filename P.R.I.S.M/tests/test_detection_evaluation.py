"""Model-free ground-truth validation and metric-accounting tests."""
import json

import pytest

from prism.detection_evaluation import (
    ANNOTATION_SCHEMA,
    COCO_2017_80_TAXONOMY,
    AnnotationFrame,
    GroundTruthObject,
    GroundTruthSet,
    ReviewStatus,
    evaluate_detections,
    iou_xyxy,
    load_ground_truth,
    write_annotation_template,
)
from prism.observations import ObjectDetection


def frame(*, frame_id="clip_01:000000", status=ReviewStatus.REVIEWED, objects=()):
    return AnnotationFrame(frame_id, "clips/clip_01.mp4", 0, 0, 100, 80, status, objects,
                           "frames/frame_000000.png")


def prediction(class_id=1, score=0.9, box=(10, 10, 40, 40)):
    return ObjectDetection(class_id, COCO_2017_80_TAXONOMY, score, box)


def dataset(*frames):
    return GroundTruthSet(COCO_2017_80_TAXONOMY, frames)


def test_template_preserves_frame_identity_and_requires_human_review(tmp_path):
    target = tmp_path / "annotations.json"
    item = frame(status=ReviewStatus.UNREVIEWED)
    write_annotation_template(target, source_video="clips/clip_01.mp4",
                              taxonomy=COCO_2017_80_TAXONOMY, frames=(item,))

    raw = json.loads(target.read_text())
    loaded = load_ground_truth(target)
    assert raw["schema_version"] == ANNOTATION_SCHEMA
    assert loaded.frames[0].frame_id == "clip_01:000000"
    assert loaded.frames[0].capture_at_ns == 0
    assert loaded.frames[0].review_status is ReviewStatus.UNREVIEWED
    assert not loaded.frames[0].evaluable


def test_invalid_boxes_and_unknown_class_identifiers_are_rejected():
    with pytest.raises(ValueError, match="strictly ordered"):
        GroundTruthObject(1, (2, 2, 2, 5))
    with pytest.raises(ValueError, match="supported"):
        dataset(frame(objects=(GroundTruthObject(80, (1, 1, 2, 2)),)))
    with pytest.raises(ValueError, match="image bounds"):
        frame(objects=(GroundTruthObject(1, (1, 1, 101, 2)),))


def test_iou_and_same_class_greedy_matching_account_for_tp_fp_fn():
    ground_truth = dataset(frame(objects=(GroundTruthObject(1, (10, 10, 40, 40)),
                                          GroundTruthObject(1, (50, 10, 80, 40)))))
    metrics = evaluate_detections(ground_truth, {
        "clip_01:000000": (prediction(1, 0.9, (10, 10, 40, 40)),
                             prediction(1, 0.8, (10, 10, 40, 40)))},
        confidence_threshold=0.25, iou_threshold=0.5)

    assert iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert (metrics.true_positives, metrics.false_positives, metrics.false_negatives) == (1, 1, 1)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.ap is None and metrics.map is None


def test_empty_detections_and_empty_ground_truth_keep_undefined_rates_honest():
    empty = dataset(frame(status=ReviewStatus.REVIEWED_EMPTY))
    metrics = evaluate_detections(empty, {"clip_01:000000": ()},
                                  confidence_threshold=0.25, iou_threshold=0.5)
    assert metrics.ground_truth_count == metrics.predicted_count == 0
    assert metrics.precision is None and metrics.recall is None

    false_alarm = evaluate_detections(empty, {"clip_01:000000": (prediction(),)},
                                     confidence_threshold=0.25, iou_threshold=0.5)
    assert false_alarm.false_positives == 1
    assert false_alarm.precision == 0.0 and false_alarm.recall is None


def test_metrics_are_deterministic_and_unreviewed_frames_are_skipped():
    reviewed = frame(objects=(GroundTruthObject(1, (10, 10, 40, 40)),))
    pending = frame(frame_id="clip_01:000010", status=ReviewStatus.UNREVIEWED)
    ground_truth = dataset(reviewed, pending)
    inputs = {"clip_01:000000": (prediction(),)}
    first = evaluate_detections(ground_truth, inputs, confidence_threshold=0.25, iou_threshold=0.5)
    second = evaluate_detections(ground_truth, inputs, confidence_threshold=0.25, iou_threshold=0.5)
    assert first == second
    assert first.frames_annotated == 1
    assert first.frames_evaluated == 1 and first.frames_skipped == 1
