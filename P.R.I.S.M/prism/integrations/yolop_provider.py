"""Optional YOLOP ONNX provider for typed, image-space vehicle detections.

YOLOP (hustvl/YOLOP, MIT repository license) is a BDD100K road-scene multitask
network: one vehicle detection class plus drivable-area and lane-line
segmentation.  Only the detection branch is exposed through the PRISM
observation contract.  The road-scene segmentation outputs are **not**
traversability evidence and are never written into terrain fields.

The provider loads a local ONNX file with OpenCV DNN and verifies its SHA-256.
It never downloads weights and never falls back to fabricated detections.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..belief_ledger import EvidenceRelation
from ..observations import (CoordinateFrame, EvidencePayload, ImageCoordinateConvention,
                           ObjectDetection, ObjectDetectionEvidence, Observation, Validity)
from ..video_observation_replay import DecodedFrame, VideoReplayInput

YOLOP_MODEL_URL = "https://github.com/hustvl/YOLOP"
YOLOP_WEIGHTS_URL = "https://github.com/hustvl/YOLOP/tree/main/weights"
YOLOP_LICENSE_NOTE = (
    "Repository declares MIT. Upstream README credits YOLOv5; helpers in "
    "lib/models/common.py are YOLOv5-derived, so the MIT declaration is treated "
    "as unverified for those files. Local evaluation only; do not redistribute.")
YOLOP_DETECTION_TAXONOMY = "bdd100k-vehicle@1"
YOLOP_CLASS_NAMES: tuple[str, ...] = ("vehicle",)
YOLOP_320_SHA256 = "86d6e8b6dfdef195c061e9bcad82d9487bb5ee1ac4a1cf9a3dc4736657a07369"
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class YOLOPProviderError(RuntimeError):
    """A YOLOP invocation or output could not be represented honestly."""


@dataclass(frozen=True, slots=True)
class YOLOPProviderConfig:
    """Explicit model identity and replay policy for one source stream."""

    source: str
    hypothesis_id: str
    relation: EvidenceRelation
    expires_after_ns: int
    weights_sha256: str
    model_identifier: str = "YOLOP"
    model_version: str = "YOLOP (BDD100K; end-to-end ONNX)"
    model_source_url: str = YOLOP_MODEL_URL
    backend: str = "opencv-dnn-cpu"
    input_size: int = 320
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    taxonomy: str = YOLOP_DETECTION_TAXONOMY
    class_names: tuple[str, ...] = YOLOP_CLASS_NAMES
    preprocessing: str = (
        "letterbox to square input with 114 padding, BGR->RGB, /255, "
        "ImageNet mean/std (YOLOP tools/demo.py convention)")

    def __post_init__(self) -> None:
        for value, name in ((self.source, "source"), (self.hypothesis_id, "hypothesis_id"),
                            (self.model_identifier, "model_identifier"),
                            (self.model_version, "model_version"),
                            (self.model_source_url, "model_source_url"),
                            (self.backend, "backend"), (self.taxonomy, "taxonomy"),
                            (self.preprocessing, "preprocessing")):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.relation, EvidenceRelation):
            raise ValueError("relation must be an EvidenceRelation")
        if (isinstance(self.expires_after_ns, bool) or not isinstance(self.expires_after_ns, int)
                or self.expires_after_ns < 0):
            raise ValueError("expires_after_ns must be a non-negative integer")
        digest = self.weights_sha256.lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("weights_sha256 must be a lowercase SHA-256 hexadecimal digest")
        object.__setattr__(self, "weights_sha256", digest)
        if (isinstance(self.input_size, bool) or not isinstance(self.input_size, int)
                or self.input_size <= 0):
            raise ValueError("input_size must be a positive integer")
        for value, name in ((self.confidence_threshold, "confidence_threshold"),
                            (self.iou_threshold, "iou_threshold")):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0):
                raise ValueError(f"{name} must be finite and in [0, 1]")
            object.__setattr__(self, name, float(value))
        names = tuple(self.class_names)
        if not names or any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("class_names must contain non-empty strings")
        object.__setattr__(self, "class_names", names)


class YOLOPDetectionProvider:
    """Turn one decoded BGR frame into detection-only replay input.

    ``forward`` receives a NCHW float32 blob and returns the network outputs in
    YOLOP order ``(detections, drivable_logits, lane_logits)``.  Injecting it
    keeps the provider testable without an ONNX runtime.
    """

    def __init__(self, forward: Callable[[np.ndarray], Sequence[Any]],
                 config: YOLOPProviderConfig) -> None:
        if not callable(forward):
            raise ValueError("forward must be callable")
        if not isinstance(config, YOLOPProviderConfig):
            raise ValueError("config must be a YOLOPProviderConfig")
        self._forward, self._config = forward, config

    @classmethod
    def from_onnx(cls, checkpoint: str | Path, *, expected_sha256: str,
                  source: str, hypothesis_id: str, relation: EvidenceRelation,
                  expires_after_ns: int, input_size: int = 320,
                  confidence_threshold: float = 0.25,
                  iou_threshold: float = 0.45) -> "YOLOPDetectionProvider":
        """Load a local ONNX file after verifying its pinned SHA-256."""
        path = Path(checkpoint)
        if not path.is_file():
            raise YOLOPProviderError(f"YOLOP ONNX does not exist: {path}")
        digest = _sha256(path)
        if digest != expected_sha256.lower():
            raise YOLOPProviderError(
                f"YOLOP ONNX SHA-256 mismatch: {digest} != {expected_sha256.lower()}")
        try:
            import cv2
            net = cv2.dnn.readNetFromONNX(str(path))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            names = net.getUnconnectedOutLayersNames()
        except Exception as error:
            raise YOLOPProviderError(f"OpenCV DNN could not load YOLOP ONNX: {error}") from error

        def forward(blob: np.ndarray) -> list[np.ndarray]:
            try:
                net.setInput(blob)
                return list(net.forward(names))
            except Exception as error:
                raise YOLOPProviderError(f"YOLOP inference failed: {error}") from error

        return cls(forward, YOLOPProviderConfig(
            source=source, hypothesis_id=hypothesis_id, relation=relation,
            expires_after_ns=expires_after_ns, weights_sha256=digest,
            input_size=input_size, confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold))

    @property
    def config(self) -> YOLOPProviderConfig:
        return self._config

    def __call__(self, frame: DecodedFrame) -> VideoReplayInput:
        return self.run_detailed(frame)[0]

    def run_detailed(self, frame: DecodedFrame) -> tuple[VideoReplayInput, list[np.ndarray]]:
        """Return replay input plus raw network outputs for offline evaluation.

        The raw outputs are for measurement only (e.g. segmentation output
        statistics); they are not part of the observation contract.
        """
        if not isinstance(frame, DecodedFrame):
            raise YOLOPProviderError("provider input must be a DecodedFrame")
        blob, letterbox = _preprocess(frame.bgr, self._config.input_size)
        outputs = self._forward(blob)
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 1:
            raise YOLOPProviderError("YOLOP must return at least the detection output")
        detections = _decode_detections(
            outputs[0], letterbox=letterbox, frame_width=frame.width,
            frame_height=frame.height, config=self._config)
        try:
            evidence = ObjectDetectionEvidence(
                self._config.taxonomy, frame.width, frame.height,
                ImageCoordinateConvention.PIXEL_XYXY_EXCLUSIVE, detections)
        except ValueError as error:
            raise YOLOPProviderError(f"invalid YOLOP image-space evidence: {error}") from error
        observation = Observation(
            observation_id=f"{self._config.source}:frame-{frame.frame_index}:{frame.capture_at_ns}",
            source=self._config.source, captured_at_ns=frame.capture_at_ns,
            coordinate_frame=CoordinateFrame.CAMERA, validity=Validity.VALID,
            provenance={
                "input_frame": {"frame_index": frame.frame_index,
                                "capture_at_ns": frame.capture_at_ns,
                                "width": frame.width, "height": frame.height},
                "detector": {
                    "implementation": "YOLOP",
                    "model_identifier": self._config.model_identifier,
                    "model_version": self._config.model_version,
                    "model_source_url": self._config.model_source_url,
                    "weights_source_url": YOLOP_WEIGHTS_URL,
                    "weights_sha256": self._config.weights_sha256,
                    "backend": self._config.backend,
                    "input_size": self._config.input_size,
                    "preprocessing": self._config.preprocessing,
                    "taxonomy": self._config.taxonomy,
                    "class_names": list(self._config.class_names),
                    "score_semantics": (
                        "objectness x class score after NMS; not observation confidence"),
                    "segmentation_outputs_used": False,
                    "segmentation_note": (
                        "drivable-area/lane logits are road-scene segmentation and are "
                        "not traversability evidence"),
                    "license_note": YOLOP_LICENSE_NOTE,
                },
            },
            evidence=EvidencePayload(object_detections=evidence))
        return VideoReplayInput(observation, self._config.hypothesis_id, self._config.relation,
                                frame.capture_at_ns + self._config.expires_after_ns), list(outputs)


def segmentation_summary(outputs: Sequence[Any]) -> dict:
    """Describe YOLOP segmentation outputs without turning them into evidence.

    Values are model output statistics only; they are not accuracy and not
    off-road traversability.
    """
    if not isinstance(outputs, (list, tuple)) or len(outputs) < 3:
        raise YOLOPProviderError("YOLOP outputs do not include both segmentation heads")
    drivable = np.asarray(outputs[1], dtype=np.float64)
    lane = np.asarray(outputs[2], dtype=np.float64)
    if drivable.ndim == 4:
        drivable = drivable[0]
    if lane.ndim == 4:
        lane = lane[0]
    if drivable.shape[0] != 2 or lane.shape[0] != 2:
        raise YOLOPProviderError("YOLOP segmentation heads must have two classes")
    drivable_mask = drivable.argmax(0) > 0
    lane_mask = lane.argmax(0) > 0
    return {
        "drivable_class_1_fraction": float(drivable_mask.mean()),
        "lane_class_1_fraction": float(lane_mask.mean()),
        "drivable_logit_mean": float(drivable.mean()),
        "lane_logit_mean": float(lane.mean()),
        "semantics": (
            "BDD100K road-scene drivable-area and lane-line outputs; not terrain "
            "labels, not traversability, and not an accuracy measurement"),
    }


@dataclass(frozen=True, slots=True)
class _Letterbox:
    scale: float
    pad_x: int
    pad_y: int


def _preprocess(bgr: np.ndarray, size: int) -> tuple[np.ndarray, _Letterbox]:
    import cv2

    if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise YOLOPProviderError("frame.bgr must be HxWx3")
    height, width = bgr.shape[:2]
    if height <= 0 or width <= 0:
        raise YOLOPProviderError("frame has no pixels")
    scale = min(size / width, size / height)
    new_w, new_h = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=interpolation)
    canvas = np.full((size, size, 3), 114, np.uint8)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None]), _Letterbox(
        scale, pad_x, pad_y)


def _decode_detections(output: Any, *, letterbox: _Letterbox, frame_width: int,
                       frame_height: int, config: YOLOPProviderConfig
                       ) -> tuple[ObjectDetection, ...]:
    array = np.asarray(output, dtype=np.float64)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] < 6 or not np.isfinite(array).all():
        raise YOLOPProviderError("detection output must be finite with shape (N, >=6)")
    objectness = array[:, 4]
    class_scores = array[:, 5:]
    class_ids = class_scores.argmax(axis=1)
    class_conf = class_scores[np.arange(len(array)), class_ids]
    confidence = objectness * class_conf
    keep = confidence >= config.confidence_threshold
    if not np.any(keep):
        return ()
    rows = array[keep]
    confidence = confidence[keep]
    class_ids = class_ids[keep]

    center_x, center_y, width, height = (rows[:, index] for index in range(4))
    boxes = np.stack([center_x - width / 2.0, center_y - height / 2.0,
                      center_x + width / 2.0, center_y + height / 2.0], axis=1)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - letterbox.pad_x) / letterbox.scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - letterbox.pad_y) / letterbox.scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(frame_width))
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(frame_height))
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, confidence, class_ids = boxes[valid], confidence[valid], class_ids[valid]
    if len(boxes) == 0:
        return ()
    keep_indices = _nms_indices(boxes, confidence, config.iou_threshold)
    detections = []
    for index in keep_indices:
        class_id = int(class_ids[index])
        if class_id >= len(config.class_names):
            raise YOLOPProviderError(
                f"YOLOP emitted class index {class_id} outside the declared taxonomy")
        name = config.class_names[class_id]
        detections.append(ObjectDetection(name, config.taxonomy,
                                          float(confidence[index]),
                                          tuple(float(value) for value in boxes[index])))
    return tuple(detections)


def _nms_indices(boxes: np.ndarray, scores: np.ndarray,
                 iou_threshold: float) -> list[int]:
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        x0 = np.maximum(boxes[current, 0], boxes[rest, 0])
        y0 = np.maximum(boxes[current, 1], boxes[rest, 1])
        x1 = np.minimum(boxes[current, 2], boxes[rest, 2])
        y1 = np.minimum(boxes[current, 3], boxes[rest, 3])
        intersection = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
        union = areas[current] + areas[rest] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection),
                        where=union > 0)
        order = rest[iou <= iou_threshold]
    return keep


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
