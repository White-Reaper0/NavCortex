"""Model-free tests for the narrow LibreYOLO detection boundary."""
import numpy as np
import pytest

from prism.belief_ledger import EvidenceRelation
from prism.libreyolo_provider import LibreYOLODetectionProvider, LibreYOLOProviderConfig, LibreYOLOProviderError
from prism.video_observation_replay import DecodedFrame

HASH = "b4d7e93f9e0393830fb42e6135c0e3464b2673b05e5ecf4b7f2374ec18e39eb2"

class FakeBoxes:
    def __init__(self, xyxy, conf, cls): self.xyxy, self.conf, self.cls = np.asarray(xyxy), np.asarray(conf), np.asarray(cls)
class FakeResult:
    def __init__(self, boxes): self.boxes = boxes
class FakeModel:
    def __init__(self, result=None, error=None): self.result, self.error, self.calls = result, error, []
    def __call__(self, image, **kwargs):
        self.calls.append((image.shape, kwargs))
        if self.error: raise self.error
        return self.result
def config(**changes):
    values = dict(source="recording:clip_01", hypothesis_id="objects-present", relation=EvidenceRelation.SUPPORTS,
                  expires_after_ns=1_000_000_000, model_identifier="LibreYOLO9t", model_version="LibreYOLO9t",
                  package_version="1.6.0", weights_sha256=HASH)
    values.update(changes)
    return LibreYOLOProviderConfig(**values)
def frame(index=2, timestamp=123_000):
    return DecodedFrame(index, timestamp, 100, 80, np.zeros((80, 100, 3), np.uint8))

def test_successful_detections_preserve_timestamp_source_and_provenance():
    model = FakeModel(FakeResult(FakeBoxes([[10, 20, 50, 70]], [0.75], [3])))
    supplied = LibreYOLODetectionProvider(model, config())(frame())
    observation, detections = supplied.observation, supplied.observation.evidence.object_detections
    assert observation.source == "recording:clip_01" and observation.captured_at_ns == 123_000
    assert observation.confidence is None and detections.taxonomy == "coco-2017@80"
    assert detections.detections[0].class_id == 3
    assert detections.detections[0].bbox_xyxy == (10.0, 20.0, 50.0, 70.0)
    assert supplied.expires_at_ns == 1_000_123_000
    assert observation.provenance["input_frame"]["frame_index"] == 2
    assert observation.provenance["detector"]["weights_sha256"] == HASH
    assert model.calls[0][1]["batch"] == 1 and model.calls[0][1]["color_format"] == "bgr"

def test_successful_zero_detection_result_is_available_not_unavailable():
    supplied = LibreYOLODetectionProvider(FakeModel(FakeResult(FakeBoxes(np.empty((0, 4)), np.empty(0), np.empty(0)))), config())(frame())
    assert supplied.observation.validity.value == "valid"
    assert supplied.observation.evidence.object_detections.detections == ()
    assert supplied.observation.evidence.has_evidence

def test_inference_error_is_not_misreported_as_zero_detections():
    with pytest.raises(LibreYOLOProviderError, match="inference failed"):
        LibreYOLODetectionProvider(FakeModel(error=RuntimeError("backend unavailable")), config())(frame())

@pytest.mark.parametrize("result", [object(), FakeResult(FakeBoxes([[1, 2, 3]], [0.8], [1])),
                                      FakeResult(FakeBoxes([[1, 2, 3, 4]], [0.8, 0.7], [1])),
                                      FakeResult(FakeBoxes([[1, 2, 3, 4]], [1.2], [1]))])
def test_malformed_model_outputs_are_rejected(result):
    with pytest.raises(LibreYOLOProviderError): LibreYOLODetectionProvider(FakeModel(result), config())(frame())

def test_same_explicit_frame_and_model_result_produce_deterministic_replay_input():
    provider = LibreYOLODetectionProvider(FakeModel(FakeResult(FakeBoxes([[1, 2, 30, 40]], [0.5], [0]))), config())
    assert provider(frame()) == provider(frame())
