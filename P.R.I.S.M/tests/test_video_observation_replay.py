"""Tests for sequential video decode and explicit observation-provider boundary."""
from pathlib import Path

import cv2
import numpy as np

from prism.belief_ledger import EvidenceRelation
from prism.observation_replay import AdvisoryDisposition
from prism.observations import CoordinateFrame, EvidencePayload, Observation, Validity
from prism.video_observation_replay import (
    FrameStatus,
    VideoObservationReplayHarness,
    VideoReplayInput,
)


ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "clips" / "clip_01.mp4"


def provider_for_every_frame(frame):
    observation = Observation(
        observation_id=f"frame-{frame.frame_index}", source="clip_01",
        captured_at_ns=frame.capture_at_ns, coordinate_frame=CoordinateFrame.CAMERA,
        validity=Validity.VALID, provenance={"clip": "clip_01", "frame": frame.frame_index},
        evidence=EvidencePayload(metric_depth_m=np.ones((2, 2), np.float32)),
    )
    return VideoReplayInput(observation, "path-clear", EvidenceRelation.SUPPORTS,
                            frame.capture_at_ns + 1_000_000_000)


def test_existing_clip_decodes_sequentially_and_replays_injected_observations():
    assert CLIP.is_file()
    report = VideoObservationReplayHarness().replay(
        CLIP, provider_for_every_frame, capture_start_ns=10_000, assessment_at_ns=100_000,
        max_frames=3)

    assert report.opened and report.fps == 30.0
    assert (report.width, report.height) == (1280, 720)
    assert report.decoded_frames == 3
    assert [record.frame_index for record in report.records] == [0, 1, 2]
    assert all(record.status is FrameStatus.PIPELINE_ADAPTED for record in report.records)
    assert report.accepted_observations == 3 and report.rejected_observations == 0
    assert report.records[1].capture_at_ns == 33_333_333 + 10_000
    assert report.records[0].adapter_result.audit.provenance["frame"] == 0
    assert report.advisory.disposition is AdvisoryDisposition.NO_ACTION


def test_missing_observations_do_not_fabricate_perception():
    report = VideoObservationReplayHarness().replay(
        CLIP, lambda _frame: None, capture_start_ns=0, assessment_at_ns=0, max_frames=2)

    assert report.decoded_frames == 2
    assert all(record.status is FrameStatus.NO_OBSERVATION for record in report.records)
    assert report.pipeline_result is None and report.advisory is None
    assert report.accepted_observations == 0


def test_frame_observation_timestamp_mismatch_is_reported_without_pipeline_entry():
    def mismatched(frame):
        observation = Observation(
            observation_id="bad-time", source="clip_01", captured_at_ns=frame.capture_at_ns + 1,
            coordinate_frame=CoordinateFrame.CAMERA, validity=Validity.VALID,
            evidence=EvidencePayload(metric_depth_m=np.ones((2, 2), np.float32)),
        )
        return VideoReplayInput(observation, "h", EvidenceRelation.SUPPORTS,
                                frame.capture_at_ns + 1_000)

    report = VideoObservationReplayHarness().replay(
        CLIP, mismatched, capture_start_ns=0, assessment_at_ns=0, max_frames=1)

    assert report.records[0].status is FrameStatus.OBSERVATION_MISMATCH
    assert report.pipeline_result is None


class _FakeCapture:
    def __init__(self, frames, declared):
        self._frames = iter(frames)
        self._declared = declared
        self.released = False

    def isOpened(self):
        return True

    def get(self, property_id):
        values = {
            cv2.CAP_PROP_FPS: 10.0,
            cv2.CAP_PROP_FRAME_WIDTH: 4.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 3.0,
            cv2.CAP_PROP_FRAME_COUNT: float(self._declared),
        }
        return values[property_id]

    def read(self):
        try:
            return True, next(self._frames)
        except StopIteration:
            return False, None

    def release(self):
        self.released = True


def test_decode_failure_before_declared_end_is_audited():
    frame = np.zeros((3, 4, 3), np.uint8)
    fake = _FakeCapture([frame], declared=2)
    harness = VideoObservationReplayHarness(capture_factory=lambda _path: fake)

    report = harness.replay("synthetic.mp4", lambda _frame: None,
                            capture_start_ns=0, assessment_at_ns=0)

    assert fake.released
    assert report.decoded_frames == 1
    assert report.records[0].status is FrameStatus.NO_OBSERVATION
    assert report.records[1].status is FrameStatus.DECODE_FAILURE
    assert report.errors == ("decode failed at frame 1",)
