"""Sequential video-to-observation replay harness.

Decoding a video produces pixels and deterministic capture-order metadata only.
It does *not* produce depth, pose, terrain, confidence, or any other perception
measurement.  A caller must inject an observation provider that explicitly
supplies a validated :class:`prism.observations.Observation`, or return ``None``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Protocol

import cv2
import numpy as np

from .doubt_engine import Recommendation
from .observation_replay import (
    AdvisoryDecision,
    ObservationReplayPipeline,
    ReplayInput,
    ReplayOutcome,
    ReplayResult,
)
from .observations import Observation
from .belief_ledger import EvidenceRelation
from .runtime_adapter import RuntimeAdapterResult


class VideoCaptureLike(Protocol):
    def isOpened(self) -> bool: ...
    def get(self, property_id: int) -> float: ...
    def read(self) -> tuple[bool, Optional[np.ndarray]]: ...
    def release(self) -> None: ...


class FrameStatus(str, Enum):
    NO_OBSERVATION = "no_observation"
    PROVIDER_ERROR = "provider_error"
    OBSERVATION_MISMATCH = "observation_mismatch"
    PIPELINE_ADAPTED = "pipeline_adapted"
    PIPELINE_REJECTED = "pipeline_rejected"
    DECODE_FAILURE = "decode_failure"


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    """One transient decoded frame passed to the provider, never retained by reports."""

    frame_index: int
    capture_at_ns: int
    width: int
    height: int
    bgr: np.ndarray


@dataclass(frozen=True, slots=True)
class VideoReplayInput:
    """How one provider observation should enter the existing replay pipeline."""

    observation: Observation
    hypothesis_id: str
    relation: EvidenceRelation
    expires_at_ns: int
    clip_id: Optional[str] = None


@dataclass(frozen=True, slots=True)
class FrameReplayRecord:
    """Serializable frame metadata and outcome; deliberately excludes image pixels."""

    frame_index: int
    capture_at_ns: int
    width: int
    height: int
    status: FrameStatus
    observation_id: Optional[str] = None
    adapter_result: Optional[RuntimeAdapterResult] = None
    adapter_error: Optional[str] = None
    recommendations: tuple[Recommendation, ...] = ()
    error: Optional[str] = None


@dataclass(frozen=True, slots=True)
class VideoReplayReport:
    """Reproducible report for a bounded, sequential decode/replay run."""

    video_path: str
    opened: bool
    fps: float
    width: int
    height: int
    declared_frame_count: int
    decoded_frames: int
    records: tuple[FrameReplayRecord, ...]
    pipeline_result: Optional[ReplayResult]
    advisory: Optional[AdvisoryDecision]
    errors: tuple[str, ...]

    @property
    def accepted_observations(self) -> int:
        return sum(record.status is FrameStatus.PIPELINE_ADAPTED for record in self.records)

    @property
    def rejected_observations(self) -> int:
        return sum(record.status is FrameStatus.PIPELINE_REJECTED for record in self.records)


ObservationProvider = Callable[[DecodedFrame], Optional[VideoReplayInput]]
CaptureFactory = Callable[[str], VideoCaptureLike]


class VideoObservationReplayHarness:
    """Decode sequentially and feed explicit provider observations into replay.

    The harness keeps only frame metadata and provider output.  The decoded BGR
    array is scoped to one provider invocation and is discarded before reading
    the next frame.
    """

    def __init__(self, pipeline: Optional[ObservationReplayPipeline] = None,
                 capture_factory: CaptureFactory = cv2.VideoCapture) -> None:
        self._pipeline = pipeline if pipeline is not None else ObservationReplayPipeline()
        self._capture_factory = capture_factory

    def replay(self, video_path: str | Path, provider: ObservationProvider, *,
               capture_start_ns: int, assessment_at_ns: int,
               max_frames: Optional[int] = None) -> VideoReplayReport:
        """Run a bounded replay with explicit times; no wall clock is consulted."""
        if not callable(provider):
            raise ValueError("provider must be callable")
        if isinstance(capture_start_ns, bool) or not isinstance(capture_start_ns, int) \
                or capture_start_ns < 0:
            raise ValueError("capture_start_ns must be a non-negative integer")
        if isinstance(assessment_at_ns, bool) or not isinstance(assessment_at_ns, int) \
                or assessment_at_ns < 0:
            raise ValueError("assessment_at_ns must be a non-negative integer")
        if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int)
                                       or max_frames < 0):
            raise ValueError("max_frames must be a non-negative integer or None")

        path = str(video_path)
        cap = self._capture_factory(path)
        if not cap.isOpened():
            cap.release()
            return VideoReplayReport(path, False, 0.0, 0, 0, 0, 0, (), None, None,
                                     ("video could not be opened",))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        declared = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if not np.isfinite(fps) or fps <= 0:
            cap.release()
            return VideoReplayReport(path, True, fps, width, height, declared, 0, (), None, None,
                                     ("video reports no positive finite FPS",))

        pending: list[tuple[int, int, int, int, VideoReplayInput]] = []
        records: list[FrameReplayRecord] = []
        errors: list[str] = []
        frame_index = 0
        try:
            while max_frames is None or frame_index < max_frames:
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    if declared > frame_index:
                        records.append(FrameReplayRecord(
                            frame_index, self._capture_timestamp(capture_start_ns, frame_index, fps),
                            width, height, FrameStatus.DECODE_FAILURE,
                            error="decode failed before declared end of video"))
                        errors.append(f"decode failed at frame {frame_index}")
                    break
                frame_height, frame_width = bgr.shape[:2]
                timestamp = self._capture_timestamp(capture_start_ns, frame_index, fps)
                decoded = DecodedFrame(frame_index, timestamp, frame_width, frame_height, bgr)
                try:
                    supplied = provider(decoded)
                except Exception as error:  # Provider is external; report and continue replay.
                    records.append(FrameReplayRecord(frame_index, timestamp, frame_width, frame_height,
                                                     FrameStatus.PROVIDER_ERROR, error=str(error)))
                    frame_index += 1
                    continue
                if supplied is None:
                    records.append(FrameReplayRecord(frame_index, timestamp, frame_width, frame_height,
                                                     FrameStatus.NO_OBSERVATION))
                elif not isinstance(supplied, VideoReplayInput) or not isinstance(
                        supplied.observation, Observation):
                    records.append(FrameReplayRecord(frame_index, timestamp, frame_width, frame_height,
                                                     FrameStatus.OBSERVATION_MISMATCH,
                                                     error="provider must return VideoReplayInput or None"))
                elif supplied.observation.captured_at_ns != timestamp:
                    records.append(FrameReplayRecord(
                        frame_index, timestamp, frame_width, frame_height,
                        FrameStatus.OBSERVATION_MISMATCH,
                        observation_id=supplied.observation.observation_id,
                        error="observation captured_at_ns does not match frame capture-order timestamp"))
                else:
                    pending.append((frame_index, timestamp, frame_width, frame_height, supplied))
                frame_index += 1
        finally:
            cap.release()

        if not pending:
            return VideoReplayReport(path, True, fps, width, height, declared, frame_index,
                                     tuple(records), None, None, tuple(errors))
        pipeline_result = self._pipeline.replay(
            [ReplayInput(item.observation, item.hypothesis_id, item.relation,
                         item.expires_at_ns, item.clip_id) for *_, item in pending],
            at_ns=assessment_at_ns,
        )
        recommendations = pipeline_result.assessment.recommendations
        for (index, timestamp, frame_width, frame_height, supplied), replay_record in zip(
                pending, pipeline_result.records):
            matching = tuple(recommendation for recommendation in recommendations
                             if recommendation.hypothesis_id == supplied.hypothesis_id)
            if replay_record.outcome is ReplayOutcome.ADAPTED:
                status = FrameStatus.PIPELINE_ADAPTED
            else:
                status = FrameStatus.PIPELINE_REJECTED
            records.append(FrameReplayRecord(
                index, timestamp, frame_width, frame_height, status,
                observation_id=supplied.observation.observation_id,
                adapter_result=replay_record.adapter_result,
                adapter_error=replay_record.adapter_error,
                recommendations=matching,
            ))
        records.sort(key=lambda record: record.frame_index)
        return VideoReplayReport(path, True, fps, width, height, declared, frame_index,
                                 tuple(records), pipeline_result, pipeline_result.advisory,
                                 tuple(errors))

    @staticmethod
    def _capture_timestamp(capture_start_ns: int, frame_index: int, fps: float) -> int:
        # This is capture-order replay time derived from indexed decode order, not
        # a claim about original camera hardware timestamps.
        return capture_start_ns + round(frame_index * 1_000_000_000 / fps)
