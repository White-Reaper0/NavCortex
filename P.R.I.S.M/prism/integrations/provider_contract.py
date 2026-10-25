"""Narrow provider boundary shared by optional perception integrations.

A provider turns one decoded frame into a validated typed observation.  It must
never fabricate detections, depth, poses, or segmentation labels when a model or
sensor is unavailable, and it must not decide vehicle motion.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..video_observation_replay import DecodedFrame, VideoReplayInput


@runtime_checkable
class FrameObservationProvider(Protocol):
    """One decoded frame in, one typed observation out; no control output."""

    @property
    def config(self) -> Any: ...

    def __call__(self, frame: DecodedFrame) -> VideoReplayInput: ...
