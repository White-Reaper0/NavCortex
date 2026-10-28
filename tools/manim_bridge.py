"""Bridge that runs a Manim scene and returns its frames as BGR numpy arrays,
bypassing Manim's own video writer entirely.

On this machine, Manim 0.19's built-in writer (PyAV-backed) produces corrupt
partial-movie files and fails combining them into a final mp4 -- verified via a
smoke test (`ffprobe` reports "Invalid data found" / "No start code" on the raw
partial files Manim itself wrote, before any file-list or path handling is even
involved). Manim's own *rendering* is fine; only its H.264 encode step is broken
here. So we monkeypatch `SceneFileWriter.write_frame` to capture each rendered
frame into a plain Python list instead of pushing it into Manim's writer queue,
and no-op `finish()` so it never attempts the broken combine step. The frames are
then handed to `prism.io_utils.VideoWriter`, the ffmpeg-subprocess writer this
project already uses everywhere else, which is known-good on this machine.
"""
from __future__ import annotations
from typing import Type
import numpy as np


def render_scene_to_frames(scene_cls: Type, width: int, height: int, fps: int,
                           background_hex: str = "#0E1012") -> list[np.ndarray]:
    """Render one Manim Scene subclass and return its frames as BGR uint8 arrays."""
    import manim.scene.scene_file_writer as sfw
    from manim import config as mconfig

    captured: list[np.ndarray] = []

    def _write_frame(self, frame_or_renderer, num_frames: int = 1) -> None:
        frame = frame_or_renderer if isinstance(frame_or_renderer, np.ndarray) else frame_or_renderer
        for _ in range(num_frames):
            captured.append(np.array(frame, copy=True))

    def _finish(self) -> None:
        pass  # skip Manim's own (broken, on this box) combine-to-mp4 step

    orig_write_frame = sfw.SceneFileWriter.write_frame
    orig_finish = sfw.SceneFileWriter.finish
    sfw.SceneFileWriter.write_frame = _write_frame
    sfw.SceneFileWriter.finish = _finish
    try:
        mconfig.pixel_width = width
        mconfig.pixel_height = height
        mconfig.frame_height = 8.0                        # manim's default world height
        mconfig.frame_width = 8.0 * width / height         # keep world units matched to pixel aspect
        mconfig.frame_rate = fps
        mconfig.background_color = background_hex
        mconfig.write_to_movie = False
        mconfig.disable_caching = True
        mconfig.verbosity = "WARNING"

        scene = scene_cls()
        scene.render()
    finally:
        sfw.SceneFileWriter.write_frame = orig_write_frame
        sfw.SceneFileWriter.finish = orig_finish

    # RGBA -> BGR (alpha is always fully opaque for a Scene with a solid background)
    return [f[..., :3][:, :, ::-1].copy() for f in captured]


def pad_or_trim(frames: list[np.ndarray], n_target: int) -> list[np.ndarray]:
    """Hold the last frame or trim so a scene occupies exactly n_target frames."""
    if not frames:
        return frames
    if len(frames) >= n_target:
        return frames[:n_target]
    return frames + [frames[-1].copy()] * (n_target - len(frames))
