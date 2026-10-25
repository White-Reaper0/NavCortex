"""Video / cache IO helpers shared by every prism module."""
from __future__ import annotations
import json, subprocess, shutil
from pathlib import Path
from typing import Iterator, Optional
import numpy as np
import cv2

from .config import (CFG, CLIP_DIR, CLIP_IDS, CACHE_DIR, WORK_DIR, PROC_W, PROC_H, CLIP_FPS,
                     EGO_MASK_BOTTOM_FRAC, EGO_MASK_WATERMARK)

# --------------------------------------------------------------------- clips


def clip_path(clip_id: str) -> Path:
    return CLIP_DIR / f"{clip_id}.mp4"


def clips_meta() -> list[dict]:
    p = CLIP_DIR / "clips.json"
    return json.loads(p.read_text()) if p.exists() else []


def read_frames(clip_id: str, resize: Optional[tuple[int, int]] = (PROC_W, PROC_H),
                max_frames: Optional[int] = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index, BGR frame). Resized to `resize` (w, h) unless None."""
    cap = cv2.VideoCapture(str(clip_path(clip_id)))
    i = 0
    while True:
        ok, f = cap.read()
        if not ok or (max_frames is not None and i >= max_frames):
            break
        if resize is not None and (f.shape[1], f.shape[0]) != resize:
            f = cv2.resize(f, resize, interpolation=cv2.INTER_AREA)
        yield i, f
        i += 1
    cap.release()


def load_clip_array(clip_id: str, resize: tuple[int, int] = (PROC_W, PROC_H)) -> np.ndarray:
    """Whole clip as (N,H,W,3) uint8. 300 frames at 640x360 is ~200 MB."""
    return np.stack([f for _, f in read_frames(clip_id, resize)])


# --------------------------------------------------------------------- ego mask

_EGO_CACHE: dict[tuple[int, int], np.ndarray] = {}


def ego_mask(h: int = PROC_H, w: int = PROC_W) -> np.ndarray:
    """True where the pixel shows the world (not the vehicle body or the watermark).

    Prefers the traced silhouette written by `tools/make_ego_mask.py`; falls back to a
    crude bottom band plus watermark box if that file has not been generated yet.
    """
    key = (h, w)
    if key not in _EGO_CACHE:
        png = WORK_DIR / "ego_mask.png"
        m = None
        if png.exists():
            raw = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
            if raw is not None:
                if raw.shape != (h, w):
                    raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
                m = raw > 127
        if m is None:
            m = np.ones((h, w), bool)
            m[int(h * (1.0 - EGO_MASK_BOTTOM_FRAC)):, :] = False
            x0, y0, x1, y1 = EGO_MASK_WATERMARK
            m[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = False
        _EGO_CACHE[key] = m
    return _EGO_CACHE[key]


# --------------------------------------------------------------------- cache


def cache_dir(clip_id: str) -> Path:
    d = CACHE_DIR / clip_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_stage(clip_id: str, stage: str, **arrays) -> Path:
    """Persist a perception stage's arrays for a clip (compressed npz)."""
    p = cache_dir(clip_id) / f"{stage}.npz"
    np.savez_compressed(p, **arrays)
    return p


def load_stage(clip_id: str, stage: str) -> dict:
    p = cache_dir(clip_id) / f"{stage}.npz"
    if not p.exists():
        raise FileNotFoundError(f"missing cache stage '{stage}' for {clip_id}: {p}")
    with np.load(p, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def has_stage(clip_id: str, stage: str) -> bool:
    return (CACHE_DIR / clip_id / f"{stage}.npz").exists()


# --------------------------------------------------------------------- video writing


class VideoWriter:
    """Writes frames through ffmpeg so the output is a clean, seekable H.264 mp4.

    Falls back to cv2.VideoWriter if ffmpeg is unavailable.
    """

    def __init__(self, path: Path | str, size: tuple[int, int], fps: int = CLIP_FPS, crf: int = 18):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = size
        self.fps = fps
        self._proc = None
        self._cv = None
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-y", "-v", "error",
                   "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-",
                   "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)]
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        else:
            self._cv = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        if self._proc is not None:
            self._proc.stdin.write(frame.tobytes())
        else:
            self._cv.write(frame)

    def close(self) -> None:
        if self._proc is not None:
            self._proc.stdin.close()
            err = self._proc.stderr.read().decode(errors="ignore")
            rc = self._proc.wait()
            if rc != 0:
                raise RuntimeError(f"ffmpeg failed writing {self.path}: {err[:600]}")
        elif self._cv is not None:
            self._cv.release()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def write_video(path: Path | str, frames, size: tuple[int, int], fps: int = CLIP_FPS) -> Path:
    with VideoWriter(path, size, fps) as w:
        for f in frames:
            w.write(f)
    return Path(path)


# --------------------------------------------------------------------- misc


def set_seed(seed: int = CFG.seed) -> None:
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> str:
    import torch
    return "cuda" if (CFG.device == "cuda" and torch.cuda.is_available()) else "cpu"
