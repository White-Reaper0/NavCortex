"""Assemble output/final_demo.mp4: a ~60s professional highlight reel combining
Manim-animated title/comparison/architecture/outro segments with a branded
montage of real footage pulled from all 9 completed pipeline stages.

Design
------
Manim renders its own frames (via tools/manim_bridge.py, which bypasses Manim's
broken-on-this-box PyAV writer and hands frames straight to the project's own
ffmpeg-subprocess VideoWriter). The montage re-reads real frames straight out of
the already-rendered, already-verified output/<stage>/<clip>.mp4 files - nothing
in the montage is synthesized, it is literally "all the video clips" cut together.

Every stage's screen time is computed from a fixed remaining-frame budget so the
whole thing lands on exactly 1800 frames (60.000s @ 30fps): render the four Manim
segments first, then split whatever frame budget is left evenly across the 9
stages. Crossfades stitch every join, including into and out of the Manim
segments, so nothing is a hard cut.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manim_bridge import render_scene_to_frames, pad_or_trim
from demo_scenes import TitleScene, CompareScene, ArchScene, OutroScene
from prism.config import OUT_DIR
from prism.io_utils import VideoWriter
from prism import viz_common as V

W, H, FPS = 1280, 720, 30
TOTAL_FRAMES = 60 * FPS  # 1800, exactly 60.000s
FADE_N = 10              # frames of crossfade at every join

# ------------------------------------------------------------------ montage plan
# (stage_dir, clip_id, start_frame, stage_no, title, subtitle)
MONTAGE = [
    ("01_depth_anything_v2",     "clip_02", 140, "01", "MONOCULAR DEPTH",
     "Depth Anything V2-S + a ground-plane metric fit solved fresh every frame"),
    ("02_terrain_segmentation",  "clip_02", 100, "02", "TERRAIN SEGMENTATION",
     "PIDNet-S, built from scratch, distilled into a 7-class off-road taxonomy"),
    ("03_traversability",        "clip_02", 100, "03", "TRAVERSABILITY",
     "Safe / risky / obstacle / unknown - from the vehicle's own clearance and step limits"),
    ("04_visual_odometry",       "clip_04", 150, "04", "VISUAL ODOMETRY",
     "ORB features + essential-matrix RANSAC, scale anchored to the metric depth map"),
    ("05_place_recognition",     "clip_02", 150, "05", "PLACE RECOGNITION",
     "A GeM descriptor recognising a corridor it has driven before"),
    ("06_uncertainty",           "clip_05", 80,  "06", "UNCERTAINTY",
     "Self-supervised confidence - measurably lower on this low-light clip"),
    ("07_lidar_like_pointcloud", "clip_02", 100, "07", "LIDAR-LIKE RECONSTRUCTION",
     "A simulated 32-beam scan, ray-cast through monocular depth"),
    ("07b_lidar_3d_view",        "clip_04", 150, "07b", "POSE-ACCUMULATED 3D VIEW",
     "260 frames fused into one persistent terrain surface via VO registration"),
    ("08_bev_25d_map",           "clip_02", 100, "08", "2.5D LOCAL MAP",
     "Colour = height/risk, vertical lift = obstacle height, rolled forward every frame"),
]


def read_segment(stage_dir: str, clip_id: str, start: int, length: int) -> list[np.ndarray]:
    path = OUT_DIR / stage_dir / f"{clip_id}.mp4"
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(length):
        ok, f = cap.read()
        if not ok:
            break
        if (f.shape[1], f.shape[0]) != (W, H):
            f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    return pad_or_trim(frames, length) if frames else []


def lower_third(frame: np.ndarray, stage_no: str, title: str, subtitle: str,
                alpha: float) -> np.ndarray:
    """Branded caption bar, matching the per-stage renderer chrome. alpha in [0,1]."""
    if alpha <= 0.001:
        return frame
    ov = frame.copy()
    bar_h = 92
    y0 = H - bar_h
    cv2.rectangle(ov, (0, y0), (W, H), (16, 15, 14), -1)
    cv2.rectangle(ov, (0, y0), (5, H), V.ACCENT, -1)
    badge_txt = f"STAGE {stage_no}"
    V.text(ov, badge_txt, (24, y0 + 26), 0.46, V.ACCENT, 1, V.FONT_B)
    V.text(ov, title, (24, y0 + 56), 0.68, V.TEXT, 1, V.FONT_B)
    V.text(ov, subtitle, (24, y0 + 80), 0.38, V.TEXT_DIM)
    # No extra top-left bug: every source frame already carries the stage
    # renderer's own "prism NN - STAGE NAME" header baked in, so a second
    # wordmark here would sit directly on top of it.
    a = float(np.clip(alpha, 0, 1))
    return cv2.addWeighted(ov, a, frame, 1 - a, 0) if a < 1.0 else ov


def build_stage_clip(entry, n_frames: int) -> list[np.ndarray]:
    stage_dir, clip_id, start, no, title, sub = entry
    raw = read_segment(stage_dir, clip_id, start, n_frames)
    out = []
    for i, f in enumerate(raw):
        if i < FADE_N:
            a = (i + 1) / FADE_N
        elif i >= n_frames - FADE_N:
            a = (n_frames - i) / FADE_N
        else:
            a = 1.0
        out.append(lower_third(f, no, title, sub, a))
    return out


def crossfade(tail: list[np.ndarray], head: list[np.ndarray], n: int) -> list[np.ndarray]:
    """Replace the last n frames of `tail` and first n of `head` with a blend,
    returning the frames that go BETWEEN the two (tail[:-n] and head[n:] stay
    where they are; caller concatenates tail_kept + blended + head_kept)."""
    n = min(n, len(tail), len(head))
    blended = []
    for i in range(n):
        a = (i + 1) / (n + 1)
        blended.append(cv2.addWeighted(head[i], a, tail[len(tail) - n + i], 1 - a, 0))
    return blended


def stitch(segments: list[list[np.ndarray]], n_fade: int = FADE_N) -> list[np.ndarray]:
    out = list(segments[0])
    for seg in segments[1:]:
        n = min(n_fade, len(out), len(seg))
        blended = crossfade(out, seg, n)
        out = out[: len(out) - n] + blended + seg[n:]
    return out


def main() -> None:
    t0 = time.perf_counter()
    print("Rendering Manim segments...")
    title_f = render_scene_to_frames(TitleScene, W, H, FPS)
    compare_f = render_scene_to_frames(CompareScene, W, H, FPS)
    arch_f = render_scene_to_frames(ArchScene, W, H, FPS)
    outro_f = render_scene_to_frames(OutroScene, W, H, FPS)
    manim_total = len(title_f) + len(compare_f) + len(arch_f) + len(outro_f)
    print(f"  title={len(title_f)} compare={len(compare_f)} arch={len(arch_f)} "
          f"outro={len(outro_f)}  (sum {manim_total} frames, {manim_total/FPS:.2f}s)")

    montage_budget = TOTAL_FRAMES - manim_total
    n_stages = len(MONTAGE)
    base = montage_budget // n_stages
    rem = montage_budget - base * n_stages
    lengths = [base + (1 if i < rem else 0) for i in range(n_stages)]
    print(f"  montage budget = {montage_budget} frames ({montage_budget/FPS:.2f}s) "
          f"across {n_stages} stages, ~{base/FPS:.2f}s each")

    print("Building montage clips...")
    stage_clips = []
    for entry, n in zip(MONTAGE, lengths):
        clip = build_stage_clip(entry, n)
        stage_clips.append(clip)
        print(f"  {entry[3]:>3s} {entry[4]:<28s} {len(clip):4d} frames from "
              f"{entry[0]}/{entry[1]}.mp4 @ frame {entry[2]}")

    print("Stitching with crossfades...")
    montage_all = stitch(stage_clips, FADE_N)

    all_segments = [title_f, compare_f, arch_f, montage_all, outro_f]
    final_frames = stitch(all_segments, FADE_N)

    # Exact-frame guarantee: crossfading trims a few frames at each of the 4 major
    # joins (n_fade each); pad the last frame if short, trim if somehow long.
    final_frames = pad_or_trim(final_frames, TOTAL_FRAMES)
    print(f"Final sequence: {len(final_frames)} frames "
          f"({len(final_frames)/FPS:.3f}s @ {FPS} fps)")

    out_path = OUT_DIR / "final_demo.mp4"
    with VideoWriter(out_path, (W, H), FPS, crf=16) as w:
        for f in final_frames:
            w.write(f)

    dt = time.perf_counter() - t0
    mb = out_path.stat().st_size / 1e6
    print(f"\nwrote {out_path}  ({mb:.1f} MB) in {dt:.1f}s")


if __name__ == "__main__":
    main()
