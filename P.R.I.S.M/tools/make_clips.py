"""Cut 5 x 10s demo clips out of the long POV source video.

Windows were chosen by inspecting a 20s-interval contact sheet of the source:
daylight, off-road/mixed terrain, no picture-in-picture inset, minimal burned-in
text, and no hard editorial cut inside the window.
"""
from __future__ import annotations
import subprocess, sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prism.config import VIDEO_IN, CLIP_DIR, CLIP_W, CLIP_H, CLIP_FPS, CLIP_SECONDS

# (clip_id, start_seconds, scene, lighting, what the clip is meant to stress)
WINDOWS = [
    ("clip_01",  90.0, "daylight gravel park trail between grass banks, pedestrians and bicycles ahead",
     "daylight", "mixed trail/grass terrain, late-appearing dynamic obstacles"),
    ("clip_02", 136.0, "wide gravel path, iron railings on the right, grass verges and tree line",
     "daylight", "clean geometry, a long straight run, a rigid lateral boundary"),
    ("clip_03", 420.5, "dusk earth path through parkland past a brick wall and grass banks",
     "dusk", "falling light, low-texture ground, raised verges either side"),
    ("clip_04", 511.5, "low-light service yard: parked vans and cars, kerbs, brick wall, grass verge",
     "low light", "static vehicle obstacles and kerb steps at low illumination"),
    ("clip_05", 665.0, "low-light path with parked cars, railings and a pedestrian crossing ahead",
     "low light", "a moving pedestrian plus degraded depth confidence"),
]


def cut(clip_id: str, start: float) -> Path:
    out = CLIP_DIR / f"{clip_id}.mp4"
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-i", str(VIDEO_IN), "-t", f"{CLIP_SECONDS:.3f}",
        "-an",
        "-vf", f"scale={CLIP_W}:{CLIP_H}:flags=bicubic,fps={CLIP_FPS}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "16",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def main() -> None:
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    meta = []
    for clip_id, start, desc, light, stress in WINDOWS:
        p = cut(clip_id, start)
        info = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate",
             "-count_frames", "-of", "json", str(p)],
            capture_output=True, text=True, check=True).stdout
        st = json.loads(info)["streams"][0]
        meta.append({"clip_id": clip_id, "source_start_s": start, "scene": desc,
                     "lighting": light, "stresses": stress,
                     "path": str(p), "width": st["width"], "height": st["height"],
                     "frames": int(st["nb_read_frames"]), "fps": st["avg_frame_rate"]})
        print(f"{clip_id}: {st['width']}x{st['height']} {st['nb_read_frames']} frames  <- t={start}s  ({desc})")
    (CLIP_DIR / "clips.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
