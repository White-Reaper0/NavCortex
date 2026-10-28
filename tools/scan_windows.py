"""Score every candidate 10 s window in the source video.

Rejects windows containing hard editorial cuts, fades to black, burned-in caption
text, or the picture-in-picture inset the creator overlays on some segments, and
prefers windows with steady forward motion over mixed terrain.
Writes work/window_stats.npz and work/window_rank.json.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prism.config import VIDEO_IN, WORK_DIR, CLIP_SECONDS

SAMPLE_FPS = 2.0
TW, TH = 480, 270


def frame_stats(bgr: np.ndarray, prev_gray: np.ndarray | None):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, W = gray.shape

    # brightness of the scene region (exclude the vehicle body band at the bottom)
    bright = float(gray[: int(0.80 * H)].mean())

    # burned-in captions are saturated red/pink or yellow text
    h, s, v = hsv[..., 0].astype(np.int16), hsv[..., 1].astype(np.int16), hsv[..., 2].astype(np.int16)
    red = (((h < 10) | (h > 165)) & (s > 140) & (v > 140))
    yellow = ((h > 20) & (h < 38) & (s > 150) & (v > 190))
    text_px = float((red | yellow).mean())

    # picture-in-picture inset: long straight horizontal edges inside the lower band
    band = gray[int(0.52 * H): int(0.94 * H)]
    edges = cv2.Canny(band, 70, 170)
    row_energy = edges.sum(axis=1) / 255.0
    pip = float(np.sort(row_energy)[-3:].mean() / band.shape[1])

    # scene cut / fade detection
    if prev_gray is None:
        cut = 0.0
        flow_mag = 0.0
    else:
        d = np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))
        cut = float(d.mean())
        f = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 2, 15, 2, 5, 1.1, 0)
        flow_mag = float(np.linalg.norm(f, axis=2).mean())

    # terrain variety: spread of hue in the lower-middle (ground) region
    ground = hsv[int(0.45 * H): int(0.80 * H)]
    variety = float(ground[..., 0].std() + 0.3 * ground[..., 1].std())
    return dict(bright=bright, text=text_px, pip=pip, cut=cut, flow=flow_mag,
                variety=variety), gray


def main() -> None:
    cap = cv2.VideoCapture(str(VIDEO_IN))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(src_fps / SAMPLE_FPS)))
    print(f"src {src_fps:.2f} fps, {total} frames, sampling every {step}")

    keys = ["bright", "text", "pip", "cut", "flow", "variety"]
    rows, times = [], []
    prev = None
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, f = cap.retrieve()
            if ok:
                f = cv2.resize(f, (TW, TH))
                st, prev = frame_stats(f, prev)
                rows.append([st[k] for k in keys])
                times.append(idx / src_fps)
        idx += 1
        if idx % 12000 == 0:
            print(f"  {idx}/{total}", flush=True)
    cap.release()

    A = np.asarray(rows, np.float32)
    t = np.asarray(times, np.float32)
    np.savez(WORK_DIR / "window_stats.npz", stats=A, times=t, keys=np.array(keys))

    n_win = int(round(CLIP_SECONDS * SAMPLE_FPS))
    cands = []
    for i in range(0, len(A) - n_win):
        w = A[i:i + n_win]
        bright, text, pip, cut, flow, variety = (w[:, j] for j in range(6))
        # hard rejects
        if bright.min() < 55:            # fade to black / night
            continue
        if cut[1:].max() > 34:           # editorial cut inside the window
            continue
        if text.mean() > 0.0016:         # persistent caption text
            continue
        if pip.mean() > 0.115:           # picture-in-picture inset present
            continue
        if flow[1:].mean() < 1.4:        # vehicle basically stationary
            continue
        score = (0.9 * variety.mean() / 20.0
                 + 0.8 * min(flow.mean(), 9.0) / 9.0
                 + 0.5 * min(bright.mean(), 150.0) / 150.0
                 - 2.5 * pip.mean()
                 - 400.0 * text.mean()
                 - 0.03 * cut[1:].max())
        cands.append(dict(t=float(t[i]), score=float(score),
                          bright=float(bright.mean()), text=float(text.mean()),
                          pip=float(pip.mean()), cutmax=float(cut[1:].max()),
                          flow=float(flow.mean()), variety=float(variety.mean())))

    cands.sort(key=lambda c: -c["score"])
    # keep the best window from each non-overlapping neighbourhood
    picked = []
    for c in cands:
        if all(abs(c["t"] - p["t"]) > 14.0 for p in picked):
            picked.append(c)
        if len(picked) >= 30:
            break
    (WORK_DIR / "window_rank.json").write_text(json.dumps(picked, indent=2))
    for c in picked[:30]:
        print(f"t={c['t']:7.1f}  score={c['score']:.3f} bright={c['bright']:5.1f} "
              f"flow={c['flow']:4.1f} var={c['variety']:5.1f} pip={c['pip']:.3f} "
              f"text={c['text']:.5f} cut={c['cutmax']:4.1f}")


if __name__ == "__main__":
    main()
