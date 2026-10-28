"""Build the ego-vehicle mask for the source rig.

The camera is rigidly mounted on the RC chassis for the whole 18-minute source video,
so the vehicle silhouette and the channel watermark occupy exactly the same pixels in
every frame. Averaging 300 frames sampled across the whole video (work/ego_median.png)
renders the scene to a blur and leaves the ego perfectly sharp, which is what the
silhouette below was traced from.

Automatic segmentation of that median was tried first (cross-scene pixel variance, and
median-image gradient energy) and both under-segment: the white bodywork is glossy, so
its brightness varies as much as the scene does. Since the rig never moves, an explicit
verified silhouette is exact where a heuristic is not.

Writes work/ego_mask.png - 255 = usable scene pixel, 0 = ego / watermark.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prism.config import CLIP_IDS, PROC_W, PROC_H, WORK_DIR
from prism.io_utils import clip_path

# Silhouette traced from work/ego_median.png at 640x360, as (x, y_top) control points.
# Between the control points the top edge is linearly interpolated; everything below is ego.
SILHOUETTE_640x360 = [
    (0, 360), (26, 344), (56, 302), (84, 276), (104, 264), (150, 256), (198, 254),
    (214, 268), (248, 262), (300, 254), (340, 254), (392, 262), (426, 268),
    (444, 254), (492, 254), (538, 256), (568, 272), (598, 302), (626, 342), (640, 360),
]
WATERMARK_640x360 = (0, 0, 56, 32)      # x0, y0, x1, y1


def build_mask(w: int = PROC_W, h: int = PROC_H) -> np.ndarray:
    sx, sy = w / 640.0, h / 360.0
    pts = [(x * sx, y * sy) for x, y in SILHOUETTE_640x360]
    poly = [(0, h)] + pts + [(w, h)]
    ego = np.zeros((h, w), np.uint8)
    cv2.fillPoly(ego, [np.array(poly, np.int32)], 1)
    x0, y0, x1, y1 = WATERMARK_640x360
    ego[int(y0 * sy):int(y1 * sy) + 1, int(x0 * sx):int(x1 * sx) + 1] = 1
    return ((1 - ego) * 255).astype(np.uint8)


def main() -> None:
    mask = build_mask()
    out = WORK_DIR / "ego_mask.png"
    cv2.imwrite(str(out), mask)
    print(f"scene pixels kept: {mask.mean()/255*100:.1f}%  ->  {out}")

    tiles = []
    for cid in CLIP_IDS:
        cap = cv2.VideoCapture(str(clip_path(cid)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
        ok, f = cap.read()
        cap.release()
        if not ok:
            continue
        f = cv2.resize(f, (PROC_W, PROC_H))
        ov = f.copy()
        ov[mask == 0] = (0, 0, 255)
        t = cv2.addWeighted(f, 0.45, ov, 0.55, 0)
        cv2.putText(t, cid, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    med = cv2.imread(str(WORK_DIR / "ego_median.png"))
    if med is not None:
        ov = med.copy()
        ov[mask == 0] = (0, 0, 255)
        tiles.append(cv2.addWeighted(med, 0.45, ov, 0.55, 0))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles) - len(tiles) % 3, 3)]
    if rows:
        cv2.imwrite(str(WORK_DIR / "ego_mask_check.png"), np.vstack(rows))
        print(f"check image -> {WORK_DIR / 'ego_mask_check.png'}")


if __name__ == "__main__":
    main()
