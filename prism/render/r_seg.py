"""Stage 02 renderer - PIDNet-S terrain semantics (distilled).

    render(packet, state) -> 1280x720 BGR

Layout
    header                     title / model / clip+frame
    left  (big)                camera frame with the prism-7 terrain overlay
    right column               legend, per-class pixel share bars, drivable-fraction
                               sparkline, model card, and the "why this is not object
                               detection" note
All chrome comes from `viz_common` so every stage video reads as one product.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..config import (CFG, IGNORE_INDEX, N_TERRAIN, PROC_H, PROC_W, TERRAIN_CLASSES,
                      TERRAIN_COLORS_BGR, TERRAIN_DRIVE_PRIOR)
from .. import viz_common as vc
from ..types import FramePacket

W, H = 1280, 720
PAD = 12
HEADER_H = 44
FOOTER_H = 24

_DRIVABLE = TERRAIN_DRIVE_PRIOR >= 0.5          # trail + grass


def _shares(label: np.ndarray) -> tuple[np.ndarray, float, float]:
    m = label != IGNORE_INDEX
    n = max(int(m.sum()), 1)
    lm = label[m]
    sh = np.array([(lm == c).sum() / n for c in range(N_TERRAIN)], np.float32)
    drivable = float(sh[_DRIVABLE].sum())
    coverage = float(n / label.size)
    return sh, drivable, coverage


def _timeline_strip(state: dict, shares: np.ndarray, w: int, h: int,
                    idx: int, n_frames: int = 300) -> np.ndarray:
    """Incrementally painted stacked-composition strip over the whole clip."""
    key = (w, h)
    if state.get("seg_strip_key") != key:
        state["seg_strip_key"] = key
        state["seg_strip"] = np.full((h, w, 3), (26, 24, 22), np.uint8)
    strip = state["seg_strip"]
    x0 = int(min(idx, n_frames - 1) / (n_frames - 1) * (w - 1))
    x1 = max(x0 + 1, int(min(idx + 1, n_frames - 1) / (n_frames - 1) * (w - 1)))
    y = 0
    for c in range(N_TERRAIN):
        hh = int(round(float(shares[c]) * h))
        if hh > 0:
            strip[y:min(y + hh, h), x0:x1] = TERRAIN_COLORS_BGR[c]
        y += hh
    return strip


def render(packet: FramePacket, state: dict) -> np.ndarray:
    img = vc.canvas(W, H)
    seg = packet.seg
    frame = packet.rgb
    if frame is None:
        frame = np.zeros((PROC_H, PROC_W, 3), np.uint8)

    if seg is None:
        vc.header(img, W, "prism", "02 - TERRAIN SEGMENTATION",
                  f"{packet.clip_id}  frame {packet.idx:03d}")
        vc.text(img, "no segmentation in packet", (PAD + 12, H // 2), 0.6, vc.BAD)
        vc.footer(img, W, H, "PIDNet-S", "")
        return img

    label = seg.label
    shares, drivable, coverage = _shares(label)

    # ---------------------------------------------------------------- history
    hist = state.setdefault("seg_drivable", [])
    hist.append(drivable)
    if len(hist) > 300:
        del hist[:-300]
    conf_hist = state.setdefault("seg_conf", [])
    m = label != IGNORE_INDEX
    mean_conf = float(seg.prob_max[m].mean()) if m.any() else 0.0
    conf_hist.append(mean_conf)
    if len(conf_hist) > 300:
        del conf_hist[:-300]

    model_name = state.get("seg_model_name", "PIDNet-S (distilled)")
    n_params = state.get("seg_params", 7_717_839)
    lat_ms = packet.timings_ms.get("seg", state.get("seg_ms", 0.0))
    agree = state.get("seg_agree_miou", None)
    backend = state.get("seg_backend", "student")

    # ---------------------------------------------------------------- header
    vc.header(img, W, "prism", "02 - TERRAIN SEGMENTATION",
              f"{packet.clip_id}   frame {packet.idx:03d}/{300}   t={packet.t:5.2f}s")

    # ---------------------------------------------------------------- left: overlay
    lw = 860
    lx, ly = PAD, HEADER_H + PAD
    lh = 512                       # sized so the 16:9 frame fills the panel exactly
    inner = vc.panel(img, lx, ly, lw, lh, "terrain overlay",
                     "prism-7 classes, alpha 0.55 over the camera frame")
    color = vc.colorize_terrain(label)
    ov = vc.overlay(frame, color, 0.55, mask=m)
    ov = ov.copy()
    ov[~m] = (frame[~m].astype(np.float32) * 0.35).astype(np.uint8)   # dim the ego mask
    # boundary of the drivable region, so the judge can see what the net committed to
    drive = np.isin(label, np.nonzero(_DRIVABLE)[0]).astype(np.uint8)
    cont, _ = cv2.findContours(drive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, [c for c in cont if cv2.contourArea(c) > 400], -1,
                     (255, 255, 255), 1, cv2.LINE_AA)
    vc.blit(img, ov, inner)

    # small inset: raw camera frame, so the overlay is always checkable
    iw = 210
    ih = int(iw * frame.shape[0] / frame.shape[1])
    ix = inner[0] + inner[2] - iw - 10
    iy = inner[1] + inner[3] - ih - 10
    img[iy:iy + ih, ix:ix + iw] = cv2.resize(frame, (iw, ih), interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (ix, iy), (ix + iw, iy + ih), vc.EDGE, 1)
    vc.text(img, "camera", (ix + 6, iy + 14), 0.36, vc.TEXT_DIM)

    # ego-mask caption
    vc.text(img, "dimmed = ego mask (traced RC chassis + watermark), label 255",
            (inner[0] + 10, inner[1] + inner[3] - 10), 0.36, vc.TEXT_DIM)

    # ---------------------------------------------------------------- left: timeline
    ty = ly + lh + PAD
    th = H - FOOTER_H - PAD - ty
    tinner = vc.panel(img, lx, ty, lw, th, "clip timeline",
                      "stacked class composition per frame, sky at the top")
    strip = _timeline_strip(state, shares, tinner[2] - 20, tinner[3] - 22, packet.idx)
    img[tinner[1] + 4:tinner[1] + 4 + strip.shape[0],
        tinner[0] + 10:tinner[0] + 10 + strip.shape[1]] = strip
    cv2.rectangle(img, (tinner[0] + 10, tinner[1] + 4),
                  (tinner[0] + 10 + strip.shape[1], tinner[1] + 4 + strip.shape[0]),
                  vc.EDGE, 1)
    cur_x = tinner[0] + 10 + int(min(packet.idx, 299) / 299 * (strip.shape[1] - 1))
    cv2.line(img, (cur_x, tinner[1] + 4), (cur_x, tinner[1] + 4 + strip.shape[0]),
             (255, 255, 255), 1)
    vc.text(img, "0 s", (tinner[0] + 10, tinner[1] + tinner[3] - 4), 0.34, vc.TEXT_DIM)
    vc.text(img, "10 s", (tinner[0] + 10 + strip.shape[1] - 26,
                          tinner[1] + tinner[3] - 4), 0.34, vc.TEXT_DIM)

    # ---------------------------------------------------------------- right column
    rx = lx + lw + PAD
    rw = W - rx - PAD
    ry = HEADER_H + PAD

    # --- legend + per-class share bars
    ph = 214
    inn = vc.panel(img, rx, ry, rw, ph, "classes", "pixel share this frame")
    yy = inn[1] + 16
    for c in range(N_TERRAIN):
        col = tuple(int(v) for v in TERRAIN_COLORS_BGR[c])
        cv2.rectangle(img, (inn[0] + 10, yy - 8), (inn[0] + 21, yy + 3), col, -1)
        cv2.rectangle(img, (inn[0] + 10, yy - 8), (inn[0] + 21, yy + 3), vc.EDGE, 1)
        vc.text(img, TERRAIN_CLASSES[c], (inn[0] + 28, yy + 2), 0.39,
                vc.TEXT if shares[c] > 0.005 else vc.TEXT_DIM)
        bx = inn[0] + 112
        bw = inn[2] - 112 - 52
        vc.bar_meter(img, bx, yy - 8, bw, 11, float(shares[c]), color=col)
        vc.text(img, f"{shares[c]*100:5.1f}%", (bx + bw + 6, yy + 2), 0.36, vc.TEXT_DIM)
        yy += 22
    vc.text(img, f"scene coverage {coverage*100:.0f}%  (rest = ego mask)",
            (inn[0] + 10, yy + 6), 0.35, vc.TEXT_DIM)
    ry += ph + PAD

    # --- drivable fraction sparkline + confidence
    ph = 146
    inn = vc.panel(img, rx, ry, rw, ph, "drivable fraction",
                   "trail + grass share")
    vc.badge(img, inn[0] + 10, inn[1] + 6, f"{drivable*100:.1f}%",
             vc.OK if drivable > 0.35 else (vc.WARN if drivable > 0.15 else vc.BAD), 0.62)
    vc.text(img, "mean class confidence", (inn[0] + 124, inn[1] + 18), 0.35, vc.TEXT_DIM)
    vc.bar_meter(img, inn[0] + 124, inn[1] + 24, inn[2] - 142, 11, mean_conf,
                 color=vc.ACCENT2)
    vc.text(img, f"{mean_conf:.2f}", (inn[0] + 124, inn[1] + 50), 0.4, vc.TEXT)
    sy = inn[1] + 60
    vc.sparkline(img, inn[0] + 10, sy, inn[2] - 20, inn[3] - 62 - 18,
                 hist, color=vc.ACCENT, lo=0.0, hi=1.0)
    vc.text(img, f"last {len(hist)} frames  (0-100%)",
            (inn[0] + 10, inn[1] + inn[3] - 5), 0.34, vc.TEXT_DIM)
    ry += ph + PAD

    # --- model card
    ph = 112
    inn = vc.panel(img, rx, ry, rw, ph, "model", backend)
    lines = [
        f"{model_name}",
        f"{n_params/1e6:.2f} M params - input 512x288 - stride 8",
        f"latency {lat_ms:.1f} ms/frame ({state.get('seg_device','cuda')} fp16)",
    ]
    if agree is not None:
        lines.append(f"teacher-agreement mIoU {agree*100:.1f}% (not ground truth)")
    else:
        lines.append("SegFormer-B0 / ADE20K remapped to prism-7")
    yy = inn[1] + 17
    for i, s in enumerate(lines):
        vc.text(img, s, (inn[0] + 10, yy), 0.375, vc.TEXT if i == 0 else vc.TEXT_DIM)
        yy += 17
    ry += ph + PAD

    # --- why-this-matters note
    vc.rounded_note(img, rx, ry, rw, [
        "A detector answers 'what object is that?'.",
        "Navigation needs 'can the wheels go there?' -",
        "so every pixel gets a surface class, including",
        "the empty ground a detector emits no box for.",
        "Grass is drivable but soft; a wall and a parked",
        "car are the same thing to the chassis.",
    ], title="WHY TERRAIN, NOT OBJECTS", accent=vc.ACCENT)

    # ---------------------------------------------------------------- footer
    vc.footer(img, W, H,
              "labels are pseudo-labels distilled from an ADE20K teacher, not RELLIS-3D/GOOSE ground truth",
              f"{PROC_W}x{PROC_H} dense map   7 classes + ignore")
    return img


# ------------------------------------------------------------------------------ self-test
if __name__ == "__main__":
    import time
    from ..config import WORK_DIR
    from .. import io_utils
    from ..models.segmentation import SegStage
    from ..types import FramePacket

    dev = io_utils.device()
    stage = SegStage(device=dev)
    state = dict(seg_model_name=stage.model_name, seg_params=stage.n_params,
                 seg_backend=stage.backend, seg_device=stage.device,
                 seg_agree_miou=stage.agree_miou)
    stage.reset()
    frames = [f for _, f in io_utils.read_frames("clip_01", max_frames=120)]
    out = None
    t0 = time.perf_counter()
    for i, f in enumerate(frames):
        pk = FramePacket(clip_id="clip_01", idx=i, t=i / 30.0, rgb=f)
        pk = stage(pk)
        out = render(pk, state)
    dt = (time.perf_counter() - t0) / len(frames) * 1000
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(WORK_DIR / "preview_seg.png"), out)
    print(f"render {out.shape} dtype={out.dtype}  "
          f"{dt:.1f} ms/frame incl. inference -> {WORK_DIR / 'preview_seg.png'}")
    stage.close()
