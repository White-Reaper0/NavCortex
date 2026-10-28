"""Shared rendering primitives so every stage video looks like one product.

Dark instrument-panel styling, one type scale, one accent colour, consistent panel
chrome. All drawing is OpenCV BGR uint8.
"""
from __future__ import annotations
from typing import Sequence, Optional
import numpy as np
import cv2

from .config import (TERRAIN_COLORS_BGR, TRAV_COLORS_BGR, TERRAIN_CLASSES, TRAV_CLASSES,
                     DECISION_COLORS_BGR, DECISIONS)

# ------------------------------------------------------------------ palette
BG = (18, 16, 14)            # near-black page
PANEL = (32, 29, 26)         # panel fill
PANEL_HI = (46, 42, 38)      # panel header
EDGE = (70, 64, 58)          # hairline
TEXT = (232, 230, 226)       # primary text
TEXT_DIM = (150, 146, 140)   # secondary text
ACCENT = (90, 190, 255)      # prism amber-orange (BGR)
ACCENT2 = (200, 230, 120)    # cool secondary
OK = (110, 230, 120)
WARN = (60, 200, 255)
BAD = (60, 60, 235)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_B = cv2.FONT_HERSHEY_DUPLEX


def canvas(w: int, h: int, color=BG) -> np.ndarray:
    c = np.empty((h, w, 3), np.uint8)
    c[:] = color
    return c


def text(img, s, org, scale=0.5, color=TEXT, thick=1, font=FONT, shadow=True):
    x, y = int(org[0]), int(org[1])
    if shadow:
        cv2.putText(img, s, (x + 1, y + 1), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), font, scale, color, thick, cv2.LINE_AA)
    return img


def text_size(s, scale=0.5, thick=1, font=FONT):
    return cv2.getTextSize(s, font, scale, thick)[0]


def panel(img, x, y, w, h, title: str = "", subtitle: str = "", accent=ACCENT, header_h: int = 26):
    """Draw panel chrome and return the inner content rectangle (x, y, w, h)."""
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), EDGE, 1)
    if title:
        cv2.rectangle(img, (x, y), (x + w, y + header_h), PANEL_HI, -1)
        cv2.line(img, (x, y + header_h), (x + w, y + header_h), EDGE, 1)
        cv2.rectangle(img, (x, y), (x + 3, y + header_h), accent, -1)
        text(img, title.upper(), (x + 10, y + header_h - 8), 0.44, TEXT, 1, FONT_B)
        if subtitle:
            tw = text_size(title.upper(), 0.44, 1, FONT_B)[0]
            text(img, subtitle, (x + 18 + tw, y + header_h - 8), 0.38, TEXT_DIM, 1)
        return (x + 1, y + header_h + 1, w - 2, h - header_h - 2)
    return (x + 1, y + 1, w - 2, h - 2)


def blit(dst, src, rect, keep_aspect=True, interp=cv2.INTER_AREA):
    """Draw `src` into `rect` = (x, y, w, h) of `dst`, letterboxed."""
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return dst
    sh, sw = src.shape[:2]
    if keep_aspect:
        s = min(w / sw, h / sh)
        nw, nh = max(1, int(sw * s)), max(1, int(sh * s))
    else:
        nw, nh = w, h
    r = cv2.resize(src, (nw, nh), interpolation=interp)
    if r.ndim == 2:
        r = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
    ox, oy = x + (w - nw) // 2, y + (h - nh) // 2
    dst[oy:oy + nh, ox:ox + nw] = r
    return dst


# ------------------------------------------------------------------ colour maps

def colorize_depth(depth_m: np.ndarray, vmin: Optional[float] = None,
                   vmax: Optional[float] = None, valid: Optional[np.ndarray] = None) -> np.ndarray:
    """Near = warm, far = cool (inverse-depth normalised, robust percentiles)."""
    d = depth_m.astype(np.float32)
    m = np.isfinite(d) & (d > 1e-3)
    if valid is not None:
        m &= valid
    if not m.any():
        return np.zeros((*d.shape, 3), np.uint8)
    inv = np.zeros_like(d)
    inv[m] = 1.0 / d[m]
    lo = np.percentile(inv[m], 2) if vmin is None else 1.0 / max(vmax, 1e-3)
    hi = np.percentile(inv[m], 98) if vmax is None else 1.0 / max(vmin, 1e-3)
    n = np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~m] = (40, 38, 35)
    return img


def colorize_height(h: np.ndarray, lo: float = -0.30, hi: float = 0.60,
                    invalid: Optional[np.ndarray] = None) -> np.ndarray:
    """Height above local ground: blue below, green at grade, red above."""
    n = np.clip((h - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if invalid is not None:
        img[invalid] = (55, 52, 48)
    return img


def colorize_conf(c: np.ndarray) -> np.ndarray:
    """Confidence: dark red (0) -> yellow -> bright green (1)."""
    n = np.clip(c, 0, 1)
    img = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_SUMMER)
    dark = (n[..., None] * 0.75 + 0.25)
    return (img.astype(np.float32) * dark).astype(np.uint8)


def colorize_label(label: np.ndarray, colors: np.ndarray) -> np.ndarray:
    lut = np.zeros((256, 3), np.uint8)
    lut[:len(colors)] = colors
    lut[255] = (60, 60, 60)
    return lut[label]


def colorize_terrain(label: np.ndarray) -> np.ndarray:
    return colorize_label(label, TERRAIN_COLORS_BGR)


def colorize_trav(label: np.ndarray) -> np.ndarray:
    return colorize_label(label, TRAV_COLORS_BGR)


def overlay(base_bgr: np.ndarray, color_map: np.ndarray, alpha: float = 0.55,
            mask: Optional[np.ndarray] = None) -> np.ndarray:
    out = cv2.addWeighted(base_bgr, 1 - alpha, color_map, alpha, 0)
    if mask is not None:
        out[~mask] = base_bgr[~mask]
    return out


# ------------------------------------------------------------------ widgets

def legend(img, x, y, names: Sequence[str], colors: np.ndarray, scale=0.38,
           swatch=11, gap=15, vertical=True, title: str = "", h_gap: int = 0):
    if title:
        text(img, title, (x, y), scale, TEXT_DIM, 1)
        y += 14
    # Horizontal mode used to hardcode 90 px per entry, which silently clipped any
    # legend with labels longer than that. Space entries by their measured text width
    # instead; `h_gap` overrides if a caller wants a fixed pitch.
    if not vertical:
        widths = [text_size(n, scale)[0] + swatch + 14 for n in names]
        offs, acc = [], 0
        for wdt in widths:
            offs.append(acc)
            acc += (h_gap if h_gap else wdt)
    for i, n in enumerate(names):
        cx, cy = (x, y + i * gap) if vertical else (x + offs[i], y)
        c = tuple(int(v) for v in colors[i])
        cv2.rectangle(img, (cx, cy - swatch + 2), (cx + swatch, cy + 2), c, -1)
        cv2.rectangle(img, (cx, cy - swatch + 2), (cx + swatch, cy + 2), EDGE, 1)
        text(img, n, (cx + swatch + 6, cy), scale, TEXT, 1)
    return img


def colorbar(img, x, y, w, h, cmap=cv2.COLORMAP_TURBO, lo_lab="", hi_lab="",
             title="", reverse=False):
    ramp = np.linspace(255, 0, w).astype(np.uint8) if reverse else np.linspace(0, 255, w).astype(np.uint8)
    bar = cv2.applyColorMap(np.tile(ramp, (h, 1)), cmap)
    img[y:y + h, x:x + w] = bar
    cv2.rectangle(img, (x, y), (x + w, y + h), EDGE, 1)
    if title:
        text(img, title, (x, y - 5), 0.36, TEXT_DIM, 1)
    if lo_lab:
        text(img, lo_lab, (x, y + h + 12), 0.34, TEXT_DIM, 1)
    if hi_lab:
        text(img, hi_lab, (x + w - text_size(hi_lab, 0.34)[0], y + h + 12), 0.34, TEXT_DIM, 1)
    return img


def bar_meter(img, x, y, w, h, value: float, color=ACCENT, label: str = "",
              lo: float = 0.0, hi: float = 1.0, warn_at: Optional[float] = None):
    v = float(np.clip((value - lo) / max(hi - lo, 1e-6), 0, 1))
    cv2.rectangle(img, (x, y), (x + w, y + h), (52, 48, 44), -1)
    c = color
    if warn_at is not None and value >= warn_at:
        c = BAD if value >= warn_at + (hi - warn_at) * 0.5 else WARN
    cv2.rectangle(img, (x, y), (x + int(w * v), y + h), c, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), EDGE, 1)
    if label:
        text(img, label, (x, y - 4), 0.36, TEXT_DIM, 1)
    return img


def badge(img, x, y, label: str, color, scale=0.5, pad=7, filled=True):
    tw, th = text_size(label, scale, 1, FONT_B)
    w, h = tw + 2 * pad, th + 2 * pad
    if filled:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        text(img, label, (x + pad, y + h - pad - 1), scale, (16, 16, 16), 1, FONT_B, shadow=False)
    else:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text(img, label, (x + pad, y + h - pad - 1), scale, color, 1, FONT_B)
    return (w, h)


def decision_badge(img, x, y, kind: int, scale=0.85):
    return badge(img, x, y, DECISIONS[kind], DECISION_COLORS_BGR[kind], scale, pad=11)


def sparkline(img, x, y, w, h, series: Sequence[float], color=ACCENT,
              lo: Optional[float] = None, hi: Optional[float] = None, fill=True):
    s = np.asarray(list(series), np.float32)
    cv2.rectangle(img, (x, y), (x + w, y + h), (26, 24, 22), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), EDGE, 1)
    if s.size < 2:
        return img
    lo = float(np.nanmin(s)) if lo is None else lo
    hi = float(np.nanmax(s)) if hi is None else hi
    n = np.clip((s - lo) / max(hi - lo, 1e-6), 0, 1)
    xs = np.linspace(x + 1, x + w - 1, s.size).astype(np.int32)
    ys = (y + h - 1 - n * (h - 2)).astype(np.int32)
    pts = np.stack([xs, ys], 1)
    if fill:
        poly = np.concatenate([pts, [[x + w - 1, y + h - 1], [x + 1, y + h - 1]]]).astype(np.int32)
        ov = img.copy()
        cv2.fillPoly(ov, [poly], tuple(int(c * 0.45) for c in color))
        cv2.addWeighted(ov, 0.6, img, 0.4, 0, img)
    cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)
    return img


def header(img, w, title="prism", subtitle="", right="", h=44):
    cv2.rectangle(img, (0, 0), (w, h), (26, 23, 21), -1)
    cv2.line(img, (0, h), (w, h), EDGE, 1)
    cv2.rectangle(img, (0, 0), (4, h), ACCENT, -1)
    text(img, title, (16, h - 15), 0.74, TEXT, 1, FONT_B)
    tw = text_size(title, 0.74, 1, FONT_B)[0]
    if subtitle:
        text(img, subtitle, (26 + tw, h - 16), 0.44, ACCENT, 1)
    if right:
        text(img, right, (w - text_size(right, 0.42)[0] - 16, h - 16), 0.42, TEXT_DIM, 1)
    return img


def footer(img, w, h_img, left="", right="", h=24):
    y = h_img - h
    cv2.rectangle(img, (0, y), (w, h_img), (24, 22, 20), -1)
    cv2.line(img, (0, y), (w, y), EDGE, 1)
    if left:
        text(img, left, (14, h_img - 8), 0.36, TEXT_DIM, 1)
    if right:
        text(img, right, (w - text_size(right, 0.36)[0] - 14, h_img - 8), 0.36, TEXT_DIM, 1)
    return img


def draw_grid_dots(img, rect, values_bgr: np.ndarray, valid: np.ndarray,
                   elev: Optional[np.ndarray] = None, dot: int = 3, elev_px: float = 0.0):
    """Render a BEV grid as coloured dots with optional vertical displacement.

    values_bgr (H,W,3) uint8, valid (H,W) bool, elev (H,W) float metres.
    Row 0 of the grid is the farthest cell; ego sits at the bottom centre, so nearer
    cells must paint over farther ones. Fully vectorised: cells are sorted far-to-near
    and written in one fancy-index assignment, where NumPy keeps the last write - which
    is the nearest cell. A per-cell Python loop here costs ~0.5 s/frame at 128x192.
    """
    x, y, w, h = rect
    gh, gw = valid.shape
    ys, xs = np.nonzero(valid)
    if ys.size == 0:
        return img
    order = np.argsort(ys, kind="stable")          # far (small row) first, near last
    ys, xs = ys[order], xs[order]

    cw, ch = w / gw, h / gh
    px = (x + (xs + 0.5) * cw).astype(np.int32)
    py = (y + (ys + 0.5) * ch).astype(np.int32)
    if elev is not None and elev_px:
        e = elev[ys, xs]
        e = np.where(np.isfinite(e), np.clip(e, -0.6, 1.2), 0.0)
        py = py - (e * elev_px).astype(np.int32)

    cols = values_bgr[ys, xs]
    r = int(max(dot, 1))
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = (dx * dx + dy * dy) <= r * r
    dy, dx = dy[disc].ravel(), dx[disc].ravel()

    Hh, Ww = img.shape[:2]
    for oy, ox in zip(dy, dx):
        yy = py + oy
        xx = px + ox
        keep = (yy >= 0) & (yy < Hh) & (xx >= 0) & (xx < Ww)
        img[yy[keep], xx[keep]] = cols[keep]
    return img


def rounded_note(img, x, y, w, lines: Sequence[str], title="", accent=ACCENT, lh=15, pad=9):
    h = pad * 2 + (14 if title else 0) + lh * len(lines)
    cv2.rectangle(img, (x, y), (x + w, y + h), (28, 26, 24), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), EDGE, 1)
    cv2.rectangle(img, (x, y), (x + 3, y + h), accent, -1)
    yy = y + pad + 10
    if title:
        text(img, title, (x + pad, yy), 0.4, accent, 1, FONT_B)
        yy += 14
    for ln in lines:
        text(img, ln, (x + pad, yy), 0.37, TEXT_DIM, 1)
        yy += lh
    return h


TERRAIN_LEGEND = (TERRAIN_CLASSES, TERRAIN_COLORS_BGR)
TRAV_LEGEND = (TRAV_CLASSES, TRAV_COLORS_BGR)
