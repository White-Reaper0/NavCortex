"""Renderer for stage 05 - lightweight visual place recognition.

Panels: the query frame, the retrieved best-match frame, the similarity score against
its decision threshold, a similarity-vs-database-index heat strip, database size, and
a note on what place recognition is for (relocalisation, drift correction) as opposed
to object detection.

`state` keeps a thumbnail ring buffer so the retrieved frame can be shown. Frames from
*other* clips are pulled lazily straight from the mp4 and cached.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import cv2

from ..config import CFG, CLIP_FPS
from ..types import FramePacket
from .. import viz_common as V
from .. import io_utils

W, H = 1280, 720
THUMB = (320, 180)


def _g(o, name, default):
    v = getattr(o, name, default)
    return default if v is None else v


# --------------------------------------------------------------------- thumbs

def _thumbs(state: dict) -> dict:
    return state.setdefault("_vpr_thumbs", {})


def _remember(state: dict, clip_id: str, idx: int, frame: np.ndarray) -> None:
    t = _thumbs(state)
    if (clip_id, idx) not in t:
        t[(clip_id, idx)] = cv2.resize(frame, THUMB, interpolation=cv2.INTER_AREA)


def _fetch(state: dict, clip_id: str, idx: int) -> Optional[np.ndarray]:
    """Thumbnail for (clip, frame). Falls back to seeking the source mp4 once."""
    t = _thumbs(state)
    key = (clip_id, idx)
    if key in t:
        return t[key]
    p = io_utils.clip_path(clip_id)
    if not p.exists():
        return None
    caps = state.setdefault("_vpr_caps", {})
    cap = caps.get(clip_id)
    if cap is None:
        cap = cv2.VideoCapture(str(p))
        caps[clip_id] = cap
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, fr = cap.read()
    if not ok:
        return None
    t[key] = cv2.resize(fr, THUMB, interpolation=cv2.INTER_AREA)
    return t[key]


# --------------------------------------------------------------------- widgets

def _sim_strip(img, rect, sims: np.ndarray, best: int, q_idx: int, excl: int,
               clip_bounds: list[tuple[str, int, int]]):
    """Similarity vs database index as a heat strip with the winner marked."""
    x, y, w, h = rect
    strip_h = h - 40
    cv2.rectangle(img, (x, y), (x + w, y + h), (24, 22, 20), -1)
    n = len(sims)
    if n == 0:
        V.text(img, "database empty - first frames of the clip",
               (x + 12, y + h // 2), 0.42, V.TEXT_DIM)
        return
    # resample the similarity vector to the strip width
    xs = np.linspace(0, n - 1, w - 4)
    s = np.interp(xs, np.arange(n), sims).astype(np.float32)
    lo, hi = 0.0, 1.0
    norm = np.clip((s - lo) / (hi - lo), 0, 1)
    bar = cv2.applyColorMap((norm * 255).astype(np.uint8).reshape(1, -1),
                            cv2.COLORMAP_INFERNO)
    img[y + 2:y + 2 + strip_h, x + 2:x + 2 + len(s)] = np.repeat(bar, strip_h, axis=0)

    # profile line on top of the heat
    ys = (y + 2 + strip_h - 1 - norm * (strip_h - 2)).astype(np.int32)
    pts = np.stack([np.arange(len(s)) + x + 2, ys], 1)
    cv2.polylines(img, [pts], False, (245, 245, 245), 1, cv2.LINE_AA)

    def to_x(i):
        return int(x + 2 + (i / max(n - 1, 1)) * (w - 5))

    # temporal exclusion window (never eligible)
    e0 = to_x(max(0, n - excl))
    cv2.rectangle(img, (e0, y + 2), (x + w - 2, y + 2 + strip_h), (60, 60, 60), 1)
    ov = img.copy()
    cv2.rectangle(ov, (e0, y + 2), (x + w - 2, y + 2 + strip_h), (35, 33, 30), -1)
    cv2.addWeighted(ov, 0.62, img, 0.38, 0, img)
    V.text(img, f"temporal exclusion ({excl} f)", (min(e0 + 4, x + w - 150), y + 16),
           0.32, V.TEXT_DIM)

    # clip boundaries in the database
    for cid, i0, _ in clip_bounds:
        bx = to_x(i0)
        cv2.line(img, (bx, y + 2), (bx, y + 2 + strip_h), (150, 150, 150), 1)
        V.text(img, cid.replace("clip_", "c"), (bx + 3, y + 2 + strip_h - 5), 0.3, V.TEXT_DIM)

    if 0 <= best < n:
        bx = to_x(best)
        cv2.line(img, (bx, y), (bx, y + strip_h + 6), V.ACCENT, 2, cv2.LINE_AA)
        cv2.circle(img, (bx, y + 2 + int((1 - norm[min(len(norm) - 1,
                   int(best / max(n - 1, 1) * (len(norm) - 1)))]) * strip_h)),
                   4, V.ACCENT, -1, cv2.LINE_AA)
        V.text(img, f"best #{best}", (max(x + 4, bx - 30), y + strip_h + 20), 0.34, V.ACCENT)

    V.text(img, "db index 0", (x + 4, y + h - 8), 0.32, V.TEXT_DIM)
    V.text(img, f"{n}", (x + w - 26, y + h - 8), 0.32, V.TEXT_DIM)
    V.colorbar(img, x + w - 240, y + h - 14, 110, 7, cv2.COLORMAP_INFERNO,
               "cos 0", "1", "")


def _score_bar(img, rect, score: float, thresh: float, is_revisit: bool,
               instant: float, seq_frames: int, note: str,
               distinct: float = 0.0, ratio_thresh: float = 1.35):
    x, y, w, h = rect
    V.text(img, note[:88], (x + 14, y + 15), 0.36,
           V.OK if is_revisit else V.TEXT_DIM)

    bx, by, bw, bh = x + 14, y + 40, w - 250, 22
    V.text(img, f"sequence score {score:.3f}", (bx, by - 8), 0.4, V.TEXT)
    V.text(img, f"single-frame cosine {instant:.3f}", (bx + 200, by - 8), 0.34, V.TEXT_DIM)
    V.text(img, f"distinctiveness {distinct:.2f}x  (need {ratio_thresh:.2f}x)",
           (bx + 390, by - 8), 0.34,
           V.OK if distinct >= ratio_thresh else V.TEXT_DIM)

    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (48, 45, 41), -1)
    f = float(np.clip(score, 0, 1))
    col = V.OK if is_revisit else (V.WARN if score > thresh * 0.97 else V.ACCENT)
    cv2.rectangle(img, (bx, by), (bx + int(bw * f), by + bh), col, -1)
    fi = float(np.clip(instant, 0, 1))       # single-frame score as a thin under-bar
    cv2.rectangle(img, (bx, by + bh + 3), (bx + int(bw * fi), by + bh + 8),
                  (120, 116, 110), -1)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), V.EDGE, 1)
    tx = bx + int(bw * float(np.clip(thresh, 0, 1)))
    cv2.line(img, (tx, by - 5), (tx, by + bh + 11), V.BAD, 2, cv2.LINE_AA)
    V.text(img, f"thr {thresh:.2f}", (min(tx - 22, bx + bw - 50), by + bh + 24), 0.32, V.BAD)
    V.text(img, "0.0", (bx - 2, by + bh + 24), 0.32, V.TEXT_DIM)
    V.text(img, "1.0", (bx + bw + 8, by + bh + 24), 0.32, V.TEXT_DIM)

    V.badge(img, x + w - 190, y + 34, "REVISIT" if is_revisit else "NO REVISIT",
            V.OK if is_revisit else V.TEXT_DIM, 0.52)
    V.text(img, f"winner stable for {seq_frames} frames",
           (x + w - 192, y + 78), 0.33, V.TEXT_DIM)


# --------------------------------------------------------------------- render

def render(packet: FramePacket, state: dict) -> np.ndarray:
    img = V.canvas(W, H)
    pl = packet.place
    if packet.rgb is not None:
        _remember(state, packet.clip_id, packet.idx, packet.rgb)

    V.header(img, W, "prism",
             "05 - PLACE RECOGNITION | GeM + learned whitening on MobileNetV3-Small "
             "(256-D)",
             right=f"{packet.clip_id}   frame {packet.idx:03d}   t={packet.t:5.2f}s")

    # ---------------------------------------------------------------- query
    r = V.panel(img, 14, 52, 616, 376, "query frame", "descriptor computed live")
    if packet.rgb is not None:
        V.blit(img, packet.rgb, (r[0] + 6, r[1] + 6, r[2] - 12, r[3] - 12))

    # ---------------------------------------------------------------- match
    best = int(pl.best_match_idx) if pl else -1
    bclip = _g(pl, "best_clip_id", packet.clip_id) if pl else packet.clip_id
    bframe = int(_g(pl, "best_frame_idx", -1)) if pl else -1
    sub = (f"{bclip} f{bframe}" if best >= 0 else "nothing eligible yet")
    r = V.panel(img, 638, 52, 616, 376, "best database match", sub)
    if best >= 0 and bframe >= 0:
        th = _fetch(state, bclip, bframe)
        if th is not None:
            V.blit(img, th, (r[0] + 6, r[1] + 6, r[2] - 12, r[3] - 12))
            if bclip != packet.clip_id:
                V.badge(img, r[0] + 14, r[1] + 14, f"CROSS-CLIP  {bclip}", V.ACCENT2, 0.42)
        else:
            V.text(img, "frame not retrievable", (r[0] + 20, r[1] + 40), 0.5, V.TEXT_DIM)
    else:
        cv2.rectangle(img, (r[0] + 6, r[1] + 6), (r[0] + r[2] - 6, r[1] + r[3] - 6),
                      (26, 24, 22), -1)
        V.text(img, "no eligible database entry yet",
               (r[0] + 30, r[1] + r[3] // 2), 0.55, V.TEXT_DIM, 1, V.FONT_B)
        V.text(img, f"every entry is inside the {_g(pl,'exclude',45)}-frame temporal "
                    f"exclusion window" if pl else "",
               (r[0] + 30, r[1] + r[3] // 2 + 26), 0.38, V.TEXT_DIM)

    # ---------------------------------------------------------------- sim strip
    r = V.panel(img, 14, 436, 760, 126, "similarity vs database index",
                "cosine, brighter = more similar")
    sims = np.asarray(_g(pl, "sims", np.zeros(0, np.float32)), np.float32) if pl else \
        np.zeros(0, np.float32)
    bounds = state.get("_vpr_clip_bounds", [])
    _sim_strip(img, r, sims, best, packet.idx, int(_g(pl, "exclude", 45)) if pl else 45,
               bounds)

    # ---------------------------------------------------------------- decision
    r = V.panel(img, 14, 570, 760, 122, "revisit decision",
                "sequence-consistency score vs threshold")
    thr = float(_g(pl, "threshold", 0.92)) if pl else 0.92
    _score_bar(img, r, float(pl.best_score) if pl else 0.0, thr,
               bool(pl.is_revisit) if pl else False,
               float(_g(pl, "instant_score", 0.0)) if pl else 0.0,
               int(_g(pl, "seq_consistent_frames", 0)) if pl else 0,
               _g(pl, "note", "no place result") if pl else "no place result",
               float(_g(pl, "distinctiveness", 0.0)) if pl else 0.0,
               float(_g(pl, "ratio_thresh", 1.35)) if pl else 1.35)

    # ---------------------------------------------------------------- info + note
    r = V.panel(img, 782, 436, 472, 256, "model + database", "what is running")
    x0 = r[0] + 12
    excl = int(_g(pl, "exclude", 45)) if pl else 45
    lines = [
        ("descriptor", "256-D L2-norm, GeM pooled"),
        ("backbone", "mobilenet_v3_small[:9], FROZEN"),
        ("trained head", "1x1 conv + GeM p + whitening (78 k)"),
        ("training", "self-supervised InfoNCE, this footage"),
        ("database", f"{int(pl.db_size) if pl else 0} descriptors"
                     f"{'  (cross-clip)' if (pl and _g(pl, 'cross_clip', False)) else ''}"),
        ("exclusion", f"{excl} frames ({excl / CLIP_FPS:.1f} s)   "
                      f"gap {int(pl.loop_gap_frames) if pl else 0}"),
    ]
    yy = r[1] + 18
    for k, v in lines:
        V.text(img, k, (x0, yy), 0.34, V.TEXT_DIM)
        V.text(img, v, (x0 + 96, yy), 0.34, V.TEXT)
        yy += 16
    V.rounded_note(img, x0, yy + 4, r[2] - 24, [
        "A detector answers 'what object is that?'. This answers",
        "'have I stood HERE before?' - one 256-D vector per frame,",
        "matched against every place already visited. It is how a",
        "camera-only robot relocalises after the odometry is lost",
        "and how it cancels accumulated drift when a route loops.",
        "A 10 s forward drive often contains NO revisit; then this",
        "panel reports the top similarity and says exactly that.",
    ], title="PLACE RECOGNITION vs OBJECT DETECTION", lh=13, pad=7)

    V.footer(img, W, H,
             left="GeM + learned whitening (NetVLAD-lite) trained self-supervised on "
                  "this footage - no place-recognition ground truth exists offline here",
             right="cosine similarity + SeqSLAM-style sequence check")
    return img


# --------------------------------------------------------------------- from cache

def packets_from_cache(clip_id: str, state: Optional[dict] = None):
    """Yield fully populated FramePackets for `clip_id` from work/cache/<clip>/vpr.npz.

        state = {}
        with io_utils.VideoWriter(path, (1280, 720)) as w:
            for pk in r_vpr.packets_from_cache("clip_01", state):
                w.write(r_vpr.render(pk, state))

    The similarity strip needs the descriptors of the whole (possibly cross-clip)
    database. Those live in the *other* clips' vpr caches, so this reassembles them
    from disk; a clip whose cache is missing leaves its slice of the strip dark.
    Pass `state` (the same dict you give to `render`) so the clip boundaries are
    drawn on the strip.
    """
    from ..config import CLIP_IDS
    from ..models.vpr import VPRParams, explain_decision
    from ..types import PlaceResult
    z = io_utils.load_stage(clip_id, "vpr")
    desc = np.asarray(z["desc"], np.float32)
    db_clip = np.asarray(z["db_clip"], np.int16)
    db_frame = np.asarray(z["db_frame"], np.int32)
    start = int(z["clip_db_start"])
    db_size = int(z["db_size"])
    prm = VPRParams()
    if "params" in z:                     # thresholds the stage actually used
        q = np.asarray(z["params"], np.float32)
        (prm.exclude_frames, prm.seq_len, prm.top_k) = (int(q[0]), int(q[1]), int(q[2]))
        (prm.sim_thresh, prm.ratio_thresh) = (float(q[3]), float(q[4]))
        (prm.min_seq_frames, prm.min_db, prm.min_gap_frames) = (int(q[5]), int(q[6]),
                                                                int(q[7]))
    n = len(desc)

    # rebuild the full database descriptor matrix from every clip's own cache
    D = np.zeros((db_size, desc.shape[1]), np.float32)
    D[start:start + n] = desc
    for ci in np.unique(db_clip):
        if ci < 0 or not (0 <= ci < len(CLIP_IDS)):
            continue
        other = CLIP_IDS[int(ci)]
        if other == clip_id:
            continue
        try:
            oz = io_utils.load_stage(other, "vpr")
        except FileNotFoundError:
            continue
        od = np.asarray(oz["desc"], np.float32)
        of = np.asarray(oz["db_frame"], np.int32)
        slot = np.nonzero(db_clip == ci)[0]
        # map each database slot of that clip to its frame index in that clip's cache
        for sl in slot:
            f = int(db_frame[sl])
            if 0 <= f < len(od):
                D[sl] = od[f]

    bounds, cur, s0 = [], None, 0
    for i, c in enumerate(db_clip):
        if c != cur:
            if cur is not None:
                bounds.append((CLIP_IDS[cur] if 0 <= cur < len(CLIP_IDS) else "?", s0, i))
            cur, s0 = int(c), i
    if cur is not None:
        bounds.append((CLIP_IDS[cur] if 0 <= cur < len(CLIP_IDS) else "?", s0, len(db_clip)))
    if state is not None:
        state["_vpr_clip_bounds"] = bounds

    for i, frame in io_utils.read_frames(clip_id):
        if i >= n:
            break
        pl = PlaceResult(descriptor=desc[i], best_match_idx=int(z["best_idx"][i]),
                         best_score=float(z["best_score"][i]),
                         is_revisit=bool(z["is_revisit"][i]),
                         loop_gap_frames=int(z["loop_gap"][i]),
                         db_size=db_size)
        bc = int(z["best_clip"][i])
        pl.best_clip_id = CLIP_IDS[bc] if 0 <= bc < len(CLIP_IDS) else clip_id
        pl.best_frame_idx = int(z["best_frame"][i])
        pl.instant_score = float(z["instant_score"][i])
        pl.distinctiveness = float(z["distinctiveness"][i])
        pl.threshold = prm.sim_thresh
        pl.ratio_thresh = prm.ratio_thresh
        pl.exclude = prm.exclude_frames
        pl.cross_clip = bool(start > 0)
        pl.seq_consistent_frames = int(z["seq_frames"][i]) if "seq_frames" in z else 0
        pl.gap_ok = bool(z["gap_ok"][i]) if "gap_ok" in z else True
        pl.sims = (D[:start + i] @ desc[i]) if (start + i) else np.zeros(0, np.float32)
        # exactly the same explainer the live stage uses, so the panel can never
        # print a reason that contradicts the numbers next to it
        pl.note = explain_decision(pl, prm)
        pk = FramePacket(clip_id=clip_id, idx=i, t=i / CLIP_FPS, rgb=frame)
        pk.place = pl
        pk.timings_ms["vpr"] = float(z["ms"][i])
        yield pk


# --------------------------------------------------------------------- self test
if __name__ == "__main__":
    from ..models.vpr import VPRStage
    from ..config import WORK_DIR
    st = VPRStage(cross_clip=True)
    state: dict = {}
    st.seed_from_clips(["clip_02"], stride=4)
    state["_vpr_clip_bounds"] = [("clip_02", 0, st.db_size)]
    out = None
    for i, fr in io_utils.read_frames("clip_01", max_frames=120):
        pk = st(FramePacket(clip_id="clip_01", idx=i, t=i / CLIP_FPS, rgb=fr))
        out = render(pk, state)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(WORK_DIR / "preview_vpr.png"), out)
    print(f"wrote {WORK_DIR / 'preview_vpr.png'}  shape={out.shape}  db={st.db_size}")
