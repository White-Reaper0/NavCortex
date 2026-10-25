"""Renderer for stage 04 - ORB monocular VO (ORB-SLAM3-style front end).

Panels: camera with ORB keypoints and inlier flow, a top-down trajectory with the
current pose and heading, speed / yaw-rate sparklines, inlier and match counts, a
tracking-health indicator that visibly changes state, a scale-source readout, and a
short navigation note.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import cv2

from ..config import CFG, CLIP_FPS
from ..types import FramePacket
from .. import viz_common as V

W, H = 1280, 720
SPARK_LEN = 160

_STATE_COLOR = {"OK": V.OK, "COAST": V.WARN, "LOST": V.BAD}
_SCALE_COLOR = {"depth-anchored": V.OK, "smoothed-prev": V.WARN,
                "const-fallback": V.WARN, "no-track": V.BAD}


def _g(o, name, default):
    v = getattr(o, name, default)
    return default if v is None else v


def _draw_camera(img, rect, packet: FramePacket):
    x, y, w, h = rect
    o = packet.odom
    frame = packet.rgb
    if frame is None:
        frame = np.full((360, 640, 3), 30, np.uint8)
    fh, fw = frame.shape[:2]
    s = min(w / fw, h / fh)
    nw, nh = int(fw * s), int(fh * s)
    ox, oy = x + (w - nw) // 2, y + (h - nh) // 2
    vis = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    vis = (vis.astype(np.float32) * 0.62).astype(np.uint8)     # dim so overlays pop

    kp = _g(o, "keypoints", np.zeros((0, 2), np.float32)) if o else np.zeros((0, 2), np.float32)
    fl = _g(o, "flow", np.zeros((0, 4), np.float32)) if o else np.zeros((0, 4), np.float32)
    kp = np.asarray(kp, np.float32).reshape(-1, 2)
    fl = np.asarray(fl, np.float32).reshape(-1, 4)

    for px, py in kp:
        cv2.circle(vis, (int(px * s), int(py * s)), 1, (150, 150, 150), -1, cv2.LINE_AA)

    if len(fl):
        AMP = 4.0                          # inter-frame motion is a few px; amplify to see it
        d = fl[:, 2:] - fl[:, :2]
        mag = np.linalg.norm(d, axis=1)
        mmax = max(float(np.percentile(mag, 92)), 0.6)
        step = max(1, len(fl) // 260)
        for i in range(0, len(fl), step):
            x0, y0 = fl[i, 0] * s, fl[i, 1] * s
            ex, ey = x0 + d[i, 0] * AMP * s, y0 + d[i, 1] * AMP * s
            t = float(np.clip(mag[i] / mmax, 0, 1))
            col = (int(255 - 120 * t), int(120 + 90 * t), int(60 + 180 * t))
            cv2.line(vis, (int(x0), int(y0)), (int(ex), int(ey)), col, 1, cv2.LINE_AA)
            cv2.circle(vis, (int(x0), int(y0)), 2, V.ACCENT, -1, cv2.LINE_AA)

    img[oy:oy + nh, ox:ox + nw] = vis
    cv2.rectangle(img, (ox, oy), (ox + nw, oy + nh), V.EDGE, 1)

    st = _g(o, "state_name", "OK" if (o and o.tracking_ok) else "LOST") if o else "LOST"
    bw, bh = V.badge(img, ox + 8, oy + 8, f"TRACK {st}", _STATE_COLOR.get(st, V.BAD), 0.46)
    n_i = int(o.n_inliers) if o else 0
    n_m = int(o.n_matches) if o else 0
    extra = f" ({len(fl)} drawn)" if len(fl) < n_i else ""
    V.text(img, f"{len(kp)} ORB keypoints   {n_i}/{n_m} inlier matches{extra}, "
                f"flow x4", (ox + 8, oy + nh - 10), 0.38, V.TEXT_DIM)
    if o is not None and not o.tracking_ok:
        V.text(img, "POSE HELD - not integrating", (ox + bw + 16, oy + 8 + bh - 7),
               0.44, V.BAD, 1, V.FONT_B)
    return (ox, oy, nw, nh)


def _draw_traj(img, rect, packet: FramePacket, state: dict):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (24, 22, 20), -1)
    o = packet.odom
    traj = _g(o, "trajectory", np.zeros((1, 2), np.float32)) if o else np.zeros((1, 2), np.float32)
    traj = np.asarray(traj, np.float32).reshape(-1, 2)
    if len(traj) == 0:
        traj = np.zeros((1, 2), np.float32)

    lo = traj.min(0); hi = traj.max(0)
    ctr = (lo + hi) / 2.0
    span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1.0)) * 1.35
    ppm = min(w, h) / span                                   # pixels per metre

    def to_px(p):
        return (int(x + w / 2 + (p[0] - ctr[0]) * ppm),
                int(y + h / 2 - (p[1] - ctr[1]) * ppm))

    # metric grid, 1 m spacing (or coarser when the path is long)
    gstep = 1.0 if span < 12 else (2.0 if span < 30 else 5.0)
    g0 = np.floor((ctr - span / 2) / gstep) * gstep
    for k in range(int(span / gstep) + 3):
        gx = g0[0] + k * gstep
        px = to_px((gx, ctr[1]))[0]
        if x < px < x + w:
            cv2.line(img, (px, y), (px, y + h), (40, 37, 34), 1)
        gy = g0[1] + k * gstep
        py = to_px((ctr[0], gy))[1]
        if y < py < y + h:
            cv2.line(img, (x, py), (x + w, py), (40, 37, 34), 1)

    pts = np.array([to_px(p) for p in traj], np.int32)
    if len(pts) > 1:
        cv2.polylines(img, [pts], False, (110, 110, 108), 3, cv2.LINE_AA)
        n = len(pts)
        seg = max(1, n // 90)
        for i in range(0, n - seg, seg):
            t = i / max(n - 1, 1)
            c = (int(90 + 60 * t), int(120 + 70 * t), int(255 - 60 * t))
            cv2.line(img, tuple(pts[i]), tuple(pts[min(i + seg, n - 1)]), c, 2, cv2.LINE_AA)
    cv2.circle(img, tuple(pts[0]), 5, V.ACCENT2, -1, cv2.LINE_AA)
    V.text(img, "start", (pts[0][0] + 8, pts[0][1] + 4), 0.34, V.TEXT_DIM)

    # current pose + heading
    pose = _g(o, "pose", None) if o else None
    yaw = float(pose.yaw) if pose is not None else 0.0
    cur = tuple(pts[-1])
    fwd = np.array([-np.sin(yaw), np.cos(yaw)])
    rgt = np.array([np.cos(yaw), np.sin(yaw)])
    L = float(np.clip(0.35 * ppm, 14.0, 26.0))
    tri = np.array([
        [cur[0] + fwd[0] * L, cur[1] - fwd[1] * L],
        [cur[0] - fwd[0] * L * 0.5 + rgt[0] * L * 0.42,
         cur[1] + fwd[1] * L * 0.5 - rgt[1] * L * 0.42],
        [cur[0] - fwd[0] * L * 0.5 - rgt[0] * L * 0.42,
         cur[1] + fwd[1] * L * 0.5 + rgt[1] * L * 0.42]], np.int32)
    ok = bool(o.tracking_ok) if o else False
    col = V.ACCENT if ok else V.BAD
    cv2.fillPoly(img, [tri], col, cv2.LINE_AA)
    cv2.polylines(img, [tri], True, (20, 20, 20), 1, cv2.LINE_AA)

    # scale bar
    bar_m = gstep
    bx, by = x + 14, y + h - 18
    cv2.line(img, (bx, by), (bx + int(bar_m * ppm), by), V.TEXT, 2)
    cv2.line(img, (bx, by - 4), (bx, by + 4), V.TEXT, 2)
    cv2.line(img, (bx + int(bar_m * ppm), by - 4), (bx + int(bar_m * ppm), by + 4), V.TEXT, 2)
    V.text(img, f"{bar_m:g} m", (bx + int(bar_m * ppm) + 8, by + 4), 0.36, V.TEXT_DIM)

    plen = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()) if len(traj) > 1 else 0.0
    V.text(img, f"path {plen:.2f} m", (x + w - 96, y + 18), 0.4, V.TEXT_DIM)
    V.text(img, "+Y forward, +X right (map frame)", (x + 14, y + 18), 0.36, V.TEXT_DIM)


def render(packet: FramePacket, state: dict) -> np.ndarray:
    img = V.canvas(W, H)
    o = packet.odom

    hist = state.setdefault("_vo", {"speed": [], "yaw": [], "inl": [], "q": [],
                                    "lost": 0, "coast": 0, "n": 0})
    if o is not None:
        hist["speed"].append(float(o.speed_mps))
        hist["yaw"].append(float(np.degrees(_g(o, "d_yaw", 0.0)) * CLIP_FPS))
        hist["inl"].append(int(o.n_inliers))
        hist["q"].append(float(o.track_quality))
        hist["n"] += 1
        st_name = _g(o, "state_name", "OK" if o.tracking_ok else "LOST")
        if st_name == "LOST":
            hist["lost"] += 1
        elif st_name == "COAST":
            hist["coast"] += 1
        for k in ("speed", "yaw", "inl", "q"):
            if len(hist[k]) > SPARK_LEN:
                del hist[k][:-SPARK_LEN]

    V.header(img, W, "prism",
             "04 - VISUAL ODOMETRY | ORB-SLAM3-style monocular front end (Python/OpenCV)",
             right=f"{packet.clip_id}   frame {packet.idx:03d}   t={packet.t:5.2f}s")

    # ---------------------------------------------------------------- camera
    r = V.panel(img, 14, 52, 706, 418, "camera + ORB tracks",
                "grid-bucketed ORB, ego mask applied")
    _draw_camera(img, r, packet)

    # ---------------------------------------------------------------- trajectory
    r = V.panel(img, 728, 52, 538, 418, "trajectory (top-down)",
                "integrated 2-D pose, no loop closure")
    _draw_traj(img, r, packet, state)

    # ---------------------------------------------------------------- sparklines
    r = V.panel(img, 14, 478, 340, 104, "speed", "m/s, EMA smoothed")
    sp = hist["speed"] or [0.0]
    V.sparkline(img, r[0] + 8, r[1] + 8, r[2] - 16, r[3] - 34, sp, V.ACCENT,
                lo=0.0, hi=max(0.4, max(sp) * 1.15))
    V.text(img, f"{sp[-1]:.2f} m/s", (r[0] + 10, r[1] + r[3] - 8), 0.46, V.TEXT, 1, V.FONT_B)
    V.text(img, f"peak {max(sp):.2f}", (r[0] + r[2] - 76, r[1] + r[3] - 8), 0.36, V.TEXT_DIM)

    r = V.panel(img, 14, 588, 340, 104, "yaw rate", "deg/s, EMA smoothed")
    yr = hist["yaw"] or [0.0]
    lim = max(12.0, max(abs(min(yr)), abs(max(yr))) * 1.2)
    V.sparkline(img, r[0] + 8, r[1] + 8, r[2] - 16, r[3] - 34, yr, V.ACCENT2, lo=-lim, hi=lim)
    V.text(img, f"{yr[-1]:+.1f} deg/s", (r[0] + 10, r[1] + r[3] - 8), 0.46, V.TEXT, 1, V.FONT_B)

    # ---------------------------------------------------------------- stats
    r = V.panel(img, 362, 478, 358, 214, "tracking health", "supervisor inputs")
    x0 = r[0] + 12
    st_name = _g(o, "state_name", "OK" if (o and o.tracking_ok) else "LOST") if o else "LOST"
    reason = _g(o, "reason", "") if o else "no odometry"
    V.badge(img, x0, 512, st_name, _STATE_COLOR.get(st_name, V.BAD), 0.56)
    lost_pct = 100.0 * hist["lost"] / max(hist["n"], 1)
    coast_pct = 100.0 * hist["coast"] / max(hist["n"], 1)
    V.text(img, f"lost {lost_pct:4.1f}%  coast {coast_pct:4.1f}%",
           (x0 + 100, 528), 0.38, V.TEXT_DIM)
    V.text(img, f"({hist['n']} frames)", (x0 + 100, 541), 0.32, V.TEXT_DIM)
    V.text(img, reason[:52], (x0, 558), 0.33,
           V.TEXT_DIM if st_name == "OK" else V.WARN)

    n_m = int(o.n_matches) if o else 0
    n_i = int(o.n_inliers) if o else 0
    V.bar_meter(img, x0, 576, 240, 9, n_i, V.OK,
                f"RANSAC inliers  {n_i} / {n_m} matches", 0, max(n_m, 1))
    q = float(o.track_quality) if o else 0.0
    V.bar_meter(img, x0, 604, 240, 9, q,
                V.OK if q > 0.55 else (V.WARN if q > 0.3 else V.BAD),
                f"track quality  {q:.2f}", 0.0, 1.0)

    par = _g(o, "parallax_deg", 0.0) if o else 0.0
    sam = _g(o, "sampson_px", 0.0) if o else 0.0
    V.text(img, f"parallax {par:5.3f} deg   epipolar resid {sam:4.2f} px",
           (x0, 632), 0.36, V.TEXT_DIM)
    src = _g(o, "scale_source_name", "depth-anchored") if o else "no-track"
    sm = _g(o, "scale_m", 0.0) if o else 0.0
    se = _g(o, "scale_se", 0.0) if o else 0.0
    V.text(img, "SCALE", (x0, 654), 0.34, V.TEXT_DIM)
    V.badge(img, x0 + 48, 642, src, _SCALE_COLOR.get(src, V.WARN), 0.36, pad=4)
    V.text(img, f"{sm*100:4.2f} cm/f  SE {se:4.2f}", (x0 + 190, 654), 0.34, V.TEXT_DIM)
    V.text(img, f"metric scale traceable to assumed h_cam = "
                f"{CFG.cam.height_above_ground_m:.2f} m", (x0, 671), 0.31, V.TEXT_DIM)
    pose = _g(o, "pose", None) if o else None
    if pose is not None:
        V.text(img, f"pose x{pose.x:+5.2f} y{pose.y:+5.2f} m  yaw"
                    f"{np.degrees(pose.yaw):+6.1f} deg", (x0, 687), 0.35, V.TEXT)

    # ---------------------------------------------------------------- note
    r = V.panel(img, 728, 478, 538, 214, "why this matters", "camera-only navigation")
    V.rounded_note(img, r[0] + 10, r[1] + 8, r[2] - 20, [
        "Off-road and under canopy there is often no GPS fix. Ego-motion has to come",
        "from the camera itself: ORB corners are matched frame to frame, an essential",
        "matrix gives rotation and a translation DIRECTION, and metric length comes",
        f"from the depth stage (anchored on an ASSUMED camera height of "
        f"{CFG.cam.height_above_ground_m:.2f} m).",
        "No IMU, no wheel encoders, no stereo baseline, no loop closure - so this drifts.",
        "",
        "If tracking fails the pose is HELD, never guessed. Downstream that suspends",
        "goal pursuit: the planner may still avoid obstacles from the live BEV map, but",
        "'drive 3 m to the waypoint' is meaningless once the odometry is unreliable.",
    ], title="GPS-DENIED EGO-MOTION")

    V.footer(img, W, H,
             left="ORB-SLAM3-style front end re-implemented in Python/OpenCV - NOT the "
                  "ORB-SLAM3 system (no bundle adjustment, no loop closing, no IMU)",
             right="depth-anchored monocular scale")
    return img


# --------------------------------------------------------------------- from cache

def packets_from_cache(clip_id: str):
    """Yield fully populated FramePackets for `clip_id` from work/cache/<clip>/odom.npz.

    Use this to render the stage video without re-running perception::

        state = {}
        with io_utils.VideoWriter(path, (1280, 720)) as w:
            for pk in r_vo.packets_from_cache("clip_01"):
                w.write(r_vo.render(pk, state))
    """
    from .. import io_utils
    from ..types import OdometryResult, Pose
    from ..perception.odometry import SCALE_SOURCE_NAMES, TRACK_STATE_NAMES
    z = io_utils.load_stage(clip_id, "odom")
    n = len(z["pose"])
    for i, frame in io_utils.read_frames(clip_id):
        if i >= n:
            break
        p = z["pose"][i]
        o = OdometryResult(
            pose=Pose(float(p[0]), float(p[1]), float(p[2])),
            d_trans=float(z["d_trans"][i]), d_yaw=float(z["d_yaw"][i]),
            speed_mps=float(z["speed"][i]), n_matches=int(z["n_matches"][i]),
            n_inliers=int(z["n_inliers"][i]), tracking_ok=bool(z["tracking_ok"][i]),
            track_quality=float(z["track_quality"][i]),
            keypoints=z["keypoints"][i][:int(z["n_keypoints"][i])],
            flow=z["flow"][i][:int(z["n_flow"][i])],
            trajectory=z["trajectory"][:i + 1])
        st = int(z["track_state"][i])
        o.state = st
        o.state_name = TRACK_STATE_NAMES[st]
        o.scale_source = int(z["scale_source"][i])
        o.scale_source_name = SCALE_SOURCE_NAMES[int(z["scale_source"][i])]
        o.scale_m = float(z["scale"][i])
        o.scale_se = 0.0
        o.parallax_deg = float(z["parallax_deg"][i])
        o.sampson_px = float(z["sampson_px"][i])
        o.reason = {0: "ok", 1: "motion-model coast (two-view direction rejected)",
                    2: "tracking lost - pose held"}[st]
        pk = FramePacket(clip_id=clip_id, idx=i, t=i / CLIP_FPS, rgb=frame)
        pk.odom = o
        pk.timings_ms["odom"] = float(z["ms"][i])
        yield pk


# --------------------------------------------------------------------- self test
if __name__ == "__main__":
    import sys
    from ..perception.odometry import OdometryStage
    from ..config import WORK_DIR
    from .. import io_utils
    cid = sys.argv[1] if len(sys.argv) > 1 else "clip_01"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    st = OdometryStage()
    st.reset()
    state: dict = {}
    out = None
    for i, fr in io_utils.read_frames(cid, max_frames=n):
        pk = st(FramePacket(clip_id=cid, idx=i, t=i / CLIP_FPS, rgb=fr))
        out = render(pk, state)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(WORK_DIR / "preview_vo.png"), out)
    print(f"wrote {WORK_DIR / 'preview_vo.png'}  shape={out.shape}  clip={cid}")
