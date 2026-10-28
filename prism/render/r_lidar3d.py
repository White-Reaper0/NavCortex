"""Renderer for stage 07b_lidar_3d_view - the immersive, pose-accumulated 3-D view.

HOW THIS DIFFERS FROM STAGE 07
------------------------------
Stage 07 (`r_lidar.py`) is an *instrument dashboard*: a small orbiting cloud panel next
to a polar PPI sweep, a range image and a statistics block, all describing **one frame**.

This stage is the opposite. The 3-D scene is the screen, and it is a **surface**, not a
cloud: every frame's gated cloud is registered into a common world frame with the
visual-odometry pose and fused into a persistent 2.5-D elevation map, which is then
triangulated into a low-poly, flat-shaded mesh. What you watch is not 300 independent
clouds but **one reconstruction of the corridor the vehicle drove down**, built up in
front of you while a chase camera follows the vehicle through it. Nothing here re-uses
the stage-07 panels.

WHAT IS ON SCREEN
-----------------
* hero      - the accumulated world surface as flat-shaded triangles, coloured by height
              above the ground plane and lit by a fixed key light so slope reads as
              brightness, with a metre grid labelled in world coordinates, the travelled
              path draped over the surface as a ribbon, and the UGV wireframe at
              `CFG.ugv` scale at the current pose. Perspective projection, painter's
              algorithm, distance fog, and an age fade so the freshest sweep is the
              brightest part of the surface.
* map       - the same triangles from a raised near-top-down camera, coloured by
              traversability class (`TRAV_COLORS_BGR`). Second colour mode, and a map.
* camera    - the single RGB frame all of it came from.
* stats/note- the counts, and the caveats.

HONESTY (this stage is the one most likely to be mistaken for a real sensor)
---------------------------------------------------------------------------
* No LiDAR runs anywhere in this pipeline. There is no time-of-flight measurement. Every
  point is monocular depth from one RGB camera, scaled to metres by an **assumed**
  camera height of `CFG.cam.height_above_ground_m` = 0.12 m.
* Registration is monocular VO: no IMU, no wheel encoders, no loop closure, and the pose
  is 2-D (x, y, yaw) - roll and pitch are fixed by each frame's own ground-plane fit.
  It drifts, and the error compounds: a point laid down at t = 8 s is placed through the
  whole chain of earlier pose estimates.
* When the VO front end declares a loss it holds its pose, so a cloud registered during
  a loss would be stamped at a stale position and smeared into the map. This renderer
  **freezes the surface** for those frames, draws the unfused live cloud as grey points,
  and says so on screen.
* The surface is a 2.5-D height field, one height per 15 cm patch of ground. It cannot
  represent an overhang, a bridge or the underside of anything, and it should not be
  read as a watertight 3-D model of the world.
* Points exist only where the camera looked, inside a ~92 deg horizontal field of view,
  and only inside `cloud_accum.VOL_R_MAX`. Empty space in this view is unseen space.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .. import viz_common as V
from ..config import CFG, TRAV_CLASSES, TRAV_COLORS_BGR
from ..io_utils import load_stage
from ..perception import cloud_accum as CA
from ..types import FramePacket

W_OUT, H_OUT = 1280, 720
STAGE_TITLE = "07b - IMMERSIVE 3D LIDAR VIEW (POSE-ACCUMULATED)"

# ---- layout ---------------------------------------------------------------
HERO = (12, 50, 946, 646)
CAM_P = (966, 50, 302, 190)
MAP_P = (966, 246, 302, 218)
STATS_P = (966, 470, 302, 106)
NOTE_P = (966, 582, 302, 114)

# ---- surface map / display ------------------------------------------------
CELL_M = 0.15                      # world cell = one pair of triangles. "Low poly" is a
                                   # deliberate choice: at 15 cm a facet is about half a
                                   # vehicle width, which is the scale a 0.34 m UGV
                                   # actually plans at, and it is coarse enough that the
                                   # facet orientation carries the slope information.
KEEP_FRAMES = 210                  # 7.0 s rolling window
MESH_HALF_M = 11.0                 # half-width of the meshed window around the camera
MAP_HALF_M = 8.0
MIN_TRACK_QUALITY = 0.25           # below this the pose is not trusted for mapping

BG3D = (16, 15, 14)

#: Measured in this repo (see the module self-test): the brick wall in clip_04 / clip_03
#: reconstructed from the accumulated map, fitted as one plane in the world frame.
WALL_CHECK = ("geometry check: the clip_04 brick wall reconstructs as ONE plane "
              "4.7 m long and 0.13 m thick, from 227 frames (0.06 m single-frame)")


# --------------------------------------------------------------------------- state


def _clip_state(packet: FramePacket, state: dict) -> dict:
    """Per-clip surface map + a pre-smoothed camera path, built once per clip."""
    st = state.get("l3d")
    if st is not None and st["clip_id"] == packet.clip_id:
        return st
    pose = None
    tok = None
    tq = None
    try:
        z = load_stage(packet.clip_id, "odom")
        pose = np.asarray(z["pose"], np.float32)
        tok = np.asarray(z["tracking_ok"], bool)
        tq = np.asarray(z["track_quality"], np.float32)
    except Exception:
        pass
    st = {
        "clip_id": packet.clip_id,
        "surface": CA.SurfaceMap(cell_m=CELL_M, keep_frames=KEEP_FRAMES),
        "pose": pose,
        "tracking_ok": tok,
        "track_quality": tq,
        # zero-phase smoothing of the whole pose track: the chase camera must not inherit
        # the VO's per-frame jitter, or the render reads as camera shake, not as a map
        "anchor": CA.chase_path(pose) if pose is not None else None,
        "chase": CA.ChaseCamera(back_m=5.2, up_m=4.6, ahead_m=3.0, look_z=0.10,
                                vfov_deg=40.0),
        "trail": [],
        "n_live": 0,
        "n_frozen": 0,
        "ms": [],
        "last_idx": -1,
    }
    state["l3d"] = st
    return st


def _pose_of(packet: FramePacket, st: dict) -> np.ndarray:
    if st["pose"] is not None and packet.idx < len(st["pose"]):
        return st["pose"][packet.idx]
    if packet.odom is not None:
        p = packet.odom.pose
        return np.array([p.x, p.y, p.yaw], np.float32)
    return np.zeros(3, np.float32)


def _anchor_of(packet: FramePacket, st: dict, pose: np.ndarray) -> np.ndarray:
    a = st["anchor"]
    if a is not None and packet.idx < len(a):
        return a[packet.idx]
    return pose


def _track_of(packet: FramePacket, st: dict) -> tuple[bool, float]:
    i = packet.idx
    if st["tracking_ok"] is not None and i < len(st["tracking_ok"]):
        return bool(st["tracking_ok"][i]), float(st["track_quality"][i])
    if packet.odom is not None:
        return bool(packet.odom.tracking_ok), float(packet.odom.track_quality)
    return True, 1.0


# --------------------------------------------------------------------------- helpers


def _scrim(img, x, y, w, h, strength: float = 0.74, fade: int = 130, bg=BG3D):
    """Darken a band of the live 3-D render so overlay text stays legible over it."""
    x, y, w, h = int(x), int(y), int(w), int(h)
    reg = img[y:y + h, x:x + w]
    if reg.size == 0:
        return
    a = np.full(reg.shape[1], strength, np.float32)
    f = min(fade, reg.shape[1])
    if f > 0:
        a[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
    a = a[None, :, None]
    reg[:] = (reg.astype(np.float32) * (1.0 - a)
              + np.array(bg, np.float32)[None, None] * a).astype(np.uint8)


def _age_alpha(mesh: CA.Mesh, frame_idx: int) -> np.ndarray:
    """Fade a facet with the age of its last observation and with the VO quality behind it.

    Both terms are honesty, not decoration: an old facet has been carried through more
    pose increments than a fresh one, and a facet fused while tracking was marginal was
    placed by a pose the front end itself did not trust.
    """
    age = np.maximum(frame_idx - mesh.face_age, 0.0)
    a = 1.0 - 0.40 * np.clip(age / float(KEEP_FRAMES), 0.0, 1.0)
    return (a * (0.62 + 0.38 * np.clip(mesh.face_qual, 0.0, 1.0))).astype(np.float32)


def _sweep_tint(base: np.ndarray, mesh: CA.Mesh, frame_idx: int,
                fresh_frames: int = 3) -> np.ndarray:
    """Lift the facets this frame just re-observed, so the live sweep stays visible."""
    fresh = (frame_idx - mesh.face_age) <= fresh_frames
    if fresh.any():
        base = base.copy()
        base[fresh] = np.clip(base[fresh] * 1.22 + 26.0, 0, 255)
    return base


def _hero_view(size, st, pose, anchor, mesh, frame_idx, live_xyz, live_ok,
               trail, trail_z, veh_z) -> np.ndarray:
    w, h = size
    img = np.empty((h, w, 3), np.uint8)
    img[:] = BG3D
    cam = st["chase"].at(anchor, (w, h))
    CA.draw_world_grid(img, cam, anchor[:2], half=9.0, step=1.0, label_every=3)
    base = _sweep_tint(CA.colorize_height_pts(mesh.face_z), mesh, frame_idx)
    n = CA.render_mesh(img, cam, mesh, CA.shade_faces(mesh, base), bg=BG3D,
                       alpha=_age_alpha(mesh, frame_idx))
    if not live_ok and live_xyz.shape[0]:
        # not fused into the surface: draw it as a grey point cloud so it is visibly
        # *not* part of the reconstruction
        grey = np.empty((live_xyz.shape[0], 3), np.uint8)
        z = np.clip((live_xyz[:, 2] - CA.HEIGHT_LO) / (CA.HEIGHT_HI - CA.HEIGHT_LO), 0, 1)
        g = (80 + 100 * z).astype(np.uint8)
        grey[:, 0], grey[:, 1], grey[:, 2] = g, g, np.clip(g.astype(np.int32) + 26, 0, 255)
        CA.render_points(img, cam, live_xyz, grey, bg=BG3D,
                         alpha=np.full(live_xyz.shape[0], 0.65, np.float32))
    # Path and vehicle are chrome, not geometry, so they go down last and are never
    # occluded by the surface: where the vehicle has been is the one thing in this view
    # that must stay readable at every frame.
    if len(trail) >= 2:
        CA.draw_trail_ribbon(img, cam, np.asarray(trail, np.float32), z=trail_z + 0.02)
    a = np.linspace(0, 2 * np.pi, 40)
    CA.draw_polyline_3d(img, cam, np.stack([pose[0] + 0.45 * np.cos(a),
                                            pose[1] + 0.45 * np.sin(a),
                                            np.full(40, veh_z + 0.015)], 1),
                        (70, 130, 175), 1)
    CA.draw_vehicle_wire(img, cam, pose, color=V.ACCENT, thick=2, z0=veh_z)
    return img, n


def _map_view(size, anchor, mesh, frame_idx, trail, trail_z) -> np.ndarray:
    """Near-top-down view of the same surface, coloured by traversability class.

    Second colour mode and a map in one panel: identical geometry, identical facets,
    only the colour rule changes - height above ground in the hero, the traversability
    head's class here.
    """
    w, h = size
    bg = (13, 12, 11)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = bg
    x, y, yaw = float(anchor[0]), float(anchor[1]), float(anchor[2])
    fwd = np.array([-np.sin(yaw), np.cos(yaw)])
    cam = CA.WorldCamera([x - fwd[0] * 1.4, y - fwd[1] * 1.4, 8.5],
                         [x + fwd[0] * 2.0, y + fwd[1] * 2.0, 0.0], (w, h), 52.0)
    CA.draw_world_grid(img, cam, (x, y), half=7.0, step=1.0,
                       color=(38, 35, 32), major=(58, 54, 49), label_every=3,
                       label_color=(92, 88, 82))
    CA.render_mesh(img, cam, mesh,
                   CA.shade_faces(mesh, TRAV_COLORS_BGR[mesh.face_trav].astype(np.float32),
                                  ambient=0.58, diffuse=0.52),
                   bg=bg, fog_m=34.0, alpha=_age_alpha(mesh, frame_idx))
    if len(trail) >= 2:
        CA.draw_trail_ribbon(img, cam, np.asarray(trail, np.float32),
                             z=trail_z + 0.03, half_w=0.09)
    CA.draw_vehicle_wire(img, cam, anchor, color=V.ACCENT, thick=1)
    return img, cam


# --------------------------------------------------------------------------- main


def render(packet: FramePacket, state: dict) -> np.ndarray:
    t_start = time.perf_counter()
    st = _clip_state(packet, state)
    img = V.canvas(W_OUT, H_OUT)
    V.header(img, W_OUT, "prism", STAGE_TITLE,
             right=f"{packet.clip_id}   frame {packet.idx:03d}   t={packet.t:5.2f}s")

    if packet.depth is None or packet.depth.depth_m is None:
        V.panel(img, 12, 60, W_OUT - 24, 560, "no depth", "nothing to reconstruct")
        V.text(img, "packet.depth.depth_m is missing - run the depth stage first",
               (40, 340), 0.6, V.TEXT_DIM, 1)
        V.footer(img, W_OUT, H_OUT, CA.ACCUM_CAVEAT, "")
        return img

    pose = _pose_of(packet, st)
    anchor = _anchor_of(packet, st, pose)
    ok_track, qual = _track_of(packet, st)
    accept = ok_track and qual >= MIN_TRACK_QUALITY

    # ------------------------------------------------------------- accumulate
    seg = packet.seg.label if packet.seg is not None else None
    trv = packet.trav.label if packet.trav is not None else None
    xyz_v, trav_pt, n_valid_px = CA.frame_cloud_vehicle(
        np.asarray(packet.depth.depth_m, np.float32),
        None if packet.depth.valid is None else np.asarray(packet.depth.valid, bool),
        getattr(packet.depth, "_normal", None), seg, trv, stride=2)
    live_world = CA.veh_to_world(xyz_v, pose) if xyz_v.shape[0] else xyz_v
    smap: CA.SurfaceMap = st["surface"]
    if packet.idx > st["last_idx"]:          # a re-render of the same frame must not double-fuse
        smap.add(live_world, trav_pt, packet.idx, qual, accept=accept)
        st["trail"].append((float(pose[0]), float(pose[1])))
        if not accept:
            st["n_frozen"] += 1
        st["last_idx"] = packet.idx
    st["n_live"] = int(xyz_v.shape[0])

    mesh = smap.mesh(anchor[:2], half_m=MESH_HALF_M)
    trail_xy = np.asarray(st["trail"], np.float32)
    trail_z = smap.height_at(trail_xy) if len(st["trail"]) >= 2 else np.zeros(1, np.float32)
    veh_z = float(smap.height_at(pose[None, :2])[0])
    rho_live = (np.hypot(xyz_v[:, 0], xyz_v[:, 1]) if xyz_v.shape[0]
                else np.zeros(1, np.float32))

    # =================================================================== hero
    inner = V.panel(img, *HERO, "accumulated 3D surface",
                    f"chase camera | colour = height above ground | "
                    f"{KEEP_FRAMES / 30:.1f} s rolling window, "
                    f"{smap.cell_m * 100:.0f} cm triangles")
    hx, hy = inner[0] + 3, inner[1] + 3
    cw, ch = inner[2] - 6, inner[3] - 6
    view, n_drawn = _hero_view((cw, ch), st, pose, anchor, mesh, packet.idx,
                               live_world, accept, st["trail"], trail_z, veh_z)
    img[hy:hy + ch, hx:hx + cw] = view

    _scrim(img, hx, hy, min(cw, 760), 96)
    _scrim(img, hx, hy + ch - 56, min(cw, 870), 56)

    V.text(img, "NO LIDAR.  This surface is monocular depth from ONE RGB camera, scaled by "
                f"an ASSUMED {CFG.cam.height_above_ground_m:.2f} m camera height.",
           (hx + 13, hy + 19), 0.40, (110, 130, 255), 1, V.FONT_B)
    V.text(img, f"{len(mesh):,} triangles over {smap.n_cells():,} mapped "
                f"{smap.cell_m * 100:.0f} cm cells   <-   {xyz_v.shape[0]:,} live points "
                f"this frame ({n_valid_px:,} valid depth px, "
                f"{smap.n_frames_used} frames fused)",
           (hx + 13, hy + 38), 0.36, V.TEXT, 1)
    V.text(img, "registered by monocular visual odometry: no IMU, no wheel encoders, "
                "no loop closure, 2-D pose only - this map drifts",
           (hx + 13, hy + 55), 0.34, V.ACCENT2, 1)
    V.text(img, f"only returns within {CA.VOL_R_MAX:.1f} m are mapped: measured on the clip_04 "
                f"wall the reconstruction is 0.19 m thick for returns taken inside 2.5 m, "
                f"0.41 m at 3.5-4.5 m",
           (hx + 13, hy + 72), 0.32, (128, 122, 114), 1)
    V.text(img, "black space is UNSEEN, not empty: the camera looks forward through "
                f"{CFG.cam.hfov_deg:.0f} deg and nothing behind it is measured",
           (hx + 13, hy + 88), 0.32, (128, 122, 114), 1)

    V.colorbar(img, hx + 13, hy + ch - 38, 214, 10, cv2.COLORMAP_TURBO,
               lo_lab=f"{CA.HEIGHT_LO:+.2f} m", hi_lab=f"{CA.HEIGHT_HI:+.2f} m",
               title="colour = height above the fitted ground plane")
    V.text(img, "amber ribbon = travelled path (raw VO pose, unsmoothed)   |   "
                f"amber box = UGV envelope {CFG.ugv.length_m:.2f} x {CFG.ugv.width_m:.2f} m   |   "
                "grid = 1 m squares, labelled in world metres",
           (hx + 13, hy + ch - 8), 0.32, (132, 126, 118), 1)

    # ---- tracking state, bottom right of the hero (kept clear of the caption block)
    bx, by = hx + cw - 322, hy + ch - 74
    _scrim(img, bx - 12, by - 14, 334, 74, strength=0.68, fade=0)
    if accept:
        V.badge(img, bx, by, f"VO TRACKING OK   quality {qual:.2f}", V.OK, 0.40, pad=6)
        V.text(img, "this frame IS being fused into the surface",
               (bx, by + 40), 0.33, (150, 190, 155), 1)
    else:
        cv2.rectangle(img, (hx - 1, hy - 1), (hx + cw, hy + ch), V.BAD, 2)
        V.badge(img, bx, by, "VO TRACKING LOST - MAPPING FROZEN", V.BAD, 0.40, pad=6)
        V.text(img, "grey points = this frame, NOT fused into the surface",
               (bx, by + 40), 0.33, (200, 200, 210), 1)
    V.text(img, f"{st['n_frozen']} of {packet.idx + 1} frames rejected so far",
           (bx, by + 56), 0.32, (150, 148, 145), 1)

    # =================================================================== camera
    inner = V.panel(img, *CAM_P, "the only sensor", "one RGB camera")
    if packet.rgb is not None:
        V.blit(img, packet.rgb, (inner[0] + 2, inner[1] + 2, inner[2] - 4, inner[3] - 4))
    else:
        V.text(img, "no rgb frame", (inner[0] + 14, inner[1] + 44), 0.45, V.TEXT_DIM, 1)

    # =================================================================== trav map
    inner = V.panel(img, *MAP_P, "same map, 2nd mode", "trav class")
    mw, mh = inner[2] - 6, inner[3] - 30
    mx, my = inner[0] + 3, inner[1] + 3
    map_img, _ = _map_view((mw, mh), anchor,
                           smap.mesh(anchor[:2], half_m=MAP_HALF_M), packet.idx,
                           st["trail"], trail_z)
    img[my:my + mh, mx:mx + mw] = map_img
    cv2.rectangle(img, (mx, my), (mx + mw, my + mh), V.EDGE, 1)
    V.text(img, "same triangles, near top-down", (mx + 5, my + 14), 0.31,
           (120, 150, 175), 1)
    V.legend(img, mx + 5, my + mh + 15, TRAV_CLASSES, TRAV_COLORS_BGR, 0.33,
             swatch=9, gap=14, vertical=False)

    # =================================================================== stats
    inner = V.panel(img, *STATS_P, "surface", "this frame")
    sx, sy, sw, sh = inner
    rows = [
        ("live points fused", f"{xyz_v.shape[0]:,}"),
        ("mapped cells", f"{smap.n_cells():,}"),
        ("triangles drawn", f"{n_drawn:,}"),
        ("live range (med / max)",
         f"{np.median(rho_live):.2f} / {rho_live.max():.2f} m"),
        ("path travelled", f"{_path_len(st['trail']):.2f} m"),
    ]
    for i, (k, v) in enumerate(rows):
        yy = sy + 15 + i * 14
        V.text(img, k, (sx + 10, yy), 0.33, V.TEXT_DIM, 1)
        V.text(img, v, (sx + sw - 12 - V.text_size(v, 0.33)[0], yy), 0.33, V.TEXT, 1)

    # =================================================================== caveat
    inner = V.panel(img, *NOTE_P, "what this is not", "read this first")
    V.rounded_note(img, inner[0] + 5, inner[1] + 3, inner[2] - 10, [
        "No LiDAR, no time-of-flight. Ranges are",
        f"monocular depth x an ASSUMED {CFG.cam.height_above_ground_m:.2f} m camera",
        "height. Registration is drifting monocular",
        "VO (no IMU / encoders / loop closure), so",
        "late points inherit every earlier pose error.",
    ], title="", accent=V.BAD, lh=13)

    st["ms"].append((time.perf_counter() - t_start) * 1e3)
    if len(st["ms"]) > 60:
        del st["ms"][:-60]
    V.footer(img, W_OUT, H_OUT,
             f"{WALL_CHECK}.",
             f"accumulate + render {np.median(st['ms']):.0f} ms/frame (CPU, numpy+OpenCV)")
    return img


def _path_len(trail) -> float:
    if len(trail) < 2:
        return 0.0
    p = np.asarray(trail, np.float32)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


# =========================================================================== self-test

def _self_test() -> None:
    from ..config import WORK_DIR
    from ..pipeline import iter_packets

    print("r_lidar3d self-test - clip_04, 300 frames, real cached perception")
    state: dict = {"clip_id": "clip_04"}
    out = None
    times = []
    for p in iter_packets("clip_04", ("depth", "seg", "trav", "odom")):
        t0 = time.perf_counter()
        out = render(p, state)
        times.append((time.perf_counter() - t0) * 1e3)
        if p.idx in (60, 180, 299):
            cv2.imwrite(str(WORK_DIR / f"preview_l3d_{p.idx:03d}.png"), out)
    st = state["l3d"]
    smap = st["surface"]
    print(f"  render {out.shape} dtype={out.dtype}  "
          f"{np.mean(times):.0f} ms/frame (median {np.median(times):.0f})")
    print(f"  surface {smap.n_cells():,} cells, {smap.n_frames_used} frames fused, "
          f"{smap.n_frames_rejected} rejected for tracking")
    print(f"  path {_path_len(st['trail']):.2f} m")
    assert out.shape == (H_OUT, W_OUT, 3) and out.dtype == np.uint8
    print(f"  wrote {WORK_DIR / 'preview_l3d_*.png'}")


if __name__ == "__main__":
    _self_test()
