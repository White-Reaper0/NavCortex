"""Renderer for stage 07_lidar_like_pointcloud - LiDAR-like 3D from one RGB camera.

Four views of the same reconstruction:

  * hero      - the point cloud in 3D from a slowly orbiting virtual camera, with a
                metre ground grid and the vehicle envelope for scale;
  * polar     - the top-down PPI sweep a rotating LiDAR would drive, with the sectors
                the camera cannot see left explicitly blank;
  * range img - the (rings x azimuth) range image, the native output format of a
                multi-beam LiDAR;
  * camera    - the single sensor all of it came from.

All four panels are clipped to one declared measuring volume (see
`perception.lidarize.sensor_gate`), so the hero cloud contains exactly the points the
beams could return and nothing else. Before that gate the hero filled with tree canopy
and sky-adjacent depth that appeared in no other panel, which made the two halves of the
frame look like two unrelated sensors.

The note panel is not decoration. Nothing here is measured by a LiDAR; every range is
inferred from a monocular network and scaled by an assumed camera height, and the
renderer says so on every frame.
"""
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from .. import viz_common as V
from ..config import CFG, TRAV_CLASSES, TRAV_COLORS_BGR
from ..perception.lidarize import (BEAM_ELEV_HI_DEG, BEAM_ELEV_LO_DEG, DISPLAY_RANGE_M,
                                   HEIGHT_HI, HEIGHT_LO, LIDAR_CAVEAT, N_AZIM, N_RINGS,
                                   SENSOR_MAX_RANGE_M,
                                   PointCloud, RingScan, depth_to_cloud, orbit_viewpoint,
                                   render_cloud, render_polar_scan, render_range_image,
                                   simulate_lidar)
from ..types import FramePacket

W_OUT, H_OUT = 1280, 720

STAGE_TITLE = "07 - LIDAR-LIKE 3D RECONSTRUCTION"

#: virtual-camera orbit across the clip - parallax is what makes a point cloud read as 3D.
#: One full sweep per 10 s clip: slow enough that a viewer can read the geometry while it
#: turns, fast enough that the parallax is unmistakable.
ORBIT_SWEEP_DEG = 27.0
ORBIT_PERIOD_S = 10.0

COLOR_MODES = {
    "height": "colour = height above ground",
    "trav": "colour = traversability class",
    "terrain": "colour = prism-7 terrain class",
    "rgb": "colour = camera pixel",
}

# ---- layout ---------------------------------------------------------------
HERO = (12, 50, 830, 464)          # x, y, w, h
CAM_P = (850, 50, 418, 188)
POLAR = (850, 244, 418, 270)
BOT_Y, BOT_H = 522, 168            # bottom band; footer starts at H_OUT-24 = 696
RANGE_P = (12, BOT_Y, 560, BOT_H)
STATS_P = (580, BOT_Y, 262, BOT_H)
NOTE_P = (850, BOT_Y, 418, BOT_H)


def _get_cloud(packet: FramePacket, state: dict) -> tuple[Optional[PointCloud], Optional[RingScan]]:
    """Use what the stage attached, otherwise build it here (renderers must stand alone).

    Whatever path is taken, `packet.timings_ms['lidarize']` ends up holding a real
    measurement of this frame's cloud + ray-cast cost - the footer prints it, so it must
    not be left at zero just because the renderer, not the stage, did the work.
    """
    cloud = getattr(packet, "cloud", None)
    scan = getattr(packet, "scan", None)
    t0 = time.perf_counter()
    built = False
    if cloud is None:
        cloud = depth_to_cloud(packet, budget=state.get("lidar_budget", 26000))
        built = True
    if cloud is not None and scan is None:
        scan = simulate_lidar(cloud, n_rings=state.get("n_rings", N_RINGS),
                              n_az=state.get("n_az", N_AZIM))
        built = True
    if built:
        packet.timings_ms["lidarize"] = (time.perf_counter() - t0) * 1e3
    return cloud, scan


def _scrim(img, x, y, w, h, strength: float = 0.72, fade: int = 110, bg=(16, 15, 14)):
    """Darken a rectangle of the 3D render so overlay text stays legible over it."""
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


def _range_axes(img, rect, scan: RingScan):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), V.EDGE, 1)
    az = scan.az_deg[scan.fov_mask]
    V.text(img, f"{az[0]:+.0f} deg", (x + 3, y + h + 13), 0.33, V.TEXT_DIM, 1)
    V.text(img, "azimuth  0", (x + w // 2 - 26, y + h + 13), 0.33, V.TEXT_DIM, 1)
    V.text(img, f"{az[-1]:+.0f} deg",
           (x + w - V.text_size(f"{az[-1]:+.0f} deg", 0.33)[0] - 3, y + h + 13), 0.33,
           V.TEXT_DIM, 1)
    V.text(img, f"beam {scan.elev_deg[0]:+.0f}", (x - 50, y + 9), 0.30, V.TEXT_DIM, 1)
    V.text(img, f"{scan.elev_deg[-1]:+.0f} deg", (x - 50, y + h), 0.30, V.TEXT_DIM, 1)


# --------------------------------------------------------------------------- main

def render(packet: FramePacket, state: dict) -> np.ndarray:
    img = V.canvas(W_OUT, H_OUT)
    V.header(img, W_OUT, "prism", STAGE_TITLE,
             right=f"{packet.clip_id}   frame {packet.idx:03d}   t={packet.t:5.2f}s")

    cloud, scan = _get_cloud(packet, state)
    if cloud is None:
        V.panel(img, 12, 60, W_OUT - 24, 560, "no depth", "nothing to reconstruct")
        V.text(img, "packet.depth.depth_m is missing - run the depth stage first",
               (40, 340), 0.6, V.TEXT_DIM, 1)
        V.footer(img, W_OUT, H_OUT, LIDAR_CAVEAT, "")
        return img

    mode = state.get("lidar_color_mode", "height")
    hist = state.setdefault("lidar_hist", {"ret": [], "pts": []})
    hist["ret"].append(float(scan.n_returns) if scan else 0.0)
    hist["pts"].append(float(len(cloud)))
    for k in hist:
        if len(hist[k]) > 150:
            state["lidar_hist"][k] = hist[k][-150:]

    # =================================================================== hero 3D
    view = orbit_viewpoint(packet.t, sweep_deg=ORBIT_SWEEP_DEG, period_s=ORBIT_PERIOD_S)
    inner = V.panel(img, *HERO, "3D point cloud",
                    f"orbiting virtual camera | {COLOR_MODES[mode]}")
    cw, ch = inner[2] - 6, inner[3] - 6
    view_img = render_cloud(cloud, view, (cw, ch), mode=mode, rmax=SENSOR_MAX_RANGE_M)
    img[inner[1] + 3:inner[1] + 3 + ch, inner[0] + 3:inner[0] + 3 + cw] = view_img
    hx, hy = inner[0] + 3, inner[1] + 3

    # scrims: the overlay text sits on top of a live 3D render, so darken what is behind
    # it rather than hoping the cloud never wanders under a caption. Faded out on the
    # right so the scrim never reads as a rectangle pasted over the scene.
    _scrim(img, hx, hy, min(cw, 700), 102)
    _scrim(img, hx, hy + ch - 50, min(cw, 330), 50)

    V.text(img, f"{len(cloud):,} points   ({cloud.n_in_fov_px:,} of "
                f"{cloud.n_valid_px:,} valid depth pixels fall inside the sensor volume)",
           (hx + 14, hy + 20), 0.38, V.TEXT, 1)
    V.text(img, f"virtual sensor  {scan.range_img.shape[0]} beams  "
                f"{BEAM_ELEV_HI_DEG:+.0f} to {BEAM_ELEV_LO_DEG:+.0f} deg elevation   "
                f"{CFG.cam.hfov_deg:.0f} deg horizontal   {SENSOR_MAX_RANGE_M:.0f} m range",
           (hx + 14, hy + 38), 0.35, V.ACCENT2, 1)
    V.text(img, f"that spec is a choice we made, not a datasheet - past ~"
                f"{SENSOR_MAX_RANGE_M:.0f} m the monocular error dominates",
           (hx + 14, hy + 55), 0.32, (128, 122, 114), 1)
    V.text(img, f"the VIRTUAL camera orbits  azim {view.azim_deg:+5.1f} deg   "
                f"elev {view.elev_deg:4.1f} deg   {view.dist_m:.1f} m back "
                f"- the vehicle itself is not turning",
           (hx + 14, hy + 73), 0.32, (128, 122, 114), 1)
    V.text(img, "outside the wedge: no camera coverage, no returns",
           (hx + 14, hy + 91), 0.32, (118, 112, 104), 1)

    V.colorbar(img, hx + 14, hy + ch - 34, 210, 10, cv2.COLORMAP_TURBO,
               lo_lab=f"{HEIGHT_LO:+.2f} m", hi_lab=f"{HEIGHT_HI:+.2f} m",
               title="colour = height above the fitted ground plane")
    if mode == "trav":
        V.legend(img, hx + cw - 96, hy + 22, TRAV_CLASSES, TRAV_COLORS_BGR, 0.34,
                 swatch=10, gap=17)

    # =================================================================== camera
    inner = V.panel(img, *CAM_P, "the only sensor", "one RGB camera")
    if packet.rgb is not None:
        V.blit(img, packet.rgb, (inner[0] + 2, inner[1] + 2, inner[2] - 4, inner[3] - 4))
    else:
        V.text(img, "no rgb frame", (inner[0] + 14, inner[1] + 44), 0.45, V.TEXT_DIM, 1)

    # =================================================================== polar
    inner = V.panel(img, *POLAR, "simulated 360 deg scan",
                    "first-hit range, top-down")
    side = min(inner[2] - 10, inner[3] - 26)
    px = inner[0] + (inner[2] - side) // 2
    py = inner[1] + 3
    img[py:py + side, px:px + side] = render_polar_scan(scan, (side, side),
                                                        max_r=DISPLAY_RANGE_M)
    cv2.rectangle(img, (px, py), (px + side, py + side), V.EDGE, 1)
    V.text(img, "BLIND", (px + side // 2 - 17, py + side - 26), 0.42, (126, 120, 112), 1)
    V.text(img, "no camera coverage here", (px + side // 2 - 64, py + side - 12), 0.32,
           (108, 102, 96), 1)
    V.text(img, "ahead", (px + side // 2 - 15, py + 12), 0.33, (120, 150, 175), 1)
    # the panel is wider than the disc; use the margins to name what the wedge is
    lx = inner[0] + 6
    rx2 = px + side + 8
    V.text(img, "left", (lx, py + side // 2 - 8), 0.33, V.TEXT_DIM, 1)
    V.text(img, f"-{CFG.cam.hfov_deg / 2:.0f}", (lx, py + side // 2 + 6), 0.33, V.TEXT_DIM, 1)
    V.text(img, "right", (rx2, py + side // 2 - 8), 0.33, V.TEXT_DIM, 1)
    V.text(img, f"+{CFG.cam.hfov_deg / 2:.0f}", (rx2, py + side // 2 + 6), 0.33, V.TEXT_DIM, 1)
    V.text(img, "seen", (rx2, py + 30), 0.32, (150, 175, 150), 1)
    V.text(img, "unseen", (lx, py + side - 40), 0.32, (120, 114, 106), 1)
    V.text(img, f"rings every 2 m to {DISPLAY_RANGE_M:.0f} m   |   "
                f"lit wedge = the {CFG.cam.hfov_deg:.0f} deg the camera sees",
           (inner[0] + 8, inner[1] + inner[3] - 8), 0.33, V.TEXT_DIM, 1)

    # =================================================================== range image
    inner = V.panel(img, *RANGE_P, "range image",
                    f"{scan.range_img.shape[0]} beams x "
                    f"{int(scan.fov_mask.sum())} azimuth bins - a LiDAR's native format")
    rw, rh = inner[2] - 68, inner[3] - 54
    rx, ry = inner[0] + 58, inner[1] + 6
    img[ry:ry + rh, rx:rx + rw] = render_range_image(scan, (rw, rh))
    _range_axes(img, (rx, ry, rw, rh), scan)
    V.colorbar(img, rx, ry + rh + 22, 150, 8, cv2.COLORMAP_TURBO,
               lo_lab="0 m", hi_lab=f"{scan.max_range_m:.1f} m")
    V.text(img, f"dark = no return: sky, past {scan.max_range_m:.0f} m, "
                f"or the masked RC body",
           (rx + 166, ry + rh + 30), 0.34, V.TEXT_DIM, 1)

    # =================================================================== stats
    inner = V.panel(img, *STATS_P, "scan", "this frame")
    sx, sy, sw, sh = inner
    x = sx + 10
    rr = scan.range_img[scan.hit]
    rows = [
        ("points shown", f"{len(cloud):,}"),
        ("beam returns", f"{scan.n_returns:,}"),
        ("in-FOV beams hit", f"{scan.fill_frac * 100:.0f}%"),
        ("beam spacing", f"{abs(scan.elev_deg[1] - scan.elev_deg[0]):.2f}"
                         f"-{abs(scan.elev_deg[-1] - scan.elev_deg[-2]):.1f} deg"),
        ("range", f"{rr.min():.2f} - {rr.max():.2f} m" if rr.size else "-"),
        ("median range", f"{np.median(rr):.2f} m" if rr.size else "-"),
    ]
    # Fixed geometry, laid out from the panel's own rect: the sparkline block is
    # reserved first and the rows fill what is left, so nothing can run off the bottom
    # edge however the numbers grow.
    spark_h, spark_lab_h, pad = 22, 14, 8
    spark_y = sy + sh - pad - spark_h
    row_pitch = 14
    row_y = sy + 15
    for i, (k, v) in enumerate(rows):
        yy = row_y + i * row_pitch
        V.text(img, k, (x, yy), 0.34, V.TEXT_DIM, 1)
        V.text(img, v, (sx + sw - 12 - V.text_size(v, 0.34)[0], yy), 0.34, V.TEXT, 1)
    sep = spark_y - spark_lab_h - 6
    cv2.line(img, (x, sep), (sx + sw - 12, sep), V.EDGE, 1)
    V.text(img, "beam returns / frame", (x, spark_y - 5), 0.32, V.TEXT_DIM, 1)
    V.sparkline(img, x, spark_y, sw - 22, spark_h, hist["ret"], color=V.ACCENT, lo=0.0)

    # =================================================================== caveat
    inner = V.panel(img, *NOTE_P, "what this is not", "read this before believing it")
    V.rounded_note(img, inner[0] + 5, inner[1] + 4, inner[2] - 10, [
        "No LiDAR. No time-of-flight. Every range here is",
        "monocular depth, scaled by an ASSUMED camera",
        f"height of {CFG.cam.height_above_ground_m:.2f} m - not a calibrated measurement.",
        "Returns exist only inside the ~92 deg camera FOV:",
        "nothing behind or beside the vehicle - the grey",
        "sectors above are unseen, not empty. One return",
        "per ray, and error grows with range far faster",
        "than a real sensor's.",
    ], title="", accent=V.BAD, lh=14)

    # rolling median so the footer reports a stable number rather than one noisy sample
    ms_hist = state.setdefault("lidar_ms", [])
    ms_hist.append(float(packet.timings_ms.get("lidarize", 0.0)))
    if len(ms_hist) > 60:
        del ms_hist[:-60]
    V.footer(img, W_OUT, H_OUT,
             "Depth Anything V2-Small monocular depth -> vehicle frame -> "
             "ray-cast into fixed beam elevations. Reconstruction, not measurement.",
             f"cloud + ray-cast {np.median(ms_hist):.0f} ms/frame "
             f"(measured here, CPU, numpy+OpenCV)")
    return img


# =========================================================================== self-test

def _self_test() -> None:
    import time
    from ..config import WORK_DIR
    from ..perception.mapping import synthetic_scene, _packet_from_scene
    from ..perception.lidarize import LidarizeStage

    print("r_lidar self-test - rendering the synthetic scene")
    depth, valid, trav, seg, truth = synthetic_scene(0.0)
    stage = LidarizeStage("cpu")
    state: dict = {}
    times = []
    out = None
    for i in range(12):
        pk = _packet_from_scene(i, depth, valid, trav, seg)
        pk.t = i / 30.0
        inv = np.clip(1.4 / np.nan_to_num(depth, nan=12.0), 0, 1)
        pk.rgb = cv2.applyColorMap((inv * 255).astype(np.uint8), cv2.COLORMAP_BONE)
        stage(pk)
        t0 = time.perf_counter()
        out = render(pk, state)
        times.append((time.perf_counter() - t0) * 1e3)
    print(f"  render: {out.shape} dtype={out.dtype}  "
          f"{np.mean(times[2:]):.1f} ms/frame (median {np.median(times[2:]):.1f})")
    print(f"  stage : {stage.last_stats}")
    assert out.shape == (H_OUT, W_OUT, 3) and out.dtype == np.uint8

    p = WORK_DIR / "preview_lidar.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), out)
    print(f"  wrote {p}")

    # a second frame far into the orbit, to confirm the virtual camera actually moves
    pk.t = ORBIT_PERIOD_S * 0.25
    v0 = orbit_viewpoint(0.0, ORBIT_SWEEP_DEG, ORBIT_PERIOD_S)
    v1 = orbit_viewpoint(pk.t, ORBIT_SWEEP_DEG, ORBIT_PERIOD_S)
    print(f"  orbit : azim {v0.azim_deg:+.1f} -> {v1.azim_deg:+.1f} deg over "
          f"{ORBIT_PERIOD_S / 4:.1f} s")
    cv2.imwrite(str(WORK_DIR / "preview_lidar_orbit.png"), render(pk, state))
    print(f"  wrote {WORK_DIR / 'preview_lidar_orbit.png'}")


if __name__ == "__main__":
    _self_test()
