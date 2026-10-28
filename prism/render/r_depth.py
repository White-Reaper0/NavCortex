"""Stage 01 renderer: Depth Anything V2-Small + ground-plane metric alignment."""
from __future__ import annotations
import numpy as np
import cv2

from ..config import CFG
from ..types import FramePacket
from .. import viz_common as V
from ..perception.geometry import unproject, to_vehicle, GroundFit, height_slope_roughness

W, H = 1280, 720


def _elevation(packet: FramePacket):
    """Height above the fitted ground plane, restricted to the mapped range."""
    d = packet.depth
    valid = d.valid if d.valid is not None else np.isfinite(d.depth_m)
    normal = getattr(d, "_normal", None)
    if normal is None:
        normal = np.array([0.0, 1.0, 0.0], np.float32)
    fit = GroundFit(a=d.scale, b=d.shift, normal=np.asarray(normal, np.float32),
                    height=CFG.cam.height_above_ground_m)
    pv = to_vehicle(unproject(np.nan_to_num(d.depth_m, nan=0.0)), fit)
    z, _, _ = height_slope_roughness(pv, valid)
    fwd = pv[..., 1]
    in_range = valid & np.isfinite(fwd) & (fwd > 0.05) & (fwd < CFG.bev.range_forward_m)
    return z, in_range, fwd


def render(packet: FramePacket, state: dict) -> np.ndarray:
    img = V.canvas(W, H)
    d = packet.depth
    valid = d.valid if d.valid is not None else np.isfinite(d.depth_m)
    z, in_range, fwd = _elevation(packet)

    hist = state.setdefault("median_depth", [])
    hist.append(float(np.nanmedian(d.depth_m[valid])) if valid.any() else np.nan)
    rhist = state.setdefault("residual", [])
    rhist.append(float(d.align_residual))

    V.header(img, W, "prism",
             "01 - MONOCULAR DEPTH  |  Depth Anything V2-Small + ground-plane metric alignment",
             right=f"{packet.clip_id}   frame {packet.idx:03d}   t={packet.t:4.1f}s")

    # ------------------------------------------------------------ row 1
    r = V.panel(img, 12, 52, 414, 290, "camera", "one RGB stream")
    V.blit(img, packet.rgb, r)

    r = V.panel(img, 434, 52, 414, 290, "metric depth", "optical-axis range")
    V.blit(img, V.colorize_depth(d.depth_m, valid=valid), r)
    V.colorbar(img, r[0] + 14, r[1] + r[3] - 24, 180, 9, cv2.COLORMAP_TURBO, "near", "far", reverse=True)

    r = V.panel(img, 856, 52, 412, 290, "height above ground", f"within {CFG.bev.range_forward_m:.1f} m")
    hm = V.colorize_height(np.nan_to_num(z, nan=0.0), -0.55, 0.55, invalid=~in_range)
    V.blit(img, hm, r)
    V.colorbar(img, r[0] + 14, r[1] + r[3] - 24, 180, 9, cv2.COLORMAP_JET, "below -0.55", "+0.55 m above")

    # ------------------------------------------------------------ row 2
    r = V.panel(img, 12, 350, 414, 250, "raw network output", "q, relative inverse depth")
    qv = np.clip(d.rel_inv / max(float(np.nanpercentile(d.rel_inv, 99)), 1e-6), 0, 1)
    V.blit(img, cv2.applyColorMap((qv * 255).astype(np.uint8), cv2.COLORMAP_BONE),
           (r[0], r[1] + 4, r[2], 150))
    yy = r[1] + 172
    for i, (s, c) in enumerate([("unitless and affine-invariant", V.TEXT_DIM),
                                ("identical for a 2 m scene and a 20 m scene", V.TEXT_DIM),
                                ("-> not usable by a planner on its own", V.ACCENT)]):
        V.text(img, s, (r[0] + 14, yy + i * 18), 0.37, c)

    r = V.panel(img, 434, 350, 414, 250, "metric alignment", "1/D = a*q + b")
    x0, y0 = r[0] + 14, r[1] + 26
    rows = [("scale  a", f"{d.scale:7.3f}", V.ACCENT),
            ("shift  b", f"{d.shift:7.3f}", V.TEXT_DIM),
            ("fit residual", f"{d.align_residual:7.3f} 1/m", V.TEXT),
            ("ground inliers", f"{d.align_inliers:7d} px", V.TEXT),
            ("valid depth", f"{valid.mean()*100:6.1f} %", V.TEXT),
            ("camera height", f"{CFG.cam.height_above_ground_m:7.2f} m", V.WARN)]
    for i, (k, val, c) in enumerate(rows):
        V.text(img, k, (x0, y0 + i * 24), 0.42, V.TEXT_DIM)
        V.text(img, val, (x0 + 196, y0 + i * 24), 0.42, c, 1, V.FONT_B)
    V.text(img, "camera height is an ASSUMPTION -", (x0, y0 + 6 * 24 + 14), 0.36, V.WARN)
    V.text(img, "every metre shown scales directly with it.", (x0, y0 + 6 * 24 + 32), 0.36, V.WARN)

    r = V.panel(img, 856, 350, 412, 250, "telemetry", "")
    V.text(img, "MEDIAN SCENE RANGE   0 - 5 m", (r[0] + 14, r[1] + 20), 0.36, V.TEXT_DIM)
    V.sparkline(img, r[0] + 14, r[1] + 26, r[2] - 28, 58, hist[-150:], V.ACCENT, lo=0, hi=5)
    V.text(img, "GROUND-FIT RESIDUAL   0 - 0.3 (1/m)", (r[0] + 14, r[1] + 106), 0.36, V.TEXT_DIM)
    V.sparkline(img, r[0] + 14, r[1] + 112, r[2] - 28, 52, rhist[-150:], V.ACCENT2, lo=0, hi=0.3)
    ms = packet.timings_ms.get("depth", 0.0)
    V.text(img, f"{ms:5.1f} ms", (r[0] + 14, r[1] + 198), 0.56, V.TEXT, 1, V.FONT_B)
    V.text(img, f"{1000.0/max(ms,1e-3):4.1f} FPS", (r[0] + 108, r[1] + 198), 0.44, V.ACCENT, 1, V.FONT_B)
    V.text(img, "RTX 4050 Laptop, FP16", (r[0] + 196, r[1] + 198), 0.36, V.TEXT_DIM)
    V.text(img, "24.8 M params  |  in 518x518  |  out 640x360",
           (r[0] + 14, r[1] + 222), 0.36, V.TEXT_DIM)

    # ------------------------------------------------------------ note
    V.rounded_note(img, 12, 610, 1256, [
        "The network returns affine-invariant relative inverse depth q - a picture of relative distance, not a measurement. prism recovers metres by solving 1/D = a*q + b",
        "jointly with the ground plane, anchored on the assumed camera height. On a single plane a and b are not separately identifiable, so the scale-only model b = 0 is used.",
        "A detector answers 'what is that?'. This stage answers 'how far, and how high off the ground?' - what a chassis with 4.5 cm of clearance actually needs to know.",
    ], title="WHY THIS STAGE EXISTS")

    V.footer(img, W, H,
             left="Depth Anything V2-Small (Yang et al., NeurIPS 2024) - pretrained, not retrained here   |   inverse-depth alignment after Marsal et al., arXiv:2412.14103",
             right="offline video perception - not physical autonomy")
    return img
