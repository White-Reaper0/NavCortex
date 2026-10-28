"""LiDAR-like 3D reconstruction from a single RGB camera.

WHAT THIS IS
------------
The vehicle carries no LiDAR. This module takes the *metric monocular depth* produced by
Depth Anything V2-Small (rescaled to metres against the assumed camera height) and turns
it into the two things a LiDAR would have given us:

1. a structured **point cloud** in the vehicle frame, subsampled to a realistic point
   budget and coloured by height / traversability / terrain / camera RGB;
2. a **simulated multi-beam scan** - N rings at fixed elevation angles x M azimuth bins,
   returning the first-hit range per (ring, azimuth), i.e. exactly the range image and
   polar sweep a rotating LiDAR emits.

Both are clipped to one declared measuring volume (`SENSOR_MAX_RANGE_M`,
`BEAM_ELEV_HI_DEG`, `BEAM_ELEV_LO_DEG`, and the camera's horizontal FOV) by
`sensor_gate`, so the cloud panel and the scan panels describe the same device. Those
numbers are a specification chosen for this camera's geometry - the virtual origin sits
at the camera, only 12 cm above the ground - not a hardware datasheet. Sky-labelled
pixels are dropped before anything else, because a real LiDAR gets no return from sky.

WHAT THIS IS NOT (say this out loud in every renderer)
------------------------------------------------------
* No time-of-flight measurement exists anywhere in this pipeline. Ranges are *inferred*
  from a monocular network, then scaled by an assumed camera height of
  `CFG.cam.height_above_ground_m`. They are a reconstruction, not a measurement.
* Returns exist **only inside the camera's field of view** (~92 deg horizontal). A real
  360 deg LiDAR sees behind and beside the vehicle; this cannot. The polar display keeps
  those sectors explicitly blank and labelled rather than filling them in.
* There is exactly one range per ray: no second returns, no dual-return foliage
  penetration, no returns through glass, dust or rain.
* Accuracy degrades with distance far faster than a real LiDAR's, because monocular
  depth error grows roughly with the square of range.

Everything here is numpy/OpenCV - no GPU, no torch.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from ..config import CFG, N_TRAV, TERRAIN_COLORS_BGR, TRAV_COLORS_BGR, OBSTACLE
from ..types import FramePacket
from ..io_utils import ego_mask
from . import geometry as geo
from .mapping import nominal_ground_fit, _scaled_K

__all__ = ["PointCloud", "RingScan", "Camera3D", "ViewPoint", "LidarizeStage",
           "beam_elevations",
           "HEIGHT_LO", "HEIGHT_HI", "DISPLAY_RANGE_M",
           "SENSOR_MAX_RANGE_M", "BEAM_ELEV_HI_DEG", "BEAM_ELEV_LO_DEG", "N_RINGS", "N_AZIM",
           "sensor_gate",
           "depth_to_cloud", "simulate_lidar", "orbit_viewpoint", "render_cloud",
           "render_range_image", "render_polar_scan", "splat_disc",
           "draw_sensor_footprint",
           "LIDAR_CAVEAT"]


# --------------------------------------------------------------------- virtual sensor spec
# One place defines what the synthetic sensor is, so the cloud, the ring scan, the polar
# display and the range image all describe *the same* device instead of three different
# ones. Numbers chosen for the geometry this camera actually has: the virtual origin sits
# at the camera, only `CFG.cam.height_above_ground_m` above the ground.
SENSOR_MAX_RANGE_M = 8.0     # declared max range. Monocular depth reaches further but its
                             # error grows ~r^2, so past this it is not worth displaying.
BEAM_ELEV_HI_DEG = 12.0      # top beam. +12 deg reaches 1.7 m of height at 8 m, which is
                             # what it takes to put a railing or a wall on the display.
BEAM_ELEV_LO_DEG = -26.0     # bottom beam, strikes the ground ~0.27 m ahead.
N_RINGS = 32
N_AZIM = 360

DISPLAY_RANGE_M = SENSOR_MAX_RANGE_M   # polar-display radius == declared sensor range

LIDAR_CAVEAT = (
    "Reconstruction from ONE RGB camera - no LiDAR, no time-of-flight. "
    "Range is inferred monocular depth scaled by an assumed "
    f"{CFG.cam.height_above_ground_m:.2f} m camera height. "
    "Returns exist only inside the ~92 deg camera FOV: nothing behind or beside the vehicle."
)


# =========================================================================== containers

@dataclass
class PointCloud:
    """A subsampled point cloud in the vehicle frame (x right, y forward, z up)."""
    xyz: np.ndarray                     # (N,3) float32 metres
    rgb: np.ndarray                     # (N,3) uint8 BGR from the camera
    height_bgr: np.ndarray              # (N,3) uint8 colour-by-height
    trav_bgr: np.ndarray                # (N,3) uint8 colour-by-traversability class
    terrain_bgr: np.ndarray             # (N,3) uint8 colour-by-prism-7 terrain class
    range_m: np.ndarray                 # (N,) float32 range from the virtual sensor
    trav: np.ndarray                    # (N,) uint8 traversability class
    n_valid_px: int = 0                 # valid depth pixels before subsampling
    n_in_fov_px: int = 0                # of those, how many fall inside the sensor volume
    budget: int = 0
    #: every valid point at full depth resolution, kept transiently (never cached) so
    #: the ring scan can be cast through the dense surface instead of the subsample
    dense_xyz: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def colors(self, mode: str) -> np.ndarray:
        return {"height": self.height_bgr, "trav": self.trav_bgr,
                "terrain": self.terrain_bgr, "rgb": self.rgb}[mode]


@dataclass
class RingScan:
    """A synthesised multi-beam scan: first-hit range per (ring, azimuth)."""
    range_img: np.ndarray               # (R,A) float32 metres, NaN = no return
    z_img: np.ndarray                   # (R,A) float32 height of the first hit, NaN = none
    intensity: np.ndarray               # (R,A) float32 0..1 SYNTHETIC return strength
    elev_deg: np.ndarray                # (R,) beam elevations
    az_deg: np.ndarray                  # (A,) azimuth bin centres, 0 = straight ahead
    fov_mask: np.ndarray                # (A,) bool, azimuths the camera can possibly see
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    max_range_m: float = 8.0

    @property
    def hit(self) -> np.ndarray:
        return np.isfinite(self.range_img)

    @property
    def n_returns(self) -> int:
        return int(self.hit.sum())

    @property
    def fill_frac(self) -> float:
        """Fraction of the *visible* (in-FOV) beams that produced a return."""
        n = int(self.hit.shape[0] * self.fov_mask.sum())
        return float(self.hit[:, self.fov_mask].sum() / max(n, 1))


# =========================================================================== cloud

_PERM_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _stable_permutation(h: int, w: int) -> np.ndarray:
    """A fixed pixel ordering, so the subsample is spatially uniform *and* temporally
    stable (a fresh random draw every frame makes the cloud boil)."""
    key = (h, w)
    if key not in _PERM_CACHE:
        _PERM_CACHE[key] = np.random.default_rng(1337).permutation(h * w).astype(np.int64)
    return _PERM_CACHE[key]


HEIGHT_LO, HEIGHT_HI = -0.15, 0.60      # point-cloud height ramp, metres above ground


def _colorize_height_pts(z: np.ndarray, lo: float = HEIGHT_LO, hi: float = HEIGHT_HI) -> np.ndarray:
    n = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
    ramp = cv2.applyColorMap((n * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)
    return ramp.reshape(-1, 3)


def sensor_gate(points_veh: np.ndarray,
                max_range_m: float = SENSOR_MAX_RANGE_M,
                elev_hi_deg: float = BEAM_ELEV_HI_DEG,
                elev_lo_deg: float = BEAM_ELEV_LO_DEG,
                min_range_m: float = 0.05) -> np.ndarray:
    """True where a vehicle-frame point lies inside the virtual sensor's measuring volume.

    The synthetic sensor has a declared range and a declared vertical beam spread, exactly
    like a catalogue LiDAR. Applying the *same* envelope to the displayed cloud and to the
    ray-cast scan is what keeps the two panels describing one device: without it the hero
    view fills with tree canopy and sky-adjacent depth that no beam could ever return, and
    the viewer is left comparing two unrelated pictures.
    """
    p = np.asarray(points_veh, np.float32).reshape(-1, 3)
    q = p - np.array([0.0, 0.0, CFG.cam.height_above_ground_m], np.float32)
    rho = np.hypot(q[:, 0], q[:, 1])
    r = np.sqrt(rho * rho + q[:, 2] * q[:, 2])
    with np.errstate(invalid="ignore", divide="ignore"):
        el = np.degrees(np.arctan2(q[:, 2], np.maximum(rho, 1e-6)))
    g = (np.isfinite(p).all(axis=1) & (r > min_range_m) & (r < max_range_m)
         & (el <= elev_hi_deg) & (el >= elev_lo_deg))
    return g.reshape(np.asarray(points_veh).shape[:-1])


def depth_to_cloud(packet: FramePacket, budget: int = 26000,
                   max_range_m: Optional[float] = None,
                   ground_normal: Optional[np.ndarray] = None,
                   keep_dense: bool = True,
                   elev_hi_deg: float = BEAM_ELEV_HI_DEG,
                   elev_lo_deg: float = BEAM_ELEV_LO_DEG,
                   drop_sky: bool = True) -> Optional[PointCloud]:
    """Metric depth -> a structured, subsampled point cloud in the vehicle frame.

    The cloud is clipped to the virtual sensor's measuring volume (`sensor_gate`) *before*
    the subsample is drawn, so the whole point budget lands inside the volume the scan
    panels describe, and the surviving points stay the same ones from frame to frame.
    """
    if packet.depth is None or packet.depth.depth_m is None:
        return None
    d = np.asarray(packet.depth.depth_m, np.float32)
    h, w = d.shape
    rmax = float(max_range_m if max_range_m is not None else SENSOR_MAX_RANGE_M)

    valid = np.isfinite(d) & (d > 0.08) & (d < rmax * 1.25)
    if packet.depth.valid is not None:
        valid &= np.asarray(packet.depth.valid, bool)
    valid &= ego_mask(h, w)
    if drop_sky and packet.seg is not None and getattr(packet.seg, "label", None) is not None:
        # a real LiDAR gets no return from the sky; neither should this one
        sky = np.asarray(packet.seg.label, np.uint8)
        if sky.shape[:2] != (h, w):
            sky = cv2.resize(sky, (w, h), interpolation=cv2.INTER_NEAREST)
        valid &= (sky != 0)
    n_valid = int(valid.sum())
    if n_valid < 32:
        return None

    if packet.geom is not None and getattr(packet.geom, "points_veh", None) is not None:
        pv = np.asarray(packet.geom.points_veh, np.float32).reshape(-1, 3)
    else:
        n = ground_normal
        if n is None:
            for src in (packet.depth, packet.geom):
                v = getattr(src, "ground_normal", None) if src is not None else None
                if v is None and src is not None:
                    v = getattr(src, "_normal", None)
                if v is not None:
                    n = np.asarray(v, np.float32).reshape(3)
                    break
        fit = (nominal_ground_fit() if n is None else
               geo.GroundFit(a=1.0, b=0.0, normal=np.asarray(n, np.float32),
                             height=CFG.cam.height_above_ground_m, ok=True))
        K = CFG.cam.K if (w, h) == (CFG.cam.width, CFG.cam.height) else _scaled_K(w, h)
        pv = geo.to_vehicle(geo.unproject(d, K), fit).reshape(-1, 3)

    valid &= sensor_gate(pv, rmax, elev_hi_deg, elev_lo_deg).reshape(h, w)
    n_in_fov = int(valid.sum())
    if n_in_fov < 32:
        return None

    perm = _stable_permutation(h, w)
    vf = valid.reshape(-1)
    sel = perm[vf[perm]]                       # valid pixels, in the stable order
    if sel.size > budget:
        sel = sel[:budget]
    xyz = pv[sel].astype(np.float32)
    keep = np.isfinite(xyz).all(axis=1)
    sel, xyz = sel[keep], xyz[keep]
    if sel.size < 16:
        return None

    # colours -------------------------------------------------------------
    if packet.rgb is not None:
        img = packet.rgb
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        rgb = img.reshape(-1, 3)[sel].astype(np.uint8)
    else:
        rgb = np.full((sel.size, 3), 160, np.uint8)

    height_bgr = _colorize_height_pts(xyz[:, 2])

    trav = np.full(sel.size, 3, np.uint8)
    if packet.trav is not None and getattr(packet.trav, "label", None) is not None:
        lab = np.asarray(packet.trav.label, np.uint8)
        if lab.shape[:2] != (h, w):
            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
        trav = np.clip(lab.reshape(-1)[sel], 0, N_TRAV - 1).astype(np.uint8)
    trav_bgr = TRAV_COLORS_BGR[trav]

    terr = np.zeros(sel.size, np.uint8)
    if packet.seg is not None and getattr(packet.seg, "label", None) is not None:
        lab = np.asarray(packet.seg.label, np.uint8)
        if lab.shape[:2] != (h, w):
            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
        terr = np.clip(lab.reshape(-1)[sel], 0, len(TERRAIN_COLORS_BGR) - 1).astype(np.uint8)
    terrain_bgr = TERRAIN_COLORS_BGR[terr]

    origin = np.array([0.0, 0.0, CFG.cam.height_above_ground_m], np.float32)
    rng = np.linalg.norm(xyz - origin, axis=1).astype(np.float32)

    dense = None
    if keep_dense:
        dv = pv[vf]
        dense = dv[np.isfinite(dv).all(axis=1)].astype(np.float32)

    return PointCloud(xyz=xyz, rgb=rgb, height_bgr=height_bgr, trav_bgr=trav_bgr,
                      terrain_bgr=terrain_bgr, range_m=rng, trav=trav,
                      n_valid_px=n_valid, n_in_fov_px=n_in_fov, budget=budget,
                      dense_xyz=dense)


# =========================================================================== ring scan

def _segment_first(idx: np.ndarray, val: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(min of `val` per bin of `idx`, index of that minimum). NaN / -1 where empty."""
    out = np.full(n, np.nan, np.float32)
    arg = np.full(n, -1, np.int64)
    if idx.size == 0:
        return out, arg
    order = np.lexsort((val, idx))
    ids = idx[order]
    first = np.concatenate(([True], ids[1:] != ids[:-1]))
    pos = order[first]
    out[ids[first]] = val[pos]
    arg[ids[first]] = pos
    return out, arg


def beam_elevations(n_rings: int = N_RINGS, elev_hi_deg: float = BEAM_ELEV_HI_DEG,
                    elev_lo_deg: float = BEAM_ELEV_LO_DEG, k: float = 1.8) -> np.ndarray:
    """Non-uniform beam elevations, dense near the horizon (descending order).

    Automotive multi-beam units (VLP-32C and friends) do exactly this: they crowd beams
    around the horizon where the useful range is and space them out steeply downward,
    where every beam hits the ground within a metre anyway. It matters a lot here -
    the virtual sensor sits at the camera, only 12 cm above the ground, so ground at
    2 m and ground at 8 m are barely 2.5 deg apart in elevation. Uniform spacing would
    burn half the rings on the first 70 cm.

    The exponential spacing gives ~0.4 deg near the horizon widening to ~2.2 deg at the
    steepest beam, in the same ballpark as a VLP-32C's 0.33-6 deg spread.
    """
    u = np.linspace(0.0, 1.0, n_rings, dtype=np.float64)
    s = (np.exp(k * u) - 1.0) / (np.exp(k) - 1.0)
    return (elev_hi_deg - s * (elev_hi_deg - elev_lo_deg)).astype(np.float32)


def simulate_lidar(cloud, n_rings: int = N_RINGS, n_az: int = N_AZIM,
                   elev_hi_deg: float = BEAM_ELEV_HI_DEG, elev_lo_deg: float = BEAM_ELEV_LO_DEG,
                   beam_tol: float = 0.85,
                   max_range_m: Optional[float] = None) -> RingScan:
    """Cast `n_rings` beams at fixed elevations through the reconstructed surface.

    For every (ring, azimuth) bin the **nearest** reconstructed point whose elevation
    falls inside the beam's angular width is taken as the first return - the same
    "first hit wins" rule a real rotating LiDAR obeys. Bins with no candidate stay NaN
    (no return), which is how the blind sectors outside the camera FOV stay honest.

    `cloud` may be a `PointCloud` (its dense surface is used when available, because the
    scan should be cast through the reconstructed *surface*, not through the display
    subsample) or a raw (N,3) vehicle-frame array.
    """
    if isinstance(cloud, PointCloud):
        pts = cloud.dense_xyz if cloud.dense_xyz is not None else cloud.xyz
    else:
        pts = np.asarray(cloud, np.float32)
    rmax = float(max_range_m if max_range_m is not None else SENSOR_MAX_RANGE_M)
    origin = np.array([0.0, 0.0, CFG.cam.height_above_ground_m], np.float32)
    elev = beam_elevations(n_rings, elev_hi_deg, elev_lo_deg)
    # per-ring angular half-width = half the gap to the nearest neighbouring beam
    gaps = np.abs(np.diff(elev)) if n_rings > 1 else np.array([1.0], np.float32)
    halfw = np.empty(n_rings, np.float32)
    halfw[:-1] = gaps
    halfw[-1] = gaps[-1] if n_rings > 1 else 1.0
    halfw[1:] = np.minimum(halfw[1:], gaps)
    halfw = np.maximum(halfw * 0.5, 0.12)
    az_edges = np.linspace(-180.0, 180.0, n_az + 1, dtype=np.float32)
    az_c = (0.5 * (az_edges[:-1] + az_edges[1:])).astype(np.float32)

    rimg = np.full((n_rings, n_az), np.nan, np.float32)
    zimg = np.full((n_rings, n_az), np.nan, np.float32)

    p = pts - origin
    rho = np.hypot(p[:, 0], p[:, 1])
    r = np.sqrt(rho ** 2 + p[:, 2] ** 2)
    ok = (r > 0.05) & (r < rmax)
    if ok.any():
        po = p[ok]
        rr = r[ok].astype(np.float32)
        az = np.degrees(np.arctan2(po[:, 0], po[:, 1]))          # 0 = forward, + = right
        el = np.degrees(np.arctan2(po[:, 2], np.maximum(rho[ok], 1e-6)))
        # nearest beam (elevations are non-uniform, so bisect), then reject any point
        # that falls outside that beam's angular width - a beam is not a plane
        asc = elev[::-1]
        j = np.clip(np.searchsorted(asc, el.astype(np.float32)), 1, n_rings - 1)
        lo_d = np.abs(el - asc[j - 1])
        hi_d = np.abs(el - asc[j])
        jj = np.where(lo_d <= hi_d, j - 1, j)
        ridx = (n_rings - 1) - jj
        good = np.abs(el - elev[ridx]) <= beam_tol * 2.0 * halfw[ridx]
        aidx = np.clip(((az + 180.0) / 360.0 * n_az).astype(np.int64), 0, n_az - 1)
        g = np.nonzero(good)[0]
        if g.size:
            flat = ridx[g] * n_az + aidx[g]
            first, arg = _segment_first(flat, rr[g], n_rings * n_az)
            rimg = first.reshape(n_rings, n_az)
            hz = np.full(n_rings * n_az, np.nan, np.float32)
            hit = arg >= 0
            hz[hit] = po[g[arg[hit]], 2] + origin[2]
            zimg = hz.reshape(n_rings, n_az)

    # SYNTHETIC return strength: a 1/r^2 falloff, not a measured reflectance.
    with np.errstate(invalid="ignore"):
        inten = np.clip(1.0 / (1.0 + (rimg / 2.5) ** 2), 0, 1).astype(np.float32)

    hfov = CFG.cam.hfov_deg * 0.5
    fov_mask = np.abs(az_c) <= hfov

    return RingScan(range_img=rimg, z_img=zimg, intensity=inten, elev_deg=elev,
                    az_deg=az_c, fov_mask=fov_mask, origin=origin, max_range_m=rmax)


# =========================================================================== 3D view

@dataclass
class ViewPoint:
    """A virtual camera orbiting the vehicle in the vehicle frame."""
    azim_deg: float = 0.0        # 0 = behind the vehicle looking forward, + = orbit right
    elev_deg: float = 33.0       # above the ground plane
    dist_m: float = 4.3          # eye distance from the look-at target
    target: tuple = (0.0, 1.55, 0.28)
    vfov_deg: float = 50.0


def orbit_viewpoint(t: float, sweep_deg: float = 27.0, period_s: float = 10.0,
                    elev_deg: float = 33.0, dist_m: float = 4.3,
                    target=(0.0, 1.55, 0.28)) -> ViewPoint:
    """A slow sinusoidal orbit across the clip - parallax is what sells the 3D.

    Both the azimuth sweep and the elevation bob run off the *same* phase, at 1x and
    0.5x, so the motion is smooth and periodic with no beat. The sweep stays well inside
    +-40 deg: swing further and the near-field ground slab turns edge-on and the cloud
    collapses to a line, which reads as a broken render rather than as parallax.
    """
    ph = 2.0 * np.pi * (t / max(period_s, 1e-6))
    return ViewPoint(azim_deg=sweep_deg * float(np.sin(ph)),
                     elev_deg=elev_deg + 5.0 * float(np.sin(ph * 0.5)),
                     dist_m=dist_m, target=tuple(target))


class Camera3D:
    """Pinhole virtual camera for rendering the cloud (vehicle frame -> pixels)."""

    def __init__(self, view: ViewPoint, size: tuple[int, int]):
        self.w, self.h = int(size[0]), int(size[1])
        tgt = np.asarray(view.target, np.float64)
        a, e = np.deg2rad(view.azim_deg), np.deg2rad(view.elev_deg)
        # azimuth 0 puts the eye behind the vehicle (-y), elevation lifts it along +z
        off = np.array([np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)])
        self.eye = tgt + off * view.dist_m
        f = tgt - self.eye
        f /= (np.linalg.norm(f) + 1e-12)
        up_w = np.array([0.0, 0.0, 1.0])
        right = np.cross(f, up_w)
        nr = np.linalg.norm(right)
        if nr < 1e-8:
            right = np.array([1.0, 0.0, 0.0])
            nr = 1.0
        right /= nr
        up = np.cross(right, f)
        self.R = np.stack([right, -up, f])          # rows -> camera X right, Y down, Z fwd
        self.fy = (self.h / 2.0) / np.tan(np.deg2rad(view.vfov_deg) / 2.0)
        self.fx = self.fy
        self.cx, self.cy = self.w / 2.0, self.h / 2.0
        self.view = view

    def project(self, pts: np.ndarray):
        """(N,3) vehicle-frame points -> (px, py, z_cam, in_front)."""
        pc = (np.asarray(pts, np.float64) - self.eye) @ self.R.T
        z = pc[:, 2]
        front = z > 0.05
        zz = np.where(front, z, 1.0)
        px = self.fx * pc[:, 0] / zz + self.cx
        py = self.fy * pc[:, 1] / zz + self.cy
        return px, py, z, front


def splat_disc(img: np.ndarray, px: np.ndarray, py: np.ndarray, colors: np.ndarray,
               radius: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Vectorised painter's-algorithm disc splatting.

    Points are ordered far -> near and written with a single fancy-index assignment, so
    near points land on top of far ones (numpy's last-write-wins on duplicate indices is
    exactly the painter's algorithm). Radius is per point, which is how the size
    attenuation with distance is done.
    """
    h, w = img.shape[:2]
    n = px.size
    if n == 0:
        return img
    rad = np.clip(np.rint(radius), 0, 4).astype(np.int32)
    order = np.argsort(-np.asarray(depth, np.float32), kind="stable")
    rank = np.empty(n, np.int64)
    rank[order] = np.arange(n)                      # rank 0 = farthest, drawn first

    pxi = np.rint(px).astype(np.int32)
    pyi = np.rint(py).astype(np.int32)
    rmax = int(rad.max())

    # group offsets by the minimum radius that includes them: one mask per level
    levels: dict[int, list[tuple[int, int]]] = {}
    for dy in range(-rmax, rmax + 1):
        for dx in range(-rmax, rmax + 1):
            need = int(np.ceil(np.sqrt(dx * dx + dy * dy) - 1e-9))
            if need <= rmax:
                levels.setdefault(need, []).append((dx, dy))

    Pf, Cf, Kf = [], [], []
    for need, offs in levels.items():
        sel = rad >= need
        if not sel.any():
            continue
        sx, sy, sc, sk = pxi[sel], pyi[sel], colors[sel], rank[sel]
        for dx, dy in offs:
            xx, yy = sx + dx, sy + dy
            ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
            if not ok.any():
                continue
            Pf.append(yy[ok].astype(np.int64) * w + xx[ok])
            Cf.append(sc[ok])
            Kf.append(sk[ok])
    if not Pf:
        return img
    P = np.concatenate(Pf)
    C = np.concatenate(Cf)
    K = np.concatenate(Kf)
    o = np.argsort(K, kind="stable")
    img.reshape(-1, 3)[P[o]] = C[o]
    return img


def _draw_polyline_3d(img, cam: Camera3D, pts: np.ndarray, color, thick=1, closed=False):
    px, py, z, front = cam.project(pts)
    if not front.all():
        return
    p = np.stack([px, py], 1).astype(np.int32)
    cv2.polylines(img, [p], closed, color, thick, cv2.LINE_AA)


def draw_ground_grid(img, cam: Camera3D, extent_x=(-3.0, 3.0), extent_y=(0.0, 8.0),
                     step=1.0, color=(52, 48, 44), axis_color=(78, 72, 66)):
    """A 1 m wireframe grid on z = 0 - the spatial reference that makes a cloud readable."""
    x0, x1 = extent_x
    y0, y1 = extent_y
    for x in np.arange(x0, x1 + 1e-6, step):
        line = np.stack([np.full(24, x), np.linspace(y0, y1, 24), np.zeros(24)], 1)
        _draw_polyline_3d(img, cam, line, axis_color if abs(x) < 1e-6 else color, 1)
    for y in np.arange(y0, y1 + 1e-6, step):
        line = np.stack([np.linspace(x0, x1, 24), np.full(24, y), np.zeros(24)], 1)
        _draw_polyline_3d(img, cam, line, color, 1)
    return img


def draw_sensor_footprint(img, cam: Camera3D, rmax: float = SENSOR_MAX_RANGE_M,
                          hfov_deg: Optional[float] = None, ring_step_m: float = 2.0,
                          color=(64, 78, 96), edge=(84, 104, 128)):
    """The measuring volume on the ground: the FOV wedge plus labelled range arcs.

    Drawing it in 3D is the only way the hero panel can show what the polar panel says in
    words - everything outside this wedge is unseen, not empty.
    """
    hf = np.deg2rad((CFG.cam.hfov_deg if hfov_deg is None else hfov_deg) * 0.5)
    for rr in np.arange(ring_step_m, rmax + 1e-6, ring_step_m):
        a = np.linspace(-hf, hf, 40)
        arc = np.stack([np.sin(a) * rr, np.cos(a) * rr, np.zeros_like(a)], 1)
        _draw_polyline_3d(img, cam, arc, color, 1)
        # label at the right-hand end of the arc, out of the way of the cloud itself
        px, py, z, front = cam.project(arc[int(len(a) * 0.88)][None])
        if front[0] and 0 <= px[0] < img.shape[1] - 34 and 12 <= py[0] < img.shape[0]:
            cv2.putText(img, f"{rr:.0f} m", (int(px[0]) + 5, int(py[0]) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (108, 122, 140), 1, cv2.LINE_AA)
    for s in (-1.0, 1.0):
        ray = np.stack([np.sin(s * hf) * np.linspace(0, rmax, 24),
                        np.cos(s * hf) * np.linspace(0, rmax, 24),
                        np.zeros(24)], 1)
        _draw_polyline_3d(img, cam, ray, edge, 1)
    return img


def draw_vehicle_box(img, cam: Camera3D, color=(90, 190, 255)):
    """The UGV envelope at scale, centred on the origin (ground under the camera)."""
    hw, L = CFG.ugv.width_m / 2.0, CFG.ugv.length_m
    hgt = max(CFG.cam.height_above_ground_m, 0.10)
    y0, y1 = -L * 0.62, L * 0.38
    base = np.array([[-hw, y0, 0], [hw, y0, 0], [hw, y1, 0], [-hw, y1, 0]], np.float64)
    top = base.copy(); top[:, 2] = hgt
    _draw_polyline_3d(img, cam, base, color, 2, closed=True)
    _draw_polyline_3d(img, cam, top, color, 1, closed=True)
    for i in range(4):
        _draw_polyline_3d(img, cam, np.stack([base[i], top[i]]), color, 1)
    return img


def render_cloud(cloud: PointCloud, view: ViewPoint, size: tuple[int, int],
                 mode: str = "height", bg=(16, 15, 14), grid: bool = True,
                 vehicle: bool = True, fog: bool = True,
                 point_scale: float = 1.0, footprint: bool = True,
                 rmax: float = SENSOR_MAX_RANGE_M) -> np.ndarray:
    """Render the cloud from a virtual viewpoint. Painter's algorithm, size attenuation."""
    w, h = int(size[0]), int(size[1])
    img = np.empty((h, w, 3), np.uint8)
    img[:] = bg
    cam = Camera3D(view, (w, h))
    if grid:
        draw_ground_grid(img, cam, extent_x=(-4.0, 4.0), extent_y=(0.0, rmax))
    if footprint:
        draw_sensor_footprint(img, cam, rmax)

    px, py, z, front = cam.project(cloud.xyz)
    m = front & (px > -8) & (px < w + 8) & (py > -8) & (py < h + 8)
    if m.any():
        col = cloud.colors(mode)[m].astype(np.float32)
        zz = z[m]
        if fog:
            # atmospheric attenuation: far points recede into the background
            a = np.clip(1.15 - zz / (view.dist_m * 3.4), 0.25, 1.0)[:, None]
            col = col * a + np.array(bg, np.float32) * (1.0 - a)
        # size attenuation: a point subtends fewer pixels the farther away it is
        rad = np.clip(point_scale * (cam.fy * 0.0085) / np.maximum(zz, 0.2), 0.6, 2.4)
        splat_disc(img, px[m], py[m], col.astype(np.uint8), rad, zz)

    if vehicle:
        draw_vehicle_box(img, cam)
    return img


# =========================================================================== 2D displays

def render_range_image(scan: RingScan, size: tuple[int, int],
                       fov_only: bool = True, bg=(24, 22, 20)) -> np.ndarray:
    """The classic LiDAR range image: rows = beams, cols = azimuth, colour = range."""
    r = scan.range_img
    if fov_only:
        r = r[:, scan.fov_mask]
    m = np.isfinite(r)
    n = np.zeros_like(r, np.float32)
    if m.any():
        n[m] = np.clip(r[m] / scan.max_range_m, 0, 1)
    img = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~m] = bg
    return cv2.resize(img, (int(size[0]), int(size[1])), interpolation=cv2.INTER_NEAREST)


def render_polar_scan(scan: RingScan, size: tuple[int, int], bg=(16, 15, 14),
                      ring_step_m: float = 2.0, max_r: Optional[float] = None,
                      blind_color=(38, 34, 31), fov_color=(23, 21, 20),
                      label_rings: bool = True) -> np.ndarray:
    """A top-down PPI sweep - the display a rotating LiDAR would drive.

    The sectors the camera cannot see are drawn as an explicit lighter-grey dead zone
    with no returns in it, because that limitation is the whole point of being honest
    about a monocular "LiDAR".
    """
    w, h = int(size[0]), int(size[1])
    img = np.empty((h, w, 3), np.uint8)
    img[:] = bg
    cx, cy = w / 2.0, h / 2.0
    R = min(w, h) * 0.45
    rmax = float(max_r if max_r is not None else scan.max_range_m)
    scale = R / max(rmax, 1e-6)
    hfov = CFG.cam.hfov_deg * 0.5
    ic = (int(cx), int(cy))

    # the whole disc is dead ground until proven otherwise ...
    cv2.circle(img, ic, int(R), blind_color, -1, cv2.LINE_AA)
    # ... and only the camera's wedge can ever contain a return
    cv2.ellipse(img, ic, (int(R), int(R)), -90.0, -hfov, hfov, fov_color, -1, cv2.LINE_AA)

    for rr in np.arange(ring_step_m, rmax + 1e-6, ring_step_m):
        cv2.circle(img, ic, int(rr * scale), (62, 58, 53), 1, cv2.LINE_AA)
        if label_rings:
            cv2.putText(img, f"{rr:.0f}", (int(cx + 3), int(cy - rr * scale - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (96, 91, 85), 1, cv2.LINE_AA)
    for a in (-hfov, 0.0, hfov):
        ar = np.deg2rad(a)
        cv2.line(img, ic, (int(cx + np.sin(ar) * R), int(cy - np.cos(ar) * R)),
                 (78, 73, 66), 1, cv2.LINE_AA)
    cv2.circle(img, ic, int(R), (72, 67, 61), 1, cv2.LINE_AA)

    m = scan.hit
    if m.any():
        ri, ai = np.nonzero(m)
        rr = np.minimum(scan.range_img[ri, ai], rmax)
        az = np.deg2rad(scan.az_deg[ai])
        px = cx + np.sin(az) * rr * scale
        py = cy - np.cos(az) * rr * scale
        z = scan.z_img[ri, ai]
        col = _colorize_height_pts(np.nan_to_num(z, nan=0.0)).astype(np.uint8)
        ok = ((px >= 0) & (px < w) & (py >= 0) & (py < h)
              & (scan.range_img[ri, ai] <= rmax))
        if ok.any():
            splat_disc(img, px[ok], py[ok], col[ok],
                       np.full(int(ok.sum()), 1.0), rr[ok])

    cv2.circle(img, ic, 3, (90, 190, 255), -1, cv2.LINE_AA)
    return img


# =========================================================================== stage

class LidarizeStage:
    """Thin stage wrapper. Attaches `packet.cloud` and `packet.scan` (extra attributes)."""

    def __init__(self, device: str = "cuda", budget: int = 26000,
                 n_rings: int = N_RINGS, n_az: int = N_AZIM, orbit: bool = True, **kw):
        self.device = device
        self.budget = int(budget)
        self.n_rings = int(n_rings)
        self.n_az = int(n_az)
        self.orbit = bool(orbit)
        self.last_stats: dict = {}
        self.reset()

    def reset(self) -> None:
        self._t = 0.0
        self.last_stats = {}

    def __call__(self, packet: FramePacket) -> FramePacket:
        t0 = time.perf_counter()
        cloud = depth_to_cloud(packet, budget=self.budget)
        scan = None if cloud is None else simulate_lidar(cloud, self.n_rings, self.n_az)
        packet.cloud = cloud
        packet.scan = scan
        dt = (time.perf_counter() - t0) * 1e3
        packet.timings_ms["lidarize"] = dt
        self.last_stats = dict(ms=dt, n_points=0 if cloud is None else len(cloud),
                               n_returns=0 if scan is None else scan.n_returns,
                               fill=0.0 if scan is None else scan.fill_frac)
        self._t = packet.t
        return packet


# =========================================================================== self-test

def _self_test() -> None:
    from .mapping import synthetic_scene, _packet_from_scene

    print("=" * 74)
    print("lidarize self-test - synthetic scene (0.30 m box @ 2.0-2.3 m, ditch @ 3.0 m)")
    print("=" * 74)
    depth, valid, trav, seg, truth = synthetic_scene(0.0)
    pk = _packet_from_scene(0, depth, valid, trav, seg)
    pk.rgb = np.full((360, 640, 3), 90, np.uint8)

    t0 = time.perf_counter()
    cloud = depth_to_cloud(pk, budget=26000)
    t_cloud = (time.perf_counter() - t0) * 1e3
    assert cloud is not None
    print(f"\ncloud: {len(cloud)} pts from {cloud.n_valid_px} valid px, "
          f"{cloud.n_in_fov_px} inside the sensor volume  ({t_cloud:.1f} ms)")

    # the displayed cloud must live inside exactly the volume the beams sweep, or the
    # hero panel and the scan panels are describing two different sensors
    o = np.array([0.0, 0.0, CFG.cam.height_above_ground_m], np.float32)
    q = cloud.xyz - o
    rho = np.hypot(q[:, 0], q[:, 1])
    el = np.degrees(np.arctan2(q[:, 2], np.maximum(rho, 1e-6)))
    rr_c = np.sqrt(rho ** 2 + q[:, 2] ** 2)
    out = int(((el > BEAM_ELEV_HI_DEG + 1e-3) | (el < BEAM_ELEV_LO_DEG - 1e-3)
               | (rr_c > SENSOR_MAX_RANGE_M + 1e-3)).sum())
    print(f"  elevation {el.min():+.1f}..{el.max():+.1f} deg "
          f"(spec {BEAM_ELEV_LO_DEG:+.0f}..{BEAM_ELEV_HI_DEG:+.0f}), "
          f"range max {rr_c.max():.2f} m (spec {SENSOR_MAX_RANGE_M:.1f}) -> "
          f"{'PASS' if out == 0 else f'FAIL ({out} pts outside the sensor volume)'}")

    # temporal stability: the same scene twice must select the identical points, because
    # a fresh random subsample every frame is what makes a rendered cloud boil
    c2 = depth_to_cloud(_packet_from_scene(1, depth, valid, trav, seg), budget=26000)
    same = (len(c2) == len(cloud)) and bool(np.array_equal(c2.xyz, cloud.xyz))
    print(f"  subsample repeatable on identical input -> {'PASS' if same else 'FAIL'}")
    print(f"  x {cloud.xyz[:,0].min():+.2f}..{cloud.xyz[:,0].max():+.2f} m   "
          f"y {cloud.xyz[:,1].min():+.2f}..{cloud.xyz[:,1].max():+.2f} m   "
          f"z {cloud.xyz[:,2].min():+.2f}..{cloud.xyz[:,2].max():+.2f} m")
    box = ((cloud.xyz[:, 0] > 0.30) & (cloud.xyz[:, 0] < 0.60)
           & (cloud.xyz[:, 1] > 1.95) & (cloud.xyz[:, 1] < 2.35))
    print(f"  points on the box footprint: {int(box.sum())}, "
          f"z max {cloud.xyz[box, 2].max():.3f} m (truth {truth['box_h']:.2f}) -> "
          f"{'PASS' if abs(cloud.xyz[box,2].max() - truth['box_h']) < 0.02 else 'FAIL'}")
    print(f"  obstacle-labelled pts      : {int((cloud.trav == OBSTACLE).sum())}")

    t0 = time.perf_counter()
    scan = simulate_lidar(cloud, n_rings=32, n_az=360)
    t_scan = (time.perf_counter() - t0) * 1e3
    print(f"\nring scan: {scan.range_img.shape} (rings x azimuth)  ({t_scan:.1f} ms)")
    print(f"  beams {scan.elev_deg[0]:+.1f}..{scan.elev_deg[-1]:+.1f} deg, non-uniform: "
          f"{abs(scan.elev_deg[1]-scan.elev_deg[0]):.2f} deg near the horizon, "
          f"{abs(scan.elev_deg[-1]-scan.elev_deg[-2]):.2f} deg at the bottom")
    print(f"  returns {scan.n_returns} / {scan.range_img.size} bins "
          f"({scan.fill_frac:.1%} of the in-FOV beams)")
    out_fov = scan.hit[:, ~scan.fov_mask].sum()
    print(f"  returns OUTSIDE the camera FOV: {out_fov}  -> "
          f"{'PASS (no invented 360 coverage)' if out_fov == 0 else 'FAIL'}")
    rr = scan.range_img[scan.hit]
    print(f"  range {rr.min():.2f}..{rr.max():.2f} m, median {np.median(rr):.2f} m")

    # Each ring has its own first hit: the low rings strike nearby ground, the rings
    # that graze the box front face must return ~2.0 m at a height above the ground.
    ai = int(np.argmin(np.abs(scan.az_deg - np.degrees(np.arctan2(0.45, 2.15)))))
    col_r, col_z = scan.range_img[:, ai], scan.z_img[:, ai]
    on_face = np.isfinite(col_r) & (col_z > 0.05)
    print(f"\n  azimuth {scan.az_deg[ai]:+.1f} deg passes through the box:")
    print(f"    rings returning above ground: {int(on_face.sum())}, "
          f"range {np.nanmin(col_r[on_face]):.2f}..{np.nanmax(col_r[on_face]):.2f} m "
          f"(box front face at y=2.00 m) -> "
          f"{'PASS' if abs(np.nanmedian(col_r[on_face]) - 2.05) < 0.25 else 'FAIL'}")
    print(f"    their heights: {np.nanmin(col_z[on_face]):.2f}..{np.nanmax(col_z[on_face]):.2f} m "
          f"(face spans 0.00..{truth['box_h']:.2f}) -> "
          f"{'PASS' if abs(np.nanmax(col_z[on_face]) - truth['box_h']) < 0.04 else 'FAIL'}")
    ground_ring = int(np.nanargmin(np.where(np.isfinite(col_r), col_r, np.inf)))
    print(f"    lowest beam ({scan.elev_deg[ground_ring]:+.1f} deg) hits ground at "
          f"{col_r[ground_ring]:.2f} m - correct: a steep beam strikes the ground first")

    # renders
    for name, fn in (("cloud", lambda: render_cloud(cloud, orbit_viewpoint(0.0), (780, 460))),
                     ("polar", lambda: render_polar_scan(scan, (300, 300))),
                     ("range", lambda: render_range_image(scan, (420, 130)))):
        t0 = time.perf_counter()
        im = fn()
        print(f"  render {name:6s}: {im.shape} in {(time.perf_counter()-t0)*1e3:.1f} ms")

    t0 = time.perf_counter()
    for i in range(10):
        render_cloud(cloud, orbit_viewpoint(i / 30.0), (780, 460))
    print(f"\n10 orbiting renders at 780x460: {(time.perf_counter()-t0)*100:.1f} ms/frame")
    print("\nCAVEAT:", LIDAR_CAVEAT)
    print("done.")


if __name__ == "__main__":
    _self_test()
