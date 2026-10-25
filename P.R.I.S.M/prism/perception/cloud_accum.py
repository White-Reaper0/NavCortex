"""Pose-registered 3-D surface reconstruction and a free 3-D camera for stage 07b.

WHAT THIS MODULE DOES
---------------------
Stage 07 (`perception/lidarize.py`) turns one frame of monocular depth into one
vehicle-frame point cloud. This module answers the next question: *what does the scene
look like after the vehicle has driven through it?* It registers every frame's cloud
into a single **world frame** using the visual-odometry pose, fuses it into a persistent
2.5-D elevation map, and triangulates that map into a low-poly, flat-shaded **surface** -
one geometric model of the corridor the vehicle drove down, not 300 stacked clouds.

A cloud is a sample of a surface; the surface is the thing a navigation stack actually
reasons about, and it is also the thing whose errors are visible. A mis-registered wall
in a point cloud looks like a slightly fatter cloud; in a mesh it turns from a cliff
into a ramp, and you can see it immediately.

FRAMES
------
vehicle : X right, Y forward, Z up, origin on the ground under the camera
world   : the VO map frame. `odom.npz/pose` is (x, y, yaw); the odometry integrates
          forward as `fwd = (-sin yaw, cos yaw)`, `right = (cos yaw, sin yaw)`, so a
          vehicle point maps to the world by a plain CCW rotation of (x, y) by yaw
          followed by a translation. See `veh_to_world`.

THE HONEST LIMITS OF THIS REGISTRATION (renderers must repeat these)
-------------------------------------------------------------------
* **The pose is 2-D.** VO estimates (x, y, yaw) only. Roll and pitch are *not* tracked;
  the vertical axis of every frame is taken from that frame's fitted ground plane, i.e.
  the reconstruction assumes the ground stays a single horizontal plane in the world.
  A real slope, a kerb climbed, or a suspension pitch is folded into z error.
* **Monocular VO drifts.** No IMU, no wheel encoders, no loop closure. Position error
  accumulates monotonically along the path, and a point registered at t = 8 s is
  registered through the entire chain of earlier pose estimates. That is why the
  surface has a *rolling window*: keeping every frame forever is not more information,
  it is more smear.
* **It is 2.5-D, one height per ground cell.** No overhangs, no bridges, no undersides.
  A branch over the trail is either the surface or it is nothing.
* **Scale is an assumption.** Every metre here traces back to
  `CFG.cam.height_above_ground_m` = 0.12 m, an assumed camera height, not a calibration.
* **Tracking loss must not be accumulated through.** When the front end declares a loss
  it *holds* the pose (see `odometry._declare_lost`), so any cloud registered during a
  loss is stamped at a stale pose and smears geometry into the map. `SurfaceMap`
  refuses those frames outright (`add(..., accept=False)`); the renderer shows the state.

Everything here is numpy / OpenCV. No torch, no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..config import CFG, TRAV_COLORS_BGR
from ..io_utils import ego_mask
from . import geometry as geo

__all__ = [
    "VOL_R_MIN", "VOL_R_MAX", "VOL_Z_MIN", "VOL_Z_MAX",
    "HEIGHT_LO", "HEIGHT_HI",
    "volume_gate", "frame_cloud_vehicle", "veh_to_world", "yaw_R2",
    "SurfaceMap", "Mesh",
    "shade_faces", "render_mesh", "LIGHT_DIR",
    "WorldCamera", "ChaseCamera", "chase_path",
    "colorize_height_pts", "render_points",
    "draw_polyline_3d", "draw_world_grid", "draw_trail_ribbon", "draw_vehicle_wire",
    "ACCUM_CAVEAT",
]

# --------------------------------------------------------------------- measuring volume
# The volume each frame contributes to the map. Deliberately *not* the stage-07 beam
# envelope: stage 07 gates by beam elevation because it is pretending to be a rotating
# multi-beam unit, and at 1 m range a +12 deg top beam clips everything above 21 cm --
# which would delete exactly the walls this stage exists to reconstruct. Here the gate is
# a plain range + height shell, which is what a mapping front end would actually use.
VOL_R_MIN = 0.30      # m, ground-plane range from the camera. Nearer than this is RC body.
VOL_R_MAX = 4.00      # m. MEASURED, not guessed. Fitting clip_04's brick wall as one
                      # world-frame plane and bucketing the residual by the range the
                      # return was taken at: 0.20 m RMS inside 1.5 m, 0.17 m at 1.5-2.5 m,
                      # 0.34 m at 2.5-3.5 m, 0.41 m at 3.5-4.5 m, 0.44 m beyond. Past
                      # ~4 m a return adds smear, not structure, so it is not mapped.
VOL_Z_MIN = -0.45     # m above the fitted ground plane (below = drop-off / bad depth)
VOL_Z_MAX = 1.30      # m. A 0.34 m UGV cannot be helped by tree canopy, and canopy
                      # is where monocular depth is worst; mapping it just buries the
                      # navigable surface under a curtain of points.

# The colour ramp spans exactly the mapped height band, so the colourbar and the volume
# gate describe the same thing and nothing on screen is off the end of the scale.
HEIGHT_LO, HEIGHT_HI = VOL_Z_MIN, VOL_Z_MAX     # height colour ramp, m above ground

ACCUM_CAVEAT = (
    "Accumulated by monocular VO: no IMU, no wheel encoders, no loop closure, "
    "2-D pose only (roll/pitch fixed by the per-frame ground-plane fit). It drifts."
)


# --------------------------------------------------------------------- per-frame cloud

def volume_gate(pts_veh: np.ndarray,
                r_min: float = VOL_R_MIN, r_max: float = VOL_R_MAX,
                z_min: float = VOL_Z_MIN, z_max: float = VOL_Z_MAX) -> np.ndarray:
    """True where a vehicle-frame point is inside the mapping volume.

    Range is measured in the ground plane from the camera position (x=0, y=0), which is
    the quantity monocular depth error actually scales with; height is measured against
    the fitted ground plane.
    """
    p = np.asarray(pts_veh, np.float32)
    flat = p.reshape(-1, 3)
    rho = np.hypot(flat[:, 0], flat[:, 1])
    g = (np.isfinite(flat).all(axis=1) & (rho >= r_min) & (rho <= r_max)
         & (flat[:, 2] >= z_min) & (flat[:, 2] <= z_max))
    return g.reshape(p.shape[:-1])


def _scaled_K(w: int, h: int) -> np.ndarray:
    """CFG intrinsics rescaled to an arbitrary map resolution."""
    K = CFG.cam.K.copy()
    sx, sy = w / float(CFG.cam.width), h / float(CFG.cam.height)
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return K


def frame_cloud_vehicle(depth_m: np.ndarray,
                        valid: Optional[np.ndarray] = None,
                        normal: Optional[np.ndarray] = None,
                        seg_label: Optional[np.ndarray] = None,
                        trav_label: Optional[np.ndarray] = None,
                        stride: int = 2,
                        r_max: float = VOL_R_MAX) -> tuple[np.ndarray, np.ndarray, int]:
    """One frame of metric depth -> (xyz_veh (N,3) float32, trav (N,) uint8, n_valid_px).

    Drops sky-labelled pixels (class 0) and everything under the ego mask, then applies
    `volume_gate`. `stride` subsamples the depth map on a fixed lattice, which keeps the
    per-frame contribution around 10-20 k points and, being a *fixed* lattice, keeps the
    surviving points identical from frame to frame instead of boiling.
    """
    d = np.asarray(depth_m, np.float32)
    h, w = d.shape
    m = np.isfinite(d) & (d > 0.10) & (d < r_max * 1.6)
    if valid is not None:
        m &= np.asarray(valid, bool)
    m &= ego_mask(h, w)
    if seg_label is not None:
        s = np.asarray(seg_label, np.uint8)
        if s.shape[:2] != (h, w):
            s = cv2.resize(s, (w, h), interpolation=cv2.INTER_NEAREST)
        m &= (s != 0)                       # no returns from sky
    n_valid = int(m.sum())

    sub = np.zeros_like(m)
    sub[::stride, ::stride] = True
    m &= sub
    if m.sum() < 16:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.uint8), n_valid

    n = np.array([0.0, 1.0, 0.0], np.float32) if normal is None else np.asarray(normal, np.float32).reshape(3)
    fit = geo.GroundFit(a=1.0, b=0.0, normal=n,
                        height=CFG.cam.height_above_ground_m, ok=True)
    K = CFG.cam.K if (w, h) == (CFG.cam.width, CFG.cam.height) else _scaled_K(w, h)
    pv = geo.to_vehicle(geo.unproject(np.nan_to_num(d, nan=0.0), K), fit)

    m &= volume_gate(pv, r_max=r_max)
    if m.sum() < 16:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.uint8), n_valid

    xyz = pv[m].astype(np.float32)
    if trav_label is not None:
        t = np.asarray(trav_label, np.uint8)
        if t.shape[:2] != (h, w):
            t = cv2.resize(t, (w, h), interpolation=cv2.INTER_NEAREST)
        trav = np.clip(t[m], 0, len(TRAV_COLORS_BGR) - 1).astype(np.uint8)
    else:
        trav = np.full(xyz.shape[0], 3, np.uint8)
    return xyz, trav, n_valid


# --------------------------------------------------------------------- registration

def yaw_R2(yaw: float) -> np.ndarray:
    """2x2 rotation taking vehicle (right, forward) into the VO world frame."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]], np.float32)


def veh_to_world(xyz_veh: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """(N,3) vehicle-frame points -> world frame using a 2-D VO pose (x, y, yaw).

    z passes through untouched: the pose carries no roll or pitch, so the reconstruction
    inherits the assumption that every frame's fitted ground plane is *the same* plane.
    """
    p = np.asarray(xyz_veh, np.float32).reshape(-1, 3)
    R = yaw_R2(float(pose[2]))
    out = np.empty_like(p)
    out[:, :2] = p[:, :2] @ R.T + np.asarray(pose[:2], np.float32)
    out[:, 2] = p[:, 2]
    return out


# --------------------------------------------------------------------- surface map

@dataclass
class Mesh:
    """A low-poly triangle surface cut out of the world elevation map."""
    verts: np.ndarray                   # (V,3) float32 world
    faces: np.ndarray                   # (F,3) int32 indices into verts
    face_z: np.ndarray                  # (F,) float32 centroid height above ground
    face_n: np.ndarray                  # (F,3) float32 unit face normals
    face_trav: np.ndarray               # (F,) uint8 dominant traversability class
    face_age: np.ndarray                # (F,) float32 frames since last observed
    face_qual: np.ndarray               # (F,) float32 VO quality behind the cell
    n_cells: int = 0                    # observed cells in the whole map

    def __len__(self) -> int:
        return int(self.faces.shape[0])


class SurfaceMap:
    """A pose-registered 2.5-D world elevation map, meshed into low-poly triangles.

    This is the actual reconstruction the stage shows. A point cloud is a *sample* of a
    surface; this is the surface. Every frame's gated cloud is registered into the world
    frame and dropped into a fixed world-aligned grid of `cell_m` cells; each cell keeps
    a temporally fused surface height, the dominant traversability class, how recently it
    was observed and the VO quality behind it. `mesh()` then walks a window of that grid
    and emits two triangles per fully-observed quad.

    Why a height field and not a full 3-D mesh: the pose is 2-D and the vertical axis
    comes from a per-frame ground-plane fit, so the honest representation of what this
    system knows is "one height per patch of ground", which is exactly a 2.5-D grid. A
    wall appears as a cliff in that field - which is right, and which is also the thing
    that shows immediately whether the registration is real, because a mis-registered
    wall becomes a *ramp* instead of a cliff.

    Per-cell height rule: the frame contribution is the **maximum** z of the samples that
    land in the cell (a mean would average an obstacle's face together with the ground in
    front of it and turn a 30 cm kerb into a 10 cm bump - the same rule
    `perception/mapping.py` uses for the BEV), fused into the map with an EMA so a single
    noisy depth pixel cannot punch a spike through the surface.
    """

    def __init__(self, cell_m: float = 0.15, n: int = 340, keep_frames: int = 210,
                 alpha: float = 0.30, min_hits: int = 1):
        self.cell_m = float(cell_m)
        self.n = int(n)
        self.keep_frames = int(keep_frames)
        self.alpha = float(alpha)
        self.min_hits = int(min_hits)
        self.reset()

    # -------------------------------------------------------------- state
    def reset(self) -> None:
        n = self.n
        self.origin: Optional[np.ndarray] = None      # world xy of cell (0,0) corner
        self.h = np.full(n * n, np.nan, np.float32)
        self.seen = np.full(n * n, -10 ** 6, np.int32)
        self.qual = np.zeros(n * n, np.float32)
        self.tcnt = np.zeros((4, n * n), np.float32)
        self.n_frames_used = 0
        self.n_frames_rejected = 0

    def n_cells(self) -> int:
        return int(np.isfinite(self.h).sum())

    def _set_origin(self, xy) -> None:
        self.origin = np.asarray(xy, np.float64) - 0.5 * self.n * self.cell_m

    # -------------------------------------------------------------- update
    def add(self, xyz_world: np.ndarray, trav: np.ndarray, frame_idx: int,
            quality: float = 1.0, accept: bool = True) -> int:
        """Fuse one registered frame into the surface. Returns cells touched.

        `accept=False` (tracking lost) drops the frame: the surface freezes rather than
        absorbing geometry stamped at a stale pose.
        """
        if not accept:
            self.n_frames_rejected += 1
            self._prune(frame_idx)
            return 0
        p = np.asarray(xyz_world, np.float32).reshape(-1, 3)
        if p.shape[0] == 0:
            self._prune(frame_idx)
            return 0
        if self.origin is None:
            self._set_origin(p[:, :2].mean(axis=0))
        ij = np.floor((p[:, :2] - self.origin) / self.cell_m).astype(np.int64)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.n) & (ij[:, 1] >= 0) & (ij[:, 1] < self.n)
        if not ok.any():
            self._prune(frame_idx)
            return 0
        flat = ij[ok, 1] * self.n + ij[ok, 0]
        z = p[ok, 2]
        t = np.asarray(trav, np.uint8).reshape(-1)[ok]

        cnt = np.bincount(flat, minlength=self.n * self.n)
        hmax = np.full(self.n * self.n, -1e9, np.float32)
        np.maximum.at(hmax, flat, z)
        obs = cnt >= self.min_hits
        if not obs.any():
            self._prune(frame_idx)
            return 0
        self.n_frames_used += 1

        fresh = obs & ~np.isfinite(self.h)
        upd = obs & np.isfinite(self.h)
        self.h[fresh] = hmax[fresh]
        a = self.alpha
        self.h[upd] = (1.0 - a) * self.h[upd] + a * hmax[upd]
        self.seen[obs] = frame_idx
        self.qual[obs] = (1.0 - a) * self.qual[obs] + a * float(quality)
        self.qual[fresh] = float(quality)
        for c in range(4):
            m = t == c
            if m.any():
                self.tcnt[c] += np.bincount(flat[m], minlength=self.n * self.n)
        self.tcnt *= 0.985                      # let a cell's class follow re-observation
        self._prune(frame_idx)
        return int(obs.sum())

    def _prune(self, frame_idx: int) -> None:
        stale = np.isfinite(self.h) & (self.seen <= frame_idx - self.keep_frames)
        if stale.any():
            self.h[stale] = np.nan
            self.tcnt[:, stale] = 0.0

    # -------------------------------------------------------------- mesh
    def height_at(self, xy: np.ndarray, default: float = 0.0) -> np.ndarray:
        """Surface height at world xy (nearest cell). Unobserved cells -> `default`."""
        p = np.asarray(xy, np.float64).reshape(-1, 2)
        out = np.full(p.shape[0], float(default), np.float32)
        if self.origin is None:
            return out
        ij = np.floor((p - self.origin) / self.cell_m).astype(np.int64)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.n) & (ij[:, 1] >= 0) & (ij[:, 1] < self.n)
        if ok.any():
            v = self.h[ij[ok, 1] * self.n + ij[ok, 0]]
            out[ok] = np.where(np.isfinite(v), v, default)
        return out

    def mesh(self, centre, half_m: float = 11.0, max_step_m: float = 2.2,
             smooth: bool = True) -> Mesh:
        """Triangulate the observed part of a window of the grid around `centre`.

        `smooth` runs a 3x3 *median* over the observed cells before meshing. A median,
        not a blur: it deletes the single-cell spikes a stray depth pixel leaves behind
        without rounding off the cliffs, which are the structures worth showing. Holes
        are never filled - an unobserved cell stays a hole in the surface, because
        unseen is not the same as flat.
        """
        empty = Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32),
                     np.zeros(0, np.float32), np.zeros((0, 3), np.float32),
                     np.zeros(0, np.uint8), np.zeros(0, np.float32),
                     np.zeros(0, np.float32), self.n_cells())
        if self.origin is None:
            return empty
        n, c = self.n, self.cell_m
        i0 = int(np.clip(np.floor((centre[0] - half_m - self.origin[0]) / c), 0, n - 2))
        i1 = int(np.clip(np.ceil((centre[0] + half_m - self.origin[0]) / c), 1, n - 1))
        j0 = int(np.clip(np.floor((centre[1] - half_m - self.origin[1]) / c), 0, n - 2))
        j1 = int(np.clip(np.ceil((centre[1] + half_m - self.origin[1]) / c), 1, n - 1))
        if i1 - i0 < 2 or j1 - j0 < 2:
            return empty

        H = self.h.reshape(n, n)[j0:j1 + 1, i0:i1 + 1]
        if smooth and H.shape[0] > 2 and H.shape[1] > 2:
            pad = np.pad(H, 1, constant_values=np.nan)
            stack = np.stack([pad[a:a + H.shape[0], b:b + H.shape[1]]
                              for a in range(3) for b in range(3)])
            # unobserved neighbours vote for the centre cell rather than being dropped:
            # that keeps `np.median` on a fully finite stack (fast, no NaN path) and
            # makes the filter a no-op at the ragged edge of the mapped area
            stack = np.where(np.isfinite(stack), stack, H[None])
            med = np.median(stack, axis=0)
            H = np.where(np.isfinite(H), med, np.nan).astype(np.float32)
        S = self.seen.reshape(n, n)[j0:j1 + 1, i0:i1 + 1]
        Q = self.qual.reshape(n, n)[j0:j1 + 1, i0:i1 + 1]
        T = self.tcnt.reshape(4, n, n)[:, j0:j1 + 1, i0:i1 + 1]
        gh, gw = H.shape
        xs = self.origin[0] + (np.arange(i0, i1 + 1) + 0.5) * c
        ys = self.origin[1] + (np.arange(j0, j1 + 1) + 0.5) * c
        X, Y = np.meshgrid(xs, ys)
        verts = np.stack([X.ravel(), Y.ravel(), np.nan_to_num(H, nan=0.0).ravel()],
                         1).astype(np.float32)

        good = np.isfinite(H)
        q = good[:-1, :-1] & good[:-1, 1:] & good[1:, :-1] & good[1:, 1:]
        if not q.any():
            return empty
        hh = np.stack([H[:-1, :-1], H[:-1, 1:], H[1:, :-1], H[1:, 1:]])
        hh = np.where(np.isfinite(hh), hh, 0.0)      # `q` already forbids NaN corners
        q &= (hh.max(axis=0) - hh.min(axis=0)) <= max_step_m
        jj, ii = np.nonzero(q)
        if jj.size == 0:
            return empty
        a = jj * gw + ii
        b = a + 1
        d = a + gw
        e = d + 1
        faces = np.concatenate([np.stack([a, b, e], 1), np.stack([a, e, d], 1)]).astype(np.int32)

        cell_t = np.argmax(T[:, :-1, :-1], axis=0)[jj, ii].astype(np.uint8)
        cell_s = S[:-1, :-1][jj, ii].astype(np.float32)
        cell_q = Q[:-1, :-1][jj, ii].astype(np.float32)
        face_trav = np.concatenate([cell_t, cell_t])
        face_seen = np.concatenate([cell_s, cell_s])
        face_qual = np.concatenate([cell_q, cell_q])

        p0 = verts[faces[:, 0]]
        p1 = verts[faces[:, 1]]
        p2 = verts[faces[:, 2]]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = nrm / np.maximum(ln, 1e-9)
        nrm[nrm[:, 2] < 0] *= -1.0                 # a height field always faces up
        face_z = ((p0[:, 2] + p1[:, 2] + p2[:, 2]) / 3.0).astype(np.float32)
        return Mesh(verts, faces, face_z, nrm.astype(np.float32), face_trav,
                    face_seen, face_qual, self.n_cells())

    def age_of(self, mesh: "Mesh", frame_idx: int) -> np.ndarray:
        return np.maximum(frame_idx - mesh.face_age, 0.0)


# --------------------------------------------------------------------- virtual camera

class WorldCamera:
    """A free pinhole camera in the world frame (eye / target / vertical FOV)."""

    def __init__(self, eye, target, size: tuple[int, int], vfov_deg: float = 46.0,
                 roll_up=(0.0, 0.0, 1.0)):
        self.w, self.h = int(size[0]), int(size[1])
        self.eye = np.asarray(eye, np.float64).reshape(3)
        self.target = np.asarray(target, np.float64).reshape(3)
        f = self.target - self.eye
        nf = np.linalg.norm(f)
        f = f / nf if nf > 1e-9 else np.array([0.0, 1.0, 0.0])
        up_w = np.asarray(roll_up, np.float64)
        right = np.cross(f, up_w)
        nr = np.linalg.norm(right)
        if nr < 1e-8:
            right, nr = np.array([1.0, 0.0, 0.0]), 1.0
        right /= nr
        up = np.cross(right, f)
        self.R = np.stack([right, -up, f])          # rows -> cam X right, Y down, Z fwd
        self.fy = (self.h / 2.0) / np.tan(np.deg2rad(vfov_deg) / 2.0)
        self.fx = self.fy
        self.cx, self.cy = self.w / 2.0, self.h / 2.0
        self.vfov_deg = float(vfov_deg)

    def to_cam(self, pts: np.ndarray) -> np.ndarray:
        return (np.asarray(pts, np.float64).reshape(-1, 3) - self.eye) @ self.R.T

    def project(self, pts: np.ndarray):
        """(N,3) world points -> (px, py, z_cam, in_front)."""
        pc = self.to_cam(pts)
        z = pc[:, 2]
        front = z > 0.06
        zz = np.where(front, z, 1.0)
        return (self.fx * pc[:, 0] / zz + self.cx,
                self.fy * pc[:, 1] / zz + self.cy, z, front)


def chase_path(pose: np.ndarray, smooth_s: float = 1.6, fps: float = 30.0) -> np.ndarray:
    """Smooth a (N,3) VO pose track into a camera anchor path (N,3) = x, y, yaw.

    A camera bolted straight onto a monocular VO pose is unwatchable: the per-frame
    jitter of the estimate becomes camera shake, and a yaw glitch becomes a whip pan.
    The anchor is therefore a zero-phase (forward+backward) box smoothing of the pose
    over ~`smooth_s` seconds, with the yaw unwrapped first so a +-pi wrap does not spin
    the camera. The vehicle marker still sits at the *raw* pose, so the wobble of the
    estimate stays visible relative to the smooth camera instead of being hidden.
    """
    p = np.asarray(pose, np.float64).reshape(-1, 3).copy()
    p[:, 2] = np.unwrap(p[:, 2])
    n = p.shape[0]
    k = max(3, int(round(smooth_s * fps)) | 1)
    if n < 3:
        return p.astype(np.float32)
    k = min(k, (n // 2) * 2 + 1)
    pad = k // 2
    out = np.empty_like(p)
    for j in range(3):
        s = np.pad(p[:, j], (pad, pad), mode="edge")
        c = np.convolve(s, np.ones(k) / k, mode="valid")     # forward
        s2 = np.pad(c, (pad, pad), mode="edge")
        out[:, j] = np.convolve(s2, np.ones(k) / k, mode="valid")   # and backward
    return out.astype(np.float32)


class ChaseCamera:
    """Follows the vehicle from behind and above, along the smoothed pose path."""

    def __init__(self, back_m: float = 4.6, up_m: float = 2.5, ahead_m: float = 2.2,
                 look_z: float = 0.30, vfov_deg: float = 46.0):
        self.back_m, self.up_m = float(back_m), float(up_m)
        self.ahead_m, self.look_z = float(ahead_m), float(look_z)
        self.vfov_deg = float(vfov_deg)

    def at(self, anchor: np.ndarray, size: tuple[int, int],
           orbit_deg: float = 0.0) -> WorldCamera:
        x, y, yaw = float(anchor[0]), float(anchor[1]), float(anchor[2])
        a = yaw + np.deg2rad(orbit_deg)
        fwd = np.array([-np.sin(a), np.cos(a)])
        eye = np.array([x - fwd[0] * self.back_m, y - fwd[1] * self.back_m, self.up_m])
        f2 = np.array([-np.sin(yaw), np.cos(yaw)])
        tgt = np.array([x + f2[0] * self.ahead_m, y + f2[1] * self.ahead_m, self.look_z])
        return WorldCamera(eye, tgt, size, self.vfov_deg)


# --------------------------------------------------------------------- point rendering

def colorize_height_pts(z: np.ndarray, lo: float = HEIGHT_LO, hi: float = HEIGHT_HI) -> np.ndarray:
    n = np.clip((np.asarray(z, np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    ramp = cv2.applyColorMap((n * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)
    return ramp.reshape(-1, 3)


def _splat(img: np.ndarray, px, py, colors, radius, depth) -> np.ndarray:
    """Painter's-algorithm disc splat: far points first, near points overwrite.

    numpy's last-write-wins on duplicate fancy indices *is* the painter's algorithm, so
    the whole cloud goes down in one assignment per offset level instead of a per-point
    Python loop.
    """
    h, w = img.shape[:2]
    n = int(np.size(px))
    if n == 0:
        return img
    rad = np.clip(np.rint(radius), 0, 3).astype(np.int32)
    order = np.argsort(-np.asarray(depth, np.float32), kind="stable")
    rank = np.empty(n, np.int64)
    rank[order] = np.arange(n)
    pxi = np.rint(px).astype(np.int32)
    pyi = np.rint(py).astype(np.int32)
    rmax = int(rad.max())
    levels: dict[int, list[tuple[int, int]]] = {}
    for dy in range(-rmax, rmax + 1):
        for dx in range(-rmax, rmax + 1):
            need = int(np.ceil(np.sqrt(dx * dx + dy * dy) - 1e-9))
            if need <= rmax:
                levels.setdefault(need, []).append((dx, dy))
    P, C, K = [], [], []
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
            P.append(yy[ok].astype(np.int64) * w + xx[ok])
            C.append(sc[ok])
            K.append(sk[ok])
    if not P:
        return img
    P, C, K = np.concatenate(P), np.concatenate(C), np.concatenate(K)
    o = np.argsort(K, kind="stable")
    img.reshape(-1, 3)[P[o]] = C[o]
    return img


def render_points(img: np.ndarray, cam: WorldCamera, xyz: np.ndarray,
                  colors_bgr: np.ndarray, bg=(16, 15, 14),
                  fog_m: float = 22.0, point_scale: float = 1.0,
                  alpha: Optional[np.ndarray] = None,
                  rad_lo: float = 0.7, rad_hi: float = 2.6,
                  fog_floor: float = 0.38) -> int:
    """Project + splat a world-frame cloud. Returns the number of points drawn.

    Point radius attenuates as 1/z (a point subtends fewer pixels the farther it is) and
    colour recedes into the background with distance, which is what gives a flat splat
    field a readable depth ordering.
    """
    if xyz.shape[0] == 0:
        return 0
    px, py, z, front = cam.project(xyz)
    h, w = img.shape[:2]
    m = front & (px > -6) & (px < w + 6) & (py > -6) & (py < h + 6)
    if not m.any():
        return 0
    col = colors_bgr[m].astype(np.float32)
    zz = z[m]
    a = np.clip(1.12 - zz / max(fog_m, 1e-3), fog_floor, 1.0)
    if alpha is not None:
        a = a * np.clip(alpha[m], 0.0, 1.0)
    a = a[:, None]
    col = col * a + np.array(bg, np.float32) * (1.0 - a)
    rad = np.clip(point_scale * (cam.fy * 0.0090) / np.maximum(zz, 0.25), rad_lo, rad_hi)
    _splat(img, px[m], py[m], col.astype(np.uint8), rad, zz)
    return int(m.sum())


# --------------------------------------------------------------------- mesh rendering

#: key light for the flat shading, in world coordinates (from above, front-left)
LIGHT_DIR = np.array([-0.45, -0.35, 0.82], np.float64)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)


def shade_faces(mesh: Mesh, base_bgr: np.ndarray, ambient: float = 0.42,
                diffuse: float = 0.72, rim: float = 0.10) -> np.ndarray:
    """Flat Lambert shading, one constant colour per triangle.

    Flat - not smooth - shading is the point: a facet that is one solid tone is what
    makes a coarse triangulation readable as a *surface* with slope and relief, instead
    of as a fog of samples. Slope becomes brightness, so a kerb, a bank and a wall each
    read as a distinct facet run even where their colours are similar.
    """
    lam = np.clip(mesh.face_n @ LIGHT_DIR, 0.0, 1.0)
    s = ambient + diffuse * lam + rim * (1.0 - np.abs(mesh.face_n[:, 2]))
    return np.clip(base_bgr.astype(np.float32) * s[:, None], 0, 255)


def render_mesh(img: np.ndarray, cam: WorldCamera, mesh: Mesh, colors_bgr: np.ndarray,
                bg=(16, 15, 14), fog_m: float = 26.0, fog_floor: float = 0.30,
                alpha: Optional[np.ndarray] = None,
                edge_color: Optional[tuple] = None, max_faces: int = 26_000) -> int:
    """Painter's-algorithm rasterisation of a flat-shaded low-poly surface.

    Triangles are projected in one vectorised pass, culled against the near plane and
    the image rectangle, then filled back-to-front. There is no z-buffer: for a 2.5-D
    height field seen from above, sorting by centroid depth is exact enough that the
    surface never self-tears, and it costs one `argsort` instead of a per-pixel test.
    """
    if len(mesh) == 0:
        return 0
    px, py, z, front = cam.project(mesh.verts)
    h, w = img.shape[:2]
    P = np.stack([np.clip(px, -1e4, 1e4), np.clip(py, -1e4, 1e4)], 1).astype(np.int32)

    f = mesh.faces
    ok = front[f].all(axis=1)
    if not ok.any():
        return 0
    zc = z[f].mean(axis=1)
    fx = P[f, 0]
    fy = P[f, 1]
    ok &= (fx.max(axis=1) >= 0) & (fx.min(axis=1) < w)
    ok &= (fy.max(axis=1) >= 0) & (fy.min(axis=1) < h)
    idx = np.nonzero(ok)[0]
    if idx.size == 0:
        return 0
    if idx.size > max_faces:                       # nearest faces win the budget
        idx = idx[np.argsort(zc[idx])[:max_faces]]
    idx = idx[np.argsort(-zc[idx], kind="stable")]

    col = colors_bgr[idx].astype(np.float32)
    d = zc[idx]
    a = np.clip(1.12 - d / max(fog_m, 1e-3), fog_floor, 1.0)
    if alpha is not None:
        a = a * np.clip(alpha[idx], 0.0, 1.0)
    col = col * a[:, None] + np.array(bg, np.float32) * (1.0 - a[:, None])
    tri = P[f[idx]]
    cols = [tuple(int(v) for v in c) for c in col]
    for i in range(idx.size):
        cv2.fillConvexPoly(img, tri[i], cols[i], cv2.LINE_8)
    if edge_color is not None:
        cv2.polylines(img, list(tri), True, edge_color, 1, cv2.LINE_AA)
    return int(idx.size)


# --------------------------------------------------------------------- 3-D chrome

def _clip_near(cam: WorldCamera, pts: np.ndarray, znear: float = 0.12) -> list[np.ndarray]:
    """Split a world polyline into runs that lie in front of the near plane.

    Without this a grid line that runs under the camera projects its behind-the-eye
    vertices to mirrored coordinates and draws a bright streak across the frame.
    """
    pc = cam.to_cam(pts)
    z = pc[:, 2]
    out, cur = [], []
    for i in range(len(pts)):
        if z[i] > znear:
            if not cur and i > 0:
                t = (znear - z[i - 1]) / (z[i] - z[i - 1])
                cur.append(pts[i - 1] + t * (pts[i] - pts[i - 1]))
            cur.append(pts[i])
        else:
            if cur:
                t = (znear - z[i - 1]) / (z[i] - z[i - 1])
                cur.append(pts[i - 1] + t * (pts[i] - pts[i - 1]))
                out.append(np.asarray(cur))
                cur = []
    if cur:
        out.append(np.asarray(cur))
    return out


def draw_polyline_3d(img, cam: WorldCamera, pts: np.ndarray, color, thick=1,
                     closed=False) -> None:
    p = np.asarray(pts, np.float64).reshape(-1, 3)
    if closed:
        p = np.concatenate([p, p[:1]])
    for run in _clip_near(cam, p):
        px, py, _, _ = cam.project(run)
        q = np.stack([np.clip(px, -3e4, 3e4), np.clip(py, -3e4, 3e4)], 1).astype(np.int32)
        if len(q) >= 2:
            cv2.polylines(img, [q], False, color, thick, cv2.LINE_AA)


def draw_world_grid(img, cam: WorldCamera, centre, half: float = 9.0, step: float = 1.0,
                    color=(46, 43, 39), major=(70, 65, 59), label_every: int = 2,
                    label_color=(112, 106, 98)) -> None:
    """A metre grid on z = 0 in world coordinates, with metre labels.

    Grid lines are snapped to absolute world metres and only the window around the
    vehicle is drawn, so the grid stays fixed to the scene (it slides past as the
    vehicle drives, which is half the evidence that the accumulation is registered) and
    the line count stays bounded however long the clip runs.
    """
    cx = float(np.round(centre[0] / step) * step)
    cy = float(np.round(centre[1] / step) * step)
    xs = np.arange(cx - half, cx + half + 1e-6, step)
    ys = np.arange(cy - half, cy + half + 1e-6, step)
    for x in xs:
        c = major if abs(x) < 1e-6 else color
        line = np.stack([np.full(28, x), np.linspace(ys[0], ys[-1], 28), np.zeros(28)], 1)
        draw_polyline_3d(img, cam, line, c, 1)
    for y in ys:
        c = major if abs(y) < 1e-6 else color
        line = np.stack([np.linspace(xs[0], xs[-1], 28), np.full(28, y), np.zeros(28)], 1)
        draw_polyline_3d(img, cam, line, c, 1)
    # metre labels along the grid crossings nearest the view centre
    h, w = img.shape[:2]
    for y in ys[::label_every]:
        for x in xs[::label_every]:
            px, py, z, front = cam.project(np.array([[x, y, 0.0]]))
            if front[0] and 30 < px[0] < w - 34 and 20 < py[0] < h - 8 and z[0] < 14.0:
                cv2.putText(img, f"{x:+.0f},{y:+.0f}", (int(px[0]) + 3, int(py[0]) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, label_color, 1, cv2.LINE_AA)


def draw_trail_ribbon(img, cam: WorldCamera, xy: np.ndarray, half_w: float = 0.12,
                      z=0.012, color=(48, 105, 160), edge=(80, 165, 225),
                      fade: int = 240) -> None:
    """The travelled path as a ribbon laid on the surface, brightening toward the present.

    `z` may be a scalar or a per-sample array, so the ribbon can be draped over the
    reconstructed height field instead of floating on the nominal ground plane.
    """
    p = np.asarray(xy, np.float64).reshape(-1, 2)
    if p.shape[0] < 2:
        return
    zz = np.full((p.shape[0], 1), float(z)) if np.isscalar(z) else \
        np.asarray(z, np.float64).reshape(-1, 1)
    d = np.diff(p, axis=0)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.maximum(n, 1e-6)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    nrm = np.concatenate([nrm, nrm[-1:]])
    left = np.concatenate([p + nrm * half_w, zz], 1)
    right = np.concatenate([p - nrm * half_w, zz], 1)
    n_pts = p.shape[0]
    for i in range(max(0, n_pts - fade), n_pts - 1):
        quad = np.stack([left[i], left[i + 1], right[i + 1], right[i]])
        runs = _clip_near(cam, quad)
        if not runs:
            continue
        f = 0.35 + 0.65 * (i - max(0, n_pts - fade)) / max(n_pts - 1 - max(0, n_pts - fade), 1)
        c = tuple(int(v * f) for v in color)
        for run in runs:
            px, py, _, _ = cam.project(run)
            poly = np.stack([np.clip(px, -3e4, 3e4), np.clip(py, -3e4, 3e4)], 1).astype(np.int32)
            cv2.fillConvexPoly(img, poly, c, cv2.LINE_AA)
    draw_polyline_3d(img, cam, left[max(0, n_pts - fade):], edge, 1)
    draw_polyline_3d(img, cam, right[max(0, n_pts - fade):], edge, 1)


def draw_vehicle_wire(img, cam: WorldCamera, pose, color=(90, 190, 255),
                      thick: int = 2, z0: float = 0.0) -> None:
    """The UGV envelope at `CFG.ugv` scale, placed at a world pose on the surface."""
    hw, L = CFG.ugv.width_m / 2.0, CFG.ugv.length_m
    hgt = max(CFG.cam.height_above_ground_m, 0.10)
    y0, y1 = -L * 0.62, L * 0.38
    base = np.array([[-hw, y0, z0], [hw, y0, z0], [hw, y1, z0], [-hw, y1, z0]])
    top = base.copy()
    top[:, 2] = z0 + hgt
    nose = np.array([[0.0, y1, z0], [0.0, y1 + 0.16, z0]])
    R = yaw_R2(float(pose[2])).astype(np.float64)
    t = np.asarray(pose[:2], np.float64)

    def w(a):
        o = a.copy()
        o[:, :2] = a[:, :2] @ R.T + t
        return o

    bw, tw, nw = w(base), w(top), w(nose)
    draw_polyline_3d(img, cam, bw, color, thick, closed=True)
    draw_polyline_3d(img, cam, tw, color, 1, closed=True)
    for i in range(4):
        draw_polyline_3d(img, cam, np.stack([bw[i], tw[i]]), color, 1)
    draw_polyline_3d(img, cam, nw, color, 1)


# =========================================================================== self-test

def _fit_line(xy: np.ndarray, band: float = 0.35, iters: int = 8):
    """Robust 2-D line fit (IRLS-trimmed PCA). Returns (centroid, unit normal)."""
    Q = np.asarray(xy, np.float64)
    c = np.median(Q, axis=0)
    _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
    n = np.array([-vt[0, 1], vt[0, 0]])
    for _ in range(iters):
        k = np.abs((Q - c) @ n) < band
        if k.sum() < 100:
            break
        c = Q[k].mean(axis=0)
        _, _, vt = np.linalg.svd(Q[k] - c, full_matrices=False)
        n = np.array([-vt[0, 1], vt[0, 0]])
    return c, n


def _line_rms(xy: np.ndarray, band: float = 0.35):
    if xy.shape[0] < 50:
        return None
    c, n = _fit_line(xy, band)
    r = (np.asarray(xy, np.float64) - c) @ n
    k = np.abs(r) < band
    return float(np.sqrt((r[k] ** 2).mean())) if k.sum() > 20 else None


def _self_test() -> None:
    import time
    from ..io_utils import load_stage
    from ..config import WORK_DIR

    print("=" * 76)
    print("cloud_accum self-test - pose-registered accumulation on real cached clips")
    print("=" * 76)

    # ---- 1. synthetic registration check: a step driven past --------------------
    # A 0.40 m kerb along world x > +1.0, observed from 60 different poses. If the
    # registration is right the surface map must reproduce the step at the right place
    # and the right height, from clouds that arrive in 60 different vehicle frames.
    rng = np.random.default_rng(0)
    sm = SurfaceMap(cell_m=0.10, keep_frames=10_000, min_hits=1)
    for i in range(60):
        pose = np.array([0.0, i * 0.05, 0.0], np.float32)
        xw = rng.uniform(-1.5, 2.5, 3000)
        yw = pose[1] + rng.uniform(0.4, 3.5, 3000)
        zw = np.where(xw > 1.0, 0.40, 0.0).astype(np.float32)
        pts_v = np.stack([xw, yw - pose[1], zw], 1).astype(np.float32)
        sm.add(veh_to_world(pts_v, pose), np.full(3000, 2, np.uint8), i, 1.0)
    m = sm.mesh((0.5, 1.5), half_m=1.4, smooth=False)
    lo = m.face_z[m.verts[m.faces[:, 0], 0] < 0.85]
    hi = m.face_z[m.verts[m.faces[:, 0], 0] > 1.15]
    print(f"\nsynthetic: {sm.n_cells()} cells, {len(m)} faces from 60 poses over a "
          f"0.40 m kerb at x=+1.00 m")
    print(f"  surface below the kerb {lo.mean():+.3f} m, above it {hi.mean():+.3f} m, "
          f"step {hi.mean()-lo.mean():.3f} m -> "
          f"{'PASS' if abs(hi.mean() - lo.mean() - 0.40) < 0.05 else 'FAIL'}")

    # ---- 2. real clip ----------------------------------------------------------
    cid = "clip_04"
    dz, oz = load_stage(cid, "depth"), load_stage(cid, "odom")
    sz, tz = load_stage(cid, "seg"), load_stage(cid, "trav")
    sm = SurfaceMap()
    t0 = time.perf_counter()
    per_frame, per_frame_world, per_frame_veh = [], [], []
    for i in range(300):
        xyz, trav, nv = frame_cloud_vehicle(dz["depth"][i].astype(np.float32),
                                            dz["valid"][i].astype(bool),
                                            dz["normal"][i], sz["label"][i], tz["label"][i])
        per_frame.append(xyz.shape[0])
        ok = bool(oz["tracking_ok"][i]) and float(oz["track_quality"][i]) > 0.25
        w = veh_to_world(xyz, oz["pose"][i])
        sm.add(w, trav, i, float(oz["track_quality"][i]), accept=ok)
        # keep the wall-ish subset for the geometry check below
        wall = ((trav == 2) & (xyz[:, 2] > 0.30) & (xyz[:, 2] < 1.25)
                & (xyz[:, 0] > 0.25) & (np.hypot(xyz[:, 0], xyz[:, 1]) < 3.0))
        if wall.sum() > 60 and ok:
            per_frame_veh.append(xyz[wall])
            per_frame_world.append((i, w[wall]))
    dt = time.perf_counter() - t0
    print(f"\n{cid}: {dt*1000/300:.1f} ms/frame to gate + register + fuse")
    print(f"  live cloud {int(np.median(per_frame)):,} pts/frame (median), surface "
          f"{sm.n_cells():,} cells of {sm.cell_m*100:.0f} cm in the "
          f"{sm.keep_frames}-frame window")
    print(f"  frames used {sm.n_frames_used}, rejected for tracking {sm.n_frames_rejected}")

    anchor = chase_path(oz["pose"])
    t0 = time.perf_counter()
    mesh = sm.mesh(anchor[299][:2], half_m=11.0)
    print(f"  mesh {len(mesh):,} triangles in {(time.perf_counter()-t0)*1e3:.0f} ms, "
          f"z {mesh.face_z.min():+.2f}..{mesh.face_z.max():+.2f} m")

    # ---- 2b. THE geometry check -------------------------------------------------
    # clip_04 has a brick wall along the right for the whole clip. If the registration
    # is real, every frame's view of it must land on ONE plane in the world frame; if
    # the poses are wrong, it fans out into a stack of duplicated surfaces. Compare the
    # accumulated thickness against the single-frame thickness of the same points in the
    # vehicle frame, which is the floor set by monocular depth noise alone.
    print("\ngeometry check - clip_04 brick wall (right side, returns inside 3 m)")
    veh_rms = [_line_rms(P[:, :2]) for P in per_frame_veh]
    veh_rms = [r for r in veh_rms if r is not None]
    allw = np.concatenate([P for _, P in per_frame_world])
    c, n = _fit_line(allw[:, :2])
    r = (allw[:, :2] - c) @ n
    inl = np.abs(r) < 0.35
    # Control: pile every frame up in the vehicle frame, i.e. pretend the vehicle never
    # moved. On THIS clip the vehicle drives roughly parallel to the wall, so the control
    # also fits a thin line - thickness alone does not prove registration here. What does
    # is the wall's *extent*: unregistered it can never be longer than the sensor range,
    # registered it must grow by the distance driven.
    ctrl = np.concatenate(per_frame_veh)[:, :2]
    cc, cn = _fit_line(ctrl)
    cr = (ctrl - cc) @ cn
    cd = np.array([-cn[1], cn[0]])
    ct = (ctrl[np.abs(cr) < 0.35] - cc) @ cd
    print(f"  NO registration (control)   : "
          f"{np.sqrt((cr[np.abs(cr) < 0.35] ** 2).mean()):.3f} m RMS but only "
          f"{ct.max() - ct.min():.1f} m long - it cannot exceed the sensor range")
    print(f"  single frame, vehicle frame : {np.median(veh_rms):.3f} m RMS "
          f"(depth noise alone, {len(veh_rms)} frames)")
    print(f"  accumulated, world frame    : {np.sqrt((r[inl]**2).mean()):.3f} m RMS over "
          f"{int(inl.sum()):,} points from {len(per_frame_world)} frames "
          f"({inl.mean()*100:.0f}% within 0.35 m), "
          f"{np.ptp((allw[inl][:, :2] - c) @ np.array([-n[1], n[0]])):.1f} m long "
          f"after a {float(np.linalg.norm(np.diff(oz['pose'][:300, :2], axis=0), axis=1).sum()):.1f} m drive")
    offs = np.array([[i, float(np.median(((P[:, :2] - c) @ n)[
        np.abs((P[:, :2] - c) @ n) < 0.6]))] for i, P in per_frame_world
        if (np.abs((P[:, :2] - c) @ n) < 0.6).sum() > 40])
    trend = np.polyfit(offs[:, 0], offs[:, 1], 1)[0]
    print(f"  per-frame wall offset       : std {offs[:,1].std():.3f} m, "
          f"drift trend {trend*300:+.2f} m over the 10 s clip  -> "
          f"{'ONE wall' if offs[:,1].std() < 0.30 else 'FANNED OUT'}")

    # ---- 3. render timing ------------------------------------------------------
    cam = ChaseCamera(back_m=5.2, up_m=4.6, ahead_m=3.0, look_z=0.10,
                      vfov_deg=40.0).at(anchor[299], (940, 619))
    img = np.full((619, 940, 3), (16, 15, 14), np.uint8)
    t0 = time.perf_counter()
    draw_world_grid(img, cam, anchor[299][:2])
    n = render_mesh(img, cam, mesh, shade_faces(mesh, colorize_height_pts(mesh.face_z)))
    xy = oz["pose"][:300, :2]
    draw_trail_ribbon(img, cam, xy, z=sm.height_at(xy) + 0.02)
    draw_vehicle_wire(img, cam, oz["pose"][299],
                      z0=float(sm.height_at(oz["pose"][299:300, :2])[0]))
    print(f"  render {n:,} shaded faces + chrome at 940x619: "
          f"{(time.perf_counter()-t0)*1e3:.0f} ms")
    p = WORK_DIR / "preview_cloud_accum.png"
    cv2.imwrite(str(p), img)
    print(f"  wrote {p}")
    print("\nCAVEAT:", ACCUM_CAVEAT)


if __name__ == "__main__":
    _self_test()
