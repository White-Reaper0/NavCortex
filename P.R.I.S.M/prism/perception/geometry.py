"""Monocular geometry: metric alignment, ground plane, height / slope / roughness, BEV indexing.

Frames
------
camera  : X right, Y down, Z forward (OpenCV)
vehicle : X right, Y forward, Z up, origin on the ground directly under the camera

Metric alignment
----------------
Depth Anything V2 emits an affine-invariant *relative inverse depth* q. Following the
inverse-depth rescaling used by Marsal et al. (arXiv:2412.14103), true optical-axis
depth D satisfies

    1 / D = a * q + b

The reference the paper uses is sparse metric landmarks from a visual-inertial
estimator. This clip has no IMU and no calibration file, so prism substitutes the
*ground-plane* reference: the drivable surface in front of the vehicle is a plane at a
known distance (the camera height above ground) below the camera.

For a pixel with normalised ray m = K^-1 [u v 1]^T (m_z = 1), the 3-D point is P = D*m.
A ground point lies on the plane n.P = h, with |n| = 1 and h the camera height, so

    n.m / h = 1 / D = a*q + b        ->     u.m - a*q - b = 0,    u = n / h

Identifiability
---------------
On a single plane, 1/D is an exact affine function of (m_x, m_y), so the data supply
only three independent numbers (the coefficients of q regressed on m_x, m_y, 1) against
four unknowns (two plane angles, a, b). One plane at a known height therefore cannot
separate the scale a from the shift b -- and naively stacking [m_x, m_y, m_z, -q, -1]
is worse than under-determined, it is exactly rank-deficient, because m_z is identically
one and duplicates the constant column.

prism resolves this by adopting the scale-only model **b = 0**, i.e. treating the
network output as scaled disparity where q -> 0 means range -> infinity. That leaves the
well-posed homogeneous system

    [m_x, m_y, 1, -q] . [u_x, u_y, u_z, a]^T = 0

whose one-dimensional null space is the solution, fixed absolutely by |u| = 1/h_cam.
Solved by SVD over thousands of candidate ground pixels and refined by Cauchy IRLS.

Two assumptions are therefore baked into every metric number downstream, and both are
assumptions rather than measurements: the camera height h_cam, to which all distances
are directly proportional, and b = 0, which mainly costs accuracy at long range where
monocular depth is least trustworthy anyway. `GroundFit.residual` reports how well the
recovered plane actually explains the observed relative depth.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2

from ..config import CFG


# --------------------------------------------------------------------- rays

_RAY_CACHE: dict[tuple, np.ndarray] = {}


def ray_grid(h: int, w: int, K: Optional[np.ndarray] = None) -> np.ndarray:
    """(h, w, 3) normalised camera rays m with m_z == 1."""
    K = CFG.cam.K if K is None else K
    key = (h, w, float(K[0, 0]), float(K[0, 2]), float(K[1, 2]))
    if key in _RAY_CACHE:
        return _RAY_CACHE[key]
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    m = np.stack([(u - K[0, 2]) / K[0, 0],
                  (v - K[1, 2]) / K[1, 1],
                  np.ones_like(u)], axis=-1).astype(np.float32)
    _RAY_CACHE[key] = m
    return m


def unproject(depth_m: np.ndarray, K: Optional[np.ndarray] = None) -> np.ndarray:
    """Depth along the optical axis -> (H, W, 3) points in the camera frame."""
    h, w = depth_m.shape
    return ray_grid(h, w, K) * depth_m[..., None].astype(np.float32)


# --------------------------------------------------------------------- metric alignment

@dataclass
class GroundFit:
    a: float = 1.0                 # 1/D = a*q + b
    b: float = 0.0
    normal: np.ndarray = None      # (3,) unit, camera frame, points toward the ground
    height: float = CFG.cam.height_above_ground_m
    residual: float = 0.0          # RMS of (u.m - a*q - b) in 1/m
    inliers: int = 0
    ok: bool = False

    def __post_init__(self):
        if self.normal is None:
            self.normal = np.array([0.0, 1.0, 0.0], np.float32)


def _candidate_ground_mask(q: np.ndarray, valid: np.ndarray,
                           seg_label: Optional[np.ndarray] = None) -> np.ndarray:
    """Pixels plausibly on the drivable surface: lower image band, plus terrain classes."""
    h, w = q.shape
    m = valid.copy()
    band = np.zeros_like(m)
    band[int(0.50 * h):, :] = True          # ground occupies the lower half of a forward view
    m &= band
    if seg_label is not None:
        # trail (1) and grass (2) are surface classes in the prism-7 taxonomy
        m &= np.isin(seg_label, (1, 2))
    return m


def fit_metric_ground(q: np.ndarray,
                      valid: np.ndarray,
                      seg_label: Optional[np.ndarray] = None,
                      K: Optional[np.ndarray] = None,
                      h_cam: Optional[float] = None,
                      max_pts: int = 6000,
                      iters: int = 6,
                      prior: Optional["GroundFit"] = None) -> GroundFit:
    """Joint solve for inverse-depth scale/shift and the ground plane (see module docstring)."""
    K = CFG.cam.K if K is None else K
    h_cam = CFG.cam.height_above_ground_m if h_cam is None else float(h_cam)
    H, W = q.shape
    gm = _candidate_ground_mask(q, valid, seg_label)
    n_avail = int(gm.sum())
    if n_avail < 400:
        gm = _candidate_ground_mask(q, valid, None)
        n_avail = int(gm.sum())
    if n_avail < 200:
        return GroundFit(ok=False)

    ys, xs = np.nonzero(gm)
    if ys.size > max_pts:
        sel = np.random.default_rng(0).choice(ys.size, max_pts, replace=False)
        ys, xs = ys[sel], xs[sel]
    m = ray_grid(H, W, K)[ys, xs]                       # (N,3), m_z == 1
    qs = q[ys, xs].astype(np.float64)

    # Scale-only model (b = 0): A @ [ux, uy, uz, a] = 0.
    # m_z is identically 1, so it already supplies the constant term; adding a separate
    # constant column for b would make the system exactly rank-deficient (see docstring).
    A = np.concatenate([m.astype(np.float64), -qs[:, None]], axis=1)
    wts = np.ones(qs.size)
    sol = None
    for _ in range(iters):
        Aw = A * wts[:, None]
        # smallest right singular vector of the weighted design matrix
        _, _, Vt = np.linalg.svd(Aw, full_matrices=False)
        sol = Vt[-1]
        r = A @ sol
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        wts = 1.0 / np.sqrt(1.0 + (r / (2.5 * s)) ** 2)   # Cauchy IRLS

    u = sol[:3]
    a, b = float(sol[3]), 0.0
    nu = np.linalg.norm(u)
    if nu < 1e-9 or not np.isfinite(nu):
        return GroundFit(ok=False)

    # |u| = 1/h_cam fixes the overall scale of the homogeneous solution
    k = (1.0 / h_cam) / nu
    u, a = u * k, a * k
    normal = (u * h_cam).astype(np.float32)
    # the ground is below the camera: +Y is down in the camera frame
    if normal[1] < 0:
        normal, a = -normal, -a

    r = (m @ (normal / h_cam)) - (a * qs + b)
    s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
    inl = np.abs(r) < 3.0 * s
    fit = GroundFit(a=float(a), b=float(b), normal=normal, height=h_cam,
                    residual=float(np.sqrt(np.mean(r[inl] ** 2))) if inl.any() else float("inf"),
                    inliers=int(inl.sum()), ok=True)

    # sanity: the plane must sit roughly under the camera and produce positive depths
    tilt = np.degrees(np.arccos(np.clip(float(normal[1]), -1, 1)))
    if tilt > 45.0 or fit.inliers < 120 or a <= 0:
        fit.ok = False
    if fit.ok and prior is not None and prior.ok:
        # temporal smoothing keeps the metric frame stable across frames
        w_new = 0.35
        fit.a = (1 - w_new) * prior.a + w_new * fit.a
        fit.b = (1 - w_new) * prior.b + w_new * fit.b
        nrm = (1 - w_new) * prior.normal + w_new * fit.normal
        fit.normal = (nrm / (np.linalg.norm(nrm) + 1e-9)).astype(np.float32)
    return fit


def depth_from_q(q: np.ndarray, fit: GroundFit,
                 d_min: float = 0.15, d_max: float = 25.0) -> tuple[np.ndarray, np.ndarray]:
    """Apply 1/D = a*q + b. Returns (depth_m, valid)."""
    inv = fit.a * q.astype(np.float32) + fit.b
    valid = np.isfinite(inv) & (inv > 1.0 / d_max)
    depth = np.full_like(inv, np.nan, np.float32)
    np.divide(1.0, inv, out=depth, where=valid)
    valid &= np.isfinite(depth) & (depth > d_min) & (depth < d_max)
    depth[~valid] = np.nan
    return depth, valid


# --------------------------------------------------------------------- vehicle frame

def vehicle_basis(normal: np.ndarray) -> np.ndarray:
    """Rotation taking camera-frame points to the vehicle frame (rows: right, forward, up)."""
    n = normal / (np.linalg.norm(normal) + 1e-9)
    up = -n                                          # ground normal points down in camera frame
    z_cam = np.array([0.0, 0.0, 1.0], np.float32)
    fwd = z_cam - np.dot(z_cam, up) * up
    ln = np.linalg.norm(fwd)
    if ln < 1e-6:                                    # degenerate: camera looking straight down
        fwd = np.array([0.0, -1.0, 0.0], np.float32) - np.dot(np.array([0.0, -1.0, 0.0], np.float32), up) * up
        ln = np.linalg.norm(fwd)
    fwd = fwd / ln
    right = np.cross(fwd, up)
    right /= (np.linalg.norm(right) + 1e-9)
    return np.stack([right, fwd, up]).astype(np.float32)


def to_vehicle(points_cam: np.ndarray, fit: GroundFit) -> np.ndarray:
    """(H,W,3) camera points -> vehicle frame with the origin on the ground under the camera."""
    R = vehicle_basis(fit.normal)
    pv = points_cam.reshape(-1, 3) @ R.T
    pv[:, 2] += fit.height                            # lift origin from camera to ground
    return pv.reshape(points_cam.shape)


def height_slope_roughness(points_veh: np.ndarray, valid: np.ndarray,
                           win: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Height above the fitted ground plane, local slope in degrees, local height std."""
    z = points_veh[..., 2].astype(np.float32).copy()
    z[~valid] = np.nan
    filled = np.nan_to_num(z, nan=0.0)
    m = valid.astype(np.float32)

    k = (win, win)
    num = cv2.boxFilter(filled, -1, k, normalize=False)
    den = cv2.boxFilter(m, -1, k, normalize=False)
    mean = num / np.maximum(den, 1.0)
    num2 = cv2.boxFilter(filled ** 2, -1, k, normalize=False)
    var = np.maximum(num2 / np.maximum(den, 1.0) - mean ** 2, 0.0)
    rough = np.sqrt(var).astype(np.float32)
    rough[den < 6] = np.nan

    # slope from dz/dx and dz/dy in metric vehicle coordinates
    x = np.nan_to_num(points_veh[..., 0], nan=0.0)
    y = np.nan_to_num(points_veh[..., 1], nan=0.0)
    gz_u = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3)
    gz_v = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3)
    gx_u = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy_v = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    dzdx = gz_u / np.where(np.abs(gx_u) < 1e-4, np.nan, gx_u)
    dzdy = gz_v / np.where(np.abs(gy_v) < 1e-4, np.nan, gy_v)
    grad = np.sqrt(np.nan_to_num(dzdx) ** 2 + np.nan_to_num(dzdy) ** 2)
    slope = np.degrees(np.arctan(np.clip(grad, 0, 20))).astype(np.float32)
    slope = cv2.medianBlur(slope, 5)
    slope[~valid] = np.nan
    return z, slope, rough


# --------------------------------------------------------------------- BEV indexing

def bev_indices(points_veh: np.ndarray, valid: np.ndarray):
    """Vehicle-frame points -> (row, col, keep) indices into the BEV grid.

    Row 0 is the farthest forward cell, row H-1 is at the vehicle; column W/2 is the
    centreline with +X to the right.
    """
    b = CFG.bev
    x = points_veh[..., 0]
    y = points_veh[..., 1]
    col = np.floor(x / b.res_m).astype(np.int32) + b.n_lateral
    row = (b.n_forward - 1) - np.floor(y / b.res_m).astype(np.int32)
    keep = (valid & np.isfinite(x) & np.isfinite(y)
            & (col >= 0) & (col < b.W) & (row >= 0) & (row < b.H) & (y > 0.02))
    return row, col, keep


def bev_to_veh(row: np.ndarray, col: np.ndarray):
    """BEV cell centres -> (x, y) metres in the vehicle frame."""
    b = CFG.bev
    x = (col - b.n_lateral + 0.5) * b.res_m
    y = ((b.n_forward - 1) - row + 0.5) * b.res_m
    return x, y


def veh_to_bev_px(x: float, y: float, rect: tuple[int, int, int, int]):
    """Vehicle-frame metres -> pixel coordinates inside a drawn BEV panel rect."""
    b = CFG.bev
    px0, py0, pw, ph = rect
    cx = (x / b.res_m + b.n_lateral + 0.5) / b.W
    cy = ((b.n_forward - 1) - (y / b.res_m) + 0.5) / b.H
    return int(px0 + cx * pw), int(py0 + cy * ph)
