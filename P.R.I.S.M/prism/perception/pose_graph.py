"""SE(2) pose graph: place-recognition loop closures pull VO drift back in.

On the footage prism's front end is ORB-SLAM3-*style* VO without loop closure,
and it drifts (see the honesty ledger in the README).  For the rover, keyframe
poses and the relative motions between them form a pose graph; when place
recognition (`models/vpr.py`) says "I have been here before" and a geometric
check confirms it, a loop-closure edge is added and the graph is re-optimised.

Backends
--------
* **GTSAM** (`pip install gtsam`) - `BetweenFactorPose2` + Levenberg-Marquardt.
  Used automatically when importable.
* **Built-in** Gauss-Newton on the sparse normal equations (NumPy + SciPy), so the
  stack still closes loops without GTSAM.

Conventions
-----------
The public API speaks prism's frame: (x right, y forward, yaw CCW from +y).
Internally everything is in the standard SE(2) frame GTSAM uses (x forward,
y left, theta CCW from +x): ``x_std = y, y_std = -x, theta = yaw``.

A false loop closure is worse than none: it bends the whole map.  So a loop edge
must pass two gates before it is optimised:

1. **geometric verification** - `relative_pose_pnp` needs enough PnP-RANSAC
   inliers between the two keyframes;
2. **drift consistency** - the loop's residual against the current estimate is
   tested with a chi-square gate (3 dof, 99.9%) under the covariance propagated
   along the odometry chain between the two keyframes (inflated by
   `drift_inflation`, because real VO drift is bias-like, not white).  A loop
   that asks for more correction than the odometry could plausibly have
   drifted is reported in `rejected_loops` and left out.

A robust kernel alone was tried first and rejected: at the first iteration a
*true* loop after metres of drift has the same huge residual as a false one, so
the kernel suppressed both.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------- SE(2)
def _to_std(p):
    x, y, yaw = float(p[0]), float(p[1]), float(p[2])
    return np.array([y, -x, yaw])


def _from_std(p):
    return np.array([-p[1], p[0], p[2]])


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _rot(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]])


def between(a_std: np.ndarray, b_std: np.ndarray) -> np.ndarray:
    """a^-1 * b in the standard frame."""
    d = _rot(a_std[2]).T @ (b_std[:2] - a_std[:2])
    return np.array([d[0], d[1], _wrap(b_std[2] - a_std[2])])


def compose(a_std: np.ndarray, d_std: np.ndarray) -> np.ndarray:
    p = a_std[:2] + _rot(a_std[2]) @ d_std[:2]
    return np.array([p[0], p[1], _wrap(a_std[2] + d_std[2])])


def relative_prism(pose_i, pose_j) -> np.ndarray:
    """Relative motion i->j (prism-frame poses in, standard-frame delta out)."""
    return between(_to_std(pose_i), _to_std(pose_j))


@dataclass
class Edge:
    i: int
    j: int
    z: np.ndarray            # (3,) measured i->j motion, standard frame
    info: np.ndarray         # (3,3) information matrix
    loop: bool = False


def _gtsam():
    try:
        import gtsam  # type: ignore
        return gtsam
    except Exception:
        return None


class PoseGraph2D:
    def __init__(self, backend: str = "auto", drift_inflation: float = 3.0,
                 gate_chi2: float = 16.27):
        if backend not in ("auto", "gtsam", "numpy"):
            raise ValueError(backend)
        if backend == "gtsam" and _gtsam() is None:
            raise ImportError("backend='gtsam' requested but `import gtsam` failed")
        self.backend = "gtsam" if (backend != "numpy" and _gtsam() is not None) else "numpy"
        self.inflation = float(drift_inflation)
        self.gate = float(gate_chi2)
        self.nodes: list[np.ndarray] = []          # standard frame
        self.edges: list[Edge] = []
        self.rejected_loops: list[tuple[int, int, float]] = []

    # ------------------------------------------------------------------ build
    def add_node(self, pose_drishti) -> int:
        self.nodes.append(_to_std(pose_drishti))
        return len(self.nodes) - 1

    def add_odometry(self, i: int, j: int, z_std=None, sigma=(0.05, 0.05, 0.02)) -> None:
        """Sequential VO edge.  Default measurement = the current estimate's delta."""
        z = between(self.nodes[i], self.nodes[j]) if z_std is None else np.asarray(z_std, float)
        self.edges.append(Edge(i, j, z, np.diag(1.0 / np.square(sigma)), loop=False))

    def add_loop(self, i: int, j: int, z_std, sigma=(0.10, 0.10, 0.05)) -> None:
        """Loop closure: node j was observed from node i with relative pose z."""
        self.edges.append(Edge(i, j, np.asarray(z_std, float),
                               np.diag(1.0 / np.square(sigma)), loop=True))

    def poses(self) -> np.ndarray:
        """(N, 3) optimised poses in the prism frame."""
        return np.array([_from_std(p) for p in self.nodes]) if self.nodes else np.zeros((0, 3))

    # ------------------------------------------------------------------ solve
    def _chain_cov(self, i: int, j: int) -> Optional[np.ndarray]:
        """First-order covariance of the i->j motion along the odometry chain."""
        odo = {(e.i, e.j): e for e in self.edges if not e.loop}
        lo, hi = (i, j) if i < j else (j, i)
        P = np.zeros((3, 3))
        acc = np.zeros(3)
        for k in range(lo, hi):
            e = odo.get((k, k + 1))
            if e is None:
                return None
            Q = np.linalg.inv(e.info) * self.inflation ** 2
            c, s_ = math.cos(acc[2]), math.sin(acc[2])
            dx, dy = e.z[0], e.z[1]
            J1 = np.array([[1, 0, -s_ * dx - c * dy], [0, 1, c * dx - s_ * dy], [0, 0, 1]])
            J2 = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
            P = J1 @ P @ J1.T + J2 @ Q @ J2.T
            acc = compose(acc, e.z)
        return P

    def loop_gate(self, e: Edge) -> float:
        """Mahalanobis chi-square of a loop edge vs. plausible odometry drift."""
        P = self._chain_cov(e.i, e.j)
        if P is None:
            return float("inf")
        r = self._residual(e)
        S = P + np.linalg.inv(e.info)
        return float(r @ np.linalg.solve(S, r))

    def optimize(self, iters: int = 20) -> dict:
        if len(self.nodes) < 2:
            return dict(backend=self.backend, iters=0, chi2=0.0, rejected_loops=0)
        self.rejected_loops = []
        keep = []
        for e in self.edges:
            if e.loop:
                chi = self.loop_gate(e)
                if chi > self.gate:
                    self.rejected_loops.append((e.i, e.j, chi))
                    continue
            keep.append(e)
        out = self._opt_gtsam(keep, iters) if self.backend == "gtsam" \
            else self._opt_numpy(keep, iters)
        out["rejected_loops"] = len(self.rejected_loops)
        out["accepted_loops"] = sum(e.loop for e in keep)
        return out

    def _residual(self, e: Edge) -> np.ndarray:
        r = between(self.nodes[e.i], self.nodes[e.j]) - e.z
        r[2] = _wrap(r[2])
        return r

    def _opt_numpy(self, edges: list, iters: int) -> dict:
        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import spsolve
        N = len(self.nodes)
        x = np.array(self.nodes, float)
        chi2 = 0.0
        for it in range(iters):
            Hm = lil_matrix((3 * N, 3 * N))
            b = np.zeros(3 * N)
            chi2 = 0.0
            for e in edges:
                xi, xj = x[e.i], x[e.j]
                Ri = _rot(xi[2])
                dt = xj[:2] - xi[:2]
                pred = np.array([*(Ri.T @ dt), _wrap(xj[2] - xi[2])])
                r = pred - e.z
                r[2] = _wrap(r[2])
                dRiT = np.array([[-math.sin(xi[2]), math.cos(xi[2])],
                                 [-math.cos(xi[2]), -math.sin(xi[2])]])
                A = np.zeros((3, 3)); B = np.zeros((3, 3))
                A[:2, :2] = -Ri.T
                A[:2, 2] = dRiT @ dt
                A[2, 2] = -1.0
                B[:2, :2] = Ri.T
                B[2, 2] = 1.0
                chi2 += float(r @ e.info @ r)
                Om = e.info
                si, sj = 3 * e.i, 3 * e.j
                Hm[si:si + 3, si:si + 3] += A.T @ Om @ A
                Hm[si:si + 3, sj:sj + 3] += A.T @ Om @ B
                Hm[sj:sj + 3, si:si + 3] += B.T @ Om @ A
                Hm[sj:sj + 3, sj:sj + 3] += B.T @ Om @ B
                b[si:si + 3] += A.T @ Om @ r
                b[sj:sj + 3] += B.T @ Om @ r
            # fix the first pose (gauge freedom)
            Hm[0:3, 0:3] += np.eye(3) * 1e9
            dx = spsolve(Hm.tocsr(), -b)
            x += dx.reshape(N, 3)
            x[:, 2] = _wrap(x[:, 2])
            if np.max(np.abs(dx)) < 1e-6:
                break
        self.nodes = [p for p in x]
        return dict(backend="numpy", iters=it + 1, chi2=chi2)

    def _opt_gtsam(self, edges: list, iters: int) -> dict:
        gtsam = _gtsam()
        graph = gtsam.NonlinearFactorGraph()
        init = gtsam.Values()
        prior = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4, 1e-4, 1e-5]))
        graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(*self.nodes[0]), prior))
        for e in edges:
            model = gtsam.noiseModel.Gaussian.Information(e.info)
            graph.add(gtsam.BetweenFactorPose2(e.i, e.j, gtsam.Pose2(*e.z), model))
        for k, p in enumerate(self.nodes):
            init.insert(k, gtsam.Pose2(*p))
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(iters)
        res = gtsam.LevenbergMarquardtOptimizer(graph, init, params).optimize()
        self.nodes = [np.array([res.atPose2(k).x(), res.atPose2(k).y(), res.atPose2(k).theta()])
                      for k in range(len(self.nodes))]
        return dict(backend="gtsam", iters=iters, chi2=float(graph.error(res)) * 2.0)


# ---------------------------------------------------------------------- geometry check
def relative_pose_pnp(bgr_i: np.ndarray, depth_i: np.ndarray, bgr_j: np.ndarray,
                      K: np.ndarray, min_inliers: int = 25,
                      max_depth_m: float = 8.0) -> Optional[tuple[np.ndarray, int]]:
    """Metric relative pose i -> j from ORB matches + frame i's metric depth.

    Returns (z_std, n_inliers) or None if the match does not verify.  This is the
    geometric check that turns a place-recognition *hint* into a loop-closure
    *edge*.  The camera is assumed level with the vehicle (yaw about camera -Y).
    """
    import cv2
    orb = cv2.ORB_create(1500)
    gi = cv2.cvtColor(bgr_i, cv2.COLOR_BGR2GRAY) if bgr_i.ndim == 3 else bgr_i
    gj = cv2.cvtColor(bgr_j, cv2.COLOR_BGR2GRAY) if bgr_j.ndim == 3 else bgr_j
    ki, di = orb.detectAndCompute(gi, None)
    kj, dj = orb.detectAndCompute(gj, None)
    if di is None or dj is None or len(ki) < min_inliers or len(kj) < min_inliers:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    m = bf.match(di, dj)
    obj, img = [], []
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    for mm in m:
        u, v = ki[mm.queryIdx].pt
        z = float(depth_i[int(round(v)), int(round(u))]) if (
            0 <= int(round(v)) < depth_i.shape[0] and 0 <= int(round(u)) < depth_i.shape[1]) else np.nan
        if not np.isfinite(z) or z <= 0.05 or z > max_depth_m:
            continue
        obj.append([(u - cx) * z / fx, (v - cy) * z / fy, z])
        img.append(kj[mm.trainIdx].pt)
    if len(obj) < min_inliers:
        return None
    ok, rvec, tvec, inl = cv2.solvePnPRansac(np.float64(obj), np.float64(img), K, None,
                                             reprojectionError=3.0, confidence=0.999,
                                             iterationsCount=200)
    if not ok or inl is None or len(inl) < min_inliers:
        return None
    R, _ = cv2.Rodrigues(rvec)
    # camera j pose in camera i frame: X_j = R X_i + t  ->  C_j = -R^T t, R_ij = R^T
    Rij = R.T
    C = (-R.T @ tvec).ravel()
    # camera (x right, y down, z forward) -> standard planar (x fwd, y left)
    dx_fwd, dy_left = float(C[2]), float(-C[0])
    dyaw = float(math.atan2(Rij[0, 2], Rij[2, 2]))       # heading change, +left
    return np.array([dx_fwd, dy_left, -dyaw]), int(len(inl))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # drive a 4 m square (prism frame), VO with a yaw bias, then close the loop
    true = []
    x, y, yaw = 0.0, 0.0, 0.0
    for leg in range(4):
        for k in range(20):
            true.append((x, y, yaw))
            x += -math.sin(yaw) * 0.2
            y += math.cos(yaw) * 0.2
        yaw += math.pi / 2
    true.append((x, y, yaw))
    true = np.array(true)
    for backend in (["numpy", "gtsam"] if _gtsam() is not None else ["numpy"]):
        pg = PoseGraph2D(backend=backend)
        est = np.array(true[0], float)
        pg.add_node(est)
        for k in range(1, len(true)):
            z = relative_prism(true[k - 1], true[k])
            z_noisy = z + np.array([rng.normal(0, 0.01), rng.normal(0, 0.01), 0.004])
            est_std = compose(_to_std(est), z_noisy)
            est = _from_std(est_std)
            pg.add_node(est)
            pg.add_odometry(k - 1, k, z_noisy, sigma=(0.02, 0.02, 0.01))
        before = float(np.linalg.norm(pg.poses()[-1, :2] - true[-1, :2]))
        pg.add_loop(0, len(true) - 1, relative_prism(true[0], true[-1]))
        pg.add_loop(5, len(true) - 30, relative_prism(true[5], true[30]) + np.array([2.0, 1.5, 1.0]))
        info = pg.optimize()
        after = float(np.linalg.norm(pg.poses()[-1, :2] - true[-1, :2]))
        print(f"[{backend}] end-point error before loop closure {before:.3f} m, "
              f"after {after:.3f} m; loops accepted {info['accepted_loops']}, "
              f"rejected {info['rejected_loops']} (one true, one deliberately false)")
        assert after < 0.5 * before
        assert info["accepted_loops"] == 1 and info["rejected_loops"] == 1
    print("self-test passed")
