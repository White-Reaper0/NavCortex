"""Keyframes + place recognition + pose graph = drift correction on the rover.

`LoopClosureManager` sits after visual odometry.  It

1. drops a keyframe every `kf_dist_m` metres or `kf_yaw_rad` of turning, and
   links it to the previous one with an odometry edge;
2. when `VPRStage` reports a revisit, finds the keyframe nearest the matched
   database frame and *verifies* the match geometrically
   (`pose_graph.relative_pose_pnp`: ORB + the old keyframe's metric depth +
   PnP-RANSAC);
3. adds a loop edge, re-optimises the graph (GTSAM if installed, else the
   built-in solver), and returns the correction for the current pose.

Keyframe images are stored as grey uint8 + float16 depth, capped at
`max_keyframes` (oldest dropped), so memory stays bounded on a Jetson.

On the recorded footage there are no true loops long enough to matter and VO
runs without this module (see README honesty ledger); it is wired into
`prism/runtime.py` for the rover.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .pose_graph import PoseGraph2D, _to_std, _from_std, between, compose


@dataclass
class LoopEvent:
    kf_from: int
    kf_to: int
    inliers: int
    accepted: bool
    correction_m: float = 0.0
    reason: str = ""


class LoopClosureManager:
    def __init__(self, K: Optional[np.ndarray] = None, kf_dist_m: float = 0.5,
                 kf_yaw_rad: float = 0.35, max_keyframes: int = 120,
                 min_gap_kf: int = 10, backend: str = "auto",
                 verifier: Optional[Callable] = None):
        from ..config import CFG
        self.K = CFG.cam.K if K is None else K
        self.kf_dist, self.kf_yaw = float(kf_dist_m), float(kf_yaw_rad)
        self.max_kf = int(max_keyframes)
        self.min_gap = int(min_gap_kf)
        self.backend = backend
        self._verify = verifier
        self.reset()

    def reset(self) -> None:
        self.graph = PoseGraph2D(backend=self.backend)
        self.kf_frame: list[int] = []                # frame index of each keyframe
        self.kf_data: "OrderedDict[int, tuple]" = OrderedDict()   # kf -> (gray, depth16)
        self._last_kf_pose: Optional[np.ndarray] = None
        self.events: list[LoopEvent] = []

    # ------------------------------------------------------------------ keyframes
    def maybe_keyframe(self, frame_idx: int, pose, gray: Optional[np.ndarray] = None,
                       depth: Optional[np.ndarray] = None) -> Optional[int]:
        p = np.asarray(pose, float)
        if self._last_kf_pose is not None:
            d = math.hypot(p[0] - self._last_kf_pose[0], p[1] - self._last_kf_pose[1])
            dy = abs((p[2] - self._last_kf_pose[2] + math.pi) % (2 * math.pi) - math.pi)
            if d < self.kf_dist and dy < self.kf_yaw:
                return None
        k = self.graph.add_node(p)
        if k > 0:
            self.graph.add_odometry(k - 1, k, sigma=(0.03, 0.03, 0.015))
        self.kf_frame.append(int(frame_idx))
        if gray is not None and depth is not None:
            self.kf_data[k] = (gray, depth.astype(np.float16))
            while len(self.kf_data) > self.max_kf:
                self.kf_data.popitem(last=False)
        self._last_kf_pose = p
        return k

    def keyframe_for_frame(self, frame_idx: int) -> int:
        if not self.kf_frame:
            return -1
        f = np.asarray(self.kf_frame)
        return int(np.argmin(np.abs(f - frame_idx)))

    # ------------------------------------------------------------------ loops
    def on_revisit(self, matched_frame: int, cur_gray: Optional[np.ndarray],
                   cur_pose) -> Optional[tuple[np.ndarray, LoopEvent]]:
        """VPR says the current view matches `matched_frame`.  Returns
        (corrected current pose, event) if a verified loop was closed, else None."""
        n = len(self.kf_frame)
        if n < self.min_gap + 1:
            return None
        i = self.keyframe_for_frame(matched_frame)
        j = n - 1                                            # latest keyframe
        if j - i < self.min_gap:
            return None
        if self._verify is not None:
            res = self._verify(i, j)
        else:
            if i not in self.kf_data or cur_gray is None:
                return None
            from .pose_graph import relative_pose_pnp
            g_i, d_i = self.kf_data[i]
            res = relative_pose_pnp(g_i, d_i.astype(np.float32), cur_gray, self.K)
        if res is None:
            ev = LoopEvent(i, j, 0, False, reason="geometric verification failed")
            self.events.append(ev)
            return None
        z, inl = res
        before = self.graph.poses()[j].copy()
        self.graph.add_loop(i, j, z)
        info = self.graph.optimize()
        if info.get("accepted_loops", 0) == 0 or any(
                (a, b) == (i, j) for a, b, _ in self.graph.rejected_loops):
            # the gate dropped it: remove the edge so it is not re-tested forever
            self.graph.edges = [e for e in self.graph.edges if not (e.loop and e.i == i and e.j == j)]
            ev = LoopEvent(i, j, inl, False, reason="inconsistent with odometry drift (chi2 gate)")
            self.events.append(ev)
            return None
        after = self.graph.poses()[j]
        # carry the keyframe's correction over to the current pose
        rel = between(_to_std(before), _to_std(cur_pose))
        corrected = _from_std(compose(_to_std(after), rel))
        ev = LoopEvent(i, j, inl, True, float(math.hypot(*(after[:2] - before[:2]))),
                       reason=f"loop closed via {info['backend']}")
        self.events.append(ev)
        self._last_kf_pose = after
        return corrected, ev
