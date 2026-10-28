"""Rover IMU fusion: fixes visual-odometry metric scale and heading drift.

Where this applies
------------------
The five demo clips come from an RC-car video with **no IMU stream**, so on the
footage prism runs camera-only and its metric scale rests on the assumed 12 cm
camera height (see `perception/geometry.py`).  The target rover (WAVE ROVER +
Jetson Orin Nano Super) has a built-in IMU.  It senses *self-motion only* - it
is not a world sensor - and this module uses it for exactly two things:

1. **Heading.**  A complementary filter: the gyro supplies the short-term yaw
   increment (smooth, but its bias integrates into drift), VO supplies the long
   term (noisy per frame, but bias-free).  The gyro bias is estimated online -
   from zero-velocity periods and from the slow gyro-minus-VO yaw-rate residual -
   and removed before blending.  When VO tracking is lost the bias-corrected gyro
   carries the heading through the dropout, so the map is not rotated wrongly
   when tracking comes back.

2. **Metric scale.**  Monocular VO + depth gets scale from the ground-plane
   assumption; an accelerometer measures metric acceleration directly.  Over a
   sliding window, the change in VO forward speed is regressed against the
   change in IMU-integrated forward speed:

        s_hat = sum(dv_imu * dv_vo) / sum(dv_vo ** 2)

   and `s_hat` is folded into a slow, clipped running correction.  Scale is only
   observable while the vehicle *accelerates* - at constant speed both sides are
   ~0 - so windows without enough excitation are skipped rather than guessed.

3. **Zero-velocity updates.**  When the vehicle is standing still, VO
   translation jitter is discarded, which stops drift accumulating at every
   stop.  "Still" needs the IMU (|a| ~ g, |gyro| ~ 0) *and* VO (~0 m/s) to agree
   for several frames in a row, or a commanded stop from the runtime: the IMU
   alone reads the same at rest and at constant speed.  Gyro / accelerometer
   biases are only learned during a commanded stop or a long quiet spell.

Frames: the IMU is assumed already rotated into the prism vehicle frame
(x right, y forward, z up, accelerometer reads +g on z at rest).  The ROS 2 node
does the REP-103 (x forward, y left) -> prism conversion before calling this.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

G = 9.80665


@dataclass
class FusedMotion:
    d_trans: float              # metres, scale-corrected
    d_yaw: float                # radians, gyro/VO blend
    scale: float                # current VO scale correction (1.0 = trust VO)
    stationary: bool
    source: str                 # "vo+imu" | "imu-only (VO lost)" | "zupt" | "vo"


class ImuFusion:
    def __init__(self, cfg=None):
        from ..nav.nav_config import NAV
        self.cfg = cfg or NAV.imu
        self.reset()

    def reset(self) -> None:
        self.scale = 1.0
        self._t_imu: Optional[float] = None
        self._dyaw_acc = 0.0                    # gyro yaw since the last VO update
        self._v_imu = 0.0                       # IMU-integrated forward speed (drifts)
        self._ay_bias = 0.0                     # slow forward-accel bias (slope, mount)
        self.gyro_bias = 0.0                    # rad/s, z axis
        self._still = deque(maxlen=20)          # recent stationary flags
        self._gz_sum = self._ay_sum = 0.0       # raw sums since the last VO update
        self._quiet_run = 0                     # consecutive VO updates that looked still
        self._n_int = 0
        self._n_imu = 0
        # (t, vo_speed, imu_speed) at every VO update, for the scale window
        self._hist: deque = deque()
        self.last_estimate: Optional[float] = None
        self.n_scale_updates = 0

    @property
    def has_imu(self) -> bool:
        return self._n_imu > 0

    # ------------------------------------------------------------------ IMU in
    def add_imu(self, t: float, gyro: np.ndarray, accel: np.ndarray) -> None:
        """One IMU sample: gyro (rad/s) and accel (m/s^2) in the vehicle frame."""
        t = float(t)
        gyro = np.asarray(gyro, np.float64)
        accel = np.asarray(accel, np.float64)
        still = (abs(np.linalg.norm(accel) - G) < self.cfg.stationary_acc_tol and
                 np.linalg.norm(gyro) < self.cfg.stationary_gyro_tol)
        self._still.append(bool(still))
        if self._t_imu is not None:
            dt = t - self._t_imu
            if 0.0 < dt < 0.5:
                self._dyaw_acc += (float(gyro[2]) - self.gyro_bias) * dt
                self._v_imu += (float(accel[1]) - self._ay_bias) * dt
                self._gz_sum += float(gyro[2])
                self._ay_sum += float(accel[1])
                self._n_int += 1
        self._t_imu = t
        self._n_imu += 1

    @property
    def imu_still(self) -> bool:
        """The IMU *alone* cannot tell rest from constant velocity (both read ~g, ~0
        rotation), so this is necessary but not sufficient; `fuse` also needs VO to
        agree before it applies a zero-velocity update."""
        return len(self._still) >= 5 and all(list(self._still)[-5:])

    # ------------------------------------------------------------------ VO in
    def fuse(self, t: float, d_trans_vo: float, d_yaw_vo: float, dt: float,
             tracking_ok: bool = True, commanded_stop: Optional[bool] = None) -> FusedMotion:
        """Combine one VO increment with the IMU samples received since the last one.

        `commanded_stop`: True if the vehicle has been *commanded* to stand still
        (the runtime knows its own last command).  It is the strongest stationarity
        cue available: the IMU alone cannot tell rest from constant velocity, and a
        slowly accelerating vehicle looks almost still to both VO and the IMU.
        """
        if not self.has_imu:
            return FusedMotion(d_trans_vo, d_yaw_vo, 1.0, False, "vo")
        dyaw_imu, self._dyaw_acc = self._dyaw_acc, 0.0
        n_int, gz_mean = max(self._n_int, 1), self._gz_sum / max(self._n_int, 1)
        ay_mean = self._ay_sum / n_int
        self._gz_sum = self._ay_sum = 0.0
        self._n_int = 0

        v_vo = float(d_trans_vo) / max(dt, 1e-3)
        quiet = self.imu_still and (not tracking_ok or abs(v_vo) < self.cfg.zupt_vo_speed)
        if commanded_stop is False:
            quiet = False
        self._quiet_run = self._quiet_run + 1 if (quiet or commanded_stop) else 0
        zupt = bool(commanded_stop) and quiet or self._quiet_run >= self.cfg.zupt_min_frames
        if zupt:
            # zero-velocity update: the vehicle is at rest, so VO translation is jitter
            self._v_imu = 0.0
            # learning *biases* needs more certainty than discarding jitter: a commanded
            # stop, or a long quiet run.  A short quiet spell can be slow acceleration
            # from rest, and learning that as accel bias corrupts every later estimate.
            if commanded_stop or self._quiet_run >= self.cfg.bias_min_frames:
                self.gyro_bias += 0.2 * (gz_mean - self.gyro_bias)
                self._ay_bias += 0.2 * (ay_mean - self._ay_bias)
            self._hist.append((float(t), 0.0, 0.0))
            self._trim(t)
            return FusedMotion(0.0, 0.0, self.scale, True, "zupt")

        if not tracking_ok:
            # VO has nothing to say: keep the heading from the gyro, hold position
            return FusedMotion(0.0, dyaw_imu, self.scale, False, "imu-only (VO lost)")

        a = self.cfg.gyro_weight
        d_yaw = a * dyaw_imu + (1.0 - a) * float(d_yaw_vo)
        # slow bias tracking: the long-run mean of (gyro - VO) yaw rate is gyro bias
        if dt > 1e-3:
            self.gyro_bias += self.cfg.gyro_bias_alpha * (dyaw_imu - float(d_yaw_vo)) / dt
        self._hist.append((float(t), v_vo, self._v_imu))
        self._trim(t)
        self._update_scale()
        return FusedMotion(self.scale * float(d_trans_vo), d_yaw, self.scale, False, "vo+imu")

    def _trim(self, t: float) -> None:
        while self._hist and t - self._hist[0][0] > self.cfg.scale_window_s:
            self._hist.popleft()

    def _update_scale(self) -> None:
        if len(self._hist) < 6:
            return
        h = np.asarray(self._hist, np.float64)
        # velocity *changes* over the window: insensitive to the IMU's velocity offset
        k = max(1, len(h) // 4)
        dv_vo = h[k:, 1] - h[:-k, 1]
        dv_imu = h[k:, 2] - h[:-k, 2]
        den = float(np.sum(dv_vo ** 2))
        # excitation gate: the VO speed must actually have changed
        travelled = float(np.sum(np.abs(h[:, 1])) * (h[-1, 0] - h[0, 0]) / max(len(h), 1))
        if den < 0.05 * len(dv_vo) * 0.1 or travelled < self.cfg.scale_min_motion_m:
            return
        s_hat = float(np.sum(dv_imu * dv_vo) / den)
        if not math.isfinite(s_hat) or s_hat <= 0:
            return
        lo, hi = self.cfg.scale_clip
        s_hat = float(np.clip(s_hat, lo, hi))
        self.last_estimate = s_hat
        self.scale += self.cfg.scale_alpha * (s_hat - self.scale)
        self.scale = float(np.clip(self.scale, lo, hi))
        self.n_scale_updates += 1


if __name__ == "__main__":
    # synthetic check: true speed profile with accelerations, VO under-scaled by 2x
    rng = np.random.default_rng(0)
    fus = ImuFusion()
    imu_hz, vo_hz, T = 200, 30, 30.0
    t, v, yaw_true, yaw_vo, yaw_fused = 0.0, 0.0, 0.0, 0.0, 0.0
    dist_true = dist_vo = dist_fused = 0.0
    gyro_bias = 0.01
    next_vo, acc_d, acc_y, t_last_vo = 0.0, 0.0, 0.0, 0.0
    while t < T:
        a = 0.6 * math.sin(2 * math.pi * t / 4.0)          # speed up / slow down
        w = 0.3 * math.sin(2 * math.pi * t / 7.0)
        dt = 1.0 / imu_hz
        v = max(0.0, v + a * dt)
        a_eff = a if v > 0 else 0.0
        fus.add_imu(t, np.array([0, 0, w + gyro_bias + rng.normal(0, 0.005)]),
                    np.array([0, a_eff + rng.normal(0, 0.05), G]))
        acc_d += v * dt
        acc_y += w * dt
        dist_true += v * dt
        yaw_true += w * dt
        t += dt
        if t >= next_vo:
            d_vo = 0.5 * acc_d * (1 + rng.normal(0, 0.03))
            dy_vo = acc_y + rng.normal(0, 0.01)
            m = fus.fuse(t, d_vo, dy_vo, t - t_last_vo)
            t_last_vo = t
            dist_vo += d_vo
            dist_fused += m.d_trans
            yaw_vo += dy_vo
            yaw_fused += m.d_yaw
            acc_d = acc_y = 0.0
            next_vo += 1.0 / vo_hz
    print(f"true distance {dist_true:.2f} m | VO (0.5x scale) {dist_vo:.2f} m | "
          f"fused {dist_fused:.2f} m | final scale {fus.scale:.2f} "
          f"({fus.n_scale_updates} updates)")
    print(f"yaw error: VO {math.degrees(yaw_vo - yaw_true):+.1f} deg, "
          f"fused {math.degrees(yaw_fused - yaw_true):+.1f} deg")
    assert abs(fus.scale - 2.0) < 0.3, fus.scale
    assert abs(yaw_fused - yaw_true) < abs(yaw_vo - yaw_true) + 0.02
    print("self-test passed")
