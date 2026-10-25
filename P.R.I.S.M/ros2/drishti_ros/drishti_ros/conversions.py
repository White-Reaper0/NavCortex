"""Frame and message conversions between ROS (REP-103) and prism.

Pure Python + NumPy, no rclpy import, so it is unit-tested outside ROS
(`tests/test_ros_conversions.py`).

    ROS base_link / odom (REP-103):  x forward, y left,  z up, yaw CCW from +x
    prism vehicle / world:         x right,   y forward, z up, yaw CCW from +y

So  x_d = -y_ros,  y_d = x_ros,  yaw_d = yaw_ros  (both CCW, both measured from
"forward"), and angular velocity about z is the same number in both.
"""
from __future__ import annotations

import math

import numpy as np


def ros_xy_to_drishti(x: float, y: float) -> tuple[float, float]:
    return -float(y), float(x)


def drishti_xy_to_ros(x: float, y: float) -> tuple[float, float]:
    return float(y), -float(x)


def ros_imu_to_drishti(gyro_xyz, accel_xyz) -> tuple[np.ndarray, np.ndarray]:
    """base_link IMU (x fwd, y left, z up) -> prism vehicle frame (x right, y fwd, z up)."""
    gx, gy, gz = (float(v) for v in gyro_xyz)
    ax, ay, az = (float(v) for v in accel_xyz)
    return np.array([-gy, gx, gz]), np.array([-ay, ax, az])


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) for a rotation of `yaw` about +z."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def image_to_bgr(encoding: str, height: int, width: int, step: int, data: bytes) -> np.ndarray:
    """sensor_msgs/Image -> (H, W, 3) uint8 BGR, without cv_bridge."""
    enc = encoding.lower()
    buf = np.frombuffer(bytes(data), np.uint8)
    if enc in ("bgr8", "rgb8"):
        img = buf.reshape(height, step)[:, :width * 3].reshape(height, width, 3)
        return np.ascontiguousarray(img[..., ::-1] if enc == "rgb8" else img)
    if enc in ("bgra8", "rgba8"):
        img = buf.reshape(height, step)[:, :width * 4].reshape(height, width, 4)
        img = img[..., :3]
        return np.ascontiguousarray(img[..., ::-1] if enc == "rgba8" else img)
    if enc in ("mono8", "8uc1"):
        img = buf.reshape(height, step)[:, :width]
        return np.ascontiguousarray(np.repeat(img[..., None], 3, axis=2))
    raise ValueError(f"unsupported image encoding {encoding!r}")


def diff_drive_wheels(v: float, w: float, wheel_base_m: float) -> tuple[float, float]:
    """Unicycle (v, w) -> left/right wheel surface speeds, m/s."""
    return v - 0.5 * w * wheel_base_m, v + 0.5 * w * wheel_base_m
