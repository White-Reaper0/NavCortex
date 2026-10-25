"""ROS <-> prism conversions (pure Python; rclpy is not needed)."""
import importlib.util
import math
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent.parent / "ros2" / "drishti_ros" / "drishti_ros" / "conversions.py"
spec = importlib.util.spec_from_file_location("drishti_ros_conversions", P)
cv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv)


def test_xy_round_trip_and_axes():
    # ROS forward (x) is prism forward (y); ROS left (+y) is prism -x
    assert cv.ros_xy_to_drishti(8.0, 0.0) == (0.0, 8.0)
    assert cv.ros_xy_to_drishti(0.0, 1.0) == (-1.0, 0.0)
    for x, y in [(1.5, -2.0), (0.0, 3.0), (-4.0, 0.5)]:
        assert np.allclose(cv.drishti_xy_to_ros(*cv.ros_xy_to_drishti(x, y)), (x, y))


def test_imu_axes():
    g, a = cv.ros_imu_to_drishti((0.0, 0.0, 0.3), (1.0, 0.0, 9.81))
    assert g[2] == 0.3                       # yaw rate is the same number
    assert a[1] == 1.0 and a[0] == 0.0       # forward accel lands on prism +y


def test_quaternion_yaw_round_trip():
    for yaw in (-2.5, -0.3, 0.0, 1.2, 3.0):
        assert math.isclose(cv.quaternion_to_yaw(*cv.yaw_to_quaternion(yaw)), yaw, abs_tol=1e-9)


def test_image_decoding():
    h, w = 4, 5
    rgb = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
    bgr = cv.image_to_bgr("rgb8", h, w, w * 3, rgb.tobytes())
    assert np.array_equal(bgr, rgb[..., ::-1])
    padded = np.zeros((h, w * 3 + 4), np.uint8)
    padded[:, :w * 3] = rgb.reshape(h, -1)
    assert np.array_equal(cv.image_to_bgr("bgr8", h, w, w * 3 + 4, padded.tobytes()), rgb)


def test_diff_drive():
    l, r = cv.diff_drive_wheels(0.5, 1.0, 0.2)
    assert math.isclose(l, 0.4) and math.isclose(r, 0.6)
