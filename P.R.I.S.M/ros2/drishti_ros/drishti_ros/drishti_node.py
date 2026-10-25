"""prism as a ROS 2 node: one RGB camera in, cmd_vel out.

Subscribes
    ~image_topic   sensor_msgs/Image        the only world sensor
    ~imu_topic     sensor_msgs/Imu          rover self-motion only (optional)
    ~goal_topic    geometry_msgs/PoseStamped  Point B, odom frame (optional; or params)
Publishes
    ~cmd_topic     geometry_msgs/Twist      linear.x = v, angular.z = w
    /prism/decision  std_msgs/String      JSON: GO/SLOW/REROUTE/STOP + rule + reason
    /prism/odom      nav_msgs/Odometry    prism's own pose estimate (REP-103)

Safety behaviour that does not depend on the navigation stack being healthy:
* a **watchdog** publishes zero velocity if no camera frame has arrived for
  `watchdog_s`, or if a control cycle raises;
* every published command is the supervisor's, never the policy's.

The heavy work runs in a fixed-rate control timer on the *latest* frame; frames
that arrive while a cycle is running are dropped, not queued, so latency does
not build up on a slow machine.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import String

from .conversions import (drishti_xy_to_ros, image_to_bgr,
                          ros_imu_to_drishti, ros_xy_to_drishti, yaw_to_quaternion)


def _import_drishti(root: str):
    """Make the prism repository importable (it is not a ROS package)."""
    if root:
        sys.path.insert(0, str(Path(root).expanduser().resolve()))
    elif os.environ.get("DRISHTI_ROOT"):
        sys.path.insert(0, os.environ["DRISHTI_ROOT"])
    from prism.runtime import PrismNavigator   # noqa: E402
    return PrismNavigator


class DrishtiNode(Node):
    def __init__(self):
        super().__init__("prism")
        p = self.declare_parameter
        p("drishti_root", "")
        p("device", "cuda")
        p("vehicle", "wave_rover")
        p("image_topic", "/camera/image_raw")
        p("imu_topic", "/imu")
        p("goal_topic", "/goal_pose")
        p("cmd_topic", "/cmd_vel")
        p("control_hz", 10.0)
        p("watchdog_s", 0.5)
        p("use_world_model", False)
        p("use_policy", False)
        p("use_imu", True)
        p("goal_x", float("nan"))        # Point B in the odom frame (REP-103), metres
        p("goal_y", float("nan"))
        g = lambda n: self.get_parameter(n).value          # noqa: E731

        Nav = _import_drishti(g("drishti_root"))
        self.nav = Nav(device=g("device"), vehicle=g("vehicle"),
                       use_world_model=g("use_world_model"), use_policy=g("use_policy"))
        self.get_logger().info(f"prism up: vehicle={g('vehicle')} device={g('device')}")

        self.cmd_pub = self.create_publisher(Twist, g("cmd_topic"), 10)
        self.dec_pub = self.create_publisher(String, "/prism/decision", 10)
        self.odom_pub = self.create_publisher(Odometry, "/prism/odom", 10)
        self.create_subscription(Image, g("image_topic"), self._on_image, qos_profile_sensor_data)
        if g("use_imu"):
            self.create_subscription(Imu, g("imu_topic"), self._on_imu, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, g("goal_topic"), self._on_goal, 10)

        self.watchdog_s = float(g("watchdog_s"))
        self._frame = None
        self._frame_t = None
        self._last_rx = None
        self._t0 = None
        gx, gy = float(g("goal_x")), float(g("goal_y"))
        if np.isfinite(gx) and np.isfinite(gy):
            self._set_goal_ros(gx, gy)
        self.create_timer(1.0 / float(g("control_hz")), self._control)

    # ------------------------------------------------------------------ inputs
    def _stamp(self, msg_stamp) -> float:
        t = msg_stamp.sec + msg_stamp.nanosec * 1e-9
        if self._t0 is None:
            self._t0 = t
        return t - self._t0

    def _on_image(self, msg: Image):
        try:
            self._frame = image_to_bgr(msg.encoding, msg.height, msg.width, msg.step, msg.data)
            self._frame_t = self._stamp(msg.header.stamp)
            self._last_rx = time.monotonic()
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)

    def _on_imu(self, msg: Imu):
        gyro, acc = ros_imu_to_drishti(
            (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z),
            (msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z))
        self.nav.add_imu(self._stamp(msg.header.stamp), gyro, acc)

    def _set_goal_ros(self, x: float, y: float):
        xd, yd = ros_xy_to_drishti(x, y)
        self.nav.set_goal(xd, yd)
        self.get_logger().info(f"Point B set: odom ({x:.2f}, {y:.2f}) m")

    def _on_goal(self, msg: PoseStamped):
        self._set_goal_ros(msg.pose.position.x, msg.pose.position.y)

    # ------------------------------------------------------------------ control
    def _stop(self, why: str):
        self.cmd_pub.publish(Twist())
        self.dec_pub.publish(String(data=json.dumps({"decision": "STOP", "rule": "WD",
                                                     "reason": why})))

    def _control(self):
        if self._last_rx is None or time.monotonic() - self._last_rx > self.watchdog_s:
            self._stop(f"watchdog: no camera frame for > {self.watchdog_s:.2f} s")
            return
        frame, t = self._frame, self._frame_t
        self._frame = None                      # use each frame once
        if frame is None:
            return                              # no new frame yet: keep the last command
        try:
            out = self.nav.step(frame, t)
        except Exception as e:                  # never leave the motors on an old command
            self.get_logger().error(f"control cycle failed: {e!r}")
            self._stop(f"control cycle failed: {type(e).__name__}")
            return
        cmd = Twist()
        cmd.linear.x = float(out.v_mps)
        cmd.angular.z = float(out.w_radps)
        self.cmd_pub.publish(cmd)
        self.dec_pub.publish(String(data=json.dumps(out.as_dict(), default=float)))

        od = Odometry()
        od.header.stamp = self.get_clock().now().to_msg()
        od.header.frame_id = "odom"
        od.child_frame_id = "base_link"
        x, y = drishti_xy_to_ros(out.pose[0], out.pose[1])
        od.pose.pose.position.x, od.pose.pose.position.y = x, y
        q = yaw_to_quaternion(out.pose[2])
        (od.pose.pose.orientation.x, od.pose.pose.orientation.y,
         od.pose.pose.orientation.z, od.pose.pose.orientation.w) = q
        od.twist.twist.linear.x = float(out.v_mps)
        od.twist.twist.angular.z = float(out.w_radps)
        self.odom_pub.publish(od)


def main(args=None):
    rclpy.init(args=args)
    node = DrishtiNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())           # leave the vehicle stopped
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
