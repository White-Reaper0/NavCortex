"""Scores one Point A -> Point B run in Gazebo: success, collisions, interventions.

Ground truth comes from the simulator, never from prism:
    /ground_truth/odom   nav_msgs/Odometry       (gz OdometryPublisher, bridged)
    /prism/contacts    ros_gz_interfaces/Contacts  (chassis contact sensor, bridged)
    /prism/intervention std_msgs/Bool          (operator take-over button, any source)
    /prism/decision    std_msgs/String         (only to log the rules that fired)

The run ends on arrival within `goal_tolerance_m`, or at `timeout_s`.  A result
JSON is appended to `results_file` so many runs can be aggregated into the
success rate / collisions / interventions the presentation reports.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, String

try:
    from ros_gz_interfaces.msg import Contacts
except ImportError:                      # contacts are optional
    Contacts = None


class MissionMonitor(Node):
    def __init__(self):
        super().__init__("drishti_mission_monitor")
        p = self.declare_parameter
        p("goal_x", 8.0)
        p("goal_y", 0.0)
        p("goal_tolerance_m", 0.30)
        p("timeout_s", 120.0)
        p("results_file", "drishti_gazebo_runs.jsonl")
        p("scenario", "default")
        p("ground_name", "ground_plane")
        g = lambda n: self.get_parameter(n).value          # noqa: E731
        self.goal = (float(g("goal_x")), float(g("goal_y")))
        self.tol = float(g("goal_tolerance_m"))
        self.timeout = float(g("timeout_s"))
        self.out = Path(g("results_file")).expanduser()
        self.scenario = g("scenario")
        self.ground = g("ground_name")

        self.t_start = None
        self.collisions = 0
        self.in_contact = False
        self.interventions = 0
        self.prev_iv = False
        self.path_m = 0.0
        self.last_xy = None
        self.rules: dict[str, int] = {}
        self.done = False

        self.create_subscription(Odometry, "/ground_truth/odom", self._on_odom, 10)
        self.create_subscription(Bool, "/prism/intervention", self._on_iv, 10)
        self.create_subscription(String, "/prism/decision", self._on_dec, 10)
        if Contacts is not None:
            self.create_subscription(Contacts, "/prism/contacts", self._on_contacts, 10)
        else:
            self.get_logger().warn("ros_gz_interfaces not found: collisions are not scored")
        self.create_timer(0.5, self._tick)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_odom(self, msg: Odometry):
        if self.done:
            return
        if self.t_start is None:
            self.t_start = self._now()
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if self.last_xy is not None:
            self.path_m += math.hypot(x - self.last_xy[0], y - self.last_xy[1])
        self.last_xy = (x, y)
        if math.hypot(x - self.goal[0], y - self.goal[1]) <= self.tol:
            self._finish("success")

    def _on_contacts(self, msg):
        hit = any(self.ground not in (c.collision1.name + c.collision2.name)
                  for c in msg.contacts)
        if hit and not self.in_contact:
            self.collisions += 1
            self.get_logger().warn(f"collision #{self.collisions}")
        self.in_contact = hit

    def _on_iv(self, msg: Bool):
        if msg.data and not self.prev_iv:
            self.interventions += 1
        self.prev_iv = bool(msg.data)

    def _on_dec(self, msg: String):
        try:
            rule = json.loads(msg.data).get("rule", "?")
        except ValueError:
            return
        self.rules[rule] = self.rules.get(rule, 0) + 1

    def _tick(self):
        if not self.done and self.t_start is not None and self._now() - self.t_start > self.timeout:
            self._finish("timeout")

    def _finish(self, outcome: str):
        self.done = True
        rec = dict(scenario=self.scenario, outcome=outcome, success=outcome == "success",
                   time_s=round(self._now() - (self.t_start or self._now()), 2),
                   path_m=round(self.path_m, 2), collisions=self.collisions,
                   interventions=self.interventions, rules=self.rules,
                   goal=self.goal, wall_clock=time.strftime("%Y-%m-%d %H:%M:%S"))
        self.out.parent.mkdir(parents=True, exist_ok=True)
        with self.out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        self.get_logger().info(f"run finished: {rec}")


def main(args=None):
    rclpy.init(args=args)
    node = MissionMonitor()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
