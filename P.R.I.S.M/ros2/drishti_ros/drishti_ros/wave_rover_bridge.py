"""cmd_vel -> Waveshare WAVE ROVER over serial, with a motor watchdog.

The WAVE ROVER's ESP32 driver board takes newline-terminated JSON commands on its
UART.  The default `cmd_format` below follows the speed-control command in
Waveshare's WAVE ROVER JSON examples (``{"T":1,"L":<left>,"R":<right>}``, values
in [-0.5, 0.5]).  It is a *parameter* because the JSON protocol can differ
between firmware versions: check the JSON command reference for the firmware on
your board before the first run, with the wheels off the ground.

Not yet run on hardware (see README honesty ledger).

Safety:
* commands older than `watchdog_s` are replaced by a stop;
* wheel commands are clipped to `max_wheel_cmd`;
* on shutdown a stop is sent.
"""
from __future__ import annotations

import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from .conversions import diff_drive_wheels


class WaveRoverBridge(Node):
    def __init__(self):
        super().__init__("wave_rover_bridge")
        p = self.declare_parameter
        p("port", "/dev/ttyTHS1")
        p("baud", 115200)
        p("wheel_base_m", 0.20)
        p("max_speed_mps", 0.6)               # speed that maps to max_wheel_cmd
        p("max_wheel_cmd", 0.5)
        p("cmd_format", '{{"T":1,"L":{left:.3f},"R":{right:.3f}}}')
        p("watchdog_s", 0.3)
        p("rate_hz", 20.0)
        g = lambda n: self.get_parameter(n).value          # noqa: E731
        import serial                                     # pyserial
        self.ser = serial.Serial(g("port"), int(g("baud")), timeout=0.05)
        self.wb = float(g("wheel_base_m"))
        self.vmax = float(g("max_speed_mps"))
        self.cmax = float(g("max_wheel_cmd"))
        self.fmt = g("cmd_format")
        self.watchdog = float(g("watchdog_s"))
        self.last = (0.0, 0.0)
        self.last_t = 0.0
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd, 10)
        self.create_timer(1.0 / float(g("rate_hz")), self._send)

    def _on_cmd(self, msg: Twist):
        self.last = (float(msg.linear.x), float(msg.angular.z))
        self.last_t = time.monotonic()

    def _write(self, left: float, right: float):
        clip = lambda u: max(-self.cmax, min(self.cmax, u))   # noqa: E731
        line = self.fmt.format(left=clip(left), right=clip(right)) + "\n"
        self.ser.write(line.encode("ascii"))

    def _send(self):
        v, w = self.last
        if time.monotonic() - self.last_t > self.watchdog:
            v, w = 0.0, 0.0                       # stale command: stop the motors
        l, r = diff_drive_wheels(v, w, self.wb)
        k = self.cmax / max(self.vmax, 1e-6)
        self._write(l * k, r * k)

    def stop(self):
        try:
            self._write(0.0, 0.0)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = WaveRoverBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
