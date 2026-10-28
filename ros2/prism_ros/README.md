# prism_ros — PRISM on ROS 2 Jazzy + Gazebo Harmonic

One RGB camera in, `cmd_vel` out. This package wraps `prism/runtime.py`
(`PrismNavigator`: depth → terrain → traversability/trust → odometry (+ IMU) →
2.5-D map → dynamic layer → goal layer / D* Lite → 17 arcs → safety supervisor).

> **Status:** written, not yet run in this repository's build environment. The
> pure-Python helpers (`conversions.py`) are unit-tested in `tests/test_ros_conversions.py`.
> Treat the first launch as bring-up.

## Nodes

| Executable | Role |
|---|---|
| `prism_node` | subscribes `/camera/image_raw`, `/imu` (optional), `/goal_pose`; publishes `/cmd_vel`, `/prism/decision` (JSON: GO/SLOW/REROUTE/STOP + rule + reason), `/prism/odom`. Watchdog: zero velocity if no frame for 0.5 s or if a cycle fails. |
| `mission_monitor` | scores a run from **simulator ground truth** (never fed to PRISM): success within 0.30 m of Point B, collisions from the chassis contact sensor, interventions from `/prism/intervention`. Appends one JSON line per run. |
| `wave_rover_bridge` | `/cmd_vel` → WAVE ROVER JSON wheel commands over serial, with a 0.3 s motor watchdog. The JSON format is a parameter: check it against your board's firmware. |

## Simulation

```bash
# ROS 2 Jazzy + Gazebo Harmonic + ros_gz installed
cd ~/ros2_ws/src && ln -s /path/to/prism/ros2/prism_ros .
cd ~/ros2_ws && colcon build --packages-select prism_ros && source install/setup.bash
pip install -r /path/to/prism/requirements.txt      # into the Python ROS uses

ros2 launch prism_ros sim.launch.py prism_root:=/path/to/PRISM
ros2 launch prism_ros sim.launch.py prism_root:=/path/to/PRISM device:=cpu headless:=true \
    results_file:=runs.jsonl scenario:=dead_end_seed0
```

`worlds/prism_offroad.sdf`: start at the origin facing +x, Point B at (8, 0). In between are a U-shaped dead end that opens towards the start, a 6 cm kerb (above the 4.5 cm clearance), rocks and a walking pedestrian. The rover matches `configs/vehicles/wave_rover.json`: 34 x 22 cm footprint, 4.5 cm clearance, camera 12 cm above the ground pitched 6° down, 92° HFOV, 640x360 @ 30 Hz, and a 200 Hz IMU with gyro bias.

Aggregate many runs into the presentation's metrics:

```bash
python -c "import json;r=[json.loads(l) for l in open('runs.jsonl')];print(len(r),'runs',
 sum(x['success'] for x in r)/len(r),'success', sum(x['collisions'] for x in r),'collisions',
 sum(x['interventions'] for x in r),'interventions')"
```

## Rover

```bash
ros2 launch prism_ros rover.launch.py prism_root:=/path/to/PRISM \
    camera_device:=/dev/video0 serial_port:=/dev/ttyTHS1
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
    "{header: {frame_id: odom}, pose: {position: {x: 8.0, y: 0.0}}}"
```

The first runs should be done with the wheels off the ground and a hand on the power switch. Speed is capped at 0.6 m/s by the `wave_rover` profile until a measured brake test raises it.

## Frames

ROS uses REP-103 (x forward, y left); PRISM uses x right, y forward. `conversions.py` maps between them (`x_d = -y_ros`, `y_d = x_ros`, same yaw and yaw rate).
