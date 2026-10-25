"""prism on the rover: USB camera + prism + WAVE ROVER serial bridge.

    ros2 launch drishti_ros rover.launch.py drishti_root:=/path/to/prism \
        camera_device:=/dev/video0 serial_port:=/dev/ttyTHS1

Point B is sent at run time:
    ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
        "{header: {frame_id: odom}, pose: {position: {x: 8.0, y: 0.0}}}"

The IMU topic is expected on /imu (sensor_msgs/Imu, base_link frame) from
whichever driver exposes the rover's IMU; without it prism falls back to
camera-only odometry, as on the recorded footage.

Not yet run on hardware: keep the wheels off the ground for the first launch,
and keep a hand on the power switch.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    L = LaunchConfiguration
    args = [
        DeclareLaunchArgument("drishti_root", default_value=os.environ.get("DRISHTI_ROOT", "")),
        DeclareLaunchArgument("device", default_value="cuda"),
        DeclareLaunchArgument("camera_device", default_value="/dev/video0"),
        DeclareLaunchArgument("serial_port", default_value="/dev/ttyTHS1"),
    ]
    camera = Node(
        package="v4l2_camera", executable="v4l2_camera_node", name="camera",
        parameters=[{"video_device": L("camera_device"), "image_size": [640, 360],
                     "pixel_format": "YUYV", "output_encoding": "bgr8"}],
        remappings=[("image_raw", "/camera/image_raw")])
    prism = Node(
        package="drishti_ros", executable="drishti_node", name="prism", output="screen",
        parameters=[{"drishti_root": L("drishti_root"), "device": L("device"),
                     "vehicle": "wave_rover", "control_hz": 10.0, "watchdog_s": 0.5}])
    rover = Node(
        package="drishti_ros", executable="wave_rover_bridge", name="wave_rover_bridge",
        output="screen", parameters=[{"port": L("serial_port"), "watchdog_s": 0.3}])
    return LaunchDescription(args + [camera, prism, rover])
