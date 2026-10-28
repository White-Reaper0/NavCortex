"""Gazebo Harmonic + prism + mission scoring, one command:

    ros2 launch prism_ros sim.launch.py prism_root:=/path/to/prism
    ros2 launch prism_ros sim.launch.py prism_root:=... device:=cpu headless:=true

Point B defaults to (8, 0) in the world/odom frame (REP-103: x forward).
Each run appends one JSON line (success, collisions, interventions, time,
path length, rules fired) to `results_file`.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("prism_ros")
    world = os.path.join(share, "worlds", "prism_offroad.sdf")
    bridge = os.path.join(share, "config", "bridge.yaml")
    params = os.path.join(share, "config", "prism.yaml")
    L = LaunchConfiguration

    args = [
        DeclareLaunchArgument("prism_root", default_value=os.environ.get("PRISM_ROOT", "")),
        DeclareLaunchArgument("device", default_value="cuda"),
        DeclareLaunchArgument("vehicle", default_value="wave_rover"),
        DeclareLaunchArgument("goal_x", default_value="8.0"),
        DeclareLaunchArgument("goal_y", default_value="0.0"),
        DeclareLaunchArgument("timeout_s", default_value="120.0"),
        DeclareLaunchArgument("results_file", default_value="prism_gazebo_runs.jsonl"),
        DeclareLaunchArgument("scenario", default_value="prism_offroad"),
        DeclareLaunchArgument("headless", default_value="false"),
    ]

    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": PythonExpression(
            ["'-r -s " + world + "' if '", L("headless"), "' == 'true' else '-r " + world + "'"])
        }.items(),
    )

    bridge_node = Node(
        package="ros_gz_bridge", executable="parameter_bridge", name="prism_bridge",
        parameters=[{"config_file": bridge, "use_sim_time": True}], output="screen")

    prism = Node(
        package="prism_ros", executable="prism_node", name="prism", output="screen",
        parameters=[params, {
            "use_sim_time": True,
            "prism_root": L("prism_root"),
            "device": L("device"),
            "vehicle": L("vehicle"),
            "goal_x": L("goal_x"),
            "goal_y": L("goal_y"),
        }])

    monitor = Node(
        package="prism_ros", executable="mission_monitor", name="prism_mission_monitor",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "goal_x": L("goal_x"),
            "goal_y": L("goal_y"),
            "timeout_s": L("timeout_s"),
            "results_file": L("results_file"),
            "scenario": L("scenario"),
        }])

    return LaunchDescription(args + [gz, bridge_node, prism, monitor])
