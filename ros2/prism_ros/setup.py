from glob import glob
from setuptools import setup

package_name = "prism_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/worlds", glob("worlds/*.sdf")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Team UniMinds",
    description="ROS 2 wrapper for the prism camera-only navigation stack",
    license="All Rights Reserved",
    entry_points={
        "console_scripts": [
            "prism_node = prism_ros.prism_node:main",
            "mission_monitor = prism_ros.mission_monitor:main",
            "wave_rover_bridge = prism_ros.wave_rover_bridge:main",
        ],
    },
)
