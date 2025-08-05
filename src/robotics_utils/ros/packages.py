"""Define utility functions for working with ROS packages."""

from __future__ import annotations

from pathlib import Path

from robotics_utils.io.logging import log_info

try:
    import rospy
    from rospkg import RosPack

    ROS_PRESENT = True
except ModuleNotFoundError:
    ROS_PRESENT = False


def resolve_package_path(relative_path: str) -> Path | None:
    """Resolve a filepath relative to a ROS package.

      Input (relative): spot_skills/launch/demo.launch
      Output (absolute): /home/Documents/catkin_ws/src/spot_skills/launch/demo.launch

    :param relative_path: Path to a file, starting with a ROS package
    :return: Path object containing the full absolute path (or None if path doesn't exist)
    """
    if not ROS_PRESENT:
        log_info(f"Cannot resolve path '{relative_path}' without ROS.")
        return None

    package_name, relative_path = relative_path.split("/", maxsplit=1)

    rospack = RosPack()
    package_path = rospack.get_path(package_name)
    full_path = Path(package_path, relative_path)

    if not full_path.exists():
        rospy.logerr(f"Could not resolve package-relative filepath: {relative_path}.")
        return None

    return full_path
