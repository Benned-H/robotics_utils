"""Define a class representing a set of named 2D navigation waypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from robotics_utils.filesystem.yaml_utils import load_yaml_data
from robotics_utils.kinematics import DEFAULT_FRAME
from robotics_utils.kinematics.poses import Pose2D


class Waypoints(Dict[str, Pose2D]):
    """A collection of named waypoints at known 2D poses."""

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> Waypoints:
        """Load a Waypoints instance from a YAML file.

        :param yaml_path: Path to a YAML file containing waypoints data
        :return: Constructed Waypoints instance
        """
        yaml_data: dict[str, Any] = load_yaml_data(yaml_path, required_keys={"waypoints"})

        # Extract the 'waypoints' and 'default_frame' values
        waypoints_data: dict[str, Any] = yaml_data["waypoints"]
        default_frame = yaml_data.get("default_frame", DEFAULT_FRAME)

        waypoints = Waypoints()
        for name, waypoint_data in waypoints_data.items():
            if isinstance(waypoint_data, list):
                waypoints[name] = Pose2D.from_list(waypoint_data, default_frame)
            elif isinstance(waypoint_data, dict):
                ref_frame: str = waypoint_data["frame"]
                pose_list: list[float] = waypoint_data["x_y_yaw"]
                waypoints[name] = Pose2D.from_list(pose_list, ref_frame)

        return waypoints

    def to_yaml_data(self, default_frame: str | None) -> dict[str, Any]:
        """Convert the waypoints into a dictionary suitable for export to YAML.

        :param default_frame: Default frame assumed in the parent YAML file (ignored if None)
        :return: Dictionary mapping waypoint names to their Pose2D data
        """
        return {name: pose.to_yaml_data(default_frame) for name, pose in self.items()}
