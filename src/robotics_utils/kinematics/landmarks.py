"""Define a class representing a set of named 2D landmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from robotics_utils.filesystem.yaml_utils import load_yaml_data
from robotics_utils.kinematics import DEFAULT_FRAME
from robotics_utils.kinematics.poses import Pose2D


class Landmarks(Dict[str, Pose2D]):
    """A collection of named landmarks at known 2D poses."""

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> Landmarks:
        """Load a Landmarks instance from a YAML file.

        :param yaml_path: Path to a YAML file containing landmarks data
        :return: Constructed Landmarks instance
        """
        yaml_data: dict[str, Any] = load_yaml_data(yaml_path, required_keys={"known_landmarks"})

        # Extract the 'known_landmarks' and 'default_frame' values
        landmarks_data: dict[str, Any] = yaml_data["known_landmarks"]
        default_frame = yaml_data.get("default_frame", DEFAULT_FRAME)

        landmarks = Landmarks()
        for name, landmark_data in landmarks_data.items():
            if isinstance(landmark_data, list):
                landmarks[name] = Pose2D.from_list(landmark_data, default_frame)
            elif isinstance(landmark_data, dict):
                ref_frame: str = landmark_data["frame"]
                pose_list: list[float] = landmark_data["x_y_yaw"]
                landmarks[name] = Pose2D.from_list(pose_list, ref_frame)

        return landmarks

    def to_yaml_data(self, default_frame: str | None) -> dict[str, Any]:
        """Convert the landmarks into a dictionary suitable for export to YAML.

        :param default_frame: Default frame assumed in the parent YAML file (or ignored if None)
        :return: Dictionary mapping landmark names to their Pose2D data
        """
        return {landmark: pose.to_yaml_data(default_frame) for landmark, pose in self.items()}
