"""Define classes to represent visual fiducial markers (e.g., AprilTags)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from robotics_utils.io.yaml_utils import load_yaml_data
from robotics_utils.spatial import Pose3D

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class FiducialMarker:
    """A visual fiducial marker used for pose estimation."""

    id: int  # Unique ID of the fiducial
    size_cm: float  # Size (centimeters) of one side of the marker's black square
    relative_frames: dict[str, Pose3D]  # Object frames w.r.t. the marker

    def __str__(self) -> str:
        """Return a human-readable string representation of the fiducial marker."""
        child_frames = ", ".join(self.relative_frames.keys())
        return f"{self.frame_name} ({self.size_cm} cm) has child frames: {child_frames}"

    @classmethod
    def from_yaml_data(cls, marker_name: str, data: dict[str, Any]) -> FiducialMarker:
        """Construct a FiducialMarker instance from imported YAML data.

        :param marker_name: Name of the marker (e.g., "marker_123")
        :param data: Marker data imported from YAML
        :return: Constructed FiducialMarker instance
        """
        if not marker_name.startswith("marker_"):
            raise ValueError(f"Visual fiducial name '{marker_name}' doesn't start with 'marker_'.")

        id_digits = "".join(re.findall(r"\d+", marker_name))
        marker_id = int(id_digits)
        size_cm = data["size_cm"]

        relative_frames: dict[str, Pose3D] = {}
        for child_frame_name, pose_data in data.get("relative_frames", {}).items():
            relative_pose = Pose3D.from_yaml_data(pose_data, default_frame=marker_name)
            relative_frames[child_frame_name] = relative_pose

        return FiducialMarker(marker_id, size_cm, relative_frames)

    @staticmethod
    def id_to_frame_name(tag_id: int) -> str:
        """Construct the name of the reference frame for a tag with the given ID."""
        return f"marker_{tag_id}"

    @property
    def frame_name(self) -> str:
        """Retrieve the name of the reference frame defined by this visual fiducial."""
        return self.id_to_frame_name(self.id)


@dataclass(frozen=True)
class FiducialSystem:
    """A system of known visual fiducial markers and fiducial-detecting cameras."""

    markers: dict[int, FiducialMarker]  # Map marker IDs to FiducialMarker instances
    camera_names: set[str]

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> FiducialSystem:
        """Load a system of visual fiducial markers from a YAML file."""
        yaml_data = load_yaml_data(yaml_path, required_keys={"markers", "cameras"})

        markers = [
            FiducialMarker.from_yaml_data(fiducial_name, data)
            for fiducial_name, data in yaml_data["markers"].items()
        ]
        camera_names = set(yaml_data["cameras"])

        return FiducialSystem(markers={m.id: m for m in markers}, camera_names=camera_names)
