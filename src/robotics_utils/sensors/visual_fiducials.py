"""Define classes to represent visual fiducial markers (e.g., AprilTags)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import Any

from robotics_utils.io.yaml_utils import load_yaml_data
from robotics_utils.kinematics import Pose3D


@dataclass(frozen=True)
class VisualFiducial:
    """A visual fiducial marker used to support pose estimation."""

    id: int  # Unique ID of the fiducial
    size_cm: float  # Size (centimeters) of one side of the marker's black square
    relative_frames: dict[str, Pose3D]  # Object frames w.r.t. the marker

    @classmethod
    def from_yaml_data(cls, marker_name: str, data: dict[str, Any]) -> VisualFiducial:
        """Construct a VisualFiducial instance from imported YAML data.

        :param marker_name: Name of the marker (e.g., "marker_123")
        :param data: Marker data imported from YAML
        :return: Constructed VisualFiducial instance
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

        return VisualFiducial(id=marker_id, size_cm=size_cm, relative_frames=relative_frames)

    @property
    def frame_name(self) -> str:
        """Retrieve the name of the reference frame defined by this visual fiducial."""
        return f"marker_{self.id}"


@dataclass(frozen=True)
class FiducialDetector:
    """A camera that detects visual fiducial markers."""

    name: str
    recognized_sizes_cm: set[float]  # Sizes of AR markers detected by the camera

    @classmethod
    def from_yaml_data(cls, camera_name: str, camera_data: dict[str, Any]) -> FiducialDetector:
        """Construct a FiducialDetector instance from imported YAML data.

        :param camera_name: Name of the relevant camera
        :param camera_data: Camera data imported from YAML
        :return: Constructed FiducialDetector instance
        """
        for key in ["recognized_sizes_cm"]:
            if key not in camera_data:
                raise KeyError(f"Expected key '{key}' in data for camera '{camera_name}'.")

        recognized_sizes_cm = set(camera_data["recognized_sizes_cm"])
        return FiducialDetector(name=camera_name, recognized_sizes_cm=recognized_sizes_cm)

    def can_recognize(self, fiducial: VisualFiducial) -> bool:
        """Evaluate whether the camera, as configured, can recognize the given fiducial marker."""
        return any(isclose(fiducial.size_cm, size_cm) for size_cm in self.recognized_sizes_cm)


@dataclass(frozen=True)
class VisualFiducialSystem:
    """A system of known visual fiducial markers and fiducial-detecting cameras."""

    markers: dict[int, VisualFiducial]  # Map marker IDs to VisualFiducial instances
    cameras: dict[str, FiducialDetector]  # Map camera names to FiducialDetector instances
    camera_detects: dict[str, set[int]]  # Map camera names to markers they can detect

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> VisualFiducialSystem:
        """Load a system of visual fiducial markers and camera detectors from a YAML file."""
        yaml_data = load_yaml_data(yaml_path, required_keys={"markers", "cameras"})

        fiducials = {
            VisualFiducial.from_yaml_data(fiducial_name, data)
            for fiducial_name, data in yaml_data["markers"].items()
        }

        detectors = {
            FiducialDetector.from_yaml_data(camera_name, data)
            for camera_name, data in yaml_data["cameras"].items()
        }

        camera_detects_markers = {
            c.name: {f.id for f in fiducials if c.can_recognize(f)} for c in detectors
        }

        return VisualFiducialSystem(
            markers={f.id: f for f in fiducials},
            cameras={d.name: d for d in detectors},
            camera_detects=camera_detects_markers,
        )
