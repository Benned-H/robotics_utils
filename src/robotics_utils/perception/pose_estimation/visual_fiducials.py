"""Define classes to represent visual fiducial markers (e.g., AprilTags)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import isclose
from typing import TYPE_CHECKING, Any

from robotics_utils.io.yaml_utils import load_yaml_data
from robotics_utils.kinematics import Pose3D

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class FiducialMarker:
    """A visual fiducial marker used for pose estimation."""

    id: int  # Unique ID of the fiducial
    size_cm: float  # Size (centimeters) of one side of the marker's black square
    relative_frames: dict[str, Pose3D]  # Object frames w.r.t. the marker

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

    @property
    def frame_name(self) -> str:
        """Retrieve the name of the reference frame defined by this visual fiducial."""
        return f"marker_{self.id}"


@dataclass(frozen=True)
class FiducialCamera:
    """A camera that detects visual fiducial markers."""

    name: str
    recognized_sizes_cm: frozenset[float]  # Sizes (cm) of AR markers detected by the camera

    @classmethod
    def from_yaml_data(cls, camera_name: str, camera_data: dict[str, Any]) -> FiducialCamera:
        """Construct a FiducialCamera instance from imported YAML data.

        :param camera_name: Name of the relevant camera
        :param camera_data: Camera data imported from YAML
        :return: Constructed FiducialCamera instance
        """
        for key in ["recognized_sizes_cm"]:
            if key not in camera_data:
                raise KeyError(f"Expected key '{key}' in data for camera '{camera_name}'.")

        return FiducialCamera(camera_name, frozenset(camera_data["recognized_sizes_cm"]))

    def can_recognize(self, fiducial: FiducialMarker) -> bool:
        """Evaluate whether the camera, as configured, can recognize the given fiducial marker."""
        return any(isclose(fiducial.size_cm, size_cm) for size_cm in self.recognized_sizes_cm)


@dataclass(frozen=True)
class FiducialSystem:
    """A system of known visual fiducial markers and fiducial-detecting cameras."""

    markers: dict[int, FiducialMarker]  # Map marker IDs to FiducialMarker instances
    cameras: dict[str, FiducialCamera]  # Map camera names to FiducialCamera instances
    camera_detects: dict[str, set[int]]  # Map camera names to markers they can detect

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> FiducialSystem:
        """Load a system of visual fiducial markers and camera detectors from a YAML file."""
        yaml_data = load_yaml_data(yaml_path, required_keys={"markers", "cameras"})

        markers = [
            FiducialMarker.from_yaml_data(fiducial_name, data)
            for fiducial_name, data in yaml_data["markers"].items()
        ]

        cameras = {
            FiducialCamera.from_yaml_data(camera_name, data)
            for camera_name, data in yaml_data["cameras"].items()
        }

        camera_detects_markers = {
            c.name: {m.id for m in markers if c.can_recognize(m)} for c in cameras
        }

        return FiducialSystem(
            markers={m.id: m for m in markers},
            cameras={c.name: c for c in cameras},
            camera_detects=camera_detects_markers,
        )
