"""Define classes to represent visual fiducial markers (e.g., AprilTags)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import isclose
from typing import TYPE_CHECKING, Any, Iterable

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

    @property
    def frame_name(self) -> str:
        """Retrieve the name of the reference frame defined by this visual fiducial."""
        return f"marker_{self.id}"


FiducialMarkers = Iterable[FiducialMarker]


@dataclass(frozen=True)
class FiducialCamera:
    """A camera that detects visual fiducial markers."""

    name: str
    recognized_sizes_cm: frozenset[float]  # Sizes (cm) of AR markers detected by the camera

    def __str__(self) -> str:
        """Return a human-readable string representation of the fiducial-detecting camera."""
        return f"Camera({self.name}, sizes_cm: {', '.join(map(str, self.recognized_sizes_cm))})"

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


FiducialCameras = Iterable[FiducialCamera]


class FiducialSystem:
    """A system of known visual fiducial markers and fiducial-detecting cameras."""

    def __init__(self, markers: FiducialMarkers, cameras: FiducialCameras) -> None:
        """Initialize the system of fiducial markers using its markers and cameras.

        :param markers: Collection of visual fiducial markers
        :param cameras: Fiducial-detecting cameras in the system
        """
        self.markers = {marker.id: marker for marker in markers}
        self.cameras = {c.name: c for c in cameras}  # Map camera names to FiducialCamera instances

        self.camera_detects = {
            c.name: {m.id for m in markers if c.can_recognize(m)} for c in cameras
        }  # Map camera names to the IDs of markers they can detect

        self.object_names = {obj_name for m in markers for obj_name in m.relative_frames}

        # Map frame names to their parent markers
        self.parent_marker: dict[str, FiducialMarker] = {}
        for marker in markers:
            for child_frame in marker.relative_frames:
                if child_frame in self.parent_marker:
                    raise ValueError(f"Child frame '{child_frame}' has two parent frames.")
                self.parent_marker[child_frame] = marker

    def __str__(self) -> str:
        """Return a readable string representation of the fiducial system."""
        markers_str = "\n\t".join(map(str, self.markers.values()))
        cameras_str = "\n\t".join(map(str, self.cameras.values()))
        return f"Markers:\n\t{markers_str}\nCameras:\n\t{cameras_str}"

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

        return FiducialSystem(markers, cameras)
