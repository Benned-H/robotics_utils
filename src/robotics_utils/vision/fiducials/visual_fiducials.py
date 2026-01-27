"""Define classes to represent visual fiducial markers (e.g., AprilTags)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from robotics_utils.io.pydantic_schemata import FiducialMarkerSchema, FiducialSystemSchema
from robotics_utils.spatial import Pose3D

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
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
    def from_schema(cls, marker_name: str, schema: FiducialMarkerSchema) -> FiducialMarker:
        """Construct a FiducialMarker instance from validated data imported from file."""
        relative_frames: dict[str, Pose3D] = {
            frame: Pose3D.from_schema(p_schema, default_frame=marker_name)
            for frame, p_schema in schema.relative_frames.items()
        }
        return FiducialMarker(schema.id, schema.size_cm, relative_frames)

    @staticmethod
    def id_to_frame_name(tag_id: int) -> str:
        """Construct the name of the reference frame for a tag with the given ID."""
        return f"marker_{tag_id}"

    @property
    def frame_name(self) -> str:
        """Retrieve the name of the reference frame defined by this visual fiducial."""
        return self.id_to_frame_name(self.id)


@dataclass
class FiducialSystem:
    """A system of known visual fiducial markers and fiducial-detecting cameras."""

    markers: dict[int, FiducialMarker]  # Map marker IDs to FiducialMarker instances
    camera_names: set[str]

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> FiducialSystem:
        """Load a system of visual fiducial markers from a YAML file."""
        schema = FiducialSystemSchema.validate_yaml(yaml_path)

        markers = [
            FiducialMarker.from_schema(m_name, m_schema)
            for m_name, m_schema in schema.markers.items()
        ]

        return FiducialSystem(markers={m.id: m for m in markers}, camera_names=schema.camera_names)
