"""Define classes to represent object-relative visual states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

from robotics_utils.spatial import Pose3D
from robotics_utils.vision import Image, ImageT

if TYPE_CHECKING:
    from robotics_utils.vision.cameras import Camera


@dataclass(frozen=True)
class ObjectViewpoint(Generic[ImageT]):
    """An object-relative viewpoint from which a camera captures visual observations."""

    camera: Camera[ImageT]
    """Interface for the camera used at the viewpoint."""

    pose_o_c: Pose3D
    """Camera pose in the object's frame."""

    @classmethod
    def from_yaml_data(
        cls,
        yaml_data: dict[str, Any],
        default_frame: str,
        cameras: dict[str, Camera],
    ) -> ObjectViewpoint:
        """Construct an object viewpoint from data imported from YAML.

        :param yaml_data: Data imported from a YAML file
        :param default_frame: Default frame used for the viewpoint's pose
        :param cameras: Map from camera names to camera objects
        :return: Constructed ObjectViewpoint instance
        """
        camera = cameras[yaml_data["camera_id"]]
        pose = Pose3D.from_yaml_data(yaml_data["pose"], default_frame=default_frame)
        return ObjectViewpoint(camera, pose)

    def to_yaml_data(self, default_frame: str) -> dict[str, Any]:
        """Convert the object viewpoint into a form suitable for export to YAML.

        :param default_frame: Default frame used for poses in the relevant YAML file
        :return: Dictionary of data representing the object viewpoint
        """
        pose_data = self.pose_o_c.to_yaml_data(default_frame=default_frame)
        return {"camera_id": self.camera.name, "pose": pose_data}


ObservationSchema = dict[str, ObjectViewpoint]
"""A schema of named camera viewpoints capturing the visual state of objects of some type."""


@dataclass(frozen=True)
class ImageObservation(Generic[ImageT]):
    """An image observation of an object from a known camera pose."""

    image: ImageT

    pose_o_c: Pose3D
    """Camera pose in the object's frame when the image was captured."""

    @classmethod
    def from_yaml_data(
        cls,
        yaml_data: dict[str, Any],
        image_type: type[Image],
        default_frame: str,
    ) -> ImageObservation:
        """Construct an image observation from data imported from YAML.

        :param yaml_data: Data imported from a YAML file
        :param image_type: Python type of the image used in the observation (e.g., `RGBImage`)
        :param default_frame: Default frame used for the observation's pose if unspecified
        :return: Constructed ImageObservation instance
        """
        image = image_type.from_file(yaml_data["image_path"])
        pose = Pose3D.from_yaml_data(yaml_data["pose"], default_frame=default_frame)
        return ImageObservation(image, pose)

    def to_yaml_data(self, default_frame: str) -> dict[str, Any]:
        """Convert the image observation into a form suitable for export to YAML."""
        if self.image.filepath is None:
            raise ValueError("Cannot export an image without a filepath to YAML.")

        pose_data = self.pose_o_c.to_yaml_data(default_frame=default_frame)
        return {"image_path": self.image.filepath.as_posix(), "pose": pose_data}


class ObjectVisualState:
    """The visual state of an object in the environment."""

    def __init__(self, schema: ObservationSchema) -> None:
        """Initialize the object-centric visual state according to the given observation schema.

        :param schema: Specifies named camera viewpoints capturing the object's visual state
        """
        self.schema = schema
        self.observations: dict[str, ImageObservation | None] = dict.fromkeys(schema)
        """A map from viewpoint names to their latest image observations (or None if unknown)."""

    @property
    def unknown_viewpoints(self) -> set[ObjectViewpoint]:
        """Retrieve the currently unknown object viewpoints in the visual state."""
        return {self.schema[v_name] for v_name, obs in self.observations.items() if obs is None}

    @classmethod
    def from_yaml_data(
        cls,
        yaml_data: dict[str, Any],
        schema: ObservationSchema,
        default_frame: str,
    ) -> ObjectVisualState:
        """Construct an object visual state from data loaded from YAML.

        :param yaml_data: Data imported from a YAML file
        :param schema: Observation schema defining viewpoints in the object visual state
        :param default_frame: Default frame used for the observation's pose if unspecified
        :return: Constructed ObjectVisualState instance
        """
        visual_state = ObjectVisualState(schema=schema)

        for v_name, viewpoint in schema.items():
            obs_data = yaml_data.get(v_name)  # Observation data at the named viewpoint
            if obs_data is not None:
                visual_state.observations[v_name] = ImageObservation.from_yaml_data(
                    yaml_data=obs_data,
                    image_type=viewpoint.camera.image_type,
                    default_frame=default_frame,
                )

        return visual_state

    def to_yaml_data(self, default_frame: str) -> dict[str, Any]:
        """Convert the object visual state into a form suitable for export to YAML.

        :param default_frame: Default frame used for poses in the relevant YAML file
        :return: Dictionary of data representing the object visual state
        """
        observed = {v_name: obs for v_name, obs in self.observations.items() if obs is not None}
        return {v: obs.to_yaml_data(default_frame) for v, obs in observed.items()}
