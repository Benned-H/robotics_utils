"""Define classes to represent object-relative visual states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Generic

from robotics_utils.io.pydantic_schemata import (
    ImageObservationSchema,
    ViewpointSchema,
    ViewpointTemplatesSchema,
)
from robotics_utils.spatial import Pose3D
from robotics_utils.vision import Image, ImageT

if TYPE_CHECKING:
    from robotics_utils.vision.cameras import Camera


@dataclass(frozen=True)
class Viewpoint(Generic[ImageT]):
    """A viewpoint from which a camera captures visual observations (typically object-relative).

    Because the same object-relative viewpoints are used for all objects of a given type,
    the `pose_c` frame `"OBJECT"` is used to denote that a viewpoint is object-relative.
    """

    camera: Camera[ImageT]
    """Interface for the camera used at the viewpoint."""

    pose_c: Pose3D
    """Camera pose, typically in the object's frame (denoted by reference frame `"OBJECT"`)."""

    @classmethod
    def from_schema(cls, schema: ViewpointSchema, cameras: dict[str, Camera]) -> Viewpoint:
        """Construct a viewpoint from validated data loaded from file.

        :param schema: Validated data representing a viewpoint
        :param cameras: Map from camera names to camera objects
        :return: Constructed Viewpoint instance
        """
        camera = cameras[schema.camera_id]
        pose = Pose3D.from_schema(schema.pose, default_frame="OBJECT")
        return Viewpoint(camera, pose)

    def to_schema(self, default_frame: str) -> ViewpointSchema:
        """Convert the viewpoint into an equivalent Pydantic schema."""
        pose_schema = self.pose_c.to_schema(default_frame=default_frame)
        return ViewpointSchema(camera_id=self.camera.name, pose=pose_schema)


NamedViewpoints = Dict[str, Viewpoint]
"""A map from viewpoint names to the corresponding viewpoints."""


class ViewpointTemplates(Dict[str, NamedViewpoints]):
    """A configuration specifying viewpoint templates for the object types in a domain."""

    @classmethod
    def from_schema(cls, schema: ViewpointTemplatesSchema, cameras: dict[str, Camera]) -> cls:
        """Construct viewpoint templates from data loaded from file."""
        templates = ViewpointTemplates()

        for obj_type, vp_schemas in schema.root.items():
            viewpoints: NamedViewpoints = {}
            for vp_name, vp_schema in vp_schemas.items():
                viewpoints[vp_name] = Viewpoint.from_schema(vp_schema, cameras)
            templates[obj_type] = viewpoints

        return templates


@dataclass(frozen=True)
class ImageObservation(Generic[ImageT]):
    """An image observation of an object from a known camera pose."""

    image: ImageT

    pose_o_c: Pose3D
    """Camera pose in the object's frame when the image was captured."""

    @classmethod
    def from_schema(
        cls,
        schema: ImageObservationSchema,
        image_type: type[Image],
        default_frame: str,
    ) -> ImageObservation:
        """Construct an image observation from validated data loaded from file.

        :param schema: Validated data representing an image observation
        :param image_type: Python type of the image used in the observation (e.g., `RGBImage`)
        :param default_frame: Default frame used for the observation's pose if unspecified
        :return: Constructed ImageObservation instance
        """
        image = image_type.from_file(schema.image_path)
        pose = Pose3D.from_schema(schema.pose, default_frame=default_frame)
        return ImageObservation(image, pose)

    def to_schema(self) -> ImageObservationSchema:
        """Convert the image observation into a schema format."""
        if self.image.filepath is None:
            raise ValueError("Cannot convert to schema due to unknown (None) image filepath.")

        pose_schema = self.pose_o_c.to_schema()
        return ImageObservationSchema(image_path=self.image.filepath, pose=pose_schema)


class ObjectVisualState:
    """The visual state of an object in the environment."""

    def __init__(self, viewpoints: dict[str, Viewpoint]) -> None:
        """Initialize the object-centric visual state according to the given viewpoints.

        :param viewpoints: Specifies named camera viewpoints capturing the object's visual state
        """
        self.viewpoints = viewpoints

        self.observations: dict[str, ImageObservation] = {}
        """A map from viewpoint names to their latest image observations."""

    @property
    def unknown_viewpoints(self) -> set[Viewpoint]:
        """Retrieve the currently unknown object viewpoints in the visual state."""
        return {self.viewpoints[v] for v in self.viewpoints if v not in self.observations}

    # TODO: These need implementation using schemas

    # @classmethod
    # def from_yaml_data(
    #     cls,
    #     yaml_data: dict[str, Any],
    #     viewpoints: Viewpoints,
    #     default_frame: str,
    # ) -> ObjectVisualState:
    #     """Construct an object visual state from data loaded from YAML.

    #     :param yaml_data: Data imported from a YAML file
    #     :param schema: Observation schema defining viewpoints in the object visual state
    #     :param default_frame: Default frame used for visual observations' poses if unspecified
    #     :return: Constructed ObjectVisualState instance
    #     """
    #     visual_state = ObjectVisualState(schema=schema)

    #     for v_name, viewpoint in schema.items():
    #         obs_data = yaml_data.get(v_name)  # Observation data at the named viewpoint
    #         if obs_data is not None:
    #             visual_state.observations[v_name] = ImageObservation.from_yaml_data(
    #                 yaml_data=obs_data,
    #                 image_type=viewpoint.camera.image_type,
    #                 default_frame=default_frame,
    #             )

    #     return visual_state

    # def to_yaml_data(self, default_frame: str) -> dict[str, Any]:
    #     """Convert the object visual state into a form suitable for export to YAML.

    #     :param default_frame: Default frame used for exported observation poses
    #     :return: Dictionary of data representing the object visual state
    #     """
    #     observed = {v_name: obs for v_name, obs in self.observations.items() if obs is not None}
    #     return {v: obs.to_yaml_data(default_frame) for v, obs in observed.items()}
