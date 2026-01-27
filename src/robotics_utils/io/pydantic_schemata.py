"""Define Pydantic models for validating environment YAML configuration files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from robotics_utils.io.yaml_utils import load_yaml_data
from robotics_utils.spatial.frames import DEFAULT_FRAME

if TYPE_CHECKING:
    from pathlib import Path

# =============================================================================
# Pose Schemata
# =============================================================================


class Pose3DDictSchema(BaseModel):
    """Schema for specifying a Pose3D as a dictionary."""

    xyz_rpy: tuple[float, float, float, float, float, float]
    frame: str

    model_config = ConfigDict(extra="forbid")


Pose3DSchema = tuple[float, float, float, float, float, float] | Pose3DDictSchema
"""A Pose3D can be specified using a 6-tuple or a dictionary with `xyz_rpy` and `frame`."""


# =============================================================================
# Primitive Shape Schemata
# =============================================================================


class BoxPrimitiveSchema(BaseModel):
    """Schema for a 3D box primitive."""

    type: Literal["box"]
    x: float = Field(gt=0, description="X dimension size (meters)")
    y: float = Field(gt=0, description="Y dimension size (meters)")
    z: float = Field(gt=0, description="Z dimension size (meters)")

    model_config = ConfigDict(extra="forbid")


class SpherePrimitiveSchema(BaseModel):
    """Schema for a sphere primitive shape."""

    type: Literal["sphere"]
    radius: float = Field(gt=0, description="Radius (meters)")

    model_config = ConfigDict(extra="forbid")


class CylinderPrimitiveSchema(BaseModel):
    """Schema for a cylinder primitive shape."""

    type: Literal["cylinder"]
    height: float = Field(gt=0, description="Height (meters)")
    radius: float = Field(gt=0, description="Radius (meters)")

    model_config = ConfigDict(extra="forbid")


PrimitiveShapeSchema = Annotated[
    Union[BoxPrimitiveSchema, SpherePrimitiveSchema, CylinderPrimitiveSchema],
    Field(discriminator="type"),
]

# =============================================================================
# Mesh Transform Schemata
# =============================================================================


class TranslateTransformSchema(BaseModel):
    """Schema for a translation transform."""

    translate: tuple[float, float, float]

    model_config = ConfigDict(extra="forbid")


class RotateTransformSchema(BaseModel):
    """Schema for a rotation transform specified using Euler angles in radians."""

    rotate: tuple[float, float, float]

    model_config = ConfigDict(extra="forbid")


class ScaleTransformSchema(BaseModel):
    """Schema for a scaling transform (uniform or non-uniform across axes)."""

    scale: float | tuple[float, float, float]

    model_config = ConfigDict(extra="forbid")


MeshTransformSchema = Union[
    TranslateTransformSchema,
    RotateTransformSchema,
    ScaleTransformSchema,
    Literal["center_mass", "center_bounds", "bottom_at_zero_z"],
]


# =============================================================================
# Mesh Schema
# =============================================================================


class MeshSchema(BaseModel):
    """Schema for a mesh loaded from file with optional transforms."""

    filepath: str = Field(description="Path to mesh file (relative to YAML file location)")
    transforms: list[MeshTransformSchema] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Collision Model Schema
# =============================================================================


class CollisionModelSchema(BaseModel):
    """Schema for a collision model containing meshes and/or primitive shapes."""

    meshes: list[MeshSchema] = Field(default_factory=list)
    primitives: list[PrimitiveShapeSchema] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def check_not_empty(self) -> CollisionModelSchema:
        """Validate that the collision model has at least one mesh or primitive."""
        if not self.meshes and not self.primitives:
            raise ValueError("Collision model must have at least one mesh or primitive.")
        return self


# =============================================================================
# Visual Fiducial Schemata
# =============================================================================


class FiducialMarkerSchema(BaseModel):
    """Schema for a fiducial marker."""

    id: int
    size_cm: float = Field(gt=0, description="Size of the marker's black square (centimeters)")
    relative_frames: dict[str, Pose3DDictSchema] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class FiducialSystemSchema(BaseModel):
    """Schema for a system of fiducial markers."""

    markers: dict[str, FiducialMarkerSchema]
    camera_names: set[str]

    @classmethod
    def validate_yaml(cls, yaml_path: Path) -> FiducialSystemSchema:
        """Validate an marker config YAML file and return the resulting schema.

        :param yaml_path: Path to a YAML file to be validated by the schema
        :return: Validated FiducialSystemSchema instance
        """
        yaml_data = load_yaml_data(yaml_path)

        try:
            return FiducialSystemSchema.model_validate(yaml_data)
        except ValidationError as v_err:
            raise ValidationError(f"Validation error in {yaml_path}: {v_err}") from v_err


# =============================================================================
# Object Visual State Schemata
# =============================================================================


class ViewpointSchema(BaseModel):
    """Schema for an objective-relative viewpoint."""

    camera_id: str
    pose: Pose3DSchema

    model_config = ConfigDict(extra="forbid")


class ImageObservationSchema(BaseModel):
    """Schema for an image observation from a known camera pose."""

    image_path: Path
    pose: Pose3DSchema

    model_config = ConfigDict(extra="forbid")


class ObjectVisualStateSchema(BaseModel):
    """Schema for the visual state of an object."""

    viewpoints: dict[str, ViewpointSchema] = Field(default_factory=dict)
    observations: dict[str, ImageObservationSchema] = Field(default_factory=dict)


# =============================================================================
# Container Schemata
# =============================================================================


class ContainedObjectSchema(BaseModel):
    """Schema for an object contained within a container."""

    pose_when_open: Pose3DSchema
    pose_when_closed: Pose3DSchema

    model_config = ConfigDict(extra="forbid")


class ContainerSchema(BaseModel):
    """Schema for a container object (e.g., cabinet)."""

    status: Literal["open", "closed"]
    open_model: CollisionModelSchema
    closed_model: CollisionModelSchema
    contains: dict[str, ContainedObjectSchema] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Object Schema
# =============================================================================


class ObjectSchema(BaseModel):
    """Schema for an object in the environment."""

    pose: Pose3DSchema | None = None
    collision_model: CollisionModelSchema | None = None
    container: ContainerSchema | None = None

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Robot Schema
# =============================================================================


class RobotSchema(BaseModel):
    """Schema for a robot in the environment."""

    base_pose: Pose3DSchema

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Object-Centric State Schema
# =============================================================================
class ObjectCentricStateSchema(BaseModel):
    """Schema for an object-centric environment state."""

    robots: dict[str, RobotSchema]
    objects: dict[str, ObjectSchema]
    default_frame: str = DEFAULT_FRAME

    model_config = ConfigDict(extra="allow")  # Allow other fields such as waypoints

    @classmethod
    def validate_yaml(cls, yaml_path: Path) -> ObjectCentricStateSchema:
        """Validate an environment YAML file and return the resulting schema.

        :param yaml_path: Path to a YAML file to be validated by the schema
        :return: Validated ObjectCentricStateSchema instance
        """
        yaml_data = load_yaml_data(yaml_path)

        try:
            return ObjectCentricStateSchema.model_validate(yaml_data)
        except ValidationError as v_err:
            raise ValidationError(f"Validation error in {yaml_path}: {v_err}") from v_err
