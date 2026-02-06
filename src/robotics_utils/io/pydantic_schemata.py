"""Define Pydantic models for validating environment YAML configuration files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Literal, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError, model_validator
from typing_extensions import Annotated

from robotics_utils.io.yaml_utils import load_yaml_data

# =============================================================================
# Pose Schemata
# =============================================================================

XY_YAW = Tuple[float, float, float]
"""A three-tuple of floats representing an SE(2) pose (x, y, yaw) (radians)."""

XYZ_RPY = Tuple[float, float, float, float, float, float]
"""A six-tuple of floats representing an SE(3) pose."""


class Pose2DDictSchema(BaseModel):
    """Schema for specifying a Pose2D as a dictionary."""

    xy_yaw: XY_YAW
    frame: str

    model_config = ConfigDict(extra="forbid")


Pose2DSchema = Union[XY_YAW, Pose2DDictSchema]
"""A Pose2D can be specified using a 3-tuple or a dictionary with `xy_yaw` and `frame`."""


class Pose3DDictSchema(BaseModel):
    """Schema for specifying a Pose3D as a dictionary."""

    xyz_rpy: XYZ_RPY
    frame: str

    model_config = ConfigDict(extra="forbid")


Pose3DSchema = Union[XYZ_RPY, Pose3DDictSchema]
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

    translate: Tuple[float, float, float]

    model_config = ConfigDict(extra="forbid")


class RotateTransformSchema(BaseModel):
    """Schema for a rotation transform specified using Euler angles in radians."""

    rotate: Tuple[float, float, float]

    model_config = ConfigDict(extra="forbid")


class ScaleTransformSchema(BaseModel):
    """Schema for a scaling transform (uniform or non-uniform across axes)."""

    scale: Union[float, Tuple[float, float, float]]

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
    transforms: List[MeshTransformSchema] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Collision Model Schema
# =============================================================================


class CollisionModelSchema(BaseModel):
    """Schema for a collision model containing meshes and/or primitive shapes."""

    meshes: List[MeshSchema] = Field(default_factory=list)
    primitives: List[PrimitiveShapeSchema] = Field(default_factory=list)

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

    size_cm: float = Field(gt=0, description="Size of the marker's black square (centimeters)")
    relative_frames: Dict[str, Pose3DSchema] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


MARKER_NAME_PATTERN = re.compile(r"^marker_(\d+)$")
"""Pattern for marker names: `'marker_'` followed by an integer ID (e.g., `'marker_19'`)."""


def parse_marker_id(marker_name: str) -> int:
    """Extract the integer ID from a validated marker name.

    :param marker_name: A marker name in the format 'marker_{id}'
    :return: The integer ID extracted from the marker name
    :raises ValueError: If the marker name doesn't match the expected format
    """
    match = MARKER_NAME_PATTERN.match(marker_name)
    if not match:
        raise ValueError(
            f"Invalid marker name '{marker_name}'. "
            "Expected format 'marker_{{id}}' (e.g., 'marker_19').",
        )
    return int(match.group(1))


class FiducialSystemSchema(BaseModel):
    """Schema for a system of fiducial markers."""

    markers: Dict[str, FiducialMarkerSchema]
    cameras: Set[str]

    @model_validator(mode="after")
    def validate_marker_names(self) -> FiducialSystemSchema:
        """Validate that all marker names follow the 'marker_{id}' format."""
        for name in self.markers:
            parse_marker_id(marker_name=name)
        return self

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
            raise RuntimeError(f"Validation error in {yaml_path}: {v_err}") from v_err


# =============================================================================
# Object Visual State Schemata
# =============================================================================


class ViewpointSchema(BaseModel):
    """Schema for an objective-relative viewpoint."""

    camera_id: str
    pose: Pose3DSchema
    """Pose data for the viewpoint (uses the object frame if unspecified)."""

    model_config = ConfigDict(extra="forbid")


NamedViewpointsSchema = Dict[str, ViewpointSchema]
"""A map from viewpoint names to the corresponding viewpoint schemas."""


class ViewpointTemplatesSchema(RootModel[Dict[str, NamedViewpointsSchema]]):
    """Schema mapping object types to their templates of viewpoints."""


class ImageObservationSchema(BaseModel):
    """Schema for an image observation from a known camera pose."""

    image_path: Path
    pose: Pose3DSchema

    model_config = ConfigDict(extra="forbid")


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

    # Allow empty collision models when they're explicitly stated to be None
    open_model: Union[CollisionModelSchema, None]
    closed_model: Union[CollisionModelSchema, None]
    contains: Dict[str, ContainedObjectSchema] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Object Schema
# =============================================================================


class ObjectSchema(BaseModel):
    """Schema for an object in the environment."""

    pose: Union[Pose3DSchema, None] = None
    collision_model: Union[CollisionModelSchema, None] = None
    container: Union[ContainerSchema, None] = None

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

    robots: Dict[str, RobotSchema]
    objects: Dict[str, ObjectSchema]
    default_frame: str

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
            raise RuntimeError(f"Validation error in {yaml_path}: {v_err}") from v_err


# =============================================================================
# Discrete Grid Schemata
# =============================================================================


class DiscreteGrid2DSchema(BaseModel):
    """Schema for a discrete 2D grid on the x-y plane."""

    origin: Pose2DSchema
    resolution_m: float = Field(gt=0, description="Cell resolution (meters)")
    width_cells: int = Field(gt=0, description="Grid width (# cells)")
    height_cells: int = Field(gt=0, description="Grid height (# cells)")
    frame_name: str = Field(default="grid", description="Name of the grid's local frame")

    model_config = ConfigDict(extra="forbid")


class OccupancyGrid2DSchema(BaseModel):
    """Schema for a 2D occupancy grid with log-odds values stored as a 16-bit image.

    The log-odds values are linearly mapped to the 16-bit range [0, 65535] using:
        pixel_value = (log_odds - log_odds_min) / (log_odds_max - log_odds_min) * 65535

    To recover log-odds from the image:
        log_odds = pixel_value / 65535 * (log_odds_max - log_odds_min) + log_odds_min
    """

    grid: DiscreteGrid2DSchema
    image_path: Path = Field(description="Path to 16-bit grayscale PNG storing log-odds")
    log_odds_min: float = Field(description="Minimum log-odds value (maps to pixel value 0)")
    log_odds_max: float = Field(description="Maximum log-odds value (maps to pixel value 65535)")
    min_obstacle_depth_m: float = Field(
        default=0.1,
        description="Minimum depth (meters) assumed for obstacles when ray-tracing",
    )

    model_config = ConfigDict(extra="forbid")
