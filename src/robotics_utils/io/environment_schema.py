"""Pydantic models for validating environment YAML configuration files.

These models provide early validation and clear error messages when loading
environment configurations, catching issues like missing required fields or
incorrect nesting at parse time rather than runtime.

Example usage:
    from robotics_utils.io.environment_schema import EnvironmentConfig

    yaml_data = load_yaml_file("environment.yaml")
    config = EnvironmentConfig.model_validate(yaml_data)  # Raises ValidationError on issues
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# =============================================================================
# Pose Configuration
# =============================================================================
class PoseDictConfig(BaseModel):
    """Pose specified as a dictionary with explicit frame."""

    model_config = ConfigDict(extra="forbid")

    xyz_rpy: tuple[float, float, float, float, float, float]
    frame: str


# Pose can be a 6-element list [x, y, z, roll, pitch, yaw] or a dict with xyz_rpy and frame
PoseConfig = list[float] | PoseDictConfig


def validate_pose_list(pose: list[float], field_name: str) -> None:
    """Validate that a pose list has exactly 6 elements."""
    if len(pose) != 6:
        msg = f"{field_name}: Pose list must have 6 elements [x, y, z, roll, pitch, yaw], got {len(pose)}"
        raise ValueError(msg)


# =============================================================================
# Primitive Shape Configuration
# =============================================================================


class BoxPrimitiveConfig(BaseModel):
    """Box primitive shape configuration."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["box"]
    x: float = Field(gt=0, description="X dimension in meters")
    y: float = Field(gt=0, description="Y dimension in meters")
    z: float = Field(gt=0, description="Z dimension in meters")


class SpherePrimitiveConfig(BaseModel):
    """Sphere primitive shape configuration."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["sphere"]
    radius: float = Field(gt=0, description="Radius in meters")


class CylinderPrimitiveConfig(BaseModel):
    """Cylinder primitive shape configuration."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["cylinder"]
    height: float = Field(gt=0, description="Height in meters")
    radius: float = Field(gt=0, description="Radius in meters")


PrimitiveConfig = Annotated[
    BoxPrimitiveConfig | SpherePrimitiveConfig | CylinderPrimitiveConfig,
    Field(discriminator="type"),
]


# =============================================================================
# Mesh Transform Configuration
# =============================================================================


class TranslateTransformConfig(BaseModel):
    """Translation transform configuration."""

    model_config = ConfigDict(extra="forbid")

    translate: tuple[float, float, float]


class RotateTransformConfig(BaseModel):
    """Rotation transform configuration (Euler angles in radians)."""

    model_config = ConfigDict(extra="forbid")

    rotate: tuple[float, float, float]


class ScaleTransformConfig(BaseModel):
    """Scale transform configuration (uniform or non-uniform)."""

    model_config = ConfigDict(extra="forbid")

    scale: float | tuple[float, float, float]


# Named transforms are simple strings; structured transforms are dicts
MeshTransformConfig = (
    Literal["center_mass", "center_bounds", "bottom_at_zero_z"]
    | TranslateTransformConfig
    | RotateTransformConfig
    | ScaleTransformConfig
)


# =============================================================================
# Mesh Configuration
# =============================================================================


class MeshConfig(BaseModel):
    """Mesh file configuration with optional transforms."""

    model_config = ConfigDict(extra="forbid")

    filepath: str = Field(description="Path to mesh file (relative to YAML file location)")
    transforms: list[MeshTransformConfig] = Field(default_factory=list)


# =============================================================================
# Collision Model Configuration
# =============================================================================


class CollisionModelConfig(BaseModel):
    """Collision model containing meshes and/or primitive shapes."""

    model_config = ConfigDict(extra="forbid")

    meshes: list[MeshConfig] = Field(default_factory=list)
    primitives: list[PrimitiveConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_not_empty(self) -> CollisionModelConfig:
        """Ensure collision model has at least one mesh or primitive."""
        if not self.meshes and not self.primitives:
            msg = "Collision model must have at least one mesh or primitive"
            raise ValueError(msg)
        return self


# =============================================================================
# Container Configuration
# =============================================================================


class ContainedObjectConfig(BaseModel):
    """Configuration for an object contained within a container."""

    model_config = ConfigDict(extra="forbid")

    pose_when_open: PoseConfig
    pose_when_closed: PoseConfig

    @model_validator(mode="after")
    def validate_poses(self) -> ContainedObjectConfig:
        """Validate pose list lengths if they are lists."""
        if isinstance(self.pose_when_open, list):
            validate_pose_list(self.pose_when_open, "pose_when_open")
        if isinstance(self.pose_when_closed, list):
            validate_pose_list(self.pose_when_closed, "pose_when_closed")
        return self


class ContainerConfig(BaseModel):
    """Configuration for a container object (e.g., dresser, cabinet)."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["open", "closed"]
    open_model: CollisionModelConfig
    closed_model: CollisionModelConfig
    contains: dict[str, ContainedObjectConfig] = Field(default_factory=dict)


# =============================================================================
# Object Configuration
# =============================================================================


class ObjectConfig(BaseModel):
    """Configuration for an object in the environment.

    Objects may have:
    - A pose (required unless contained in a container)
    - A collision_model (required to appear in the planning scene)
    - A container specification (if the object is a container like a cabinet)
    """

    model_config = ConfigDict(extra="forbid")

    pose: PoseConfig | None = None
    collision_model: CollisionModelConfig | None = None
    container: ContainerConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def check_misplaced_primitives(cls, data: Any) -> Any:
        """Catch common mistake of putting 'primitives' at the object level."""
        if isinstance(data, dict) and "primitives" in data and "collision_model" not in data:
            msg = (
                "Found 'primitives' at object level. "
                "Did you mean to put it under 'collision_model'?\n"
                "Expected:\n"
                "  collision_model:\n"
                "    primitives: [...]\n"
                "Got:\n"
                "  primitives: [...]"
            )
            raise ValueError(msg)
        return data

    @model_validator(mode="after")
    def validate_pose(self) -> ObjectConfig:
        """Validate pose list length if it's a list."""
        if isinstance(self.pose, list):
            validate_pose_list(self.pose, "pose")
        return self


# =============================================================================
# Robot Configuration
# =============================================================================


class RobotConfig(BaseModel):
    """Configuration for a robot in the environment."""

    model_config = ConfigDict(extra="forbid")

    base_pose: PoseConfig

    @model_validator(mode="after")
    def validate_pose(self) -> RobotConfig:
        """Validate pose list length if it's a list."""
        if isinstance(self.base_pose, list):
            validate_pose_list(self.base_pose, "base_pose")
        return self


# =============================================================================
# Environment Configuration (Root Model)
# =============================================================================


class EnvironmentConfig(BaseModel):
    """Root configuration model for environment YAML files.

    Example YAML structure:
        robots:
          spot:
            base_pose: [3, 0, 0.55, 0, 0, 3.14159]

        objects:
          eraser1:
            collision_model:
              primitives: [{ type: box, x: 0.05, y: 0.125, z: 0.03 }]

          board1:
            pose: [7, -0.35, -0.18, 0, 0, -1.5708]
            collision_model:
              primitives: [{ type: box, x: 0.03, y: 1.3, z: 0.9 }]

        default_frame: map
    """

    model_config = ConfigDict(extra="allow")  # Allow extra fields like known_landmarks

    robots: dict[str, RobotConfig]
    objects: dict[str, ObjectConfig]
    default_frame: str = "map"


# =============================================================================
# Validation Helper Function
# =============================================================================


def validate_environment_yaml(
    yaml_data: dict[str, Any],
    yaml_path: Path | None = None,
) -> EnvironmentConfig:
    """Validate environment YAML data and return a validated configuration.

    :param yaml_data: Dictionary of data loaded from a YAML file
    :param yaml_path: Optional path to the YAML file (for error messages)
    :return: Validated EnvironmentConfig instance
    :raises pydantic.ValidationError: If the YAML data is invalid
    """
    try:
        return EnvironmentConfig.model_validate(yaml_data)
    except Exception as e:
        if yaml_path:
            msg = f"Validation error in {yaml_path}: {e}"
            raise type(e)(msg) from e
        raise
