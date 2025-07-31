"""Define classes to represent primitive 3D shapes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from robotics_utils.kinematics.collision_models.aabb import AxisAlignedBoundingBox
from robotics_utils.kinematics.point3d import Point3D


class PrimitiveShape(Protocol):
    """Protocol for primitive shapes."""

    @property
    def aabb(self) -> AxisAlignedBoundingBox:
        """Get the axis-aligned bounding box (AABB) of the primitive shape."""
        ...

    def to_dimensions(self) -> list[float]:
        """Convert the primitive shape into a list of its dimensions."""
        ...


@dataclass(frozen=True)
class Box(PrimitiveShape):
    """Box primitive shape with (x,y,z) dimensions (in meters)."""

    x_m: float
    y_m: float
    z_m: float

    @property
    def aabb(self) -> AxisAlignedBoundingBox:
        """Get the axis-aligned bounding box (AABB) of the box."""
        half_dims = np.array(self.to_dimensions()) / 2
        return AxisAlignedBoundingBox(
            min_xyz=Point3D.from_array(-half_dims),
            max_xyz=Point3D.from_array(half_dims),
        )

    def to_dimensions(self) -> list[float]:
        """Convert the box into a list of its (x,y,z) dimensions."""
        return [self.x_m, self.y_m, self.z_m]


@dataclass(frozen=True)
class Sphere(PrimitiveShape):
    """Sphere primitive shape with a radius (in meters)."""

    radius_m: float

    @property
    def aabb(self) -> AxisAlignedBoundingBox:
        """Get the axis-aligned bounding box (AABB) of the sphere."""
        return AxisAlignedBoundingBox(
            min_xyz=Point3D(-self.radius_m, -self.radius_m, -self.radius_m),
            max_xyz=Point3D(self.radius_m, self.radius_m, self.radius_m),
        )

    def to_dimensions(self) -> list[float]:
        """Convert the sphere into a list containing its radius."""
        return [self.radius_m]


@dataclass(frozen=True)
class Cylinder(PrimitiveShape):
    """Cylinder primitive shape with a height and radius (in meters)."""

    height_m: float
    radius_m: float

    @property
    def aabb(self) -> AxisAlignedBoundingBox:
        """Get the axis-aligned bounding box (AABB) of the cylinder."""
        half_height_m = self.height_m / 2.0
        return AxisAlignedBoundingBox(
            min_xyz=Point3D(-self.radius_m, -self.radius_m, -half_height_m),
            max_xyz=Point3D(self.radius_m, self.radius_m, half_height_m),
        )

    def to_dimensions(self) -> list[float]:
        """Convert the cylinder into a list of its dimensions."""
        return [self.height_m, self.radius_m]


def create_primitive_shape(shape_type: str, params: dict[str, float]) -> PrimitiveShape:
    """Create a primitive shape from its type and parameters."""
    if shape_type == "box":
        return Box(x_m=params["x"], y_m=params["y"], z_m=params["z"])

    if shape_type == "sphere":
        return Sphere(radius_m=params["radius"])

    if shape_type == "cylinder":
        return Cylinder(height_m=params["height"], radius_m=params["radius"])

    raise ValueError(f"Unknown primitive shape type: {shape_type}")
