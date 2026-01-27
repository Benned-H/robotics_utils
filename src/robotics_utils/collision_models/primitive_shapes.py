"""Define classes to represent primitive 3D shapes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from robotics_utils.geometry import AxisAlignedBoundingBox, Point3D
from robotics_utils.io.pydantic_schemata import (
    BoxPrimitiveSchema,
    PrimitiveShapeSchema,
    SpherePrimitiveSchema,
)


class PrimitiveShape(ABC):
    """Protocol for primitive shapes."""

    @property
    @abstractmethod
    def aabb(self) -> AxisAlignedBoundingBox:
        """Get the axis-aligned bounding box (AABB) of the primitive shape."""

    @abstractmethod
    def to_dimensions(self) -> list[float]:
        """Convert the primitive shape into a list of its dimensions."""

    @classmethod
    def from_schema(cls, schema: PrimitiveShapeSchema) -> PrimitiveShape:
        """Construct a primitive shape from the given validated data."""
        if isinstance(schema, BoxPrimitiveSchema):
            return Box(x_m=schema.x, y_m=schema.y, z_m=schema.z)
        if isinstance(schema, SpherePrimitiveSchema):
            return Sphere(radius_m=schema.radius)

        return Cylinder(height_m=schema.height, radius_m=schema.radius)


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
