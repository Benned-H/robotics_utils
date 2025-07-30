"""Define classes representing primitive shapes used for collision-checking."""

from __future__ import annotations

from dataclasses import astuple, dataclass
from typing import Protocol

import numpy as np
import trimesh

from robotics_utils.kinematics.collision_models import AxisAlignedBoundingBox
from robotics_utils.kinematics.point3d import Point3D


class PrimitiveShape(Protocol):
    """Protocol for primitive shapes."""

    @property
    def aabb(self) -> AxisAlignedBoundingBox:
        """Get the axis-aligned bounding box (AABB) of the primitive shape."""

    def to_mesh(self) -> trimesh.Trimesh:
        """Convert the primitive shape into a mesh."""


@dataclass(frozen=True)
class Box(PrimitiveShape):
    """Box primitive shape with (x,y,z) dimensions (in meters)."""

    x_m: float
    y_m: float
    z_m: float

    @property
    def aabb(self) -> AxisAlignedBoundingBox:
        """Get the axis-aligned bounding box (AABB) of the box."""
        half_dims = np.array(astuple(self)) / 2
        return AxisAlignedBoundingBox(
            min_xyz=Point3D.from_array(-half_dims),
            max_xyz=Point3D.from_array(half_dims),
        )

    def to_mesh(self) -> trimesh.Trimesh:
        """Convert the box into a mesh."""
        return trimesh.creation.box(extents=astuple(self))


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

    def to_mesh(self) -> trimesh.Trimesh:
        """Convert the sphere into a mesh."""
        return trimesh.creation.icosphere(radius=self.radius_m, subdivisions=3)


dataclass(frozen=True)


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

    def to_mesh(self) -> trimesh.Trimesh:
        """Convert the cylinder into a mesh."""
        return trimesh.creation.cylinder(radius=self.radius_m, height=self.height_m, sections=32)


def create_primitive_shape(shape_type: str, params: dict[str, float]) -> PrimitiveShape:
    """Create a primitive shape from its type and parameters."""
    if shape_type == "box":
        return Box(x_m=params["x"], y_m=params["y"], z_m=params["z"])

    if shape_type == "sphere":
        return Sphere(radius_m=params["radius"])

    if shape_type == "cylinder":
        return Cylinder(height_m=params["height"], radius_m=params["radius"])

    raise ValueError(f"Unknown primitive shape type: {shape_type}")
