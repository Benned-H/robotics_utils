"""Define classes to represent primitive 3D shapes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import trimesh

from robotics_utils.geometry import AxisAlignedBoundingBox, Point3D
from robotics_utils.io.pydantic_schemata import (
    BoxPrimitiveSchema,
    PrimitiveShapeSchema,
    SpherePrimitiveSchema,
)
from robotics_utils.spatial import Pose3D


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

    @classmethod
    def from_schema_with_pose(cls, schema: PrimitiveShapeSchema) -> tuple[PrimitiveShape, Pose3D]:
        """Construct a primitive shape and its object-relative local pose from schema data."""
        primitive = cls.from_schema(schema=schema)
        primitive_pose = Pose3D.from_sequence(schema.pose)
        return primitive, primitive_pose


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


def get_shape_center_pose_wrt_primitive(primitive: PrimitiveShape) -> Pose3D:
    """Get the pose of the shape geometry's center relative the primitive's local frame.

    i.e., where is the primitive's center w.r.t. its frame, which is on the shape's bottom?
    """
    if isinstance(primitive, Box):
        return Pose3D.from_xyz_rpy(z=primitive.z_m / 2.0)
    if isinstance(primitive, Sphere):
        return Pose3D.from_xyz_rpy(z=primitive.radius_m)
    if isinstance(primitive, Cylinder):
        return Pose3D.from_xyz_rpy(z=primitive.height_m / 2.0)
    raise ValueError(f"Unexpected primitive shape type: {primitive} (type {type(primitive)}).")


def primitive_to_mesh(primitive: PrimitiveShape) -> trimesh.Trimesh:
    """Convert a primitive shape into a mesh in a local frame with bottom at z = 0.

    :raises ValueError: If primitive type is not recognized
    """
    if isinstance(primitive, Box):
        mesh = trimesh.primitives.Box(
            extents=[primitive.x_m, primitive.y_m, primitive.z_m],
        ).to_mesh()
        mesh.apply_translation([0, 0, primitive.z_m / 2.0])
        return mesh

    if isinstance(primitive, Sphere):
        mesh = trimesh.primitives.Sphere(radius=primitive.radius_m).to_mesh()
        mesh.apply_translation([0, 0, primitive.radius_m])
        return mesh

    if isinstance(primitive, Cylinder):
        mesh = trimesh.primitives.Cylinder(
            radius=primitive.radius_m,
            height=primitive.height_m,
        ).to_mesh()
        mesh.apply_translation([0, 0, primitive.height_m / 2.0])
        return mesh

    raise ValueError(f"Unexpected primitive shape type: {primitive} (type {type(primitive)}).")
