"""Define classes to represent and process collision-geometry meshes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import trimesh

from robotics_utils.geometry import AxisAlignedBoundingBox, Point3D
from robotics_utils.io.pydantic_schemata import (
    MeshSchema,
    MeshTransformSchema,
    RotateTransformSchema,
    ScaleTransformSchema,
    TranslateTransformSchema,
)
from robotics_utils.spatial import EulerRPY


def load_mesh_from_schema(schema: MeshSchema, yaml_path: Path) -> trimesh.Trimesh:
    """Load a trimesh.Trimesh instance from validated data imported from a YAML file."""
    mesh_path = yaml_path.parent / schema.filepath
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    mesh = trimesh.load_mesh(mesh_path)
    for transform_schema in schema.transforms:
        transform = MeshTransform.from_schema(transform_schema)
        transform.apply(mesh)

    return mesh


def compute_aabb(mesh: trimesh.Trimesh) -> AxisAlignedBoundingBox:
    """Compute the axis-aligned bounding box (AABB) of the mesh."""
    min_bounds, max_bounds = mesh.bounds
    return AxisAlignedBoundingBox(
        min_xyz=Point3D.from_sequence(min_bounds),
        max_xyz=Point3D.from_sequence(max_bounds),
    )


class MeshTransform(ABC):
    """Abstract base class for transform operations on meshes."""

    @abstractmethod
    def apply(self, mesh: trimesh.Trimesh) -> None:
        """Apply the transform to the given mesh in-place."""

    @classmethod
    def from_schema(cls, schema: MeshTransformSchema) -> MeshTransform:
        """Construct a MeshTransform from the given validated data."""
        if isinstance(schema, TranslateTransformSchema):
            return Translate(Point3D.from_sequence(schema.translate))
        if isinstance(schema, RotateTransformSchema):
            return Rotate(rpy=EulerRPY.from_sequence(schema.rotate))
        if isinstance(schema, ScaleTransformSchema):
            return Scale(schema.scale)

        if schema == "center_mass":
            return CenterMass()
        if schema == "center_bounds":
            return CenterBounds()
        if schema == "bottom_at_zero_z":
            return BottomAtZeroZ()

        raise TypeError(f"Unexpected mesh transform schema: {schema}")


@dataclass(frozen=True)
class Translate(MeshTransform):
    """Translate a mesh by some (x,y,z) vector."""

    xyz: Point3D

    def apply(self, mesh: trimesh.Trimesh) -> None:
        """Translate the given mesh in-place."""
        mesh.apply_translation(self.xyz.to_array())


@dataclass(frozen=True)
class Rotate(MeshTransform):
    """Rotate a mesh based on Euler angles specified in radians."""

    rpy: EulerRPY  # 3D rotation represented using fixed-frame Euler angles

    def apply(self, mesh: trimesh.Trimesh) -> None:
        """Rotate the given mesh in-place."""
        matrix = self.rpy.to_homogeneous_matrix()
        mesh.apply_transform(matrix)


@dataclass(frozen=True)
class Scale(MeshTransform):
    """Scale a mesh either by a constant factor or non-uniform factors in (x,y,z)."""

    factor: float | tuple[float, float, float]

    def apply(self, mesh: trimesh.Trimesh) -> None:
        """Scale the given mesh in-place."""
        mesh.apply_scale(self.factor)


@dataclass(frozen=True)
class CenterMass(MeshTransform):
    """Center a mesh at the origin based on its center of mass (i.e., centroid)."""

    def apply(self, mesh: trimesh.Trimesh) -> None:
        """Center the mass of the given mesh in-place."""
        mesh.apply_translation(-mesh.centroid)


@dataclass(frozen=True)
class CenterBounds(MeshTransform):
    """Center a mesh at the origin based on its bounding box."""

    def apply(self, mesh: trimesh.Trimesh) -> None:
        """Center the bounding box of the given mesh in-place."""
        bounds_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
        mesh.apply_translation(-bounds_center)


@dataclass(frozen=True)
class BottomAtZeroZ(MeshTransform):
    """Place the bottom of a mesh at z = 0."""

    def apply(self, mesh: trimesh.Trimesh) -> None:
        """Bottom-normalize the given mesh in-place."""
        _, _, min_z = mesh.bounds[0]
        mesh.apply_translation([0, 0, -min_z])
