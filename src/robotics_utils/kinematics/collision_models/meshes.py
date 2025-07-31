"""Define classes to represent and process collision-geometry meshes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import trimesh

from robotics_utils.kinematics.collision_models.aabb import AxisAlignedBoundingBox
from robotics_utils.kinematics.point3d import Point3D
from robotics_utils.kinematics.rotations import EulerRPY


class Mesh(trimesh.Trimesh):
    """A mesh used as geometry for collision-checking."""

    def __init__(self, source_path: Path, *args: tuple, **kwargs: dict[str, Any]) -> None:
        """Initialize the mesh with the filepath it was loaded from."""
        super().__init__(*args, **kwargs)
        self.source_path = source_path

    @classmethod
    def from_file(cls, mesh_path: Path) -> Mesh:
        """Load a mesh from the given file.

        :param mesh_path: Filepath containing mesh data
        :return: Loaded Mesh instance
        """
        if not mesh_path.exists():
            raise FileNotFoundError(f"Cannot load mesh from nonexistent file: {mesh_path}")

        return Mesh(mesh_path, trimesh.load_mesh(mesh_path))  # TODO: Does this work?

    @classmethod
    def from_yaml_data(cls, yaml_data: dict[str, Any]) -> Mesh:
        """Construct a Mesh instance from data imported from YAML."""
        if "filepath" not in yaml_data:
            raise KeyError(f"No mesh filepath was provided in the YAML data: {yaml_data}")

        source_path = Path(yaml_data["filepath"])
        if not source_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {source_path}")

        mesh = Mesh.from_file(source_path)

        transforms = parse_mesh_transforms(yaml_data.get("transforms", []))
        for transform in transforms:
            transform.apply(mesh)

        return mesh

    @property
    def aabb(self) -> AxisAlignedBoundingBox:
        """Get the axis-aligned bounding box (AABB) of the mesh."""
        min_bounds, max_bounds = self.bounds
        return AxisAlignedBoundingBox(
            min_xyz=Point3D.from_sequence(min_bounds),
            max_xyz=Point3D.from_sequence(max_bounds),
        )


class MeshTransform(Protocol):
    """Protocol for transform operations on meshes."""

    def apply(self, mesh: Mesh) -> None:
        """Apply the transform to the given mesh in-place."""


@dataclass(frozen=True)
class Translate(MeshTransform):
    """Translate a mesh by some (x,y,z) vector."""

    xyz: Point3D

    def apply(self, mesh: Mesh) -> None:
        """Translate the given mesh in-place."""
        mesh.apply_translation(self.xyz.to_array())

    @classmethod
    def from_list(cls, values: list[float]) -> Translate:
        """Construct a Translate instance from a list of values."""
        return Translate(xyz=Point3D.from_sequence(values))


@dataclass(frozen=True)
class Rotate(MeshTransform):
    """Rotate a mesh based on Euler angles specified in radians."""

    rpy: EulerRPY  # 3D rotation represented using fixed-frame Euler angles

    def apply(self, mesh: Mesh) -> None:
        """Rotate the given mesh in-place."""
        matrix = self.rpy.to_homogeneous_matrix()
        mesh.apply_transform(matrix)

    @classmethod
    def from_list(cls, values: list[float]) -> Rotate:
        """Construct a Rotate instance from a list of values."""
        return Rotate(rpy=EulerRPY.from_list(values))


@dataclass(frozen=True)
class Scale(MeshTransform):
    """Scale a mesh either by a constant factor or non-uniform factors in (x,y,z)."""

    factor: float | tuple[float, float, float]

    def apply(self, mesh: Mesh) -> None:
        """Scale the given mesh in-place."""
        mesh.apply_scale(self.factor)

    @classmethod
    def from_value(cls, value: float | list[float]) -> Scale:
        """Construct a Scale instance from a single value or list of values."""
        if isinstance(value, list):
            if len(value) != 3:
                raise ValueError(f"Non-uniform scale expects 3 values, got {len(value)}")
            return Scale(factor=(value[0], value[1], value[2]))

        return Scale(factor=value)


@dataclass(frozen=True)
class CenterMass(MeshTransform):
    """Center a mesh at the origin based on its center of mass (i.e., centroid)."""

    def apply(self, mesh: Mesh) -> None:
        """Center the mass of the given mesh in-place."""
        mesh.apply_translation(-mesh.centroid)


@dataclass(frozen=True)
class CenterBounds(MeshTransform):
    """Center a mesh at the origin based on its bounding box."""

    def apply(self, mesh: Mesh) -> None:
        """Center the bounding box of the given mesh in-place."""
        bounds_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
        mesh.apply_translation(-bounds_center)


@dataclass(frozen=True)
class BottomAtZeroZ(MeshTransform):
    """Place the bottom of a mesh at z = 0."""

    def apply(self, mesh: Mesh) -> None:
        """Bottom-normalize the given mesh in-place."""
        _, _, min_z = mesh.bounds[0]
        mesh.apply_translation([0, 0, -min_z])


def parse_mesh_transforms(transform_specs: list[dict[str, Any] | str]) -> list[MeshTransform]:
    """Parse mesh transform specifications from YAML data."""
    transforms: list[MeshTransform] = []

    for spec in transform_specs:
        if isinstance(spec, str):  # Named transforms
            if spec == "center_mass":
                transforms.append(CenterMass())
            elif spec == "center_bounds":
                transforms.append(CenterBounds())
            elif spec == "bottom_at_zero_z":
                transforms.append(BottomAtZeroZ())
            else:
                raise ValueError(f"Unknown named mesh transform: {spec}")
        elif isinstance(spec, dict):  # Structured transforms
            if "translate" in spec:
                transforms.append(Translate.from_list(spec["translate"]))
            elif "rotate" in spec:
                transforms.append(Rotate.from_list(spec["rotate"]))
            elif "scale" in spec:
                transforms.append(Scale.from_value(spec["scale"]))
            else:
                raise ValueError(f"Unknown mesh transform type: {spec}")
        else:
            raise TypeError(f"Invalid transform specification type: {spec}")

    return transforms
