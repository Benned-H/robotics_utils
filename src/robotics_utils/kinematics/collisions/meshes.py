"""Define classes for representing and manipulating collision meshes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import trimesh

from robotics_utils.kinematics.collision_models import AxisAlignedBoundingBox
from robotics_utils.kinematics.point3d import Point3D
from robotics_utils.kinematics.poses import Pose3D
from robotics_utils.kinematics.rotations import EulerRPY


class MeshSimplifier(Protocol):
    """Protocol for mesh simplification strategies."""

    def simplify(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Simplify the given mesh."""
        ...


@dataclass(frozen=True)
class RatioSimplifier(MeshSimplifier):
    """Simplify a mesh to a ratio of its original face count."""

    ratio: float = 0.2  # Proportion of mesh faces kept in the simplified mesh
    min_faces: int = 1000  # Minimum number of faces in the simplified mesh

    def simplify(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Simplify a mesh by reducing its face count to a target ratio of its original count."""
        target_count = max(int(self.ratio * len(mesh.faces)), self.min_faces)
        if len(mesh.faces) <= target_count:
            return mesh

        return mesh.simplify_quadric_decimation(face_count=target_count)


@dataclass
class MeshData:
    """A processed mesh with metadata."""

    mesh: trimesh.Trimesh
    source_path: Path  # Filepath from which the mesh was imported
    transforms_applied: list[MeshTransform]

    @classmethod
    def from_yaml_data(
        cls,
        yaml_data: dict[str, Any],
        simplifier: MeshSimplifier | None,
    ) -> MeshData:
        """Load a MeshData instance from data imported from YAML."""
        if "filepath" not in yaml_data:
            raise KeyError(f"No mesh filepath was provided in the YAML data: {yaml_data}")
        source_path = Path(yaml_data["filepath"])

        if not source_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {source_path}")

        mesh = trimesh.load_mesh(source_path)

        transforms = parse_mesh_transforms(yaml_data.get("transforms", []))
        for transform in transforms:
            transform.apply(mesh)

        if simplifier is not None:
            mesh = simplifier.simplify(mesh)

        return MeshData(mesh, source_path, transforms)

    @classmethod
    def from_mesh_path(cls, mesh_path: Path, simplifier: MeshSimplifier | None) -> MeshData:
        """Load a MeshData directly from a mesh file.

        :param mesh_path: Filepath to a mesh file (e.g., .OBJ)
        :param simplifier: Used to simplify the imported mesh (optional)
        :return: Constructed MeshData instance
        """
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        mesh = trimesh.load_mesh(mesh_path)
        if simplifier is not None:
            mesh = simplifier.simplify(mesh)

        return MeshData(mesh, mesh_path, transforms_applied=[])

    @property
    def aabb(self) -> AxisAlignedBoundingBox:
        """Get the axis-aligned bounding box (AABB) of the mesh."""
        min_bounds, max_bounds = self.mesh.bounds
        return AxisAlignedBoundingBox(
            min_xyz=Point3D.from_sequence(min_bounds),
            max_xyz=Point3D.from_sequence(max_bounds),
        )


class MeshTransform(Protocol):
    """Protocol for transform operations on meshes."""

    def apply(self, mesh: trimesh.Trimesh) -> None:
        """Apply the transform to the given mesh in-place."""
        ...


@dataclass(frozen=True)
class Translate(MeshTransform):
    """Translate a mesh by some (x,y,z) vector."""

    xyz: Point3D

    def apply(self, mesh: trimesh.Trimesh) -> None:
        """Translate the given mesh in-place."""
        mesh.apply_translation(self.xyz.to_array())

    @classmethod
    def from_list(cls, values: list[float]) -> Translate:
        """Construct a Translate instance from a list of values."""
        return Translate(xyz=Point3D.from_sequence(values))


@dataclass(frozen=True)
class Rotate(MeshTransform):
    """Rotate a mesh based on Euler angles specified in radians."""

    rpy: EulerRPY  # A 3D rotation represented using fixed-frame Euler angles

    def apply(self, mesh: trimesh.Trimesh) -> None:
        """Rotate the given mesh in-place."""
        matrix = self.rpy.to_homogeneous_matrix()
        mesh.apply_transform(matrix)

    @classmethod
    def from_list(cls, values: list[float]) -> Rotate:
        """Construct a Rotate instance from a list of values."""
        return Rotate(rpy=EulerRPY.from_list(values))


@dataclass(frozen=True)
class ApplyPose(MeshTransform):
    """Transform the mesh by applying a relative pose."""

    pose: Pose3D

    def apply(self, mesh: trimesh.Trimesh) -> None:
        """Adjust the pose of the given mesh in-place."""
        matrix = self.pose.to_homogeneous_matrix()
        mesh.apply_transform(matrix)


@dataclass(frozen=True)
class Scale(MeshTransform):
    """Scale a mesh either by a constant factor or non-uniform factors in (x,y,z)."""

    factor: float | tuple[float, float, float]

    def apply(self, mesh: trimesh.Trimesh) -> None:
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

    def apply(self, mesh: trimesh.Trimesh) -> None:
        """Center the mass of the given mesh in-place."""
        mesh.apply_translation(-mesh.centroid)


@dataclass(frozen=True)
class CenterBounds(MeshTransform):
    """Center a mesh at the origin based on its bounding box."""

    def apply(self, mesh: trimesh.Trimesh) -> None:
        """Center the bounding box of the given mesh in-place."""
        bounds_center = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-bounds_center)


@dataclass(frozen=True)
class BottomAtZeroZ(MeshTransform):
    """Place the bottom of a mesh at z = 0."""

    def apply(self, mesh: trimesh.Trimesh) -> None:
        """Bottom-normalize the given mesh in-place."""
        _, _, min_z = mesh.bounds[0]
        mesh.apply_translation([0, 0, -min_z])


@dataclass(frozen=True)
class OrientToAxes(MeshTransform):
    """Orient a mesh using its oriented bounding box."""

    def apply(self, mesh: trimesh.Trimesh) -> None:
        """Apply the oriented bounding box transform to the given mesh in-place."""
        mesh.apply_obb()


MeshTransformSpec = dict[str, Any] | str
"""Type representing a YAML specification for a mesh transform."""


def parse_mesh_transforms(transform_specs: list[MeshTransformSpec]) -> list[MeshTransform]:
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
            elif spec == "orient_to_axes":
                transforms.append(OrientToAxes())
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
            raise TypeError(f"Invalid mesh transform specification: {spec}")

    return transforms
