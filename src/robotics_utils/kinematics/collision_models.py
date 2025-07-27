"""Define classes to represent collision models supporting multiple primitives and meshes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Tuple

import numpy as np
import trimesh
from trimesh.transformations import euler_matrix


class MeshTransform(Protocol):
    """Protocol for transform operations on meshes."""

    def apply(self, mesh: trimesh.Trimesh) -> None:
        """Apply the transform to the given mesh in-place."""
        ...


@dataclass(frozen=True)
class Translate(MeshTransform):
    """Translate a mesh by some (x,y,z) vector."""

    x: float
    y: float
    z: float

    def apply(self, mesh: trimesh.Trimesh) -> None:
        """Apply the translation to the given mesh in-place."""
        mesh.apply_translation(np.array([self.x, self.y, self.z]))

    @classmethod
    def from_list(cls, values: list[float]) -> Translate:
        """Construct a Translate instance from a list of values."""
        if len(values) != 3:
            raise ValueError(f"Translate expects 3 values, got {len(values)}")
        return Translate(x=values[0], y=values[1], z=values[2])


@dataclass(frozen=True)
class Rotate(MeshTransform):
    """Rotate a mesh based on Euler angles in radians."""

    roll_rad: float  # Rotation about the x-axis
    pitch_rad: float  # Rotation about the y-axis
    yaw_rad: float  # Rotation about the z-axis

    def apply(self, mesh: trimesh.Trimesh) -> None:
        """Apply the rotation to the given mesh in-place."""
        matrix = euler_matrix(self.roll_rad, self.pitch_rad, self.yaw_rad, axes="sxyz")
        mesh.apply_transform(matrix)

    @classmethod
    def from_list(cls, values: list[float]) -> Rotate:
        """Construct a Rotate instance from a list of values."""
        if len(values) != 3:
            raise ValueError(f"Rotate expects 3 values, got {len(values)}")
        return Rotate(roll_rad=values[0], pitch_rad=values[1], yaw_rad=values[2])


@dataclass(frozen=True)
class Scale(MeshTransform):
    """Scale a mesh either by a constant factor or non-uniform factors in (x,y,z)."""

    factor: float | tuple[float, float, float]

    def apply(self, mesh: trimesh.Trimesh) -> None:
        """Scale the given mesh in-place."""
        mesh.apply_scale(self.factor)

    @classmethod
    def from_value(cls, value: float | list[float]) -> Scale:
        """Construct a Scale instance from a single value or list."""
        if isinstance(value, list):
            if len(value) != 3:
                raise ValueError(f"Non-uniform scale expects 3 values, got {len(value)}")
            return Scale(factor=(value[0], value[1], value[2]))
        return Scale(factor=value)


@dataclass(frozen=True)
class CenterMass(MeshTransform):
    """Center a mesh at the origin based on its center of mass (centroid)."""

    def apply(self, mesh: trimesh.Trimesh) -> None:
        """Center the mass of the given mesh in-place."""
        mesh.apply_translation(-mesh.centroid)


@dataclass(frozen=True)
class CenterBounds(MeshTransform):
    """Center a mesh at the origin based on its bounding box."""

    def apply(self, mesh: trimesh.Trimesh) -> None:
        """Center the given mesh in-place."""
        bounds_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
        mesh.apply_translation(-bounds_center)


@dataclass(frozen=True)
class BottomAtZeroZ(MeshTransform):
    """Place the bottom of a mesh at z = 0."""

    def apply(self, mesh: trimesh.Trimesh) -> None:
        """Bottom-normalize the given mesh in-place."""
        _, _, min_z = mesh.bounds[0]
        mesh.apply_translation(np.array([0, 0, -min_z]))


@dataclass(frozen=True)
class OrientToAxes(MeshTransform):
    """Orient a mesh using its oriented bounding box."""

    def apply(self, mesh: trimesh.Trimesh) -> None:
        """Apply the oriented bounding box transform to the given mesh in-place."""
        mesh.apply_obb()


def parse_mesh_transforms(transform_specs: list[dict[str, Any] | str]) -> list[MeshTransform]:
    """Parse mesh transform specifications from YAML."""
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
            raise TypeError(f"Invalid transform specification type: {spec}")

    return transforms


DimsXYZ = Tuple[float, float, float]


@dataclass(frozen=True)
class Box:
    """Box primitive shape with (x,y,z) dimensions."""

    x: float
    y: float
    z: float

    @property
    def bounds(self) -> DimsXYZ:
        """Get the bounding box dimensions of the box."""
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class Sphere:
    """Sphere primitive shape with a radius."""

    radius: float

    @property
    def bounds(self) -> DimsXYZ:
        """Get the bounding box dimensions of the sphere."""
        d = 2.0 * self.radius
        return (d, d, d)


@dataclass(frozen=True)
class Cylinder:
    """Cylinder primitive shape with a height and radius."""

    height: float
    radius: float

    @property
    def bounds(self) -> DimsXYZ:
        """Get the bounding box dimensions of the cylinder."""
        d = 2.0 * self.radius
        return (d, d, self.height)


PrimitiveShape = Box | Sphere | Cylinder


class MeshSimplifier(Protocol):
    """Protocol for mesh simplification strategies."""

    def simplify(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Simplify the given mesh."""
        ...


@dataclass(frozen=True)
class RatioSimplifier(MeshSimplifier):
    """Simplify a mesh to a ratio of its original face count."""

    ratio: float = 0.1  # Proportion of mesh faces kept in the simplified mesh
    min_faces: int = 100  # Minimum number of faces in the simplified mesh

    def simplify(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Simplify a mesh by reducing its face count to a target ratio of its original count."""
        target_count = max(int(self.ratio * len(mesh.faces)), self.min_faces)
        if len(mesh.faces) <= target_count:
            return mesh

        return mesh.simplify_quadric_decimation(face_count=target_count)


@dataclass(frozen=True)
class MeshData:
    """A processed mesh with metadata."""

    mesh: trimesh.Trimesh
    source_path: Path  # Filepath from which the mesh was imported
    transforms_applied: list[MeshTransform]

    @classmethod
    def from_yaml_data(cls, yaml_data: dict[str, Any], simplifier: MeshSimplifier) -> MeshData:
        """Load a MeshData instance from the given data imported from YAML."""
        if "filepath" not in yaml_data:
            raise KeyError(f"No mesh filepath was provided in the YAML data: {yaml_data}")
        source_path = Path(yaml_data["filepath"])

        if not source_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {source_path}")

        mesh = trimesh.load_mesh(source_path)

        transforms = parse_mesh_transforms(yaml_data.get("transforms", []))
        for transform in transforms:
            transform.apply(mesh)

        mesh = simplifier.simplify(mesh)

        return MeshData(mesh, source_path, transforms_applied=transforms)


@dataclass
class CollisionModel:
    """A collision model supporting multiple meshes and geometric primitives."""

    meshes: list[MeshData] = field(default_factory=list)
    primitives: list[PrimitiveShape] = field(default_factory=list)

    @property
    def bounds(self) -> DimsXYZ:
        """Get the combined bounding box of all elements of the model."""
        if not self.meshes and not self.primitives:
            return (0.0, 0.0, 0.0)

        all_bounds: list[DimsXYZ] = []

        for mesh_data in self.meshes:
            b = mesh_data.mesh.bounds
            dims = b[1] - b[0]
            all_bounds.append(tuple(dims))

        all_bounds.extend(primitive.bounds for primitive in self.primitives)

        return (
            max(b[0] for b in all_bounds),
            max(b[1] for b in all_bounds),
            max(b[2] for b in all_bounds),
        )

    @classmethod
    def from_yaml_data(cls, data: dict[str, Any], simplifier: MeshSimplifier) -> CollisionModel:
        """Create a collision model from data loaded from YAML."""
        meshes_data = data.get("meshes", [])
        meshes = [MeshData.from_yaml_data(mesh_data, simplifier) for mesh_data in meshes_data]

        primitives = [
            create_primitive_shape(shape_type=prim_spec["type"], params=prim_spec["params"])
            for prim_spec in data.get("primitives", [])
        ]

        if not meshes and not primitives:
            raise ValueError("Collision model must have at least one mesh or geometric primitive")

        return CollisionModel(meshes=meshes, primitives=primitives)


def create_primitive_shape(shape_type: str, params: dict[str, float]) -> PrimitiveShape:
    """Create a primitive shape from its type and parameters."""
    if shape_type == "box":
        return Box(x=params["x"], y=params["y"], z=params["z"])

    if shape_type == "sphere":
        return Sphere(radius=params["radius"])

    if shape_type == "cylinder":
        return Cylinder(height=params["height"], radius=params["radius"])

    raise ValueError(f"Unknown primitive shape type: {shape_type}")
