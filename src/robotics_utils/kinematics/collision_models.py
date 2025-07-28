"""Define classes to represent collision models supporting multiple primitive shapes and meshes."""

from __future__ import annotations

from dataclasses import astuple, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import trimesh

from robotics_utils.kinematics.point3d import Point3D
from robotics_utils.kinematics.rotations import EulerRPY


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
        bounds_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
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


@dataclass(frozen=True)
class AxisAlignedBoundingBox:
    """An axis-aligned bounding box (AABB) comprised of minimum and maximum (x,y,z) coordinates."""

    min_xyz: Point3D
    max_xyz: Point3D

    def to_mesh(self) -> trimesh.Trimesh:
        """Create a mesh visualizing the axis-aligned bounding box."""
        bounds = (astuple(self.min_xyz), astuple(self.max_xyz))
        return trimesh.creation.box(bounds=bounds)


class PrimitiveShape(Protocol):
    """Protocol for primitive shapes."""

    @property
    def aabb(self) -> AxisAlignedBoundingBox:
        """Get the axis-aligned bounding box (AABB) of the primitive shape."""
        ...

    def to_list(self) -> list[float]:
        """Convert the primitive shape into a list of its dimensions."""
        ...

    def to_mesh(self) -> trimesh.Trimesh:
        """Convert the primitive shape into a mesh."""
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
        half_x_m = self.x_m / 2.0
        half_y_m = self.y_m / 2.0
        half_z_m = self.z_m / 2.0
        return AxisAlignedBoundingBox(
            min_xyz=Point3D(-half_x_m, -half_y_m, -half_z_m),
            max_xyz=Point3D(half_x_m, half_y_m, half_z_m),
        )

    def to_list(self) -> list[float]:
        """Convert the box into a list of its (x,y,z) dimensions."""
        return [self.x_m, self.y_m, self.z_m]

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

    def to_list(self) -> list[float]:
        """Convert the sphere into a list containing its radius."""
        return [self.radius_m]

    def to_mesh(self) -> trimesh.Trimesh:
        """Convert the sphere into a mesh."""
        return trimesh.creation.icosphere(radius=self.radius_m, subdivisions=3)


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

    def to_list(self) -> list[float]:
        """Convert the cylinder into a list of its dimensions."""
        return [self.height_m, self.radius_m]

    def to_mesh(self) -> trimesh.Trimesh:
        """Convert the cylinder into a mesh."""
        return trimesh.creation.cylinder(radius=self.radius_m, height=self.height_m, sections=32)


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

    @property
    def aabb(self) -> AxisAlignedBoundingBox:
        """Get the axis-aligned bounding box (AABB) of the mesh."""
        min_bounds, max_bounds = self.mesh.bounds
        return AxisAlignedBoundingBox(
            min_xyz=Point3D.from_sequence(min_bounds),
            max_xyz=Point3D.from_sequence(max_bounds),
        )


@dataclass
class CollisionModel:
    """A collision model supporting multiple meshes and geometric primitives."""

    meshes: list[MeshData] = field(default_factory=list)
    primitives: list[PrimitiveShape] = field(default_factory=list)

    @property
    def aabb(self) -> AxisAlignedBoundingBox:
        """Get the combined axis-aligned bounding box (AABB) of all elements in the model."""
        combined_min = np.array([0.0, 0.0, 0.0])  # Initialize an empty combined bounding box
        combined_max = np.array([0.0, 0.0, 0.0])

        for mesh in self.meshes:
            combined_min = np.minimum(combined_min, mesh.aabb.min_xyz.to_array())
            combined_max = np.maximum(combined_max, mesh.aabb.max_xyz.to_array())

        for p in self.primitives:
            combined_min = np.minimum(combined_min, p.aabb.min_xyz.to_array())
            combined_max = np.maximum(combined_max, p.aabb.max_xyz.to_array())

        return AxisAlignedBoundingBox(
            min_xyz=Point3D.from_array(combined_min),
            max_xyz=Point3D.from_array(combined_max),
        )

    @classmethod
    def from_yaml_data(cls, data: dict[str, Any], simplifier: MeshSimplifier) -> CollisionModel:
        """Create a collision model from data loaded from YAML."""
        meshes = [MeshData.from_yaml_data(m_data, simplifier) for m_data in data.get("meshes", [])]

        primitives = [
            create_primitive_shape(shape_type=prim_spec["type"], params=prim_spec["params"])
            for prim_spec in data.get("primitives", [])
        ]

        if not meshes and not primitives:
            raise ValueError("Collision model must have at least one mesh or geometric primitive")

        return CollisionModel(meshes=meshes, primitives=primitives)

    def visualize(
        self,
        show_bounding_box: bool = True,
        resolution: tuple[int, int] = (1920, 1080),
    ) -> None:
        """Visualize the collision model using the given configuration."""
        scene = trimesh.Scene()

        for i, mesh_data in enumerate(self.meshes):
            mesh = mesh_data.mesh.copy()
            mesh.visual = (0.7, 0.7, 0.7, 0.8)
            scene.add_geometry(mesh, node_name=f"mesh_{i}")

        for i, primitive in enumerate(self.primitives):
            primitive_mesh = primitive.to_mesh()
            primitive_mesh.visual = (0.2, 0.6, 1.0, 0.5)
            scene.add_geometry(primitive_mesh, node_name=f"primitive_{i}")

        if show_bounding_box:
            bbox_mesh = self.aabb.to_mesh()
            bbox_mesh.visual = (1.0, 0.5, 0.0, 1.0)  # TODO: High alpha here? Or not alpha?
            scene.add_geometry(bbox_mesh, node_name="bounding_box")

        scene.set_camera(resolution=resolution)
        scene.show()


def create_primitive_shape(shape_type: str, params: dict[str, float]) -> PrimitiveShape:
    """Create a primitive shape from its type and parameters."""
    if shape_type == "box":
        return Box(x_m=params["x"], y_m=params["y"], z_m=params["z"])

    if shape_type == "sphere":
        return Sphere(radius_m=params["radius"])

    if shape_type == "cylinder":
        return Cylinder(height_m=params["height"], radius_m=params["radius"])

    raise ValueError(f"Unknown primitive shape type: {shape_type}")
