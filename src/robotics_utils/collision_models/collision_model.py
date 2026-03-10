"""Define a class to represent collision models supporting multiple meshes and primitive shapes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from robotics_utils.collision_models.meshes import compute_aabb, load_mesh_from_schema
from robotics_utils.collision_models.primitive_shapes import PrimitiveShape, primitive_to_mesh
from robotics_utils.geometry import AxisAlignedBoundingBox
from robotics_utils.spatial import Pose3D

if TYPE_CHECKING:
    from pathlib import Path

    import trimesh

    from robotics_utils.io.pydantic_schemata import CollisionModelSchema


@dataclass
class CollisionModel:
    """A collision model supporting multiple meshes and geometric primitives."""

    meshes: list[trimesh.Trimesh] = field(default_factory=list)
    primitives: list[PrimitiveShape] = field(default_factory=list)
    primitive_poses: list[Pose3D] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate and initialize primitive local poses."""
        if not self.primitive_poses:
            self.primitive_poses = [Pose3D.identity() for _ in self.primitives]

        if len(self.primitive_poses) != len(self.primitives):
            raise ValueError(
                "CollisionModel expects one primitive pose per primitive shape. "
                f"Received {len(self.primitives)} primitives and "
                f"{len(self.primitive_poses)} primitive poses.",
            )

    @property
    def aabb(self) -> AxisAlignedBoundingBox:
        """Get the combined axis-aligned bounding box (AABB) of all elements in the model."""
        all_meshes = self.meshes.copy()

        for primitive, pose_o_p in zip(self.primitives, self.primitive_poses):
            primitive_mesh = primitive_to_mesh(primitive)
            primitive_mesh.apply_transform(pose_o_p.to_homogeneous_matrix())
            all_meshes.append(primitive_mesh)

        all_aabbs = [compute_aabb(mesh) for mesh in all_meshes]
        return AxisAlignedBoundingBox.union(aabb_iter=all_aabbs)

    @classmethod
    def from_schema(cls, schema: CollisionModelSchema, yaml_path: Path) -> CollisionModel:
        """Construct a CollisionModel using validated data imported from a YAML file."""
        meshes = [load_mesh_from_schema(mesh_schema, yaml_path) for mesh_schema in schema.meshes]
        primitives: list[PrimitiveShape] = []
        primitive_poses: list[Pose3D] = []
        for primitive_schema in schema.primitives:
            primitive_shape, primitive_pose = PrimitiveShape.from_schema_with_pose(primitive_schema)
            primitives.append(primitive_shape)
            primitive_poses.append(primitive_pose)

        return CollisionModel(meshes=meshes, primitives=primitives, primitive_poses=primitive_poses)
