"""Define a class to represent collision models supporting multiple meshes and primitive shapes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from robotics_utils.collision_models.meshes import compute_aabb, load_mesh_from_schema
from robotics_utils.collision_models.primitive_shapes import PrimitiveShape
from robotics_utils.geometry import AxisAlignedBoundingBox

if TYPE_CHECKING:
    from pathlib import Path

    import trimesh

    from robotics_utils.io.pydantic_schemata import CollisionModelSchema


@dataclass
class CollisionModel:
    """A collision model supporting multiple meshes and geometric primitives."""

    meshes: list[trimesh.Trimesh] = field(default_factory=list)
    primitives: list[PrimitiveShape] = field(default_factory=list)

    @property
    def aabb(self) -> AxisAlignedBoundingBox:
        """Get the combined axis-aligned bounding box (AABB) of all elements in the model."""
        combined_mesh_aabb = AxisAlignedBoundingBox.union(compute_aabb(m) for m in self.meshes)
        combined_primitive_aabb = AxisAlignedBoundingBox.union(p.aabb for p in self.primitives)
        return AxisAlignedBoundingBox.union((combined_mesh_aabb, combined_primitive_aabb))

    @classmethod
    def from_schema(cls, schema: CollisionModelSchema, yaml_path: Path) -> CollisionModel:
        """Construct a CollisionModel using validated data imported from a YAML file."""
        meshes = [load_mesh_from_schema(mesh_schema, yaml_path) for mesh_schema in schema.meshes]
        primitives = [PrimitiveShape.from_schema(p_schema) for p_schema in schema.primitives]
        return CollisionModel(meshes=meshes, primitives=primitives)
