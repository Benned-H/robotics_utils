"""Define a class to represent collision models supporting multiple meshes and primitive shapes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from robotics_utils.collision_models.aabb import AxisAlignedBoundingBox
from robotics_utils.collision_models.meshes import compute_aabb, load_trimesh_from_yaml_data
from robotics_utils.collision_models.primitive_shapes import PrimitiveShape, create_primitive_shape

if TYPE_CHECKING:
    from pathlib import Path

    import trimesh


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
    def from_yaml_data(cls, data: dict[str, Any], yaml_path: Path) -> CollisionModel:
        """Create a collision model from data loaded from the specified YAML file."""
        meshes = [load_trimesh_from_yaml_data(md, yaml_path) for md in data.get("meshes", [])]

        primitives = [
            create_primitive_shape(shape_data) for shape_data in data.get("primitives", [])
        ]

        if not meshes and not primitives:
            raise ValueError("Collision model must have at least one mesh or geometric primitive")

        return CollisionModel(meshes=meshes, primitives=primitives)
