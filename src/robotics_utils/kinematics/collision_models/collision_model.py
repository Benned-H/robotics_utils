"""Define a class to represent collision models supporting multiple meshes and primitive shapes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robotics_utils.kinematics.collision_models import (
    AxisAlignedBoundingBox,
    Mesh,
    PrimitiveShape,
    create_primitive_shape,
)


@dataclass
class CollisionModel:
    """A collision model supporting multiple meshes and geometric primitives."""

    meshes: list[Mesh] = field(default_factory=list)
    primitives: list[PrimitiveShape] = field(default_factory=list)

    @property
    def aabb(self) -> AxisAlignedBoundingBox:
        """Get the combined axis-aligned bounding box (AABB) of all elements in the model."""
        return AxisAlignedBoundingBox.union(
            entity.aabb for entity in (self.meshes + self.primitives)
        )

    @classmethod
    def from_yaml_data(cls, data: dict[str, Any]) -> CollisionModel:
        """Create a collision model from data loaded from YAML."""
        meshes = [Mesh.from_yaml_data(mesh_data) for mesh_data in data.get("meshes", [])]

        primitives = [
            create_primitive_shape(shape_type=shape_data["type"], params=shape_data["params"])
            for shape_data in data.get("primitives", [])
        ]

        if not meshes and not primitives:
            raise ValueError("Collision model must have at least one mesh or geometric primitive")

        return CollisionModel(meshes=meshes, primitives=primitives)
