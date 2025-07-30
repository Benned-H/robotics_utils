"""Define classes to represent collision models supporting multiple primitive shapes and meshes."""

from __future__ import annotations

from dataclasses import astuple, dataclass, field
from typing import Any

import numpy as np
import trimesh

from robotics_utils.kinematics.collisions.meshes import MeshData, MeshSimplifier
from robotics_utils.kinematics.collisions.primitives import PrimitiveShape
from robotics_utils.kinematics.point3d import Point3D
from robotics_utils.kinematics.poses import Pose3D


@dataclass(frozen=True)
class AxisAlignedBoundingBox:
    """An axis-aligned bounding box (AABB) comprised of minimum and maximum (x,y,z) coordinates."""

    min_xyz: Point3D
    max_xyz: Point3D

    # TODO: Implement matrix multiplication!

    def to_mesh(self) -> trimesh.Trimesh:
        """Create a mesh visualizing the axis-aligned bounding box."""
        bounds = (astuple(self.min_xyz), astuple(self.max_xyz))
        return trimesh.creation.box(bounds=bounds)


@dataclass
class CollisionModel:
    """A collision model supporting multiple meshes and geometric primitives."""

    meshes: list[MeshData] = field(default_factory=list)
    mesh_poses: list[Pose3D | None] = field(default_factory=list)
    """Optional respective poses expressing pose_c_m: mesh w.r.t. collision model frame."""

    primitives: list[PrimitiveShape] = field(default_factory=list)
    primitive_poses: list[Pose3D | None] = field(default_factory=list)
    """Optional respective poses expressing pose_c_p: primitive w.r.t. collision model frame."""

    # @property
    # def aabb(self) -> AxisAlignedBoundingBox:
    #     """Get the combined axis-aligned bounding box (AABB) of all elements in the model."""
    #     combined_min = np.array([0.0, 0.0, 0.0])  # Initialize an empty combined bounding box
    #     combined_max = np.array([0.0, 0.0, 0.0])

    #     for mesh, pose_c_m in zip(self.meshes, self.mesh_poses, strict=True):
    #         aabb_m = mesh.aabb

    #         point_m_min = mesh.aabb.min_xyz
    #         point_m_max = mesh.aabb.max_xyz

    #     #     for mesh in self.meshes:
    #     #         combined_min = np.minimum(combined_min, mesh.aabb.min_xyz.to_array())
    #     #         combined_max = np.maximum(combined_max, mesh.aabb.max_xyz.to_array())

    #     #     for p in self.primitives:
    #     #         combined_min = np.minimum(combined_min, p.aabb.min_xyz.to_array())
    #     #         combined_max = np.maximum(combined_max, p.aabb.max_xyz.to_array())

    #     return AxisAlignedBoundingBox(
    #         min_xyz=Point3D.from_array(combined_min),
    #         max_xyz=Point3D.from_array(combined_max),
    #     )


#     @classmethod
#     def from_yaml_data(
#         cls,
#         data: dict[str, Any],
#         simplifier: MeshSimplifier | None,
#     ) -> CollisionModel:
#         """Create a collision model from data loaded from YAML."""
#         print("Into CollisionModel.from_yaml_data...")
#         meshes = [MeshData.from_yaml_data(m_data, simplifier) for m_data in data.get("meshes", [])]
#         print("Finished with meshes...")

#         primitives = [
#             create_primitive_shape(shape_type=prim_spec["type"], params=prim_spec["params"])
#             for prim_spec in data.get("primitives", [])
#         ]
#         print("Finished with primitives...")

#         if not meshes and not primitives:
#             raise ValueError("Collision model must have at least one mesh or geometric primitive")

#         return CollisionModel(meshes=meshes, primitives=primitives)
