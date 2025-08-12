"""Define a class to represent axis-aligned bounding boxes."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from robotics_utils.kinematics import Point3D

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


@dataclass(frozen=True)
class AxisAlignedBoundingBox:
    """An axis-aligned bounding box (AABB) comprised of minimum and maximum (x,y,z) coordinates."""

    min_xyz: Point3D
    max_xyz: Point3D

    @classmethod
    def union(cls, aabb_iter: Iterable[AxisAlignedBoundingBox]) -> AxisAlignedBoundingBox:
        """Compute the union of the given axis-aligned bounding boxes (AABBs).

        :param aabb_iter: Iterable collection of AABBs
        :return: Resulting axis-aligned bounding box containing all given AABBs
        """
        combined_min = np.array([0.0, 0.0, 0.0])  # Initialize an empty combined bounding box
        combined_max = np.array([0.0, 0.0, 0.0])

        for aabb in aabb_iter:
            combined_min = np.minimum(combined_min, aabb.min_xyz.to_array())
            combined_max = np.maximum(combined_max, aabb.max_xyz.to_array())

        return AxisAlignedBoundingBox(
            min_xyz=Point3D.from_array(combined_min),
            max_xyz=Point3D.from_array(combined_max),
        )

    @property
    def vertices(self) -> Iterator[Point3D]:
        """Provide an iterator over the eight vertices of the axis-aligned bounding box."""
        min_x, min_y, min_z = self.min_xyz
        max_x, max_y, max_z = self.max_xyz
        all_xyzs = itertools.product((min_x, max_x), (min_y, max_y), (min_z, max_z))
        return (Point3D.from_sequence(xyz) for xyz in all_xyzs)

    def contains(self, entity: Point3D | AxisAlignedBoundingBox) -> bool:
        """Evaluate whether the bounding box contains the given entity."""
        if isinstance(entity, Point3D):
            return self._contains_point3d(entity)
        if isinstance(entity, AxisAlignedBoundingBox):
            return self._contains_aabb(entity)
        raise NotImplementedError(f"Unsupported type: {type(entity)}")

    def _contains_point3d(self, p: Point3D) -> bool:
        """Evaluate whether the bounding box contains the given 3D point."""
        min_x, min_y, min_z = self.min_xyz
        max_x, max_y, max_z = self.max_xyz
        return min_x <= p.x <= max_x and min_y <= p.y <= max_y and min_z <= p.z <= max_z

    def _contains_aabb(self, aabb: AxisAlignedBoundingBox) -> bool:
        """Evaluate whether this bounding box contains another."""
        return all(self.contains(v) for v in aabb.vertices)
