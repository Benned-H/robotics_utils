"""Define a class to represent axis-aligned bounding boxes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from robotics_utils.kinematics.point3d import Point3D


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
