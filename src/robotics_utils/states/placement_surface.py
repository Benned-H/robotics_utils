"""Define a class representing a surface onto which objects can be placed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from robotics_utils.math import ClosedInterval

if TYPE_CHECKING:
    from robotics_utils.states.object_states import ObjectKinematicState


@dataclass(frozen=True)
class PlacementSurface:
    """A surface onto which objects can be placed.

    This class assumes that its coordinate frame is 'level' (i.e., the surface is at a fixed z).
    """

    height_m: float
    """Height (m) of the surface in its reference frame."""

    x_range: ClosedInterval
    """Range of x-values defining the surface's extent (meters)."""

    y_range: ClosedInterval
    """Range of y-values defining the surface's extent (meters)."""

    frame: str
    """Name of the surface object's coordinate frame."""

    @classmethod
    def from_object_aabb(cls, obj_kin_state: ObjectKinematicState) -> PlacementSurface:
        """Construct a PlacementSurface using the top of an object's axis-aligned bounding box.

        :param obj_kin_state: Kinematic state of an object
        :return: Constructed PlacementSurface instance
        """
        aabb = obj_kin_state.collision_model.aabb
        return PlacementSurface(
            height_m=aabb.max_xyz.z,
            x_range=ClosedInterval(minimum=aabb.min_xyz.x, maximum=aabb.max_xyz.x),
            y_range=ClosedInterval(minimum=aabb.min_xyz.y, maximum=aabb.max_xyz.y),
            frame=obj_kin_state.name,
        )
