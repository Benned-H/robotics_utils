"""Define a dataclass to represent motion planning queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_utils.kinematics import Configuration, Pose3D


@dataclass(frozen=True)
class MotionPlanningQuery:
    """A motion planning query specifying an end-effector target and collisions to ignore."""

    ee_target: Pose3D | Configuration

    ignored_objects: set[str] = field(default_factory=set)
    """Set of names of objects to be completely ignored during collision checking."""

    ignore_all_collisions: bool = False
    """Should all collision checking be disabled when computing this motion plan?"""
