"""Define dataclasses to represent skeletons of motion plans to be recomputed online."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_utils.spatial import Pose3D


@dataclass(frozen=True)
class ManipulationPlanSkeleton:
    """A skeleton for an object-relative manipulation plan to be recomputed online."""

    robot_name: str  # Name of the robot to be used to execute the plan
    manipulator_name: str  # Name of the relevant manipulator
    target_o_ee: Pose3D  # Target end-effector pose w.r.t. object


@dataclass(frozen=True)
class NavigationPlanSkeleton:
    """A skeleton for a navigation plan to be recomputed online."""

    robot_name: str  # Name of the robot to be used to execute the plan
    target_base_pose: Pose3D  # Target robot base pose of the navigation plan
