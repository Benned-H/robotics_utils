"""Define dataclasses to represent skeletons of motion plans to be recomputed online."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_utils.kinematics import Pose3D


@dataclass(frozen=True)
class NavigationPlanSkeleton:
    """A skeleton for a navigation plan to be recomputed online."""

    robot_name: str  # Name of the robot to be used to execute the plan
    target_base_pose: Pose3D  # Target robot base pose of the navigation plan
