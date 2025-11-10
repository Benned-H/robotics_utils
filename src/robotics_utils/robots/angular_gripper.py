"""Define an interface for an angular robot gripper."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class GripperAngleLimits:
    """Specifies joint limits (in radians) for a robot gripper."""

    open_rad: float
    """Angle (radians) at which the gripper is fully open."""

    closed_rad: float
    """Angle (radians) at which the gripper is fully closed."""


class AngularGripper(ABC):
    """An interface for an angular robot gripper."""

    def __init__(self, limits: GripperAngleLimits) -> None:
        """Initialize the angular gripper with its joint limits."""
        self.joint_limits = limits

    @property
    @abstractmethod
    def link_names(self) -> list[str]:
        """Retrieve the names of the links in the gripper."""
        ...

    @abstractmethod
    def move_to_angle_rad(self, target_rad: float, timeout_s: float) -> None:
        """Move the gripper to a target angle (radians).

        :param target_rad: Target angle (radians) for the gripper
        :param timeout_s: Duration (seconds) after which the motion times out
        """
        ...

    def open(self, timeout_s: float = 10.0) -> None:
        """Open the gripper to its fully-open position."""
        self.move_to_angle_rad(self.joint_limits.open_rad, timeout_s)

    def close(self, timeout_s: float = 10.0) -> None:
        """Close the gripper to its fully-closed position."""
        self.move_to_angle_rad(self.joint_limits.closed_rad, timeout_s)
