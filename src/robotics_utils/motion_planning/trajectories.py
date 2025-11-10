"""Define classes to represent planned paths and trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from robotics_utils.kinematics import Configuration, Pose3D

Path = Sequence[Configuration]
"""A path is a sequence of robot configurations."""

CartesianPath = Sequence[Pose3D]
"""A Cartesian path is a sequence of target poses (e.g., for an end-effector)."""


@dataclass
class TrajectoryPoint:
    """A planned state of joint values at a specified time in a trajectory."""

    time_s: float
    """Time (seconds) since the trajectory started."""

    positions: Configuration
    velocities: Configuration

    @property
    def joint_names(self) -> list[str]:
        """Retrieve the names of the joints specified by the point."""
        return list(self.positions.keys())


@dataclass
class Trajectory:
    """A sequence of planned configurations at specified times."""

    points: list[TrajectoryPoint]

    def __post_init__(self) -> None:
        """Verify properties expected of any valid trajectory."""
        if not self.points:
            return

        # All points in any non-empty trajectory should use the same joint names
        j0_names = self.points[0].joint_names
        for p in self.points[1:]:
            jn_names = p.joint_names
            if j0_names != jn_names:
                raise ValueError(f"Trajectory points used joint names: {j0_names} and {jn_names}.")

    @property
    def joint_names(self) -> list[str]:
        """Retrieve the names of the joints specified by the trajectory."""
        return [] if not self.points else self.points[0].joint_names
