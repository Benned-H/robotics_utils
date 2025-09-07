"""Define classes to represent planned paths and trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_utils.kinematics import Configuration, Pose3D


@dataclass
class Path:
    """A sequence of planned robot configurations."""

    configurations: list[Configuration]


@dataclass
class CartesianPath:
    """A sequence of planned poses in the robot's workspace."""

    poses: list[Pose3D]


@dataclass
class StampedConfiguration:
    """A time-stamped configuration in a trajectory."""

    time_s: float
    """Time (seconds) since the trajectory started."""

    configuration: Configuration

    @property
    def joint_names(self) -> list[str]:
        """Retrieve the list of joint names specified by the configuration."""
        return list(self.configuration.keys())


@dataclass
class StampedPose3D:
    """A time-stamped 3D pose in a Cartesian trajectory."""

    time_s: float
    """Time (seconds) since the Cartesian trajectory started."""

    pose: Pose3D


@dataclass
class Trajectory:
    """A time-specified sequence of planned configurations."""

    points: list[StampedConfiguration]

    def __post_init__(self) -> None:
        """Verify properties expected of any valid trajectory."""
        if not self.points:
            return

        # If not empty, all points in the trajectory should use the same joint names
        j0_names = self.points[0].joint_names
        for p in self.points[1:]:
            jn_names = p.joint_names
            if j0_names != jn_names:
                raise ValueError(f"Trajectory points used joints: {j0_names} and {jn_names}.")

    @property
    def joint_names(self) -> list[str]:
        """Retrieve the names of the joints specified by the trajectory."""
        return [] if not self.points else self.points[0].joint_names


@dataclass
class CartesianTrajectory:
    """A time-specified sequence of planned poses in the robot's workspace."""

    points: list[StampedPose3D]
