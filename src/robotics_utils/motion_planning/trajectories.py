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


@dataclass
class CartesianTrajectory:
    """A time-specified sequence of planned poses in the robot's workspace."""

    points: list[StampedPose3D]
