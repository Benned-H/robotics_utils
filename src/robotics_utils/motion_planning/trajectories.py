"""Define classes to represent planned paths and trajectories."""

from __future__ import annotations

from dataclasses import dataclass

from robotics_utils.kinematics import Configuration


@dataclass
class TrajectoryPoint:
    """A time-stamped joint state in a trajectory."""

    time_s: float
    """Time (seconds) since the trajectory started."""

    position: Configuration
    velocities: Configuration

    @property
    def joint_names(self) -> list[str]:
        """Retrieve the list of joint names specified by the position."""
        return list(self.position.keys())


@dataclass
class Trajectory:
    """A time-specified dynamic sequence of planned configurations."""

    points: list[TrajectoryPoint]

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
