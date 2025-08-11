"""Define classes to represent robot trajectories."""

from typing import Sequence

from robotics_utils.kinematics import Configuration, Pose3D

Path = Sequence[Configuration]
"""A path is a sequence of robot configurations."""

CartesianPath = Sequence[Pose3D]
"""A Cartesian path is a sequence of target poses (e.g., for an end-effector)."""
