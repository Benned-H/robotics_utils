"""Define classes to represent robot trajectories."""

from collections.abc import Sequence

from robotics_utils.kinematics.kinematics_core import Configuration
from robotics_utils.kinematics.poses import Pose3D

Path = Sequence[Configuration]
"""A path is a sequence of robot configurations."""

CartesianPath = Sequence[Pose3D]
"""A Cartesian path is a sequence of target poses (e.g., for an end-effector)."""
