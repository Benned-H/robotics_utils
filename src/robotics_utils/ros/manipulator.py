"""Define a class to interface with robot manipulators using MoveIt."""

import sys

from moveit_commander import MoveGroupCommander, roscpp_initialize

from robotics_utils.kinematics import Configuration


class Manipulator:
    """A MoveIt-based interface for a robot manipulator."""

    def __init__(self, move_group_name: str) -> None:
        """Initialize an interface for the manipulator's move group."""
        roscpp_initialize(sys.argv)
        self._move_group = MoveGroupCommander(move_group_name, wait_for_servers=30)
        self.joint_names: tuple[str] = tuple(self._move_group.get_active_joints())

    def get_configuration(self) -> Configuration:
        """Retrieve the manipulator's current configuration."""
        joint_values = self._move_group.get_current_joint_values()
        assert len(self.joint_names) == len(joint_values)

        return dict(zip(self.joint_names, joint_values))
