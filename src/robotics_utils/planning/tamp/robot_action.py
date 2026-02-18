"""Define a protocol to represent low-level executable robot actions."""

from typing import Generic, List, Protocol

from robotics_utils.planning.tamp.states import StateT


class RobotAction(Protocol, Generic[StateT]):
    """A general interface for any low-level action to be executed on a robot."""

    def apply(self, curr_state: StateT) -> StateT:
        """Directly apply the effects of the robot action onto the low-level state.

        :param curr_state: Current low-level state (not to be modified)
        :return: Updated low-level state reflecting the action's effects
        """
        ...

    def execute(self, curr_state: StateT) -> None:
        """Execute the full action on the robot.

        :param curr_state: Current low-level state of the environment
        """
        ...


Trajectory = List[RobotAction[StateT]]
"""A trajectory is a sequence of low-level actions to be executed on a robot."""
