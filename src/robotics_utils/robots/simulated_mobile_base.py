"""Implement an interface for a simulated mobile robot base."""

from __future__ import annotations

from typing import TYPE_CHECKING

from robotics_utils.motion_planning import NavigationQuery, RectangularFootprint, plan_se2_path
from robotics_utils.robots import MobileRobot
from robotics_utils.skills import Outcome

if TYPE_CHECKING:
    from robotics_utils.perception import OccupancyGrid2D
    from robotics_utils.spatial import Pose2D
    from robotics_utils.states import ObjectCentricState


class SimulatedMobileBase(MobileRobot):
    """An interface modeling a simulated mobile robot base."""

    def __init__(
        self,
        env_state: ObjectCentricState,
        occupancy_grid: OccupancyGrid2D,
        robot_footprint: RectangularFootprint,
        *,
        robot_name: str,
        base_frame: str,
    ) -> None:
        """Initialize the simulated mobile base with the current environment state.

        :param env_state: State of the simulated environment (used to update the robot's base pose)
        :param occupancy_grid: 2D occupancy grid used for collision-aware path planning
        :param robot_footprint: Robot footprint used for collision checking during navigation
        :param robot_name: Name of the robot
        :param base_frame: Name of the reference frame of the robot's mobile base
        """
        if robot_name not in env_state.robot_names:
            env_state.add_robot(robot_name)

        self.env_state = env_state
        self.occupancy_grid = occupancy_grid
        self.robot_footprint = robot_footprint
        self.robot_name = robot_name
        self.base_frame = base_frame

    @property
    def current_base_pose(self) -> Pose2D:
        """Retrieve the robot's current base pose."""
        base_pose = self.env_state.get_robot_base_pose(robot_name=self.robot_name)
        if base_pose is None:
            raise RuntimeError(
                f"Environment state does not contain the base pose of robot '{self.robot_name}'.",
            )

        return base_pose.to_2d()

    def compute_navigation_plan(self, initial: Pose2D, goal: Pose2D) -> list[Pose2D] | None:
        """Compute a navigation plan between the two given robot base poses.

        :param initial: Robot base pose from which the plan begins
        :param goal: Target base pose to be reached by the navigation plan
        :return: Navigation plan (list of base pose waypoints), or None if no plan is found
        """
        return [initial, goal]

        # TODO: Replace dummy planner with actual below implementation!
        # query = NavigationQuery(
        #     start_pose=initial,
        #     goal_pose=goal,
        #     occupancy_grid=self.occupancy_grid,
        #     robot_footprint=self.robot_footprint,
        # )
        # return plan_se2_path(query)

    def execute_navigation_plan(self, nav_plan: list[Pose2D], timeout_s: float = 60.0) -> Outcome:
        """Execute the given navigation plan on the simulated robot base.

        :param nav_plan: Navigation plan of 2D base pose waypoints
        :param timeout_s: Duration (seconds) after which the plan times out (default: 60 seconds)
        :return: Boolean success indicator and explanatory message
        """
        if not nav_plan:
            return Outcome(success=False, message="Cannot execute an empty navigation plan.")

        final_pose = nav_plan[-1].to_3d()
        self.env_state.set_robot_base_pose(self.robot_name, base_pose=final_pose)

        success = self.env_state.get_robot_base_pose(self.robot_name) == final_pose
        message = (
            f"Successfully moved '{self.robot_name}' to the end of the navigation plan."
            if success
            else f"Unable to move robot '{self.robot_name}' to the end of the navigation plan."
        )
        return Outcome(success=success, message=message)
