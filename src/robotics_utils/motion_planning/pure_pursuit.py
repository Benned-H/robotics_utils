"""Pure pursuit path follower for holonomic mobile robots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from robotics_utils.spatial import Pose2D


@dataclass
class PurePursuitConfig:
    """Configuration for pure pursuit controller."""

    lookahead_distance_m: float = 0.8
    """Distance (meters) ahead of robot to target on the path."""

    goal_tolerance_m: float = 0.2
    """Distance threshold (meters) for reaching the final goal."""

    min_lookahead_m: float = 0.3
    """Minimum lookahead when close to goal."""

    lookahead_steps: int = 5
    """Number of waypoints ahead of the closest point to target."""


class PurePursuitFollower:
    """Pure pursuit path follower for holonomic robots.

    For holonomic robots, pure pursuit simplifies to finding and targeting
    a base pose a few steps ahead on the path.

    The algorithm:
    1. Find the closest point on the path to the robot's current position
    2. Step forward a fixed number of waypoints from that point
    3. That waypoint becomes the target - command the robot directly toward it
    4. Repeat until the goal is reached

    Since Spot is holonomic (can move in any direction), we don't need to
    compute steering angles - we just set the target pose directly.
    """

    def __init__(self, path: list[Pose2D], config: PurePursuitConfig | None = None) -> None:
        """Initialize the pure pursuit follower with a path.

        :param path: List of Pose2D waypoints defining the path to follow
        :param config: Configuration parameters (uses defaults if None)
        """
        if len(path) < 2:
            raise ValueError("Path must have at least 2 waypoints.")

        self.path = path
        self.config = config or PurePursuitConfig()
        self._path_xy = np.array([p.position.to_array() for p in path])  # Shape (N, 2)
        self._ref_frame = path[0].ref_frame
        self._furthest_target_idx = 0
        """Index of the furthest path waypoint yet targeted during path following."""

    def get_target_pose(self, robot_pose: Pose2D) -> tuple[Pose2D, bool]:
        """Get the target pose for the robot to navigate toward.

        Finds the closest waypoint on the path, then steps forward a fixed number
        of waypoints to produce a reactive target that stays close to the path.

        :param robot_pose: Current robot pose (must be in same frame as path)
        :return: Tuple of (target_pose, is_complete)
                 - target_pose: The pose to navigate toward
                 - is_complete: True if robot has reached the goal
        """
        robot_xy = robot_pose.position.to_array()
        dist_to_goal_m = np.linalg.norm(robot_xy - self._path_xy[-1])

        # If we're sufficiently close to the goal, return it
        if dist_to_goal_m < self.config.goal_tolerance_m:
            self._furthest_target_idx = len(self.path) - 1
            return (self.path[-1], True)

        # Find the waypoint nearest to the robot
        distances_from_robot_m = np.linalg.norm(self._path_xy - robot_xy, axis=1)
        closest_idx = int(np.argmin(distances_from_robot_m))

        # Ensure monotonic progress along the path
        closest_idx = max(closest_idx, self._furthest_target_idx)

        # Step forward a fixed number of waypoints from the closest point
        target_idx = min(closest_idx + self.config.lookahead_steps, len(self.path) - 1)
        self._furthest_target_idx = target_idx

        # If we've reached the last waypoint, signal completion
        if target_idx >= len(self.path) - 1:
            return (self.path[-1], True)

        # Compute heading from the path direction at the target
        target_pose = self.path[target_idx]
        next_xy = self._path_xy[min(target_idx + 1, len(self.path) - 1)]
        diff_xy = next_xy - self._path_xy[target_idx]
        target_yaw_rad = np.arctan2(diff_xy[1], diff_xy[0])

        return (
            Pose2D(target_pose.x, target_pose.y, target_yaw_rad, self._ref_frame),
            False,
        )
