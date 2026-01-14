"""Define a class representing robot footprints for collision checking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotics_utils.spatial import Pose2D


@dataclass(frozen=True)
class RectangularFootprint:
    """Rectangular robot footprint for collision checking during path planning.

    The footprint is defined by half-lengths in the robot's local frame.
    During collision checking, occupied cells are transformed into the robot's
    body frame and checked against the rectangular bounds.
    """

    half_length_x: float  # Half-length along robot's x-axis (forward)
    half_length_y: float  # Half-length along robot's y-axis (left)

    @classmethod
    def spot_footprint(cls) -> RectangularFootprint:
        """Create Spot's rectangular footprint (1.1m × 0.5m).

        Spot's body is approximately 1.1m long (x-axis) and 0.5m wide (y-axis).
        The footprint is centered at the origin of the robot's body frame.

        :return: RectangularFootprint representing Spot's body
        """
        return cls(half_length_x=1.1 / 2.0, half_length_y=0.5 / 2.0)

    def check_collision(
        self,
        pose: Pose2D,
        occupancy_mask: np.ndarray,
        origin: Pose2D,
        resolution_m: float,
    ) -> bool:
        """Check if footprint at given pose collides with occupied cells.

        Transforms occupied cell positions into the robot's body frame and
        checks if any fall within the rectangular bounds.

        :param pose: Robot pose in world frame
        :param occupancy_mask: Boolean grid where True indicates occupied cells
        :param origin: Origin pose of the occupancy grid
        :param resolution_m: Grid resolution in meters
        :return: True if collision detected, False otherwise
        """
        # Get indices of occupied cells
        occupied_rows, occupied_cols = np.where(occupancy_mask)
        if len(occupied_rows) == 0:
            return False

        # Convert grid indices to world coordinates
        # Grid cell (col, row) -> world position relative to origin
        cos_origin = np.cos(origin.yaw_rad)
        sin_origin = np.sin(origin.yaw_rad)

        # Local coordinates in grid frame
        local_x = occupied_cols * resolution_m
        local_y = occupied_rows * resolution_m

        # Rotate to world frame and translate by origin
        world_x = origin.x + local_x * cos_origin - local_y * sin_origin
        world_y = origin.y + local_x * sin_origin + local_y * cos_origin

        # Transform world points into robot body frame
        # First translate relative to robot position
        dx = world_x - pose.x
        dy = world_y - pose.y

        # Rotate by -robot_yaw to get body frame coordinates
        cos_neg_yaw = np.cos(-pose.yaw_rad)
        sin_neg_yaw = np.sin(-pose.yaw_rad)

        body_x = dx * cos_neg_yaw - dy * sin_neg_yaw
        body_y = dx * sin_neg_yaw + dy * cos_neg_yaw

        # Check if any points are within rectangular bounds
        in_x = np.abs(body_x) < self.half_length_x
        in_y = np.abs(body_y) < self.half_length_y
        collision = np.any(in_x & in_y)

        return bool(collision)
