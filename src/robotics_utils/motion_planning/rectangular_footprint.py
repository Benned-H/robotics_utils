"""Define a class representing rectangular robot footprints for collision checking."""

from dataclasses import dataclass

import numpy as np

from robotics_utils.perception import OccupancyGrid2D
from robotics_utils.spatial import Pose2D


@dataclass(frozen=True)
class RectangularFootprint:
    """A rectangular robot footprint used for collision checking during path planning.

    During collision checking, occupied cells are transformed into the robot's body frame
    and checked against the bounds of the rectangular footprint.
    """

    max_x_m: float
    """Maximum x-coordinate (meters) of the footprint in the robot body frame."""

    min_x_m: float
    """Minimum x-coordinate (meters) of the footprint in the robot body frame."""

    half_length_y_m: float
    """Half-length (meters) of the robot along its body-frame y-axis."""

    def check_collision(self, robot_pose: Pose2D, occupancy_grid: OccupancyGrid2D) -> bool:
        """Check whether the robot collides with obstacles at the given pose.

        :param robot_pose: Robot pose in world frame
        :param occupancy_grid: Occupancy grid representing occupied cells
        :return: True if a collision is detected, else False
        """
        grid = occupancy_grid.grid
        if robot_pose.ref_frame != grid.origin.ref_frame:
            raise ValueError(
                f"Cannot check collisions when robot pose is in frame '{robot_pose.ref_frame}' "
                f"and occupancy grid origin is in frame '{grid.origin.ref_frame}'.",
            )

        # Compute the occupied cells' coordinates in the robot body frame
        occupied_mask = occupancy_grid.get_occupied_mask()

        occupied_rows, occupied_cols = np.where(occupied_mask)
        occupied_homogeneous_g = np.array(
            [
                grid.col_to_local_x(occupied_cols),
                grid.row_to_local_y(occupied_rows),
                np.ones_like(occupied_cols),
            ],
        )

        transform_w_g = grid.origin.to_homogeneous_matrix()  # Grid w.r.t. world
        transform_w_r = robot_pose.to_homogeneous_matrix()  # Robot body w.r.t. world
        transform_r_g = np.linalg.inv(transform_w_r) @ transform_w_g  # Grid w.r.t. robot body

        occupied_homogeneous_r = transform_r_g @ occupied_homogeneous_g  # w.r.t. robot body
        occupied_x_r, occupied_y_r = occupied_homogeneous_r[0], occupied_homogeneous_r[1]

        in_x = (self.min_x_m < occupied_x_r) & (occupied_x_r < self.max_x_m)
        in_y = np.abs(occupied_y_r) < self.half_length_y_m

        return bool(np.any(in_x & in_y))
