"""Define a class representing 2D occupancy grids using log-odds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from robotics_utils.geometry import Point2D
from robotics_utils.motion_planning import DiscreteGrid2D, GridCell

if TYPE_CHECKING:
    from robotics_utils.perception.laser_scan import LaserScan2D
    from robotics_utils.spatial import Pose2D


def bresenham_line(c0: GridCell, c1: GridCell) -> list[GridCell]:
    """Compute grid cells along a line using Bresenham's algorithm with integer arithmetic.

    Reference: https://zingl.github.io/bresenham.html ("Line" algorithm)

    :param c0: Start grid cell indices
    :param c1: End grid cell indices
    :return: List of (row, col) cell indices along the line
    """
    x0 = c0.col
    y0 = -c0.row
    x1 = c1.col
    y1 = -c1.row

    cells = []

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    x, y = x0, y0
    while True:
        cells.append(GridCell(-y, x))

        if x == x1 and y == y1:
            break

        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx

        if e2 <= dx:
            err += dx
            y += sy

    return cells


class OccupancyGrid2D:
    """A 2D occupancy grid using log-odds to represent the probability of occupancy."""

    def __init__(self, grid: DiscreteGrid2D) -> None:
        """Initialize an occupancy grid.

        :param grid: Defines the origin, resolution, height, and width of the grid
        """
        self.grid = grid

        # Log-odds representation: L = log( p(occupied) / p(free) )
        # Initialize to log( 0.5 / 0.5 ) = log(1) = 0 (equal probability of occupied and free)
        # Reference: Chapter 4.2 (pg. 94) of Probabilistic Robotics by Thrun, Burgard, and Fox
        self.log_odds = np.zeros((grid.height_cells, grid.width_cells), dtype=np.float32)

    def copy(self) -> OccupancyGrid2D:
        """Create a deep copy of this occupancy grid."""
        grid_copy = DiscreteGrid2D(
            origin=self.grid.origin,
            resolution_m=self.grid.resolution_m,
            width_cells=self.grid.width_cells,
            height_cells=self.grid.height_cells,
            frame_name=self.grid.frame_name,
        )
        occ_grid = OccupancyGrid2D(grid_copy)
        occ_grid.log_odds = np.copy(self.log_odds)
        return occ_grid

    def update(self, scan: LaserScan2D, *, p_free: float = 0.1, p_occupied: float = 0.9) -> None:
        """Update the occupancy grid using an inverse sensor model with ray tracing.

        :param scan: Laser scan to be incorporated into the grid
        :param p_free: Probability that a cell is occupied given a ray passes through it
        :param p_occupied: Probability that a cell is occupied given a laser hits in it
        """
        if not scan.num_points:
            return

        # Convert probabilities into log-odds (see pg. 286 of ProbRob)
        l_free = np.log(p_free / (1 - p_free))
        l_occupied = np.log(p_occupied / (1 - p_occupied))

        sensor_world_x = scan.sensor_pose.x
        sensor_world_y = scan.sensor_pose.y
        sensor_yaw_rad = scan.sensor_pose.yaw_rad

        sensor_grid_cell = self.grid.world_to_cell(Point2D(sensor_world_x, sensor_world_y))

        for i in range(scan.num_points):
            range_m, bearing_rad = scan.beam_data[i]

            beam_yaw_rad = sensor_yaw_rad + bearing_rad  # World-frame yaw of the beam
            end_w_x = sensor_world_x + range_m * np.cos(beam_yaw_rad)  # Endpoint in world frame
            end_w_y = sensor_world_y + range_m * np.sin(beam_yaw_rad)

            end_grid_cell = self.grid.world_to_cell(Point2D(end_w_x, end_w_y))

            # Ray trace from sensor to endpoint
            ray_cells = bresenham_line(sensor_grid_cell, end_grid_cell)

            # Update free-space cells along the ray (excluding the endpoint)
            for cell in ray_cells[:-1]:
                if self.grid.is_valid_cell(cell):
                    self.log_odds[cell.row, cell.col] += l_free

            # Update occupied cell at the endpoint of the beam
            if self.grid.is_valid_cell(end_grid_cell):
                self.log_odds[end_grid_cell.row, end_grid_cell.col] += l_occupied

    def get_occupied_mask(self, p_threshold: float = 0.5) -> np.ndarray:
        """Compute a Boolean mask of occupied cells (occupancy probability > threshold).

        :param p_threshold: Probability threshold for occupancy (0.0 to 1.0)
        :return: Boolean array where True indicates occupied cells
        """
        # Reference: Equation (4.14) on pg. 95 of ProbRob
        p_occupied = 1 - 1 / (1 + np.exp(self.log_odds))
        return p_occupied > p_threshold

    def mask_as_free(self, mask: np.ndarray) -> OccupancyGrid2D:
        """Create a copy of the occupancy grid in which the masked cells are set to free space.

        :param mask: Boolean mask specifying free grid cells
        :return: New OccupancyGrid2D with the masked cells set to free space
        """
        grid_copy = self.copy()
        grid_copy.log_odds[mask] = -10.0  # Use a large (but finite) negative value for stability
        return grid_copy
