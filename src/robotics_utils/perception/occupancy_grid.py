"""Define a class representing 2D occupancy grids using log-odds."""

from __future__ import annotations

import numpy as np

from robotics_utils.geometry import Point2D
from robotics_utils.perception.laser_scan import LaserScan2D
from robotics_utils.spatial import Pose2D


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Compute grid cells along a line using Bresenham's algorithm with integer arithmetic.

    Reference: https://zingl.github.io/bresenham.html ("Line" algorithm)

    :param x0: Start x-coordinate
    :param y0: Start y-coordinate
    :param x1: End x-coordinate
    :param y1: End y-coordinate
    :return: List of (x, y) cell indices along the line
    """
    cells = []

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    x, y = x0, y0
    while True:
        cells.append((x, y))

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

    def __init__(
        self,
        origin: Pose2D,
        resolution_m: float,
        width_cells: int,
        height_cells: int,
    ) -> None:
        """Initialize an occupancy grid.

        :param origin: Origin pose of the grid in a global frame (lower-left corner)
        :param resolution_m: Size of cells in the grid (meters)
        :param width_cells: Grid width in cells (along x-axis)
        :param height_cells: Grid height in cells (along y-axis)
        """
        self.origin = origin  # pose_w_g (grid. w.r.t. world)
        self.resolution_m = resolution_m
        self.width_cells = width_cells
        self.height_cells = height_cells

        # Log-odds representation: L = log( p(occupied) / p(free) )
        # Initialize to log( 0.5 / 0.5 ) = log(1) = 0 (equal probability of occupied and free)
        # Reference: Chapter 4.2 (pg. 94) of Probabilistic Robotics by Thrun, Burgard, and Fox
        self.log_odds = np.zeros((height_cells, width_cells), dtype=np.float32)

    def copy(self) -> OccupancyGrid2D:
        """Create a deep copy of this occupancy grid."""
        grid = OccupancyGrid2D(
            origin=self.origin,
            resolution_m=self.resolution_m,
            width_cells=self.width_cells,
            height_cells=self.height_cells,
        )
        grid.log_odds = np.copy(self.log_odds)
        return grid

    def world_to_grid(self, world_x: float, world_y: float) -> tuple[int, int]:
        """Convert a point in the world frame to grid cell indices.

        :param world_x: x-coordinate in the world frame
        :param world_y: y-coordinate in the world frame
        :return: (grid_x, grid_y) cell indices
        """
        dx_w = world_x - self.origin.x  # Point relative to grid origin expressed in world frame
        dy_w = world_y - self.origin.y

        cos_yaw = np.cos(-self.origin.yaw_rad)  # Rotate by -origin.yaw to convert to grid frame
        sin_yaw = np.sin(-self.origin.yaw_rad)

        local_x = dx_w * cos_yaw - dy_w * sin_yaw  # Position in grid frame
        local_y = dx_w * sin_yaw + dy_w * cos_yaw

        grid_x = int(np.floor(local_x / self.resolution_m))  # Convert to grid indices
        grid_y = int(np.floor(local_y / self.resolution_m))

        return grid_x, grid_y

    def grid_to_world(self, grid_x: int, grid_y: int) -> Point2D:
        """Convert grid cell indices to a position in the world frame at the cell center.

        :param grid_x: Grid x index
        :param grid_y: Grid y index
        :return: Point in world frame at cell center
        """
        local_x = (grid_x + 0.5) * self.resolution_m  # Cell center in grid frame
        local_y = (grid_y + 0.5) * self.resolution_m

        cos_yaw = np.cos(self.origin.yaw_rad)  # Rotate by origin.yaw to align with world frame
        sin_yaw = np.sin(self.origin.yaw_rad)

        rotated_x = local_x * cos_yaw - local_y * sin_yaw
        rotated_y = local_x * sin_yaw + local_y * cos_yaw

        world_x = self.origin.x + rotated_x  # Translate to world frame
        world_y = self.origin.y + rotated_y

        return Point2D(world_x, world_y)

    def is_valid_cell(self, grid_x: int, grid_y: int) -> bool:
        """Check whether the given cell coordinate is within the grid."""
        return 0 <= grid_x < self.width_cells and 0 <= grid_y < self.height_cells

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

        sensor_grid_x, sensor_grid_y = self.world_to_grid(sensor_world_x, sensor_world_y)

        for i in range(scan.num_points):
            range_m, bearing_rad = scan.ranges_m[i]

            beam_yaw_rad = sensor_yaw_rad + bearing_rad  # World-frame yaw of the beam
            end_w_x = sensor_world_x + range_m * np.cos(beam_yaw_rad)  # Endpoint in world frame
            end_w_y = sensor_world_y + range_m * np.sin(beam_yaw_rad)

            end_grid_x, end_grid_y = self.world_to_grid(end_w_x, end_w_y)

            # Ray trace from sensor to endpoint
            ray_cells = bresenham_line(sensor_grid_x, sensor_grid_y, end_grid_x, end_grid_y)

            # Update free-space cells along the ray (excluding the endpoint)
            for cell_x, cell_y in ray_cells[:-1]:
                if self.is_valid_cell(cell_x, cell_y):
                    self.log_odds[cell_x, cell_y] += l_free

            # Update occupied cell at the endpoint of the beam
            if self.is_valid_cell(end_grid_x, end_grid_y):
                self.log_odds[end_grid_x, end_grid_y] += l_occupied

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
