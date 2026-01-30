"""Define a class representing 2D occupancy grids using log-odds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from robotics_utils.geometry import Point2D
from robotics_utils.motion_planning.discretization import DiscreteGrid2D, GridCell

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from robotics_utils.collision_models import CollisionModelRasterizer
    from robotics_utils.perception.laser_scan import LaserScan2D
    from robotics_utils.states import ObjectKinematicState


def bresenham_line(c0: GridCell, c1: GridCell) -> list[GridCell]:
    """Compute grid cells along a line using Bresenham's algorithm with integer arithmetic.

    Reference: https://zingl.github.io/bresenham.html ("Line" algorithm)

    :param c0: Start grid cell indices
    :param c1: End grid cell indices
    :return: List of (row, col) cell indices along the line
    """
    x0 = c0.col
    y0 = c0.row
    x1 = c1.col
    y1 = c1.row

    cells = []

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    x, y = x0, y0
    while True:
        cells.append(GridCell(y, x))

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


def log_odds(p: float) -> float:
    """Compute the log-odds of the given probability p."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"Invalid probability value: {p} (expected 0 < p < 1).")
    return np.log(p / (1 - p))


OCCUPIED_LOG_ODDS = 10.0  # TODO: It would be nice to derive these from probabilities instead
FREE_SPACE_LOG_ODDS = -10.0


class OccupancyGrid2D:
    """A 2D occupancy grid using log-odds to represent the probability of occupancy."""

    def __init__(self, grid: DiscreteGrid2D, min_obstacle_depth_m: float = 0.1) -> None:
        """Initialize an occupancy grid.

        :param grid: Defines the origin, resolution, height, and width of the grid
        :param min_obstacle_depth_m: Minimum depth (meters) assumed for obstacles when ray-tracing
        """
        self.grid = grid
        self.min_obstacle_depth_m = min_obstacle_depth_m
        """Minimum depth (meters) assumed for any obstacle when ray-tracing."""

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

    def update(
        self,
        scan: LaserScan2D,
        *,
        clearing_scan: LaserScan2D | None = None,
        p_free: float = 0.1,
        p_occupied: float = 0.75,
    ) -> None:
        """Update the occupancy grid using an inverse sensor model with ray tracing.

        :param scan: Laser scan used exclusively to mark occupied cells (all beams)
        :param clearing_scan: Optional scan used for ray-tracing free space (if None, uses `scan`)
        :param p_free: Probability that a cell is occupied given a ray passes through it
        :param p_occupied: Probability that a cell is occupied given a laser hits in it
        """
        if scan.sensor_pose.ref_frame != self.grid.origin.ref_frame:
            raise RuntimeError(
                f"Laser scan is expressed in frame '{scan.sensor_pose.ref_frame}' "
                f"but this occupancy grid is in frame '{self.grid.origin.ref_frame}'.",
            )
        if clearing_scan is not None:
            scan_frame = scan.sensor_pose.ref_frame
            clearing_frame = clearing_scan.sensor_pose.ref_frame
            if scan_frame != clearing_frame:
                raise RuntimeError(
                    f"Occupancy-filling laser scan is expressed in frame '{scan_frame}' "
                    f"but free-space-clearing laser scan is in frame '{clearing_frame}'.",
                )

        # Convert probabilities into log-odds (see pg. 286 of ProbRob)
        l_free = log_odds(p_free)
        l_occupied = log_odds(p_occupied)

        # Use the clearing scan for free-space ray-tracing
        free_scan = clearing_scan if clearing_scan is not None else scan
        free_sensor_w_x = free_scan.sensor_pose.x
        free_sensor_w_y = free_scan.sensor_pose.y
        free_sensor_grid_cell = self.grid.world_to_cell(Point2D(free_sensor_w_x, free_sensor_w_y))

        for i in range(free_scan.num_beams):
            range_m, bearing_rad = free_scan.beam_data[i]
            beam_yaw_rad = free_scan.sensor_pose.yaw_rad + bearing_rad
            end_w_x = free_sensor_w_x + range_m * np.cos(beam_yaw_rad)
            end_w_y = free_sensor_w_y + range_m * np.sin(beam_yaw_rad)
            end_grid_cell = self.grid.world_to_cell(Point2D(end_w_x, end_w_y))

            # Ray trace from sensor to endpoint
            ray_cells = bresenham_line(free_sensor_grid_cell, end_grid_cell)

            # Update free-space cells along the ray (excluding the endpoint)
            for cell in ray_cells[:-1]:
                if self.grid.is_valid_cell(cell):
                    self.log_odds[cell.row, cell.col] += l_free

        # Use all beam "hits" to mark occupied cells
        for i in range(scan.num_beams):
            range_m, bearing_rad = scan.beam_data[i]
            beam_yaw_rad = scan.sensor_pose.yaw_rad + bearing_rad  # World-frame yaw of the beam
            cos_yaw = np.cos(beam_yaw_rad)
            sin_yaw = np.sin(beam_yaw_rad)
            end_w_x = scan.sensor_pose.x + range_m * cos_yaw  # Endpoint in world frame
            end_w_y = scan.sensor_pose.y + range_m * sin_yaw
            end_grid_cell = self.grid.world_to_cell(Point2D(end_w_x, end_w_y))

            # Ray trace past the endpoint by the minimum depth of any obstacle
            past_end_x = end_w_x + self.min_obstacle_depth_m * cos_yaw
            past_end_y = end_w_y + self.min_obstacle_depth_m * sin_yaw
            past_end_cell = self.grid.world_to_cell(Point2D(past_end_x, past_end_y))

            obstacle_cells = bresenham_line(end_grid_cell, past_end_cell)
            for cell in obstacle_cells:
                if self.grid.is_valid_cell(cell):
                    self.log_odds[cell.row, cell.col] += l_occupied

    def get_occupied_mask(self, p_threshold: float = 0.5) -> NDArray[np.bool]:
        """Compute a Boolean mask of occupied cells (occupancy probability > threshold).

        :param p_threshold: Probability threshold for occupancy (0.0 to 1.0)
        :return: Boolean array where True indicates occupied cells
        """
        # Reference: Equation (4.14) on pg. 95 of ProbRob
        p_occupied = 1 - 1 / (1 + np.exp(self.log_odds))
        return p_occupied > p_threshold

    def mask_as_free(self, mask: NDArray[np.bool]) -> OccupancyGrid2D:
        """Create a copy of the occupancy grid in which the masked cells are set to free space.

        :param mask: Boolean mask specifying free grid cells
        :return: New OccupancyGrid2D with the masked cells set to free space
        """
        grid_copy = self.copy()
        grid_copy.log_odds[mask] = FREE_SPACE_LOG_ODDS
        return grid_copy

    def mask_as_occupied(self, mask: NDArray[np.bool]) -> OccupancyGrid2D:
        """Create a copy of the occupancy grid in which the masked cells are set as occupied.

        :param mask: Boolean mask specifying occupied grid cells
        :return: New OccupancyGrid2D with the masked cells set as occupied
        """
        grid_copy = self.copy()
        grid_copy.log_odds[mask] = OCCUPIED_LOG_ODDS
        return grid_copy

    def stamp_as_free(
        self,
        obj: ObjectKinematicState,
        rasterizer: CollisionModelRasterizer,
    ) -> OccupancyGrid2D:
        """Rasterize an object into an occupancy mask and stamp it as free space in a grid copy.

        :param obj: Kinematic state of the object (pose + collision model)
        :param rasterizer: Rasterizes the object's collision model into a 2D occupancy mask
        :return: Copy of the occupancy grid where the object's mask is marked as free space
        """
        mask = rasterizer.rasterize_object(obj=obj, grid=self.grid)
        return self.mask_as_free(mask)

    def stamp_as_occupied(
        self,
        obj: ObjectKinematicState,
        rasterizer: CollisionModelRasterizer,
    ) -> OccupancyGrid2D:
        """Rasterize an object into an occupancy mask and stamp it as occupied in a grid copy.

        :param obj: Kinematic state of the object (pose + collision model)
        :param rasterizer: Rasterizes the object's collision model into a 2D occupancy mask
        :return: Copy of the occupancy grid where the object's mask is marked as occupied
        """
        mask = rasterizer.rasterize_object(obj=obj, grid=self.grid)
        return self.mask_as_occupied(mask)
