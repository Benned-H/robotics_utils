"""Define a class representing 2D occupancy grids using log-odds."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from robotics_utils.geometry import Point2D
from robotics_utils.spatial import DEFAULT_FRAME, Pose2D
from robotics_utils.states import ObjectKinematicState

from .laser_scan import LaserScan2D


class OccupancyGrid2D:
    """2D occupancy grid using log-odds representation for probabilistic mapping.

    The grid maintains a probabilistic representation of occupied vs. free space
    using log-odds values that are updated incrementally from laser scans using
    an inverse sensor model with ray tracing.
    """

    def __init__(
        self,
        origin: Pose2D,
        resolution_m: float,
        width_cells: int,
        height_cells: int,
        ref_frame: str = DEFAULT_FRAME,
    ):
        """Initialize an occupancy grid.

        :param origin: Origin pose of the grid in the reference frame (lower-left corner)
        :param resolution_m: Cell size in meters
        :param width_cells: Grid width in cells
        :param height_cells: Grid height in cells
        :param ref_frame: Reference frame name (default: "map")
        """
        self.origin = origin
        self.resolution_m = resolution_m
        self.width_cells = width_cells
        self.height_cells = height_cells
        self.ref_frame = ref_frame

        # Log-odds representation: L = log(P(occupied) / P(free))
        # Initialize to 0 (equal probability of occupied and free)
        self.log_odds: np.ndarray = np.zeros((height_cells, width_cells), dtype=np.float32)

    def update(
        self,
        scan: LaserScan2D,
        prob_occupied: float = 0.7,
        prob_free: float = 0.3,
    ) -> None:
        """Update occupancy grid using inverse sensor model with ray tracing.

        :param scan: Laser scan to incorporate into the grid
        :param prob_occupied: Probability of occupancy at hit points
        :param prob_free: Probability of free space along rays
        """
        if scan.num_points() == 0:
            return

        # Convert probabilities to log-odds updates
        log_odds_occupied = np.log(prob_occupied / (1 - prob_occupied))
        log_odds_free = np.log(prob_free / (1 - prob_free))

        # Get sensor position in grid coordinates
        sensor_world_x = scan.sensor_pose.x
        sensor_world_y = scan.sensor_pose.y
        sensor_yaw = scan.sensor_pose.yaw_rad

        sensor_grid_x, sensor_grid_y = self.world_to_grid(
            Point2D(sensor_world_x, sensor_world_y)
        )

        # Process each scan point
        for i in range(scan.num_points()):
            range_m, angle_rad = scan.ranges_m[i]

            # Convert from sensor frame to world frame
            angle_world = angle_rad + sensor_yaw
            endpoint_world_x = sensor_world_x + range_m * np.cos(angle_world)
            endpoint_world_y = sensor_world_y + range_m * np.sin(angle_world)

            # Convert endpoint to grid coordinates
            endpoint_grid_x, endpoint_grid_y = self.world_to_grid(
                Point2D(endpoint_world_x, endpoint_world_y)
            )

            # Check if endpoint is within grid bounds
            if not self._is_valid_cell(endpoint_grid_x, endpoint_grid_y):
                continue

            # Ray trace from sensor to endpoint using Bresenham's algorithm
            ray_cells = self._bresenham_line(
                sensor_grid_x, sensor_grid_y, endpoint_grid_x, endpoint_grid_y
            )

            # Update free space along the ray (all cells except the endpoint)
            for cell_x, cell_y in ray_cells[:-1]:
                if self._is_valid_cell(cell_x, cell_y):
                    self.log_odds[cell_y, cell_x] += log_odds_free

            # Update occupied space at endpoint
            if self._is_valid_cell(endpoint_grid_x, endpoint_grid_y):
                self.log_odds[endpoint_grid_y, endpoint_grid_x] += log_odds_occupied

    def get_occupied_mask(self, threshold: float = 0.5) -> np.ndarray:
        """Get Boolean mask of occupied cells (probability > threshold).

        :param threshold: Probability threshold for occupancy (0.0 to 1.0)
        :return: Boolean array where True indicates occupied cells
        """
        # Convert log-odds to probability: p = 1 / (1 + exp(-L))
        # Equivalently: p = sigmoid(L)
        probabilities = 1.0 / (1.0 + np.exp(-self.log_odds))
        return probabilities > threshold

    def remove_object(
        self,
        obj: ObjectKinematicState,
        min_height_m: float,
        max_height_m: float,
    ) -> OccupancyGrid2D:
        """Return copy of grid with object's footprint removed (set to free space).

        :param obj: Object to remove from the grid
        :param min_height_m: Minimum height for rasterization
        :param max_height_m: Maximum height for rasterization
        :return: New OccupancyGrid2D with object removed
        """
        # Import here to avoid circular dependency
        from .rasterizer import CollisionModelRasterizer

        # Create a copy of this grid
        grid_copy = self.copy()

        # Rasterize the object's collision model to a Boolean mask
        object_mask = CollisionModelRasterizer.rasterize_object(
            obj=obj,
            resolution_m=self.resolution_m,
            origin=self.origin,
            grid_shape=(self.height_cells, self.width_cells),
            min_height_m=min_height_m,
            max_height_m=max_height_m,
        )

        # Set cells within the object footprint to free space (log-odds = -inf)
        # Using a large negative value instead of -inf for numerical stability
        grid_copy.log_odds[object_mask] = -10.0

        return grid_copy

    def world_to_grid(self, point: Point2D) -> tuple[int, int]:
        """Convert world coordinates to grid cell indices.

        :param point: Point in world frame
        :return: (grid_x, grid_y) cell indices
        """
        # Transform point relative to grid origin
        dx = point.x - self.origin.x
        dy = point.y - self.origin.y

        # Rotate by -origin.yaw to align with grid axes
        cos_yaw = np.cos(-self.origin.yaw_rad)
        sin_yaw = np.sin(-self.origin.yaw_rad)

        local_x = dx * cos_yaw - dy * sin_yaw
        local_y = dx * sin_yaw + dy * cos_yaw

        # Convert to grid indices
        grid_x = int(np.floor(local_x / self.resolution_m))
        grid_y = int(np.floor(local_y / self.resolution_m))

        return grid_x, grid_y

    def grid_to_world(self, grid_x: int, grid_y: int) -> Point2D:
        """Convert grid cell indices to world coordinates (cell center).

        :param grid_x: Grid x index
        :param grid_y: Grid y index
        :return: Point in world frame at cell center
        """
        # Cell center in local grid frame
        local_x = (grid_x + 0.5) * self.resolution_m
        local_y = (grid_y + 0.5) * self.resolution_m

        # Rotate by origin.yaw to align with world frame
        cos_yaw = np.cos(self.origin.yaw_rad)
        sin_yaw = np.sin(self.origin.yaw_rad)

        rotated_x = local_x * cos_yaw - local_y * sin_yaw
        rotated_y = local_x * sin_yaw + local_y * cos_yaw

        # Translate to world frame
        world_x = self.origin.x + rotated_x
        world_y = self.origin.y + rotated_y

        return Point2D(world_x, world_y)

    def copy(self) -> OccupancyGrid2D:
        """Create a deep copy of this occupancy grid."""
        grid_copy = OccupancyGrid2D(
            origin=self.origin,
            resolution_m=self.resolution_m,
            width_cells=self.width_cells,
            height_cells=self.height_cells,
            ref_frame=self.ref_frame,
        )
        grid_copy.log_odds = np.copy(self.log_odds)
        return grid_copy

    def _is_valid_cell(self, grid_x: int, grid_y: int) -> bool:
        """Check if grid cell indices are within bounds."""
        return 0 <= grid_x < self.width_cells and 0 <= grid_y < self.height_cells

    def _bresenham_line(
        self, x0: int, y0: int, x1: int, y1: int
    ) -> list[tuple[int, int]]:
        """Compute cells along a line using Bresenham's algorithm.

        :param x0: Start x coordinate
        :param y0: Start y coordinate
        :param x1: End x coordinate
        :param y1: End y coordinate
        :return: List of (x, y) cell coordinates along the line
        """
        cells = []

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            cells.append((x, y))

            if x == x1 and y == y1:
                break

            e2 = 2 * err

            if e2 > -dy:
                err -= dy
                x += sx

            if e2 < dx:
                err += dx
                y += sy

        return cells
