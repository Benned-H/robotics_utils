"""Define a class representing robot footprints for collision checking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from robotics_utils.geometry import Point2D
from robotics_utils.spatial import Pose2D


@dataclass(frozen=True)
class Footprint:
    """Robot footprint for collision checking during path planning.

    The footprint is defined as a polygon with vertices specified in the robot's
    local frame. During collision checking, the footprint is transformed to the
    world frame and checked against the occupancy grid.
    """

    vertices: list[Point2D]  # Vertices in robot frame (e.g., rectangular)

    @classmethod
    def spot_footprint(cls) -> Footprint:
        """Create Spot's rectangular footprint (1.1m × 0.5m).

        Spot's body is approximately 1.1m long (x-axis) and 0.5m wide (y-axis).
        The footprint is centered at the origin of the robot's body frame.

        :return: Footprint representing Spot's body
        """
        # Define 4 corners of rectangle centered at origin
        half_length = 1.1 / 2.0  # 0.55m
        half_width = 0.5 / 2.0  # 0.25m

        vertices = [
            Point2D(half_length, half_width),  # Front-right
            Point2D(half_length, -half_width),  # Front-left
            Point2D(-half_length, -half_width),  # Rear-left
            Point2D(-half_length, half_width),  # Rear-right
        ]

        return cls(vertices=vertices)

    def check_collision(
        self,
        pose: Pose2D,
        occupancy_mask: np.ndarray,
        origin: Pose2D,
        resolution_m: float,
    ) -> bool:
        """Check if footprint at given pose collides with occupied cells.

        :param pose: Robot pose in world frame
        :param occupancy_mask: Boolean grid where True indicates occupied cells
        :param origin: Origin pose of the occupancy grid
        :param resolution_m: Grid resolution in meters
        :return: True if collision detected, False otherwise
        """
        # Transform vertices to world frame
        world_vertices = []
        cos_yaw = np.cos(pose.yaw_rad)
        sin_yaw = np.sin(pose.yaw_rad)

        for vertex in self.vertices:
            # Rotate vertex by robot yaw
            rotated_x = vertex.x * cos_yaw - vertex.y * sin_yaw
            rotated_y = vertex.x * sin_yaw + vertex.y * cos_yaw

            # Translate to robot position
            world_x = pose.x + rotated_x
            world_y = pose.y + rotated_y

            world_vertices.append(Point2D(world_x, world_y))

        # Convert vertices to grid coordinates
        grid_vertices = []
        for vertex in world_vertices:
            grid_x, grid_y = self._world_to_grid(vertex, origin, resolution_m)
            grid_vertices.append((grid_x, grid_y))

        # Check if any cells within the footprint polygon are occupied
        # Use a simple rasterization approach: check all cells in the bounding box
        # and test if they're inside the polygon

        # Get bounding box of footprint in grid coordinates
        grid_xs = [v[0] for v in grid_vertices]
        grid_ys = [v[1] for v in grid_vertices]

        min_x = int(np.floor(min(grid_xs)))
        max_x = int(np.ceil(max(grid_xs)))
        min_y = int(np.floor(min(grid_ys)))
        max_y = int(np.ceil(max(grid_ys)))

        height, width = occupancy_mask.shape

        # Check all cells in bounding box
        for grid_y in range(max(0, min_y), min(height, max_y + 1)):
            for grid_x in range(max(0, min_x), min(width, max_x + 1)):
                # Check if cell is inside the footprint polygon
                if self._point_in_polygon(grid_x, grid_y, grid_vertices):
                    # Check if cell is occupied
                    if occupancy_mask[grid_y, grid_x]:
                        return True  # Collision detected

        return False  # No collision

    @staticmethod
    def _world_to_grid(
        point: Point2D, origin: Pose2D, resolution_m: float
    ) -> tuple[float, float]:
        """Convert world coordinates to grid coordinates (continuous).

        :param point: Point in world frame
        :param origin: Grid origin pose
        :param resolution_m: Grid resolution in meters
        :return: (grid_x, grid_y) continuous coordinates
        """
        # Transform point relative to grid origin
        dx = point.x - origin.x
        dy = point.y - origin.y

        # Rotate by -origin.yaw to align with grid axes
        cos_yaw = np.cos(-origin.yaw_rad)
        sin_yaw = np.sin(-origin.yaw_rad)

        local_x = dx * cos_yaw - dy * sin_yaw
        local_y = dx * sin_yaw + dy * cos_yaw

        # Convert to grid coordinates (continuous, not discrete)
        grid_x = local_x / resolution_m
        grid_y = local_y / resolution_m

        return grid_x, grid_y

    @staticmethod
    def _point_in_polygon(
        px: float, py: float, polygon: list[tuple[float, float]]
    ) -> bool:
        """Check if point is inside polygon using ray casting algorithm.

        :param px: Point x coordinate
        :param py: Point y coordinate
        :param polygon: List of (x, y) polygon vertices
        :return: True if point is inside polygon
        """
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]

            if py > min(p1y, p2y):
                if py <= max(p1y, p2y):
                    if px <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x

                        if p1x == p2x or px <= xinters:
                            inside = not inside

            p1x, p1y = p2x, p2y

        return inside
