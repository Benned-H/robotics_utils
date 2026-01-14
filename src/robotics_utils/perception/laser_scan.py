"""Define a class representing 2D laser scans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from robotics_utils.reconstruction import PointCloud
    from robotics_utils.spatial import Pose2D


@dataclass(frozen=True)
class LaserScan2D:
    """A 2D laser scan with range and angle measurements relative to a known sensor pose."""

    sensor_pose: Pose2D
    """Pose of the sensor at scan time (global frame)."""

    ranges_m: np.ndarray
    """Array of (r, θ) values (ranges in meters; angles in radians) of shape (N, 2)."""

    range_min_m: float
    """Minimum valid range measurement (meters)."""

    range_max_m: float
    """Maximum valid range measurement (meters)."""

    @classmethod
    def from_pointcloud(
        cls,
        pointcloud: PointCloud,
        sensor_pose: Pose2D,
        *,
        min_height_m: float = 0.1,
        max_height_m: float = 2.0,
        range_min_m: float = 0.5,
        range_max_m: float = 100.0,
    ) -> LaserScan2D:
        """Convert a 3D point cloud collected by a laser range finder into a 2D laser scan.

        :param pointcloud: 3D point cloud collected using LiDAR
        :param sensor_pose: Pose of the sensor at scan time (global frame)
        :param min_height_m: Minimum height of included points (meters)
        :param max_height_m: Maximum height of included points (meters)
        :param range_min_m: Minimum valid range (meters)
        :param range_max_m: Maximum valid range (meters)
        :return: Constructed LaserScan2D with filtered and polar-coordinate points
        """
        if not pointcloud.points.shape[0]:
            raise ValueError("Cannot construct a LaserScan2D from an empty point cloud.")

        # Filter by height (z-coordinate w.r.t. sensor)
        z_coords = pointcloud.points[:, 2]
        valid_height_mask = (z_coords >= min_height_m) & (z_coords <= max_height_m)
        filtered_points = pointcloud.points[valid_height_mask]

        if not filtered_points.shape[0]:  # No points after filtering --> return empty scan
            return LaserScan2D(sensor_pose, np.empty((0, 2)), range_min_m, range_max_m)

        # Convert to 2D polar coordinates (r, θ) in the sensor frame (ignore z-coordinate)
        x_coords = filtered_points[:, 0]
        y_coords = filtered_points[:, 1]

        ranges_m = np.sqrt(x_coords**2 + y_coords**2)
        angles_rad = np.arctan2(y_coords, x_coords)

        # Filter by range limits
        valid_range_mask = (ranges_m >= range_min_m) & (ranges_m <= range_max_m)
        valid_ranges = ranges_m[valid_range_mask]
        valid_angles = angles_rad[valid_range_mask]

        ranges_arr = np.stack([valid_ranges, valid_angles], axis=1).astype(np.float32)  # (r, θ)

        return LaserScan2D(
            sensor_pose=sensor_pose,
            ranges_m=ranges_arr,
            range_min_m=range_min_m,
            range_max_m=range_max_m,
        )

    @property
    def num_points(self) -> int:
        """Get the number of valid points in the laser scan."""
        return self.ranges_m.shape[0]
