"""Define a class representing 2D laser scans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotics_utils.spatial import Pose2D


@dataclass(frozen=True)
class LaserScan2D:
    """2D laser scan with range and angle measurements in a reference frame.

    This class represents a 2D slice of a 3D point cloud, filtered by height
    and converted to polar coordinates for use in occupancy grid mapping.
    """

    sensor_pose: Pose2D  # Pose of sensor at scan time (world frame)
    ranges_m: np.ndarray  # Array of (r, θ) values, shape (N, 2)
    range_min_m: float  # Minimum valid range
    range_max_m: float  # Maximum valid range

    @classmethod
    def from_point_cloud_3d(
        cls,
        points_3d: np.ndarray,
        sensor_pose: Pose2D,
        min_height_m: float = 0.1,
        max_height_m: float = 2.0,
        range_min_m: float = 0.5,
        range_max_m: float = 100.0,
    ) -> LaserScan2D:
        """Convert 3D point cloud to 2D laser scan by height filtering.

        :param points_3d: (N, 3) array of (x, y, z) points in sensor frame
        :param sensor_pose: Pose of sensor at scan time (world frame)
        :param min_height_m: Minimum height to include (meters)
        :param max_height_m: Maximum height to include (meters)
        :param range_min_m: Minimum valid range (meters)
        :param range_max_m: Maximum valid range (meters)
        :return: LaserScan2D with filtered and converted points
        """
        if points_3d.shape[0] == 0:
            # No points - return empty scan
            return cls(
                sensor_pose=sensor_pose,
                ranges_m=np.empty((0, 2), dtype=np.float32),
                range_min_m=range_min_m,
                range_max_m=range_max_m,
            )

        # Filter by height (z coordinate in sensor frame)
        z_coords = points_3d[:, 2]
        height_mask = (z_coords >= min_height_m) & (z_coords <= max_height_m)
        filtered_points = points_3d[height_mask]

        if filtered_points.shape[0] == 0:
            # No points after filtering - return empty scan
            return cls(
                sensor_pose=sensor_pose,
                ranges_m=np.empty((0, 2), dtype=np.float32),
                range_min_m=range_min_m,
                range_max_m=range_max_m,
            )

        # Convert to polar coordinates (r, θ) in sensor frame
        # Ignore z coordinate for 2D projection
        x_coords = filtered_points[:, 0]
        y_coords = filtered_points[:, 1]

        ranges = np.sqrt(x_coords**2 + y_coords**2)
        angles = np.arctan2(y_coords, x_coords)

        # Filter by range limits
        range_mask = (ranges >= range_min_m) & (ranges <= range_max_m)
        valid_ranges = ranges[range_mask]
        valid_angles = angles[range_mask]

        # Stack into (N, 2) array
        ranges_array = np.stack([valid_ranges, valid_angles], axis=1).astype(np.float32)

        return cls(
            sensor_pose=sensor_pose,
            ranges_m=ranges_array,
            range_min_m=range_min_m,
            range_max_m=range_max_m,
        )

    def num_points(self) -> int:
        """Get the number of valid scan points."""
        return self.ranges_m.shape[0]
