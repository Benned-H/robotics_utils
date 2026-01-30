"""Define a class representing 2D laser scans."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from robotics_utils.perception.pointcloud import PointCloud
    from robotics_utils.spatial import Pose2D


class LaserScan2D:
    """A 2D laser scan with range and angle measurements relative to a known sensor pose."""

    def __init__(
        self,
        sensor_pose: Pose2D,
        beam_data: NDArray[np.floating],
        range_min_m: float,
        range_max_m: float,
    ) -> None:
        """Initialize the laser scan by applying the given range limits.

        :param sensor_pose: Pose of the sensor at scan time (global frame)
        :param beam_data: Array of (r, θ) values (in meters; in radians) of shape (N, 2)
        :param range_min_m: Minimum valid range measurement (meters)
        :param range_max_m: Maximum valid range measurement (meters)
        """
        self.sensor_pose = sensor_pose
        """Pose of the sensor at scan time (global frame)."""

        # Filter beams using the provided range limits
        ranges_m = beam_data[:, 0]
        valid_range_mask = (ranges_m >= range_min_m) & (ranges_m <= range_max_m)

        self.beam_data = beam_data[valid_range_mask]
        """Array of (r, θ) values (ranges in meters; angles in radians) of shape (N, 2)."""

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
        beam_data = np.stack([ranges_m, angles_rad], axis=1).astype(np.float32)  # (r, θ)

        return LaserScan2D(
            sensor_pose=sensor_pose,
            beam_data=beam_data,
            range_min_m=range_min_m,
            range_max_m=range_max_m,
        )

    @property
    def num_beams(self) -> int:
        """Get the number of beams in the laser scan."""
        return self.beam_data.shape[0]

    def filter_per_angle_bin(self, angle_resolution_rad: float | None = None) -> LaserScan2D:
        """Filter the laser scan to keep only the minimum-range beam per angular bin.

        :param angle_resolution_rad: Optional angular bin width (radians) (default: 1 degree)
        :return: New LaserScan2D with only the minimum-range beam per angle bin
        """
        if angle_resolution_rad is None:
            angle_resolution_rad = np.deg2rad(1.0)

        if not self.num_beams:
            return LaserScan2D(self.sensor_pose, np.empty((0, 2)), 0.0, float("inf"))

        ranges_m = self.beam_data[:, 0]
        angles_rad = self.beam_data[:, 1]

        # Discretize the angle values to create bins
        bin_indices = np.floor(angles_rad / angle_resolution_rad).astype(np.int32)

        # Find the minimum-range index for each unique bin
        unique_bins, inverse_indices = np.unique(bin_indices, return_inverse=True)
        num_bins = len(unique_bins)

        min_ranges_m = np.full(num_bins, np.inf)  # Initialize to infinity, which will get replaced
        min_indices = np.zeros(num_bins, dtype=np.int64)  # Index of each bin's minimum-range beam

        for beam_i, (range_m, bin_idx) in enumerate(zip(ranges_m, inverse_indices)):
            if range_m < min_ranges_m[bin_idx]:
                min_ranges_m[bin_idx] = range_m
                min_indices[bin_idx] = beam_i

        filtered_beam_data = self.beam_data[min_indices]

        return LaserScan2D(
            sensor_pose=self.sensor_pose,
            beam_data=filtered_beam_data,
            range_min_m=0.0,
            range_max_m=float("inf"),  # Use permissive range limits to skip re-filtering
        )
