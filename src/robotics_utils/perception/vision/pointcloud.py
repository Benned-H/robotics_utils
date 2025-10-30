"""Define a class to represent 3D pointclouds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from robotics_utils.perception.vision.cameras import CameraIntrinsics
    from robotics_utils.perception.vision.images import DepthImage


class Pointcloud:
    """A pointcloud of 3D points."""

    def __init__(self, points: NDArray[np.float64]) -> None:
        """Initialize the pointcloud using a NumPy array of shape (N, 3)."""
        if len(points.shape) != 2 or points.shape[1] != 3:
            raise ValueError(f"Pointcloud expects an array of shape (N, 3), got {points.shape}")

        self.points = points

    def __len__(self) -> int:
        """Retrieve the length of (i.e., number of points in) the pointcloud."""
        return self.points.shape[0]

    @classmethod
    def from_depth_image(cls, depth: DepthImage, intrinsics: CameraIntrinsics) -> Pointcloud:
        """Construct a pointcloud from the given depth image.

        Note that zero-depth pixels are filtered out during this conversion.

        Reference: https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html
            See the equations in the description of the create_from_rgbd_image() function.

        :param depth: Depth image of shape (H, W)
        :param intrinsics: Camera intrinsic parameters
        :return: Pointcloud consisting of 3D points represented as an array of shape (N, 3)
        """
        z = depth.data  # (H, W)
        v, u = np.indices((depth.height, depth.width))  # V gives row indices and U gives columns
        x = (u - intrinsics.x0) * z / intrinsics.fx  # (H, W)
        y = (v - intrinsics.y0) * z / intrinsics.fy  # (H, W)
        pointmap = np.stack([x, y, z], axis=-1)  # Stack along last axis to get (H, W, 3)

        assert pointmap.shape == (depth.height, depth.width, 3)  # Sanity-check dimensions

        nonzero_points = pointmap[depth.data > 0]  # Filter out zero-depth pixels, giving (N, 3)

        return Pointcloud(nonzero_points)
