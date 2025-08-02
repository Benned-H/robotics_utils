"""Define classes to represent and visualize pointclouds."""

from __future__ import annotations

from types import TracebackType

import numpy as np
import open3d as o3d
from numpy.typing import NDArray
from typing_extensions import Self

from robotics_utils.sensors.cameras import CameraIntrinsics
from robotics_utils.vision.images import DepthImage


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


class PointcloudVisualizer:
    """A manager context for live visualization of pointclouds using Open3D."""

    def __init__(self) -> None:
        """Initialize an Open3D visualizer for pointclouds."""
        self.vis = o3d.visualization.Visualizer()
        self.o3d_pcd = o3d.geometry.PointCloud()

        self.uncalled_while_active: bool | None = None
        """None = Visualizer inactive; True = Uncalled but live; False = Active and been called."""

    def __enter__(self) -> Self:
        """Enter a managed context for live pointcloud visualization."""
        self.vis.create_window()
        self.uncalled_while_active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        """Exit a managed context for live pointcloud visualization.

        :param exc_type: Type of exception raised in the context (None if no exception)
        :param exc_value: Value of the exception raised in the context (None if no exception)
        :param traceback: Traceback of the exception raised in the context (None if no exception)
        :return: True if exception is suppressed, False if exception should propagate, else None
        """
        self.vis.destroy_window()
        self.uncalled_while_active = None
        return None

    def visualize(self, pointcloud: Pointcloud) -> None:
        """Visualize the given pointcloud if the visualizer is active."""
        if self.uncalled_while_active is None:
            return  # Exit if the visualizer isn't active

        self.o3d_pcd.points = o3d.utility.Vector3dVector(pointcloud.points)

        if self.uncalled_while_active:
            self.vis.add_geometry(self.o3d_pcd)
            self.uncalled_while_active = False
        else:
            self.vis.update_geometry(self.o3d_pcd)

        self.vis.poll_events()
        self.vis.update_renderer()
