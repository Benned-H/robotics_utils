"""Define a class to represent 3D pointclouds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import open3d as o3d

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from robotics_utils.vision import DepthImage, RGBDImage
    from robotics_utils.vision.cameras import CameraIntrinsics
    from robotics_utils.vision.vlms import ObjectSegmentation


class Pointcloud:
    """A pointcloud of 3D points."""

    def __init__(
        self,
        points: NDArray[np.float64],
        colors: NDArray[np.uint8] | None = None,
    ) -> None:
        """Initialize the pointcloud using a NumPy array of shape (N, 3).

        :param points: Array containing 3D point data
        :param colors: Optional array containing point colors of shape (N, 3)
        """
        if len(points.shape) != 2 or points.shape[1] != 3:
            raise ValueError(f"Pointcloud expects an array of shape (N, 3), got {points.shape}")

        self.points = points
        """Points in the pointcloud; shape (N, 3)."""

        self.colors = colors
        """Optional RGB colors (0-255) of each point in the pointcloud; shape (N, 3)."""

    def __len__(self) -> int:
        """Retrieve the length of (i.e., number of points in) the pointcloud."""
        return self.points.shape[0]

    @classmethod
    def from_depth_image(cls, depth: DepthImage, depth_intrinsics: CameraIntrinsics) -> Pointcloud:
        """Construct a pointcloud from the given depth image.

        Reference: https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html
            See the equations in the description of the create_from_rgbd_image() function.

        :param depth: Depth image of shape (H, W)
        :param depth_intrinsics: Depth camera intrinsic parameters
        :return: Pointcloud consisting of 3D points represented as an array of shape (N, 3)
        """
        z = depth.data  # (H, W)
        v, u = np.indices((depth.height, depth.width))  # V gives row indices and U gives columns
        x = (u - depth_intrinsics.x0) * z / depth_intrinsics.fx  # (H, W)
        y = (v - depth_intrinsics.y0) * z / depth_intrinsics.fy  # (H, W)
        pointmap = np.stack([x, y, z], axis=-1)  # Stack along last axis to get (H, W, 3)

        assert pointmap.shape == (depth.height, depth.width, 3)  # Sanity-check dimensions

        return Pointcloud(pointmap.reshape(-1, 3, order="C"))  # Convert to (N, 3)

    @classmethod
    def from_rgbd_image(cls, rgbd: RGBDImage, depth_intrinsics: CameraIntrinsics) -> Pointcloud:
        """Construct a pointcloud from the given RGB-D image.

        :param rgbd: RGB-D image with RGB and depth data
        :param depth_intrinsics: Depth camera intrinsic parameters
        :return: Pointcloud consisting of 3D points with associated RGB colors
        :raises ValueError: If the provided RGB and depth images have different dimensions (W x H)
        """
        if not rgbd.same_dimensions:
            raise ValueError("Cannot construct Pointcloud; RGB and depth images differ in size.")

        cloud = Pointcloud.from_depth_image(rgbd.depth, depth_intrinsics)
        cloud.colors = rgbd.rgb.data.reshape(-1, 3, order="C")
        return cloud

    @classmethod
    def from_segmented_rgbd(
        cls,
        rgbd: RGBDImage,
        depth_intrinsics: CameraIntrinsics,
        segmentation: ObjectSegmentation,
    ) -> Pointcloud:
        """Construct a pointcloud from an object instance segmentation in the given RGB-D image.

        :param rgbd: RGB-D image with RGB and depth data
        :param depth_intrinsics: Depth camera intrinsic parameters
        :param segmentation: Object instance segmentation including a pixel mask
        :return: Pointcloud consisting of masked 3D points with associated RGB colors
        """
        if not rgbd.same_dimensions:
            raise ValueError("Cannot construct Pointcloud; RGB and depth images differ in size.")

        # Get pixel coordinates where the mask is True
        mask_coords = np.argwhere(segmentation.mask)  # (N, 2)
        v = mask_coords[:, 0]  # Row indices for the masked pixels
        u = mask_coords[:, 1]  # Column indices for the masked pixels

        # Extract RGB and depth values at masked pixels
        rgb_values = rgbd.rgb.data[v, u]  # (N, 3)
        z = rgbd.depth.data[v, u]  # (N,)

        # Same math as in Pointcloud.from_depth_image
        x = (u - depth_intrinsics.x0) * z / depth_intrinsics.fx  # (N,)
        y = (v - depth_intrinsics.y0) * z / depth_intrinsics.fy  # (N,)

        # Stack along last axis to get (N, 3)
        points_xyz = np.stack([x, y, z], axis=-1).astype(np.float64)

        return Pointcloud(points=points_xyz, colors=rgb_values)

    def to_o3d(self) -> o3d.geometry.PointCloud:
        """Convert the pointcloud into an Open3D pointcloud."""
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(self.points)
        if self.colors is not None:
            o3d_pcd.colors = o3d.utility.Vector3dVector(self.colors / 255.0)
        return o3d_pcd
