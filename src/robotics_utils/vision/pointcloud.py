"""Define classes to represent 3D pointclouds and related concepts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.backends.backend_agg import FigureCanvasAgg

from robotics_utils.kinematics import Plane3D
from robotics_utils.visualization import Displayable

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from robotics_utils.vision.cameras import CameraIntrinsics
    from robotics_utils.vision.image_processing import DepthImage, RGBDImage
    from robotics_utils.vision.vlms import InstanceSegmentation


@dataclass(frozen=True)
class PlaneEstimate(Displayable):
    """A plane estimated based on a pointcloud."""

    pointcloud: Pointcloud
    plane: Plane3D
    inlier_indices: list[int]

    def get_inlier_text(self) -> str:
        """Retrieve a text description of the plane estimate's inlier points."""
        n_inliers = len(self.inlier_indices)
        total_points = len(self.pointcloud)
        inlier_ratio = 100 * n_inliers / total_points
        return f"Inliers: {n_inliers}/{total_points} ({inlier_ratio:.1f}%)"

    def project_inliers_onto_plane(self) -> NDArray[np.float64]:
        """Project inlier points onto the plane's 2D coordinate system.

        Each 3D inlier point is projected onto the plane using the plane's basis vectors,
        resulting in 2D coordinates (u, v) relative to the plane center.

        :return: Array of shape (N_inliers, 2) with projected 2D coordinates in meters
        """
        basis = self.plane.find_basis()
        inlier_points = self.pointcloud.points[self.inlier_indices]  # (N_inliers, 3)

        offsets = inlier_points - self.plane.point  # Inliers w.r.t. plane center in 3D
        coords_2d = np.zeros((len(inlier_points), 2))
        coords_2d[:, 0] = np.dot(offsets, basis.u)
        coords_2d[:, 1] = np.dot(offsets, basis.v)

        return coords_2d

    def convert_for_visualization(self) -> NDArray[np.uint8]:
        """Create a 2D visualization by projecting inlier points onto the estimated plane.

        If the pointcloud has RGB color data, the points are colored accordingly.
        Otherwise, all points are displayed in blue.

        :return: BGR image array suitable for display with OpenCV
        """
        coords_2d = self.project_inliers_onto_plane()  # (N_inliers, 2)

        # Get colors for inlier points if available (else default to blue)
        point_colors = (
            self.pointcloud.colors[self.inlier_indices] / 255.0
            if self.pointcloud.colors is not None
            else "blue"
        )

        fig, ax = plt.subplots(figsize=(8, 8))  # TODO: Dynamic default figure size
        ax.scatter(x=coords_2d[:, 0], y=coords_2d[:, 1], s=20, c=point_colors, alpha=0.6)
        ax.set_xlabel("U axis (m)")
        ax.set_ylabel("V axis (m)")
        ax.set_title("2D Projection of Inlier Points onto Estimated Plane")
        ax.grid(visible=True, alpha=0.3)
        ax.set_aspect(aspect="equal", adjustable="box")

        info_text = self.get_inlier_text()
        bbox_props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8}
        ax.text(x=0.02, y=0.98, s=info_text, transform=ax.transAxes, va="top", bbox=bbox_props)

        # Convert the matplotlib figure into a NumPy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        image_rgba = np.asarray(canvas.buffer_rgba())
        plt.close(fig)

        image_rgb = image_rgba[:, :, :3]
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        return image_bgr.astype(np.uint8)


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
        segmentation: InstanceSegmentation,
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

    def fit_plane_ransac(
        self,
        inlier_threshold_m: float = 0.01,
        ransac_n: int = 3,
        iterations: int = 1000,
    ) -> PlaneEstimate | None:
        """Fit a plane to the pointcloud using RANSAC.

        :param inlier_threshold_m: Maximum distance (m) an inlier point can be from the plane
        :param ransac_n: Number of initial points to sample in each iteration of RANSAC
        :param iterations: Number of RANSAC iterations
        :return: Estimated plane, including a list of inlier indices (or None if no plane found)
        """
        if len(self) < ransac_n:  # Cannot fit a plane without N points
            return None

        # Run RANSAC plane segmentation
        plane_model, inlier_indices = self.to_o3d().segment_plane(
            distance_threshold=inlier_threshold_m,
            ransac_n=ransac_n,
            num_iterations=iterations,
        )
        if not inlier_indices:  # Don't trust a plane estimate without any inliers
            return None

        # Extract plane parameters: [a, b, c, d] for ax + by + cz + d = 0
        # Reference: https://www.open3d.org/docs/0.19.0/tutorial/geometry/pointcloud.html#Plane-segmentation
        a, b, c, d = plane_model

        normal = np.array([a, b, c])  # Compute the unit-length normal vector of the plane
        normal = normal / np.linalg.norm(normal)

        # Select a point on the plane using the centroid of the inlier points
        inlier_points = self.points[inlier_indices]
        centroid = inlier_points.mean(axis=0)

        estimated_plane = Plane3D(point=centroid, normal=normal)

        return PlaneEstimate(self, estimated_plane, inlier_indices)
