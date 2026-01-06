"""Define a class to represent and compute plane estimates from 3D pointclouds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from robotics_utils.geometry import Plane3D
from robotics_utils.visualization import Displayable

if TYPE_CHECKING:
    from robotics_utils.reconstruction.pointcloud import Pointcloud


@dataclass(frozen=True)
class PlaneEstimate(Displayable):
    """A plane estimated based on a pointcloud."""

    pointcloud: Pointcloud
    plane: Plane3D
    inlier_indices: list[int]

    @classmethod
    def fit_plane_ransac(
        cls,
        pointcloud: Pointcloud,
        inlier_threshold_m: float = 0.01,
        ransac_n: int = 3,
        iterations: int = 1000,
    ) -> PlaneEstimate | None:
        """Fit a plane to the given pointcloud using RANSAC.

        :param pointcloud: 3D pointcloud used to estimate a 3D plane
        :param inlier_threshold_m: Maximum distance (m) an inlier point can be from the plane
        :param ransac_n: Number of initial points to sample in each iteration of RANSAC
        :param iterations: Number of RANSAC iterations
        :return: Estimated plane, including a list of inlier indices (or None if no plane found)
        """
        if len(pointcloud) < ransac_n:  # Cannot fit a plane without N points
            return None

        # Run RANSAC plane segmentation
        plane_model, inlier_indices = pointcloud.to_o3d().segment_plane(
            distance_threshold=inlier_threshold_m,
            ransac_n=ransac_n,
            num_iterations=iterations,
        )
        if not inlier_indices:  # Don't trust a plane estimate without any inliers
            return None

        # Extract plane parameters: [a, b, c, d] for ax + by + cz + d = 0
        # Reference:
        #   https://www.open3d.org/docs/0.19.0/tutorial/geometry/pointcloud.html#Plane-segmentation
        a, b, c, d = plane_model

        normal = np.array([a, b, c])  # Compute the unit-length normal vector of the plane
        normal = normal / np.linalg.norm(normal)

        # Select a point on the plane using the centroid of the inlier points
        inlier_points = pointcloud.points[inlier_indices]
        centroid = inlier_points.mean(axis=0)

        estimated_plane = Plane3D(point=centroid, normal=normal)

        return PlaneEstimate(pointcloud, estimated_plane, inlier_indices)

    def get_inlier_text(self) -> str:
        """Retrieve a text description of the plane estimate's inlier points."""
        n_inliers = len(self.inlier_indices)
        total_points = len(self.pointcloud)
        inlier_ratio = 100 * n_inliers / total_points
        return f"Inliers: {n_inliers}/{total_points} ({inlier_ratio:.1f}%)"

    def project_inliers_onto_plane(self) -> np.typing.NDArray[np.float64]:
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

    def convert_for_visualization(self) -> np.typing.NDArray[np.uint8]:
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
