"""Define functions to support 3D-to-image projection."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import cv2
import numpy as np

from robotics_utils.io import console
from robotics_utils.kinematics import Point3D, Pose3D
from robotics_utils.vision.images import PixelXY, RGBImage

if TYPE_CHECKING:
    from robotics_utils.vision.cameras import CameraIntrinsics


def project_3d_to_image(
    points: list[Point3D],
    intrinsics: CameraIntrinsics,
    pose_wrt_camera: Pose3D,
) -> list[PixelXY]:
    """Project 3D points onto the image plane based on the given intrinsics and transform.

    :param points: 3D points in the world frame to be projected to the image plane
    :param intrinsics: Camera intrinsics used for projection
    :param pose_wrt_camera: Pose of the frame used for the points (w.r.t. camera)
    :return: List of (x,y) pixel coordinates corresponding to the 3D points
    """
    points_arr = np.stack([p.to_array() for p in points], axis=0).astype(np.float32)  # N x 3
    points_arr = points_arr.reshape(-1, 1, 3)  # Reshape to (N, 1, 3) for cv2.projectPoints
    rvec, _ = cv2.Rodrigues(pose_wrt_camera.orientation.to_rotation_matrix())
    tvec = pose_wrt_camera.position.to_array()
    distortion = np.zeros((4, 1), dtype=np.float32)

    console.print(f"{rvec} {tvec} {distortion}")

    projections, _ = cv2.projectPoints(points_arr, rvec, tvec, intrinsics.to_matrix(), distortion)
    return [PixelXY(row.ravel()) for row in projections]


def draw_axes(
    length: float,
    pose_c_a: Pose3D,
    intrinsics: CameraIntrinsics,
    image: RGBImage,
) -> RGBImage:
    """Visualize the given pose in the given image, returned as a new image.

    :param length: Length (m) of the drawn axes in the world
    :param pose_c_a: Pose of the axes frame (frame a) relative to the camera (frame c)
    :param intrinsics: Camera intrinsics used to project the axes to the image plane
    :param image: Image on which the axes are drawn (not modified)
    :return: New image containing the pose visualized using RGB axes
    """
    axes_xyz = [(0, 0, 0), (length, 0, 0), (0, length, 0), (0, 0, length)]
    axes_pixels = project_3d_to_image([Point3D(*p) for p in axes_xyz], intrinsics, pose_c_a)
    axes_tuples = [(p.x, p.y) for p in axes_pixels]

    image_data = deepcopy(image.data)
    rgb_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, color in enumerate(rgb_colors):
        axis_endpoint = axes_tuples[i + 1]
        cv2.line(image_data, axes_tuples[0], axis_endpoint, color, thickness=3)

    return RGBImage(image_data)
