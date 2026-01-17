"""Unit tests for the PointCloud class."""

from hypothesis import given

from robotics_utils.reconstruction import PointCloud
from robotics_utils.vision import DepthImage
from robotics_utils.vision.cameras import CameraIntrinsics

from .strategies.vision_strategies import camera_intrinsics, depth_images


@given(depth_images(), camera_intrinsics())
def test_pointcloud_from_depth_image(depth: DepthImage, intrinsics: CameraIntrinsics) -> None:
    """Verify that a point cloud can be constructed from a depth image."""
    # Arrange/Act - Given a depth image, attempt to construct a point cloud
    cloud = PointCloud.from_depth_image(depth=depth, depth_intrinsics=intrinsics)

    # Assert - Verify that the pointcloud's data has the expected shape
    num_pixels = depth.height * depth.width
    assert cloud.points.shape == (num_pixels, 3)
