"""Define functions to interface with Open3D without introducing dependencies elsewhere."""

import open3d as o3d

from robotics_utils.reconstruction.pointcloud import PointCloud


def pointcloud_to_o3d(pcd: PointCloud) -> o3d.geometry.PointCloud:
    """Convert the given pointcloud into an Open3D pointcloud."""
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.points)
    if pcd.colors is not None:
        o3d_pcd.colors = o3d.utility.Vector3dVector(pcd.colors / 255.0)
    return o3d_pcd
