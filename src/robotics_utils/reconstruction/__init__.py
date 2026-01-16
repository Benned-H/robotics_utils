"""Import classes and definitions for 3D reconstruction from sensor data."""

from .pointcloud import PointCloud as PointCloud

try:
    import open3d

    OPEN3D_PRESENT = True
except ModuleNotFoundError:
    OPEN3D_PRESENT = False

if OPEN3D_PRESENT:
    from .open3d_utils import pointcloud_to_o3d as pointcloud_to_o3d
    from .plane_estimation import PlaneEstimate as PlaneEstimate
