"""Import classes for perception including occupancy mapping and 3D reconstruction."""

from .laser_scan import LaserScan2D as LaserScan2D
from .occupancy_grid import OccupancyGrid2D as OccupancyGrid2D
from .pointcloud import PointCloud as PointCloud

try:
    import open3d

    OPEN3D_PRESENT = True
except ModuleNotFoundError:
    OPEN3D_PRESENT = False

if OPEN3D_PRESENT:
    from .open3d_utils import pointcloud_to_o3d as pointcloud_to_o3d
    from .plane_estimation import PlaneEstimate as PlaneEstimate
