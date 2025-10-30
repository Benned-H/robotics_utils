"""Import classes and functions for visualization."""

from .display_images import Displayable as Displayable
from .display_images import display_window as display_window

# Attempt to import the PointcloudVisualizer class, but allow failure if Open3D is unavailable
try:
    import open3d

    OPEN3D_PRESENT = True
except ModuleNotFoundError:
    OPEN3D_PRESENT = False

if OPEN3D_PRESENT:
    from .pointcloud_visualizer import PointcloudVisualizer as PointcloudVisualizer
