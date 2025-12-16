"""Import classes related to machine vision."""

try:
    import open3d

    OPEN3D_PRESENT = True
except ModuleNotFoundError:
    OPEN3D_PRESENT = False

from .cameras import Camera as Camera
from .cameras import CameraIntrinsics as CameraIntrinsics
from .cameras import DepthCamera as DepthCamera
from .cameras import DepthCameraSpec as DepthCameraSpec
from .cameras import Resolution as Resolution
from .cameras import RGBCamera as RGBCamera
from .image_processing import BoundingBox as BoundingBox
from .image_processing import DepthImage as DepthImage
from .image_processing import Image as Image
from .image_processing import ImagePatch as ImagePatch
from .image_processing import PixelXY as PixelXY
from .image_processing import RGBDImage as RGBDImage
from .image_processing import RGBImage as RGBImage
from .projections import draw_axes as draw_axes
from .projections import project_3d_to_image as project_3d_to_image
from .vision_utils import RGB as RGB
from .vision_utils import determine_pytorch_device as determine_pytorch_device
from .vision_utils import get_rgb_colors as get_rgb_colors

if OPEN3D_PRESENT:
    from .pointcloud import Pointcloud as Pointcloud
