"""Import classes defining parameters and interfaces for camera."""

try:
    import pyrealsense2

    REALSENSE_PRESENT = True
except ModuleNotFoundError:
    REALSENSE_PRESENT = False

from .camera_params import CameraFOV as CameraFOV
from .camera_params import CameraIntrinsics as CameraIntrinsics
from .camera_params import DepthCameraSpec as DepthCameraSpec
from .camera_params import Resolution as Resolution
from .cameras import Camera as Camera
from .cameras import DepthCamera as DepthCamera
from .cameras import RGBCamera as RGBCamera

if REALSENSE_PRESENT:
    from .realsense import D415_SPEC as D415_SPEC
    from .realsense import D455_SPEC as D455_SPEC
    from .realsense import RealSense as RealSense
