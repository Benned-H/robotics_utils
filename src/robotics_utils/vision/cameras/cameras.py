"""Define interfaces for different kinds of cameras."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic

from robotics_utils.vision.depth_image import DepthImage
from robotics_utils.vision.image import ImageT
from robotics_utils.vision.rgb_image import RGBImage

if TYPE_CHECKING:
    from robotics_utils.vision.cameras.camera_params import CameraIntrinsics, DepthCameraSpec


@dataclass
class Camera(ABC, Generic[ImageT]):
    """An interface for a robot camera that returns a particular type of image.

    Camera frame convention:
        - x-axis points to the right in the image frame
        - y-axis points down in the image frame
        - z-axis points forward from the camera

    Reference (e.g.): https://opensfm.org/docs/geometry.html#camera-coordinates
    """

    name: str
    intrinsics: CameraIntrinsics
    image_type: type[ImageT]
    """Type of image captured by the camera (e.g., `DepthImage`)."""

    frame_name: str | None = None
    """Name of the coordinate frame associated with the camera."""

    @abstractmethod
    def get_image(self) -> ImageT:
        """Capture and return an image using the camera."""


@dataclass
class RGBCamera(Camera[RGBImage]):
    """An interface for an RGB camera."""


@dataclass
class DepthCamera(Camera[DepthImage]):
    """An interface for a depth camera."""

    depth_spec: DepthCameraSpec | None = None
