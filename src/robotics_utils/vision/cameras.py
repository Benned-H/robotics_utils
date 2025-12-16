"""Define classes to model and interface with camera sensors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import astuple, dataclass
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from robotics_utils.kinematics import Pose3D
from robotics_utils.vision.image_processing import DepthImage, Image, RGBImage


@dataclass(frozen=True)
class CameraIntrinsics:
    """Intrinsic parameters for a pinhole model camera.

    Definitions:
        - Principal axis - Line perpendicular to the image plane through the camera pinhole.
        - Principal point - Where the principal axis intersects with the image plane, relative
            to the origin of the film (i.e., the pinhole's location if projected onto the film).

    Reference: https://ksimek.github.io/2013/08/13/intrinsic/
    """

    fx: float  # Focal length (pixels) in x
    fy: float  # Focal length (pixels) in y
    x0: float  # Principal point offset in x
    y0: float  # Principal point offset in y

    def __iter__(self) -> Iterator[float]:
        """Provide an iterator over the camera intrinsics: [fx, fy, x0, y0]."""
        yield from astuple(self)

    def to_matrix(self) -> NDArray[np.float64]:
        """Convert the camera intrinsic parameters into a 3x3 intrinsic matrix."""
        return np.array([[self.fx, 0, self.x0], [0, self.fy, self.y0], [0, 0, 1]])


@dataclass(frozen=True)
class CameraFOV:
    """A camera's angular field of view (FOV)."""

    horizontal_deg: float
    vertical_deg: float


@dataclass(frozen=True)
class Resolution:
    """A camera's resolution, with sensible sorting."""

    width: int
    height: int

    def __lt__(self, other: Resolution) -> bool:
        """Evaluate whether the Resolution is less than another."""
        return self.pixels < other.pixels

    def __str__(self) -> str:
        """Return a readable string representation of the resolution."""
        return f"{self.width}x{self.height}"

    @property
    def pixels(self) -> int:
        """Calculate the total number of pixels in the camera resolution."""
        return self.width * self.height


@dataclass(frozen=True)
class DepthCameraSpec:
    """Operational specifications for a specific model of depth camera."""

    fov: CameraFOV

    min_range_m: float
    """Absolute minimum range (meters) for the camera, below which data should be discarded."""

    max_range_m: float
    """Absolute maximum range (meters) for the camera, beyond which data should be discarded."""


ImageT = TypeVar("ImageT", bound=Image)


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
