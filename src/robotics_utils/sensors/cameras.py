"""Define classes to model and interface with camera sensors."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import astuple, dataclass

import numpy as np
from numpy.typing import NDArray


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
class DepthCameraSpec:
    """Operational specifications for a specific model of depth camera."""

    fov: CameraFOV

    min_range_m: float
    """Absolute minimum range (meters) for the camera, below which data should be discarded."""

    max_range_m: float
    """Absolute maximum range (meters) for the camera, beyond which data should be discarded."""


# Reference: Range values read directly from D415 box (Product #82635ASRCDVKHV). Depth FOV from:
#   https://www.intel.com/content/www/us/en/products/sku/128256/intel-realsense-depth-camera-d415/specifications.html
D415_SPEC = DepthCameraSpec(
    CameraFOV(horizontal_deg=69.4, vertical_deg=42.5),
    min_range_m=0.3,
    max_range_m=10,
)


# Reference: Range and Depth FOV both provided by:
#   https://www.bhphotovideo.com/c/product/1570532-REG/intel_82635dsd455_realsense_depth_camera_d455.html/specs
# Depth FOV confirmed by:
#   https://www.intel.com/content/www/us/en/products/sku/205847/intel-realsense-depth-camera-d455/specifications.html
D455_SPEC = DepthCameraSpec(
    CameraFOV(horizontal_deg=86, vertical_deg=57),
    min_range_m=0.4,
    max_range_m=20,
)
