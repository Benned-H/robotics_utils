"""Define a class to represent depth images."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from robotics_utils.vision.image_processing.image import Image

if TYPE_CHECKING:
    from pathlib import Path


class DepthImage(Image):
    """A depth image represented as a NumPy array of shape (H, W)."""

    def __init__(self, data: np.typing.NDArray, filepath: Path | None = None) -> None:
        """Initialize the depth image using an array of shape (H, W) of depth data (in meters)."""
        super().__init__(data, filepath)

        # Verify expected properties of depth image data
        if len(self.data.shape) != 2:
            raise ValueError(f"DepthImage expects 2-dim. data, got {self.data.shape}.")

        if self.data.dtype != np.float64:  # Expect floats because the depth data is in meters
            raise TypeError(f"DepthImage expects datatype np.float64, got {self.data.dtype}.")

    @property
    def min_depth_m(self) -> float:
        """Retrieve the minimum non-zero depth (meters) in the image."""
        return np.min(self.data[self.data > 0])

    @property
    def max_depth_m(self) -> float:
        """Retrieve the maximum depth (meters) in the image."""
        return np.max(self.data)

    def convert_for_visualization(self) -> np.typing.NDArray[np.uint8]:
        """Convert the depth image into a form that can be visualized."""
        normalized_depth = cv2.normalize(self.data, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        convert_to_uint8 = np.astype(normalized_depth, np.uint8)
        return cv2.applyColorMap(convert_to_uint8, colormap=cv2.COLORMAP_JET)
