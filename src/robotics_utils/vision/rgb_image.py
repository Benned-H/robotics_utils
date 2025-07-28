"""Define a class to represent RGB images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


class RGBImage:
    """An RGB image represented as a NumPy array."""

    def __init__(self, data: NDArray) -> None:
        """Initialize the RGB image using the given array of data."""
        if data.dtype != np.uint8:
            data = (data * 255).astype(np.uint8)

        self.data = data

        ### Verify expected properties of the RGB image data ###
        if len(self.data.shape) != 3:
            raise ValueError(f"RGBImage expects 3-dimensional data, got {self.data.shape}")

        if self.data.shape[2] != 3:
            raise ValueError(f"RGBImage expects 3 channels of color, got {self.data.shape[2]}")

        if self.data.dtype != np.uint8:
            raise TypeError(f"RGBImage expects datatype np.uint8, got {self.data.dtype}")

    @classmethod
    def from_file(cls, image_path: str | Path) -> RGBImage:
        """Load an RGB image from the given filepath."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Cannot load image from nonexistent file: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to load image from path: {image_path}")

        rgb_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return RGBImage(rgb_data)

    @property
    def height(self) -> int:
        """Retrieve the height (in pixels) of the image."""
        return self.data.shape[0]

    @property
    def width(self) -> int:
        """Retrieve the width (in pixels) of the image."""
        return self.data.shape[1]

    def visualize(self, window_title: str) -> None:
        """Visualize the image in an OpenCV2 window with the given title."""
        image_bgr = cv2.cvtColor(self.data, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_title, image_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
