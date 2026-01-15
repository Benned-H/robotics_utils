"""Define a class to represent depth images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from robotics_utils.vision.image import Image


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

    @classmethod
    def from_file(cls, image_path: str | Path) -> DepthImage:
        """Load a depth image from the given filepath.

        Assumes the file stores depth as 16-bit unsigned integers in millimeters.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Cannot load image from nonexistent file: {image_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH)
        if image is None:
            raise RuntimeError(f"Failed to load image from path: {image_path}")

        # Convert from millimeters (uint16) to meters (float64)
        depth_data = image.astype(np.float64) / 1000.0
        return DepthImage(depth_data, image_path)

    def to_file(self, image_path: str | Path) -> None:
        """Save the depth image to the given filepath.

        Saves depth as 16-bit unsigned integers in millimeters.
        """
        image_path = Path(image_path)
        image_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert from meters (float64) to millimeters (uint16)
        depth_mm = (self.data * 1000.0).astype(np.uint16)
        success = cv2.imwrite(str(image_path), depth_mm)
        if not success:
            raise RuntimeError(f"Failed to save image to path: {image_path}")

        # Populate the image's filepath if it didn't have one
        if self.filepath is None:
            self.filepath = image_path
