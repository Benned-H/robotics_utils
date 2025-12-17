"""Define a class to represent RGB images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from robotics_utils.vision.image import Image


class RGBImage(Image):
    """An RGB image represented as a NumPy array of shape (H, W, 3)."""

    def __init__(self, data: np.typing.NDArray, filepath: Path | None = None) -> None:
        """Initialize the RGB image using the given array of data."""
        super().__init__(data, filepath)

        # Verify expected properties of RGB image data
        if len(self.data.shape) != 3:
            raise ValueError(f"RGBImage expects 3-dim. data, got {self.data.shape}.")

        if self.channels != 3:
            raise ValueError(f"RGBImage expects 3 channels of color, got {self.channels}.")

        if self.data.dtype != np.uint8:
            raise TypeError(f"RGBImage expects datatype np.uint8, got {self.data.dtype}.")

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
        return RGBImage(rgb_data, image_path)

    def to_file(self, image_path: str | Path) -> None:
        """Save the RGB image to the given filepath."""
        image_path = Path(image_path)
        image_path.parent.mkdir(parents=True, exist_ok=True)

        bgr_data = cv2.cvtColor(self.data, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(str(image_path), bgr_data)
        if not success:
            raise RuntimeError(f"Failed to save image to path: {image_path}")

        # Populate the image's filepath if it didn't have one
        if self.filepath is None:
            self.filepath = image_path

    def convert_for_visualization(self) -> np.typing.NDArray[np.uint8]:
        """Convert the RGB image into a form that can be visualized."""
        color_converted = cv2.cvtColor(self.data, cv2.COLOR_RGB2BGR)
        return color_converted.astype(np.uint8)
