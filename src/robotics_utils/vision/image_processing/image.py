"""Define an abstract base class to represent images using NumPy arrays."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from PIL import Image as PILImage

from robotics_utils.vision.image_processing.pixel_xy import PixelXY
from robotics_utils.visualization.display_images import Displayable

if TYPE_CHECKING:
    from pathlib import Path


class Image(ABC, Displayable):
    """An image represented as a NumPy array of shape (H, W, ...)."""

    @abstractmethod
    def convert_for_visualization(self) -> np.typing.NDArray[np.uint8]:
        """Convert the image data into a form that can be visualized."""

    def __init__(self, data: np.typing.NDArray, filepath: Path | None = None) -> None:
        """Initialize the image using the given data."""
        if len(data.shape) < 2:
            raise ValueError(f"Image expects at least 2-dim. data, got {data.shape}.")

        self.data = data
        self.filepath = filepath
        """Optional filepath from which this image was loaded."""

    @property
    def width(self) -> int:
        """Retrieve the width (in pixels) of the image."""
        return self.data.shape[1]

    @property
    def height(self) -> int:
        """Retrieve the height (in pixels) of the image."""
        return self.data.shape[0]

    @property
    def channels(self) -> int:
        """Retrieve the number of color channels in the image."""
        return 1 if len(self.data.shape) == 2 else self.data.shape[2]

    @property
    def resolution(self) -> tuple[int, int]:
        """Retrieve the resolution of the image in the form (width, height)."""
        return (self.width, self.height)

    def clip_x(self, pixel_x: int) -> int:
        """Clip a pixel x-coordinate into the image."""
        return np.clip(pixel_x, a_min=0, a_max=self.width - 1)

    def clip_y(self, pixel_y: int) -> int:
        """Clip a pixel y-coordinate into the image."""
        return np.clip(pixel_y, a_min=0, a_max=self.height - 1)

    def clip_pixel(self, pixel_xy: PixelXY) -> PixelXY:
        """Clip the given (x,y) coordinate of a pixel into the image."""
        return PixelXY((self.clip_x(pixel_xy.x), self.clip_y(pixel_xy.y)))

    def fit_into(self, max_width_px: int | None = None, max_height_px: int | None = None) -> None:
        """Resize the image (in-place) to fit within the given dimensions (in pixels).

        :param max_width_px: Optional maximum width (pixels) of the result (defaults to None)
        :param max_height_px: Optional maximum height (pixels) of the result (defaults to None)
        """
        # First, fit within maximum width if one was provided
        if max_width_px is not None and self.width > max_width_px:
            new_height_px = int(self.height * max_width_px / self.width)

            pil_image = PILImage.fromarray(self.data)
            resized = pil_image.resize((max_width_px, new_height_px), PILImage.Resampling.LANCZOS)
            self.data = np.asarray(resized)

        # Second, fit within the maximum height if one was provided
        if max_height_px is not None and self.height > max_height_px:
            new_width_px = int(self.width * max_height_px / self.height)

            pil_image = PILImage.fromarray(self.data)
            resized = pil_image.resize((new_width_px, max_height_px), PILImage.Resampling.LANCZOS)
            self.data = np.asarray(resized)


ImageT = TypeVar("ImageT", bound=Image)
"""Type variable representing a specific type of Image."""
