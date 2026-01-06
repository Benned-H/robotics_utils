"""Define a class to represent object instance segmentations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

from robotics_utils.vision import RGB, BoundingBox, RGBImage, assign_random_colors
from robotics_utils.visualization import Displayable


@dataclass(frozen=True)
class ObjectSegmentation:
    """A segmentation of an object instance in an image."""

    query: str
    mask: np.typing.NDArray[np.bool]
    bbox: BoundingBox
    score: float

    def draw(self, image: RGBImage, color: RGB, opacity: float = 0.3) -> None:
        """Draw the instance segmentation on the given image.

        :param image: Image on which the segmentation is drawn (modified in-place)
        :param color: RGB color of the visualization
        :param opacity: Transparency value in the range [0, 1] used for the mask overlay
        """
        # Draw the mask by shading the corresponding area the given color
        mask_3d = self.mask[:, :, np.newaxis]  # Shape: (H, W, 1), broadcasts to (H, W, 3)

        color_arr = np.asarray(color, dtype=image.data.dtype)
        blended = (image.data * (1 - opacity) + color_arr * opacity).astype(image.data.dtype)

        # Blend the color with the original image where the mask is True
        image.data = np.where(mask_3d, blended, image.data)

        # Label the mask with its query and score
        label = f"{self.query}: {self.score:.2f}"
        text_xy = (self.bbox.top_left.x - 50, self.bbox.top_left.y - 10)
        cv2.putText(
            img=image.data,
            text=label,
            org=text_xy,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=color,
            thickness=2,
        )


@dataclass(frozen=True)
class ObjectSegmentations(Displayable):
    """A collection of object instance segmentations in an image."""

    segmentations: list[ObjectSegmentation]
    image: RGBImage
    """Image in which the object segmentations were found."""

    def __iter__(self) -> Iterator[ObjectSegmentation]:
        """Provide an iterator over the object instance segmentations."""
        yield from self.segmentations

    @property
    def queries(self) -> list[str]:
        """Retrieve all unique text queries describing the segmented object(s)."""
        return list({s.query for s in self.segmentations})

    def convert_for_visualization(self) -> np.typing.NDArray[np.uint8]:
        """Visualize the object segmentations by drawing them on the image."""
        query_colors = assign_random_colors(self.queries)

        vis_image = RGBImage(self.image.data.copy())
        for seg in self.segmentations:
            color: RGB = query_colors[seg.query]
            seg.draw(image=vis_image, color=color)

        bgr_data = cv2.cvtColor(vis_image.data, cv2.COLOR_RGB2BGR)
        return bgr_data.astype(np.uint8)
