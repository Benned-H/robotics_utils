"""Define a class providing open-vocabulary object bounding box detection."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import cv2
import numpy as np

from robotics_utils.visualization import Displayable

if TYPE_CHECKING:
    from robotics_utils.perception.vision import RGB, BoundingBox, RGBImage


@dataclass(frozen=True)
class ObjectBoundingBox:
    """The bounding box of a detected object in an image."""

    query: str
    bounding_box: BoundingBox
    score: float | None = None

    def draw(self, image: RGBImage, color: RGB = (255, 255, 0), thickness: int = 3) -> None:
        """Visualize the object bounding box on the given image.

        :param image: Image on which the object detection is drawn (modified in-place)
        :param color: RGB color of the visualization
        :param thickness: Thickness (pixels) of the drawn bounding box
        """
        self.bounding_box.draw(image, color, thickness)

        label = f"{self.query}: {self.score:.2f}" if self.score is not None else self.query
        text_xy = (self.bounding_box.top_left.x - 50, self.bounding_box.top_left.y - 10)
        cv2.putText(image.data, label, text_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


@dataclass(frozen=True)
class ObjectBoundingBoxes(Displayable):
    """A collection of detected object bounding boxes in an image."""

    detections: list[ObjectBoundingBox]
    image: RGBImage
    """Image in which the object bounding boxes were found."""

    @property
    def queries(self) -> set[str]:
        """Retrieve the set of text queries describing the detected object(s)."""
        return {d.query for d in self.detections}

    def convert_for_visualization(self) -> np.typing.NDArray[np.uint8]:
        """Convert the detected object bounding boxes into a form that can be visualized."""
        rng = np.random.default_rng()

        q_colors = {q: tuple(int(n) for n in rng.integers(0, 255, size=3)) for q in self.queries}

        rgb_image = deepcopy(self.image)
        for detection in self.detections:
            color: RGB = q_colors[detection.query]
            detection.draw(rgb_image, color)

        bgr_data = cv2.cvtColor(rgb_image.data, cv2.COLOR_RGB2BGR)
        return bgr_data.astype(np.uint8)


class BoundingBoxDetector(Protocol):
    """Detect bounding boxes for objects in images based on text queries."""

    def detect(self, image: RGBImage, queries: list[str]) -> ObjectBoundingBoxes:
        """Detect object bounding boxes matching text queries in the given image.

        :param image: RGB image to detect objects within
        :param queries: Text queries describing the object(s) to be detected
        :return: Collection of detected object bounding boxes matching the queries
        """
        ...
