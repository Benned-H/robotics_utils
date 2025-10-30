"""Define a class providing open-vocabulary object keypoint detection."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import cv2
import numpy as np

from robotics_utils.visualization import Displayable

if TYPE_CHECKING:
    from robotics_utils.perception.vision import RGB, PixelXY, RGBImage


@dataclass(frozen=True)
class ObjectKeypoint:
    """A pixel keypoint for a detected object in an image."""

    query: str
    keypoint: PixelXY
    score: float | None = None

    def draw(self, image: RGBImage, color: RGB, radius: int = 5, thickness: int = 3) -> None:
        """Draw the object keypoint as a labeled circle on the given image.

        :param image: Image on which the keypoint is drawn (modified in-place)
        :param color: RGB color of the visualization
        :param radius: Radius (pixels) of the drawn circle (defaults to 5)
        :param thickness: Thickness (pixels) of the drawn circle (defaults to 3)
        """
        cv2.circle(image.data, tuple(self.keypoint.xy), radius, color, thickness)

        label = f"{self.query}: {self.score:.2f}" if self.score is not None else self.query
        text_xy = (self.keypoint.x - 50, self.keypoint.y - 10)
        cv2.putText(image.data, label, text_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


@dataclass(frozen=True)
class ObjectKeypoints(Displayable):
    """A collection of detected object keypoints in an image."""

    detections: list[ObjectKeypoint]
    image: RGBImage
    """Image in which the object keypoints were found."""

    @property
    def queries(self) -> set[str]:
        """Retrieve the set of text queries describing the detected object(s)."""
        return {d.query for d in self.detections}

    def convert_for_visualization(self) -> np.typing.NDArray[np.uint8]:
        """Visualize the detected object keypoints."""
        rng = np.random.default_rng()

        q_colors = {q: tuple(int(n) for n in rng.integers(0, 255, size=3)) for q in self.queries}

        vis_image = deepcopy(self.image)
        for detection in self.detections:
            color: RGB = q_colors[detection.query]
            detection.draw(vis_image, color)

        return vis_image.data


class KeypointDetector(Protocol):
    """Detect keypoints for objects in images based on text queries."""

    def detect(self, image: RGBImage, queries: list[str]) -> ObjectKeypoints:
        """Detect object keypoints matching text queries in the given image.

        :param image: RGB image to detect objects within
        :param queries: Text queries describing the object(s) to be detected
        :return: Collection of detected object keypoints matching the queries
        """
        ...
