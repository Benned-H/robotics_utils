"""Define a class providing open-vocabulary object keypoint detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import cv2
import numpy as np

from robotics_utils.vision import PixelXY, RGBImage, get_rgb_colors
from robotics_utils.visualization import Displayable

if TYPE_CHECKING:
    from robotics_utils.vision import RGB


@dataclass(frozen=True)
class KeypointDetection:
    """A pixel keypoint for a detected object in an image."""

    query: str
    keypoint: PixelXY
    score: float | None = None

    @classmethod
    def from_json(cls, json_data: dict) -> KeypointDetection:
        """Construct a KeypointDetection from JSON data."""
        return KeypointDetection(
            query=json_data["query"],
            keypoint=PixelXY.from_json(json_data["keypoint"]),
            score=float(json_data["score"]) if "score" in json_data else None,
        )

    def to_json(self) -> dict:
        """Convert the detected keypoint to a dictionary of JSON data."""
        json_data = {"query": self.query, "keypoint": self.keypoint.to_json()}
        if self.score is not None:
            json_data["score"] = self.score
        return json_data

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
class KeypointDetections(Displayable):
    """A collection of detected object keypoints in an image."""

    detections: list[KeypointDetection]
    image: RGBImage
    """Image in which the object keypoints were found."""

    @property
    def queries(self) -> set[str]:
        """Retrieve the set of text queries describing the detected object(s)."""
        return {d.query for d in self.detections}

    def convert_for_visualization(self) -> np.typing.NDArray[np.uint8]:
        """Visualize the detected object keypoints."""
        query_colors = dict(zip(self.queries, get_rgb_colors(len(self.queries))))

        rgb_image = RGBImage(self.image.data.copy())
        for detection in self.detections:
            color: RGB = query_colors[detection.query]
            detection.draw(rgb_image, color)

        bgr_data = cv2.cvtColor(rgb_image.data, cv2.COLOR_RGB2BGR)
        return bgr_data.astype(np.uint8)

    @classmethod
    def from_json(cls, json_data: dict) -> KeypointDetections:
        """Construct a KeypointDetections from JSON data."""
        return KeypointDetections(
            detections=[KeypointDetection.from_json(d) for d in json_data["detections"]],
            image=RGBImage.from_file(json_data["image_path"]),
        )

    def to_json(self) -> dict:
        """Convert the collection of keypoint detections into a dictionary of JSON data."""
        if self.image.filepath is None:
            raise ValueError("Cannot convert keypoint detections to JSON; image has no filepath.")

        return {
            "detections": [det.to_json() for det in self.detections],
            "image_path": str(self.image.filepath.resolve()),
        }


class KeypointDetector(Protocol):
    """Detect keypoints for objects in images based on text queries."""

    def detect_keypoints(self, image: RGBImage, queries: list[str]) -> KeypointDetections:
        """Detect object keypoints matching text queries in the given image.

        :param image: RGB image to detect objects within
        :param queries: Text queries describing the object(s) to be detected
        :return: Collection of object keypoint detections matching the queries
        """
        ...
