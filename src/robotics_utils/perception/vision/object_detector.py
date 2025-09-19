"""Define a class providing open-vocabulary object detection using OWL-ViT."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTImageProcessorFast, OwlViTProcessor

from robotics_utils.perception.vision.bounding_box import BoundingBox
from robotics_utils.perception.vision.vision_utils import RGB, determine_pytorch_device
from robotics_utils.visualization.display_images import Displayable

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

    from robotics_utils.perception.vision.images import RGBImage


class TextQueries:
    """A set of text queries for an object detection model."""

    def __init__(self) -> None:
        """Initialize an empty list (acting as a set) of text queries."""
        self._queries: list[str] = []

    def __bool__(self) -> bool:
        """Return a Boolean indicating if the set of text queries is empty (empty = False)."""
        return bool(self._queries)

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over the set of text queries."""
        return iter(self._queries)

    def __str__(self) -> str:
        """Return a readable string representation of the text queries."""
        return "\t" + "\n\t".join(self._queries)

    def add(self, query: str) -> None:
        """Add the given text query, or multiple comma-separated queries, to the set."""
        new_queries = [q.strip() for q in query.split(",")]
        for q in new_queries:
            if q and q not in self._queries:
                self._queries.append(q)
        self._queries = sorted(self._queries)

    def remove(self, query: str) -> bool:
        """Remove the given text query from the set.

        :param query: Text query to be removed
        :return: Boolean value indicating if the given query was removed
        """
        try:
            self._queries.remove(query.strip())
        except ValueError as _:
            return False
        return True

    def clear(self) -> None:
        """Clear the set of text queries."""
        self._queries.clear()


@dataclass(frozen=True)
class ObjectDetection:
    """A successful object detection for a text query."""

    query: str
    score: float
    bounding_box: BoundingBox

    def draw(self, image: RGBImage, color: RGB = (255, 255, 0), thickness: int = 3) -> None:
        """Visualize the object detection on the given image.

        :param image: Image on which the object detection is drawn (modified in-place)
        :param color: RGB color of the visualization
        :param thickness: Thickness (pixels) of the drawn bounding box
        """
        self.bounding_box.draw(image, color, thickness)

        label = f"{self.query}: {self.score:.2f}"
        text_xy = (self.bounding_box.top_left.x - 50, self.bounding_box.top_left.y - 10)
        cv2.putText(image.data, label, text_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


@dataclass(frozen=True)
class ObjectDetections(Displayable):
    """A collection of successful object detections from text queries."""

    detections: list[ObjectDetection]
    image: RGBImage
    """Image in which the detections were found."""

    @property
    def queries(self) -> set[str]:
        """Retrieve the set of text queries describing the detected object(s)."""
        return {d.query for d in self.detections}

    def convert_for_visualization(self) -> NDArray[np.uint8]:
        """Convert the object detections into a form that can be visualized."""
        rng = np.random.default_rng()

        q_colors = {q: tuple(int(n) for n in rng.integers(0, 255, size=3)) for q in self.queries}

        vis_image = deepcopy(self.image)
        for detection in self.detections:
            color = RGB(q_colors[detection.query])
            detection.draw(vis_image, color)

        return vis_image.data


class ObjectDetector:
    """Detect objects in images using the OWL-ViT model."""

    def __init__(self, model_name: str = "google/owlvit-base-patch32") -> None:
        """Initialize the specified OWL-ViT model."""
        self.device = determine_pytorch_device()
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set the model into evaluation mode
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.fast_image_processor = OwlViTImageProcessorFast()

    def detect(self, image: RGBImage, queries: list[str] | TextQueries) -> ObjectDetections:
        """Detect objects matching text queries in the given image.

        :param image: RGB image to detect objects within
        :param queries: Text queries describing the object(s) to be detected
        :return: Collection of all successful object detections for the queries
        """
        if isinstance(queries, TextQueries):
            queries = list(queries)

        pil_image = Image.fromarray(image.data)

        # Process all text queries at once
        inputs = self.processor(
            text=queries,
            images=pil_image,
            return_tensors="pt",  # Return PyTorch tensors
        ).to(self.device)

        with torch.no_grad():
            raw_outputs = self.model(**inputs)  # Single forward pass

        outputs: list[dict] = self.fast_image_processor.post_process_object_detection(
            raw_outputs,
            threshold=0.1,
        )[0]  # Index 0 because we only provided one image
        scores = outputs["scores"].tolist()
        labels = outputs["labels"].tolist()  # Integer class labels (indices into `queries`)
        boxes = outputs["boxes"].tolist()

        detections = [
            ObjectDetection(
                queries[query_idx],
                score,
                BoundingBox.from_ratios(box_data, image.data.shape),
            )
            for score, query_idx, box_data in zip(scores, labels, boxes)
        ]
        return ObjectDetections(detections, image)
