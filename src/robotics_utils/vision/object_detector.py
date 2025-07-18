"""Define a class providing open-vocabulary object detection using OWL-ViT."""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTImageProcessorFast, OwlViTProcessor

from robotics_utils.vision.bounding_box import BoundingBox
from robotics_utils.vision.vision_utils import determine_pytorch_device


@dataclass(frozen=True)
class ObjectDetection:
    """A successful object detection for a text query."""

    query: str
    score: float
    bounding_box: BoundingBox


def visualize_detections(
    image: np.ndarray,
    detections: list[ObjectDetection],
    color: tuple[int, int, int] = (255, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    """Draw a collection of object detection results on an image.

    :param image: RGB image over which the detections are visualized
    :param detections: List of object detection results
    :param color: Color of the visualized bounding boxes
    :param thickness: Thickness (pixels) of the drawn bounding boxes
    :return: Updated image with detection visualizations added
    """
    vis_image = image.copy()
    for result in detections:
        top_left_xy = result.bounding_box.top_left_xy
        bottom_right_xy = result.bounding_box.bottom_right_xy

        cv2.rectangle(vis_image, top_left_xy, bottom_right_xy, color, thickness)

        # Add a text label for each detection result
        label = f"{result.query}: {result.score:.2f}"
        text_xy = (result.bounding_box.top_left_x - 50, result.bounding_box.top_left_y - 10)
        cv2.putText(vis_image, label, text_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return vis_image


class ObjectDetector:
    """Detect objects in images using the OWL-ViT model."""

    def __init__(self, model_name: str = "google/owlvit-base-patch32") -> None:
        """Initialize the specified OWL-ViT model."""
        self.device = determine_pytorch_device()
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set the model into evaluation mode
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.fast_image_processor = OwlViTImageProcessorFast()

    def detect(self, image: np.ndarray, text_queries: list[str]) -> list[ObjectDetection]:
        """Detect objects matching text queries in the given image.

        :param image: RGB image to detect objects within
        :param text_queries: Text queries describing the object(s) to be detected
        :return: Dataclass storing the result of each object detection query
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)

        # Process all text queries at once
        inputs = self.processor(
            text=text_queries,
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
        labels = outputs["labels"].tolist()  # Integer class labels (indices into `text_queries`)
        boxes = outputs["boxes"].tolist()

        results = []
        for score, label_idx, box_data in zip(scores, labels, boxes, strict=True):
            query = text_queries[label_idx]
            bounding_box = BoundingBox.from_ratios(box_data, image.shape)
            results.append(ObjectDetection(query, score, bounding_box))

        return results
