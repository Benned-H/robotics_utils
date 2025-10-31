"""Import classes providing interfaces for VLMs."""

from .bounding_box_detection import DetectedBoundingBox as DetectedBoundingBox
from .bounding_box_detection import DetectedBoundingBoxes as DetectedBoundingBoxes
from .keypoint_detection import KeypointDetector as KeypointDetector
from .keypoint_detection import ObjectKeypoint as ObjectKeypoint
from .keypoint_detection import ObjectKeypoints as ObjectKeypoints
from .owl_vit_bounding_box_detector import OwlViTBoundingBoxDetector as OwlViTBoundingBoxDetector
from .queries import TextQueries as TextQueries
