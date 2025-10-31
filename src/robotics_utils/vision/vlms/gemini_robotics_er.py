"""Define a class providing an interface to Gemini Robotics-ER 1.5.

Reference:
    https://github.com/google-gemini/cookbook/blob/main/quickstarts/gemini-robotics-er.ipynb
"""

from __future__ import annotations

import json
import textwrap
import time
from copy import deepcopy
from typing import Any

import numpy as np
import PIL.Image

try:
    from google import genai
    from google.genai.types import GenerateContentConfig, HttpOptions, ThinkingConfig
    from httpx import ConnectTimeout

    GEN_AI_PRESENT = True
except ModuleNotFoundError:
    GEN_AI_PRESENT = False


from robotics_utils.io.logging import console
from robotics_utils.vision import BoundingBox, PixelXY, RGBImage
from robotics_utils.vision.vlms.bounding_box_detection import (
    BoundingBoxDetector,
    DetectedBoundingBox,
    DetectedBoundingBoxes,
)
from robotics_utils.vision.vlms.keypoint_detection import (
    DetectedKeypoint,
    DetectedKeypoints,
    KeypointDetector,
)


def parse_json(json_output: str | None) -> Any | None:
    """Parse a JSON output from a VLM into Python data structures.

    :param json_output: String containing JSON data (or None if VLM didn't return text)
    :return: Parsed JSON output as a list or dictionary (or None if JSON not found)
    """
    if json_output is None:
        return None

    lines = json_output.splitlines()
    json_data: str | None = None
    for i, line in enumerate(lines):
        if line == "```json":
            # Ignore everything before "```json"
            json_block_onwards = "\n".join(lines[i + 1 :])

            # Ignore everything after the closing "```"
            json_data = json_block_onwards.split("```")[0]
            break

    try:
        if json_data is not None:
            return json.loads(json_data)
    except json.JSONDecodeError:
        console.print_exception()
        console.print("[yellow]⚠️ Warning: Invalid JSON response. Skipping.[/yellow]")

    return None


MIN_TIMEOUT_MS = 10_000
"""Google Gen AI prohibits setting the request timeout to less than 10 seconds."""


class GeminiRoboticsER(KeypointDetector, BoundingBoxDetector):
    """An interface for Gemini Robotics-ER 1.5."""

    def __init__(
        self,
        api_key: str,
        model_id: str = "gemini-robotics-er-1.5-preview",
        object_limit: int = 20,
        timeout_s: float = 10.0,
    ) -> None:
        """Initialize an interface for Gemini Robotics-ER 1.5 (preview version).

        :param api_key: Google API key used to access Gemini Robotics-ER 1.5
        :param model_id: String specifying which Gemini model to use
        :param object_limit: Limit on the number of objects per response (defaults to 20)
        :param timeout_s: Maximum duration (seconds) of any Gemini call (defaults to 10 sec)
        """
        if not GEN_AI_PRESENT:
            raise ImportError("Cannot run GeminiRoboticsER without google-genai.")

        timeout_ms = int(1000 * timeout_s)
        if timeout_ms < MIN_TIMEOUT_MS:
            raise ValueError(f"timeout_s cannot be less than 10 seconds; it was {timeout_s}.")

        self.client = genai.Client(api_key=api_key, http_options=HttpOptions(timeout=timeout_ms))
        self.model_id = model_id
        self.object_limit = object_limit
        self.timeout_s = timeout_s

    def call(
        self,
        image: RGBImage,
        prompt: str,
        config: GenerateContentConfig | None = None,
    ) -> Any | None:
        """Call Gemini Robotics ER 1.5 on the given visual-language prompt.

        :param image: RGB image used in the prompt
        :param prompt: Natural language prompt
        :param config: Optional Gemini configuration (defaults to None; zero thinking budget)
        :return: Data parsed from the response JSON, or None if no JSON was output
        """
        start_time = time.time()

        if config is None:
            config = GenerateContentConfig(
                temperature=0.5,
                thinking_config=ThinkingConfig(thinking_budget=0),
            )

        pil_image = PIL.Image.fromarray(image.data)

        output = None
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[pil_image, prompt],
                config=config,
            )

        except ConnectTimeout:
            console.print(f"[yellow]Call timed out after {self.timeout_s} seconds.[/yellow]")

        except Exception:
            console.print_exception()
            raise

        else:
            if response.text is not None:
                console.print(f"Raw output from Gemini Robotics ER 1.5:\n{response.text}")

            output = parse_json(response.text)

        console.print(f"\nTotal processing time: {(time.time() - start_time):.4f} seconds.")
        return output

    def detect_keypoints(self, image: RGBImage, queries: list[str]) -> DetectedKeypoints:
        """Detect object keypoints matching text queries in the given image.

        :param image: RGB image to detect objects within
        :param queries: Text queries describing the object(s) to be detected
        :return: Collection of detected object keypoints matching the queries
        """
        copied = deepcopy(image)
        copied.resize(max_width_px=800)
        console.print(f"Resized image copy is {copied.width} x {copied.height} pixels (W x H).")

        prompt = textwrap.dedent(f"""\
            Get all points matching the following objects: {", ".join(queries)}. The label
            returned should be an identifying name for the object detected.
            Limit to {self.object_limit} objects.

            The answer should follow the JSON format:
            [{{"point": , "label": }}, ...]

            The points are in [y, x] format normalized to 0-1000.
            """)

        points_data = self.call(copied, prompt)
        if points_data is None or not isinstance(points_data, list):
            console.print(f"Unexpected output from Gemini: {points_data}")
            return DetectedKeypoints([], image)

        # Convert the JSON data to object keypoints (i.e., convert 0-1000 to image pixel coords)
        keypoints = []
        for det in points_data:
            yx_output = det["point"]
            if len(yx_output) != 2:
                raise ValueError(f"Points should be in [y, x] format; found {yx_output}.")
            y_ratio, x_ratio = np.asarray(yx_output, dtype=np.float32) / 1000.0

            pixel_x = int(x_ratio * image.width)
            pixel_y = int(y_ratio * image.height)
            pixel_xy = image.clip_pixel(PixelXY((pixel_x, pixel_y)))

            keypoints.append(DetectedKeypoint(query=det["label"], keypoint=pixel_xy))

        return DetectedKeypoints(detections=keypoints, image=image)

    def detect_bounding_boxes(self, image: RGBImage, queries: list[str]) -> DetectedBoundingBoxes:
        """Detect object bounding boxes matching text queries in the given image.

        :param image: RGB image to detect objects within
        :param queries: Text queries describing the object(s) to be detected
        :return: Collection of detected object bounding boxes matching the queries
        """
        copied = deepcopy(image)
        copied.resize(max_width_px=800)
        console.print(f"Resized image copy is {copied.width} x {copied.height} pixels (W x H).")

        prompt = textwrap.dedent(f"""\
            Return bounding boxes for the following objects: {", ".join(queries)}. The
            label for each box should be an identifying name for the detected object.
            If an object is present multiple times, use the same corresponding label.
            Limit to {self.object_limit} objects.

            The answer should follow the JSON format:
            [{{"box_2d": [ymin, xmin, ymax, xmax], "label": }}, ...]
            normalized to 0-1000. The values in box_2d must only be integers.
            """)

        boxes_data = self.call(copied, prompt)
        if boxes_data is None or not isinstance(boxes_data, list):
            console.print(f"Unexpected output from Gemini: {boxes_data}")
            return DetectedBoundingBoxes([], image)

        # Convert the JSON data to bounding boxes (i.e., convert 0-1000 to image pixel coords)
        detections: list[DetectedBoundingBox] = []
        image_hw = np.array([image.height, image.width], dtype=np.float32)
        for det in boxes_data:
            box_2d: tuple[int] = tuple(det["box_2d"])
            if len(box_2d) != 4:
                raise ValueError(f"Bounding box was {box_2d}; expected [ymin, xmin, ymax, xmax].")
            box_2d_ratio = np.asarray(box_2d, dtype=np.float32) / 1000.0

            ymin, xmin = box_2d_ratio[:2] * image_hw
            ymax, xmax = box_2d_ratio[2:] * image_hw

            min_xy = PixelXY((xmin, ymin))
            max_xy = PixelXY((xmax, ymax))
            bb = BoundingBox(top_left=min_xy, bottom_right=max_xy)

            detections.append(DetectedBoundingBox(query=det["label"], bounding_box=bb))

        return DetectedBoundingBoxes(detections, image)
