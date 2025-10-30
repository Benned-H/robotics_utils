"""Define a class providing an interface to Gemini Robotics-ER 1.5.

Reference:
    https://github.com/google-gemini/cookbook/blob/main/quickstarts/gemini-robotics-er.ipynb
"""

from __future__ import annotations

import json
import textwrap
import time

import PIL.Image
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig
from rich.console import Console

from robotics_utils.io.logging import log_info
from robotics_utils.perception.vision import RGBImage
from robotics_utils.perception.vision.vlms.keypoint_detector import (
    KeypointDetector,
    ObjectKeypoints,
)


def parse_json(json_output: str | None) -> str | None:
    """Parse a JSON output from a VLM.

    :param json_output: String containing JSON data (or None if VLM didn't return text)
    :return: Cleaned JSON output (or None if JSON not found)
    """
    if json_output is None:
        return None

    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":  # Ignore everything before "```json"
            output = "\n".join(lines[i + 1 :])
            return output.split("```")[0]  # Ignore everything after the closing "```"
    return None


class GeminiRoboticsER(KeypointDetector):
    """An interface for Gemini Robotics-ER 1.5."""

    def __init__(self, api_key: str, model_id: str = "gemini-robotics-er-1.5-preview") -> None:
        """Initialize the keypoint detector to use Gemini Robotics-ER 1.5 (preview version)."""
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id

    def call(
        self,
        image: RGBImage,
        prompt: str,
        config: GenerateContentConfig | None = None,
    ) -> str | None:
        """Call Gemini Robotics ER 1.5 on the given visual-language prompt.

        :param image: RGB image used in the prompt
        :param prompt: Natural language prompt
        :param config: Optional Gemini configuration (defaults to None; zero thinking budget)
        :return: JSON text from the response, or None if no JSON was output
        """
        if config is None:
            config = GenerateContentConfig(
                temperature=0.5,
                thinking_config=ThinkingConfig(thinking_budget=0),
            )

        pil_image = PIL.Image.fromarray(image.data)

        image_response = self.client.models.generate_content(
            model=self.model_id,
            contents=[pil_image, prompt],
            config=config,
        )
        if image_response.text is not None:
            log_info(image_response.text)

        return parse_json(image_response.text)

    def detect_keypoints(
        self,
        image: RGBImage,
        queries: list[str],
        console: Console
        | None = None,  # TODO: Instead use https://rich.readthedocs.io/en/latest/console.html
    ) -> ObjectKeypoints:
        """Detect object keypoints matching text queries in the given image.

        :param image: RGB image to detect objects within
        :param queries: Text queries describing the object(s) to be detected
        :param console: Optional CLI console used for logging (defaults to None)
        :return: Collection of detected object keypoints matching the queries
        """
        image.resize()  # Resize image for smaller API calls
        if console is not None:
            console.print(f"Image post-resizing is {image.width} x {image.height} pixels (W x H).")

        prompt = textwrap.dedent(f"""\
            Get all points matching the following objects: {", ".join(queries)}. The label
            returned should be an identifying name for the object detected.

            The answer should follow the JSON format:
            [{{"point": , "label": }}, ...]

            The points are in [y, x] format normalized to 0-1000.
            """)

        start_time = time.time()
        json_output = self.call(image, prompt)

        points_data = []
        try:
            if json_output is not None:
                data = json.loads(json_output)
                points_data.extend(data)
        except json.JSONDecodeError:
            if console is not None:
                console.print_exception()
                console.print("[yellow]⚠️ Warning: Invalid JSON response. Skipping.[/yellow]")

        if console is not None:
            console.print(f"\nTotal processing time: {(time.time() - start_time):.4f} seconds.")
            console.print(str(points_data))

        return None  # TODO
