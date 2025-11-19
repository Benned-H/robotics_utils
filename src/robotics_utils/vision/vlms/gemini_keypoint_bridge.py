"""Bridge to call Gemini keypoint detection from ROS 1 Noetic (Python 3.8) environment.

This module provides a compatibility layer that allows Python 3.8 code to use
Gemini Robotics-ER 1.5 by spawning a subprocess with a newer Python version.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_utils.vision import PixelXY, RGBImage


class GeminiKeypointBridge:
    """Bridge class to call Gemini keypoint detection via subprocess.

    This class works in Python 3.8 but delegates the actual Gemini calls
    to a subprocess running Python 3.12+ with google-genai installed.
    """

    def __init__(self, uv_python: str = "3.12") -> None:
        """Initialize the Gemini keypoint bridge.

        :param uv_python: Python version for uv to use (defaults to "3.12")
        """
        self.uv_python = uv_python
        self.service_script = Path(__file__).parent / "gemini_keypoint_service.py"

        if not self.service_script.exists():
            raise FileNotFoundError(
                f"Gemini service script not found: {self.service_script}",
            )

    def detect_keypoints(
        self,
        image: RGBImage,
        queries: list[str],
    ) -> list[tuple[str, PixelXY]]:
        """Detect keypoints in an image using Gemini Robotics-ER.

        :param image: RGB image to detect keypoints in
        :param queries: List of text queries describing objects to detect
        :return: List of (query, keypoint) tuples where keypoint is (x, y) pixel coordinate
        :raises RuntimeError: If the subprocess call fails or returns invalid data
        """
        # Save image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            image.to_file(tmp_path)

        try:
            # Build the command to run via uv
            # uv will handle creating/managing the venv with Python 3.12
            # Dependencies are specified inline in the script using PEP 723 metadata
            cmd = [
                "uv",
                "run",
                "--python",
                self.uv_python,
                str(self.service_script),
                str(tmp_path),
                *queries,
            ]

            # Run the subprocess
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=60.0,  # 60 second timeout
            )

            # Check for errors
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                raise RuntimeError(
                    f"Gemini keypoint detection failed: {error_msg}",
                )

            # Parse the JSON output
            try:
                output = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Failed to parse JSON output: {e}\nOutput: {result.stdout}",
                ) from e

            if not output.get("success"):
                error = output.get("error", "Unknown error")
                raise RuntimeError(f"Gemini detection failed: {error}")

            # Extract keypoints from the response
            detections = output.get("detections", [])
            keypoints = []
            for det in detections:
                query = det["query"]
                kp = det["keypoint"]
                pixel_xy = (kp["x"], kp["y"])
                keypoints.append((query, pixel_xy))

            return keypoints

        finally:
            # Clean up temporary file
            if tmp_path.exists():
                tmp_path.unlink()

    def detect_handle_keypoint(
        self,
        image: RGBImage,
        handle_query: str = "door handle",
    ) -> PixelXY | None:
        """Detect a single door handle keypoint in an image.

        Convenience method specifically for door handle detection.

        :param image: RGB image containing a door handle
        :param handle_query: Text query for the handle (defaults to "door handle")
        :return: (x, y) pixel coordinate of the handle, or None if not detected
        """
        keypoints = self.detect_keypoints(image, [handle_query])

        if not keypoints:
            return None

        # Return the first detected keypoint
        _, pixel_xy = keypoints[0]
        return pixel_xy
