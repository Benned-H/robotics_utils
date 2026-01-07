"""Bridge to interface with Gemini Robotics-ER 1.5 from an older version of Python.

This module provides a compatibility layer that allows Python 3.8 code to use
Gemini Robotics-ER 1.5 by spawning a subprocess with a newer Python version.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import IO, TYPE_CHECKING, Literal

from robotics_utils.io import console, error_console
from robotics_utils.meta import ROBOTICS_UTILS_ROOT
from robotics_utils.vision.vlms.keypoint_detection import KeypointDetections

if TYPE_CHECKING:
    from rich.console import Console

    from robotics_utils.vision import RGBImage

DetectionType = Literal["keypoint"]

GEMINI_SCRIPTS = {
    "keypoints": Path(__file__).parent / "gemini_keypoint_service.py",
}


class GeminiRoboticsBridge:
    """Bridge class to call Gemini Robotics-ER 1.5 via subprocess.

    This class works in Python 3.8 but delegates the actual Gemini calls
    to a subprocess running Python 3.10+ with google-genai installed.
    """

    @staticmethod
    def call_script(api_key: str, script: Path, image_path: Path, queries: list[str]) -> dict:
        """Call Gemini Robotics-ER using the specified script and the given inputs.

        :param api_key: Google API key used to access Gemini Robotics-ER 1.5
        :param script: Path to the Python executable to be run as a subprocess
        :param image_path: Path to the RGB image used in the relevant detections
        :param queries: List of object text queries passed to Gemini
        :return: Dictionary of JSON data output from the subprocess
        """
        # Build a command to run the script via uv
        # Dependencies are specified inline in the script using PEP 723 metadata
        # Use --reinstall-package to pick up local edits to robotics_utils
        cmd = [
            "uv",
            "run",
            "--reinstall-package",
            "robotics_utils",
            str(script),
            str(image_path),
            *queries,
        ]

        env = os.environ.copy()
        env["GOOGLE_API_KEY"] = api_key
        env["ROBOTICS_UTILS_ROOT"] = str(ROBOTICS_UTILS_ROOT.resolve())

        # Run the subprocess with a 3-minute timeout (gives uv time to set up its venv)
        process = subprocess.Popen(
            cmd,
            env=env,  # Pass the API key while preserving PATH (allows finding uv)
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Stream stdout and stderr to the parent process in real-time
        stdout_lines = []
        stderr_lines = []

        def stream_output(
            pipe: IO[str] | None,
            console_obj: Console,
            lines_list: list[str],
        ) -> None:
            """Read lines from the given pipe and print them in real time."""
            if pipe:
                for line in iter(pipe.readline, ""):
                    console_obj.print(line.rstrip())
                    lines_list.append(line)
                pipe.close()

        # Create threads to stream both stdout and stderr simultaneously
        stdout_thread = threading.Thread(
            target=stream_output,
            args=(process.stdout, console, stdout_lines),
        )
        stderr_thread = threading.Thread(
            target=stream_output,
            args=(process.stderr, error_console, stderr_lines),
        )

        try:
            stdout_thread.start()
            stderr_thread.start()

            # Wait for process to complete with timeout
            process.wait(timeout=180.0)

            # Wait for threads to finish reading all output
            stdout_thread.join()
            stderr_thread.join()

        except subprocess.TimeoutExpired as timeout:
            process.kill()
            stdout_thread.join()
            stderr_thread.join()
            raise RuntimeError(f"Gemini script {script.name} timed out.") from timeout

        final_stdout_line = stdout_lines[-1].strip()
        try:  # Parse the JSON output
            json_output: dict = json.loads(final_stdout_line)
        except json.JSONDecodeError as err:
            stdout_text = "".join(stdout_lines).strip()
            stderr_text = "".join(stderr_lines).strip() if stderr_lines else ""
            error_msg = (
                f"Failed to parse JSON from {script.name}:\n\t{err}"
                f"\n\tstdout: {stdout_text}\n\tstderr: {stderr_text}"
            )
            raise RuntimeError(error_msg) from err

        # Check if the subprocess reported an error
        if not json_output.get("success"):
            error = json_output.get("error", "Unknown error")
            raise RuntimeError(f"Gemini script {script.name} failed:\n\t{error}")

        return json_output["output"]  # Extract and return detections JSON data

    @staticmethod
    def detect_keypoints(api_key: str, image: RGBImage, queries: list[str]) -> KeypointDetections:
        """Detect keypoints for the specified objects in the given image."""
        script = GEMINI_SCRIPTS["keypoints"]

        image_path = image.filepath
        if image_path is None:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                image_path = Path(tmp_file.name)
                image.to_file(image_path)

        try:
            result_json = GeminiRoboticsBridge.call_script(api_key, script, image_path, queries)
        finally:
            if image_path.parent.name == "tmp" and image_path.exists():
                image_path.unlink()

        return KeypointDetections.from_json(result_json)
