# robotics_utils

Catch-all repository for general-purpose robotics utilities.

## Demos

_Open-Vocabulary Bounding Box Detection_ - Launch the open-vocabulary bounding box detection demo by running:

```bash
uv venv --clear && uv sync --extra vision
uv run scripts/bounding_box_detection_demo.py interactive IMAGE_PATH
```

- Here, `IMAGE_PATH` specifies the path to the image you'd like to detect objects in.

_Intel RealSense Demo_ - Install the dependencies for the RealSense demo using [these instructions](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md#installing-the-packages). Then reopen the terminal and reconnect the RealSense to the computer.

Launch the RealSense demo by running:

```bash
uv venv --clear && uv sync --extra realsense
uv run scripts/realsense_demo.py
```

_AprilTag Detection Demo_ - Run the AprilTag detection demo by running:

```bash
uv venv --clear && uv sync --extra vision
uv run scripts/apriltag_detection_demo.py
```

## Development Commands

To set up the virtual environment for running the unit tests, run:

```bash
uv venv --clear && uv sync --extra vision
```

To generate a dependency graph for the codebase, run:

```bash
uv run pyreverse --colorized src/robotics_utils
```
