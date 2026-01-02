# robotics_utils

Catch-all repository for general-purpose robotics utilities.

## Demos

### Open-Vocabulary Bounding Box Detection

Launch the open-vocabulary bounding box detection demo by running:

```bash
uv venv --clear && uv sync --extra vision
uv run scripts/bounding_box_detection_demo.py IMAGE_PATH
```

- `IMAGE_PATH` denotes the path to the image you'd like to detect objects in.
- To use the Gemini Robotics ER 1.5 backend, instead run:
  ```bash
  uv venv --clear && uv sync --extra gemini
  uv run scripts/bounding_box_detection_demo.py --backend gemini --api-key API_KEY IMAGE_PATH
  ```

### Open-Vocabulary Object Keypoint Detection

Launch the open-vocabulary keypoint detection demo by running:

```bash
uv venv --clear && uv sync --extra gemini
uv run scripts/keypoint_detection_demo.py API_KEY IMAGE_PATH
```

- `API_KEY` denotes your Google API key.
- `IMAGE_PATH` denotes the path to the image you'd like to detect objects in.

### Intel RealSense Demo

Install the dependencies for the RealSense demo using [these instructions](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md#installing-the-packages). Then reopen the terminal and reconnect the RealSense to the computer.

Launch the RealSense demo by running:

```bash
uv venv --clear && uv sync --extra realsense
uv run scripts/realsense_demo.py
```

### AprilTag Detection Demo

Run the AprilTag detection demo by running:

```bash
uv venv --clear && uv sync --extra vision
uv run scripts/apriltag_detection_demo.py
```

### PDDL Parsing

To set up the environment to run the PDDL-related code, use the command:

```bash
uv venv --clear --python 3.13 && uv sync
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
