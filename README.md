# robotics_utils

Catch-all repository for general-purpose robotics utilities.

## Demos

_Open-Vocab. Object Detection_ - Launch the open-vocabulary object detection demo by running:

```bash
uv venv && uv sync && uv pip install robotics-utils[vision]
uv run scripts/object_detection_demo.py interactive IMAGE_PATH
```

- Here, `IMAGE_PATH` specifies the path to the image you'd like to detect objects in.

## Development Commands

To set up the virtual environment for running the unit tests, run:

```bash
uv venv && uv sync && uv pip install robotics-utils[vision]
```

To generate a dependency graph for the codebase, run:

```bash
uv run pyreverse --colorized src/robotics_utils
```
