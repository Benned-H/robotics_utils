"""Unit tests for classes representing visual fiducials."""

from pathlib import Path

import pytest

from robotics_utils.perception.sensors.visual_fiducials import VisualFiducialSystem


@pytest.fixture
def markers_yaml() -> Path:
    """Specify a path to an example environment YAML file."""
    yaml_path = Path(__file__).parent.parent / "test_data/markers_example.yaml"
    assert yaml_path.exists(), f"Expected to find file: {yaml_path}"
    return yaml_path


def test_visual_fiducial_system_from_yaml(markers_yaml: Path) -> None:
    """Verify that a VisualFiducialSystem can be loaded from an example YAML file."""
    system = VisualFiducialSystem.from_yaml(markers_yaml)

    # Assert - Expect that the loaded data matches what's specified in the YAML file
    assert len(system.markers) == 2

    marker_1 = system.markers.get(1)
    assert marker_1 is not None
    assert marker_1.id == 1
    assert marker_1.size_cm == pytest.approx(4)
    assert len(marker_1.relative_frames) == 2
    assert "eraser" in marker_1.relative_frames
    assert "table" in marker_1.relative_frames

    marker_12 = system.markers.get(12)
    assert marker_12 is not None
    assert marker_12.id == 12
    assert marker_12.size_cm == pytest.approx(18.5)
    assert len(marker_12.relative_frames) == 1
    assert "sink" in marker_12.relative_frames

    assert len(system.cameras) == 3
    frontleft = system.cameras.get("frontleft")
    assert frontleft is not None
    assert len(frontleft.recognized_sizes_cm) == 1
    assert any(s == pytest.approx(18.5) for s in frontleft.recognized_sizes_cm)

    frontright = system.cameras.get("frontright")
    assert frontright is not None
    assert len(frontright.recognized_sizes_cm) == 1
    assert any(s == pytest.approx(18.5) for s in frontright.recognized_sizes_cm)

    hand = system.cameras.get("hand")
    assert hand is not None
    assert len(hand.recognized_sizes_cm) == 2
    assert any(s == pytest.approx(4) for s in hand.recognized_sizes_cm)
    assert any(s == pytest.approx(18.5) for s in hand.recognized_sizes_cm)
