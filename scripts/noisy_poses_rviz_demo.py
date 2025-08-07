"""Define a ROS node to visualize noisy averaged pose estimates in RViz."""

from robotics_utils.ros.transform_manager import TransformManager


def main() -> None:
    """Launch a ROS node to visualize example averaged poses in RViz."""
    TransformManager.init_node("noisy_poses_demo")


if __name__ == "__main__":
    main()
