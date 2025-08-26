"""Script to simply call Spot skills."""

from robotics_utils.skills.gmu_skills import SpotSkillsExecutor


def main() -> None:
    """Run the specified skill(s)."""
    executor = SpotSkillsExecutor()
    executor.pick("waterbottle")


if __name__ == "__main__":
    main()
