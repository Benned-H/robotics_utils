"""Hacked solutions for Dorfl data processing (9/12)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from robotics_utils.abstractions.learning import SkillTransition
from robotics_utils.abstractions.objects.object_type import BaseObjectType
from robotics_utils.perception.vision import RGBImage
from robotics_utils.skills import (
    Skill,
    SkillInstance,
    SkillsInventory,
    SkillsProtocol,
    skill_method,
)


def load_skill_transition(
    parent_dir: Path,
    skill_instance: SkillInstance,
    idx: int,
) -> SkillTransition[RGBImage]:
    """Load the image for the specified skill transition.

    :param parent_dir: Parent directory of the image files
    :param skill_instance: Skill instance executed for the transition
    :param idx: Integer index identifying pre/post image pairs
    :return: Imported SkillTransition using RGB images as states
    """
    skill_name = skill_instance.skill.name.lower()

    paths: list[Path] = []
    for path in parent_dir.iterdir():
        if skill_name not in str(path):
            continue

        path_split = path.stem.split("_")
        if not path_split or path_split[-1] != str(idx):
            continue

        paths.append(path)

    if len(paths) != 2:
        raise RuntimeError(f"Found {len(paths)} images for {skill_instance} with index {idx}.")

    pre_path = None
    post_path = None
    for p in paths:
        if "pre" in str(p):
            pre_path = p
        if "post" in str(p):
            post_path = p

    if pre_path is None or post_path is None:
        raise RuntimeError(f"Could not find pre and post images in paths: {paths}.")

    return SkillTransition[RGBImage](
        state_before=RGBImage.from_file(pre_path),
        skill_instance=skill_instance,
        success=True,
        state_after=RGBImage.from_file(post_path),
    )


class Pickable(BaseObjectType): ...


class Bread(BaseObjectType): ...


class Jar(BaseObjectType): ...  # TODO: Make pickable once images include pre/post-grasp of jar


# class Jar(Pickable): ...


class Knife(Pickable): ...


class DorflSkillsProtocol(SkillsProtocol):
    """Define the structure of skills for Dorfl."""

    @skill_method
    def grasp(self, picked: Pickable) -> None:
        """Grasp and pick an object using one of Dorfl's grippers.

        :param picked: Object picked up
        """

    @skill_method
    def spread(self, bread: Bread, knife: Knife) -> None:
        """Spread PB onto a slice of bread using a knife.

        :param bread: Slice of bread spread with peanut butter
        :param knife: Knife used to spread the PB
        """

    @skill_method
    def scoop(self, jar: Jar, knife: Knife) -> None:
        """Scoop peanut butter from a jar using a knife.

        :param jar: Jar from which PB is scooped
        :param knife: Knife used to scoop peanut butter
        """

    @skill_method
    def open(self, jar: Jar) -> None:
        """Open the specified jar.

        :param jar: Jar to be opened
        """


Universe = dict[str, object]


def main() -> None:
    """Process example images from Dorfl into a format for learning visual S2S symbols."""
    images_path = Path.home() / "Downloads/dorfl_images"
    if not images_path.exists() and images_path.is_dir():
        raise RuntimeError(f"Cannot find directory at path: {images_path}")

    # Create skills based on the above Protocol
    dorfl_skills = SkillsInventory.from_protocol(DorflSkillsProtocol)
    print("Skills for Dorfl have been constructed as:")
    for skill in dorfl_skills:
        print(skill)
    print("\n\n")

    # Instantiate the skills using all objects in the environment
    knife = Knife("knife", visual_description="Metal knife with red plastic handle")
    pb_jar = Jar("pb_jar", visual_description="Peanut butter jar")
    bread = Bread("bread", visual_description="Slice of white bread")

    objects = [knife, pb_jar, bread]
    print(objects)

    instances_per_skill = {skill: skill.create_all_instances(objects) for skill in dorfl_skills}
    for skill, instances in instances_per_skill.items():
        instances_str = "\n\t".join(map(str, instances))
        print(f"\nSkill {skill} has been instantiated {len(instances)} times as: {instances_str}.")

    # Load all relevant transitions for each skill instance
    transitions_per_skill: dict[Skill, list[SkillTransition]] = defaultdict(list)
    for skill, all_instances in instances_per_skill.items():
        for i in range(1):
            transitions_per_skill[skill].extend(
                load_skill_transition(images_path, instance, i) for instance in all_instances
            )

    for skill, transitions in transitions_per_skill.items():
        print(f"Loaded {len(transitions)} transitions for skill '{skill.name}'.")


if __name__ == "__main__":
    main()
