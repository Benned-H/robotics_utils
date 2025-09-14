"""Define a skills protocol for the Dorfl bimanual manipulator."""

from robotics_utils.abstractions.objects import BaseObjectType
from robotics_utils.skills import SkillsProtocol, skill_method


# Define object types for the "Spread PB" task
class Pickable(BaseObjectType): ...


class Bread(BaseObjectType): ...


class Jar(Pickable): ...


class Knife(Pickable): ...


class DorflSkillsProtocol(SkillsProtocol):
    """Define the structure of skills for Dorfl."""

    @skill_method
    def grasp(self, picked: Pickable) -> None:
        """Grasp and pick an object using one of Dorfl's grippers.

        :param picked: Object to be picked up
        """

    @skill_method
    def spread(self, bread: Bread, knife: Knife) -> None:
        """Spread peanut butter onto a slice of bread using a knife.

        :param bread: Slice of bread on which peanut butter is spread
        :param knife: Knife used to spread peanut butter
        """

    @skill_method
    def scoop(self, jar: Jar, knife: Knife) -> None:
        """Scoop peanut butter from a jar using a knife.

        :param jar: Jar from which peanut butter is scooped
        :param knife: Knife used to scoop peanut butter
        """

    @skill_method
    def open(self, jar: Jar) -> None:
        """Open the specified jar.

        :param jar: Jar to be opened
        """
