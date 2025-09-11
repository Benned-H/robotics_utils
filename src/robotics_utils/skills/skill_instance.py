@dataclass(frozen=True)
class SkillInstance:
    """A skill instantiated using particular concrete objects."""

    skill: Skill
    """Specifies the skill instance's parameter signature."""

    bindings: Bindings
    """Maps each skill parameter name to the name of its bound object."""

    def __str__(self) -> str:
        """Return a readable string representation of the skill instance."""
        args_string = ", ".join(self.bindings[p.name] for p in self.skill.parameters)
        return f"{self.skill.name}({args_string})"

    @classmethod
    def from_string(
        cls,
        string: str,
        available_skills: SkillsInventory,
        objects: Objects,
    ) -> SkillInstance:
        """Construct a SkillInstance from the given string.

        :param string: String description of a skill instance
        :param available_skills: Inventory specifying the available skills
        :param objects: Collection of all objects in the environment
        :return: Constructed SkillInstance instance
        """
        match = re.match(r"^(\w+)\(([^)]*)\)$", string.strip())
        if not match:
            raise ValueError(f"Could not parse SkillInstance string: '{string}'.")

        skill_name = match.group(1)
        if skill_name not in available_skills.skills:
            raise ValueError(f"Invalid skill name parsed from string: '{skill_name}'.")
        skill = available_skills.skills[skill_name]

        args_string = match.group(2).strip()
        args = [arg.strip() for arg in args_string.split(",")] if args_string else []

        if len(skill.parameters) != len(args):
            len_param = len(skill.parameters)
            raise ValueError(f"Skill '{skill_name}' expects {len_param} args, not {len(args)}.")

        bindings: Bindings = {}
        for bound_object, param in zip(args, skill.parameters, strict=True):
            if bound_object not in objects:
                raise ValueError(f"Object '{bound_object}' not found in the environment.")

            obj_types = objects.get_types_of(bound_object)

            if param.object_type not in obj_types:
                raise ValueError(
                    f"Cannot parse skill instance from '{string}' because skill parameter "
                    f"'{param.name}' expects type {param.object_type} but the provided "
                    f"argument object '{bound_object}' only has type(s) {obj_types}.",
                )
            bindings[param.name] = bound_object

        return SkillInstance(skill, bindings)

    def execute(self, executor: SkillsProtocol) -> object | None:
        """Execute this skill instance."""
        return self.skill.execute(executor, self.bindings)
