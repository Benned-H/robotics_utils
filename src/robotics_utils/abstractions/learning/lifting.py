"""Define a class to manage the process of lifting grounded abstract transitions."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from robotics_utils.abstractions.learning.canonical_order import canonical_order_key
from robotics_utils.abstractions.symbols.discrete_parameter import DiscreteParameter
from robotics_utils.abstractions.symbols.ground_atom import GroundAtom, GroundLiteral
from robotics_utils.abstractions.symbols.predicate import Predicate

if TYPE_CHECKING:
    from robotics_utils.abstractions.symbols.abstract_states import AbstractState
    from robotics_utils.abstractions.symbols.objects import ObjectMapping, ObjectSymbol
    from robotics_utils.skills import SkillInstance


class LiftingContext:
    """Manages the state of the lifting process for a grounded abstract transition.

    The lifting context tracks which object symbols have been assigned to which
    lifted parameters, ensuring consistent parameter assignment per transition.
    """

    def __init__(self, skill_instance: SkillInstance, obj_to_symbol: ObjectMapping) -> None:
        """Initialize the lifting context based on the given skill instance.

        :param skill_instance: Skill instance providing concrete argument bindings
        :param obj_to_symbol: Maps Python objects to corresponding ObjectSymbols
        """
        discrete_params = skill_instance.skill.discrete_parameters

        self.skill_arguments = tuple(obj_to_symbol(arg) for arg in skill_instance.arguments)
        """Object arguments of the skill executed to create the transition."""

        self.obj_to_param = dict(zip(self.skill_arguments, discrete_params, strict=True))
        self.used_param_names = {p.name for p in self.obj_to_param.values()}

        self.type_to_next_param_idx: dict[str, int] = defaultdict(int)

    def lift_object(self, obj: ObjectSymbol) -> DiscreteParameter:
        """Lift a concrete object symbol to create a discrete parameter.

        :param obj: Concrete object symbol to lift
        :return: Discrete parameter corresponding to the lifted object in the transition
        """
        if obj in self.obj_to_param:  # Has this object been assigned a lifted parameter?
            return self.obj_to_param[obj]

        # If not, create a new parameter with a canonical name based on the object's type
        idx = self.type_to_next_param_idx[obj.type_]
        new_param_name = f"?{obj.type_}{idx}"

        # Ensure the name doesn't collide with existing parameters
        while new_param_name in self.used_param_names:
            idx += 1
            new_param_name = f"?{obj.type_}{idx}"

        new_param = DiscreteParameter(new_param_name, obj.type_)
        self.obj_to_param[obj] = new_param
        self.used_param_names.add(new_param_name)
        self.type_to_next_param_idx[obj.type_] = idx + 1

        return new_param

    def lift_ground_atom(self, ground_atom: GroundAtom) -> Predicate:
        """Lift a ground atom to an analogous predicate with lifted parameters.

        Each argument object of the ground atom is replaced by the corresponding lifted parameter.
        The resulting predicate uses the same name as the original but with parameter bindings
        derived from the lifting context for the transition.

        :param ground_atom: Ground atom (i.e., grounded predicate) to lift
        :return: Predicate with lifted parameter bindings
        """
        lifted_params = tuple(self.lift_object(arg) for arg in ground_atom.arguments)
        return Predicate(ground_atom.name, parameters=lifted_params)

    def lift_abstract_state(self, state: AbstractState) -> frozenset[Predicate]:
        """Lift the given abstract state using the lifting context.

        :param state: Set of ground atoms to be lifted
        :return: Lifted predicates corresponding to the abstract state's facts
        """
        ground_literals = [GroundLiteral(g_atom, negated=False) for g_atom in state.facts]
        sorted_literals = sorted(
            ground_literals,
            key=lambda lit: canonical_order_key(lit, self.skill_arguments),
        )

        for literal in sorted_literals:
            for arg in literal.ground_atom.arguments:
                self.lift_object(arg)

        return frozenset(self.lift_ground_atom(lit.ground_atom) for lit in sorted_literals)
