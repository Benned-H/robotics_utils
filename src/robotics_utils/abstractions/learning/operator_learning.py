"""Define functions to induce symbolic operators from skill transition data.

This module implements algorithms for learning lifted STRIPS-style operators from
observed state transitions. The core approach:
    1. Partition transitions by their lifted effects (using canonical ordering for consistency)
    2. Learn preconditions by intersecting the lifted pre-states within each partition
    3. Return a set of learned operators, one per partition

Assumptions and limitations:
- STRIPS-style operators only (no conditional effects or quantifiers)
- Closed-world assumption for abstract states
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic

from robotics_utils.abstractions.grounding import ActionT
from robotics_utils.abstractions.learning.canonical_order import canonical_order_key
from robotics_utils.abstractions.learning.lifting import LiftingContext
from robotics_utils.abstractions.symbols.abstract_states import AbstractState, AbstractStateSpace
from robotics_utils.abstractions.symbols.ground_atom import GroundLiteral
from robotics_utils.abstractions.symbols.operators import Effects, Operator, Preconditions

if TYPE_CHECKING:
    from robotics_utils.abstractions.learning.transitions import AbstractTransition
    from robotics_utils.abstractions.symbols.discrete_parameter import DiscreteParameter
    from robotics_utils.abstractions.symbols.objects import ObjectMapping
    from robotics_utils.abstractions.symbols.predicate import Predicate


def lift_effects(
    transition: AbstractTransition[ActionT],
    obj_to_symbol: ObjectMapping,
) -> tuple[Effects, LiftingContext[ActionT]]:
    """Lift the ground effects of an abstract transition to a symbolic Effects signature.

    :param transition: Successful abstract transition with before and after states
    :param obj_to_symbol: Mapping from Python objects to ObjectSymbols
    :return: Tuple of (lifted Effects, LiftingContext used for lifting)
    :raises ValueError: If the given transition was unsuccessful (no post-state)
    """
    if not transition.success or transition.after is None:
        raise ValueError("Cannot lift effects of an unsuccessful transition.")

    # Compute the ground atoms added to or deleted from the abstract state
    add_atoms = transition.after.facts - transition.before.facts
    delete_atoms = transition.before.facts - transition.after.facts

    add_literals = [GroundLiteral(g_atom, negated=False) for g_atom in add_atoms]
    delete_literals = [GroundLiteral(g_atom, negated=True) for g_atom in delete_atoms]
    all_literals = add_literals + delete_literals

    # Create a lifting context, seeded using the transition action's arguments
    context = LiftingContext(transition.action, obj_to_symbol)

    # Sort by canonical key for deterministic parameter assignment
    sorted_literals = sorted(
        all_literals,
        key=lambda lit: canonical_order_key(lit, context.action_arguments),
    )

    for g_literal in sorted_literals:
        for arg in g_literal.ground_atom.arguments:
            context.lift_object(arg)

    # Create lifted effects using the lifting context
    add_predicates = frozenset(context.lift_ground_atom(g_atom) for g_atom in add_atoms)
    delete_predicates = frozenset(context.lift_ground_atom(g_atom) for g_atom in delete_atoms)

    return Effects(add=add_predicates, delete=delete_predicates), context


@dataclass(frozen=True)
class LiftedTransition(Generic[ActionT]):
    """A lifted abstract transition with its associated lifting context."""

    transition: AbstractTransition[ActionT]
    context: LiftingContext[ActionT]


@dataclass(frozen=True)
class PartitionID:
    """An identifier for a partition: (action name, lifted effects signature)."""

    action_name: str
    effects_signture: Effects


Partition = list[LiftedTransition[ActionT]]
Partitions = dict[PartitionID, Partition[ActionT]]
"""Groups abstract transitions by their lifted effects signatures."""


def learn_preconditions(partition: Partition, state_space: AbstractStateSpace) -> Preconditions:
    """Learn preconditions by intersecting lifted pre-states across transitions.

    :param partition: A group of abstract transitions with the same lifted effects
    :param state_space: Abstract state space defining all possible ground atoms
    :return: Learned preconditions (positive and negative)
    """
    if not partition:
        return Preconditions(positive=frozenset(), negative=frozenset())

    # Lift each pre-state using the corresponding lifting context
    lifted_before_states: list[frozenset[Predicate]] = []
    lifted_before_state_complements: list[frozenset[Predicate]] = []

    for lifted_t in partition:
        # Use the same context from effects lifting to maintain consistent parameters
        lifted_before_state = lifted_t.context.lift_abstract_state(lifted_t.transition.before)
        lifted_before_states.append(lifted_before_state)

        # Compute before-state complements with the abstract state space (closed-world assumption)
        absent_ground_atoms = state_space.possible_facts - lifted_t.transition.before.facts
        before_state_complement = AbstractState(frozenset(absent_ground_atoms))
        lifted_complement = lifted_t.context.lift_abstract_state(before_state_complement)
        lifted_before_state_complements.append(lifted_complement)

    # Compute the intersection of all lifted 'before' states
    positive_preconditions = lifted_before_states[0]
    for lifted_state in lifted_before_states[1:]:
        positive_preconditions = positive_preconditions.intersection(lifted_state)

    # Compute the intersection of all lifted 'before' state complements
    negative_preconditions = lifted_before_state_complements[0]
    for lifted_state_complement in lifted_before_state_complements[1:]:
        negative_preconditions = negative_preconditions.intersection(lifted_state_complement)

    return Preconditions(positive_preconditions, negative_preconditions)


def learn_operators(
    transitions: list[AbstractTransition[ActionT]],
    obj_to_symbol: ObjectMapping,
    state_space: AbstractStateSpace,
) -> set[Operator]:
    """Learn lifted operators from a collection of grounded abstract transitions.

    :param transitions: Collection of abstract transitions (may include multiple skills)
    :param obj_to_symbol: Mapping from Python objects to corresponding ObjectSymbols
    :param state_space: Abstract state space defining all possible ground atoms
    :return: Set of learned Operator instances
    """
    partitions: Partitions[ActionT] = defaultdict(list)

    # Partition transitions by their lifted effects
    for transition in transitions:
        if not transition.success:
            continue  # Skip unsuccessful transitions for now

        effects, context = lift_effects(transition, obj_to_symbol)
        partition_id = PartitionID(transition.action.name, effects)
        lifted_transition = LiftedTransition(transition, context)
        partitions[partition_id].append(lifted_transition)

    # Learn preconditions and operators for each partition
    learned_operators: set[Operator] = set()
    operator_name_counts: dict[str, int] = defaultdict(int)  # For naming: action name -> count

    for partition_id, partition in partitions.items():
        preconditions = learn_preconditions(partition, state_space)
        effects = partition_id.effects_signture

        # Determine operator name and parameters
        partition_idx = operator_name_counts[partition_id.action_name]
        operator_name_counts[partition_id.action_name] += 1
        operator_name = f"{partition_id.action_name}_{partition_idx}"

        # Collect all operator parameters from preconditions and effects
        all_params: set[DiscreteParameter] = set()
        for pred in preconditions.positive | preconditions.negative:
            all_params.update(pred.parameters)
        for pred in effects.add | effects.delete:
            all_params.update(pred.parameters)

        skill_params = [
            p for p in partition[0].transition.action.discrete_params if p in all_params
        ]
        aux_params = sorted((p for p in all_params if p not in skill_params), key=lambda p: p.name)

        # Include skill parameters in skill order, then auxiliary parameters alphabetically
        parameters = tuple(skill_params + aux_params)

        operator = Operator(operator_name, parameters, preconditions, effects)
        learned_operators.add(operator)

    return learned_operators
