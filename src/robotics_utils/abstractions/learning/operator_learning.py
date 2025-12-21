"""Operator learning algorithms for inducing symbolic operators from transition data.

This module implements algorithms for learning lifted STRIPS-style operators from
observed state transitions. The core approach:
1. Partition transitions by their lifted effects (using canonical ordering for consistency)
2. Learn preconditions by intersecting lifted "before" states within each partition
3. Return a set of learned operators, one per partition

Assumptions and limitations:
- STRIPS-style operators only (no conditional effects, no quantifiers)
- Closed-world assumption for abstract states
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from robotics_utils.abstractions.learning.transition_data import AbstractTransition
from robotics_utils.abstractions.symbols.discrete_parameter import DiscreteParameter
from robotics_utils.abstractions.symbols.ground_atom import GroundAtom, GroundLiteral
from robotics_utils.abstractions.symbols.objects import ObjectSymbol
from robotics_utils.abstractions.symbols.operators import Effects, Operator, Preconditions
from robotics_utils.abstractions.symbols.predicate import Predicate

if TYPE_CHECKING:
    from robotics_utils.abstractions.symbols.abstract_states import AbstractStateSpace
    from robotics_utils.skills.skill import Skill
    from robotics_utils.skills.skill_instance import SkillInstance


# =============================================================================
# Type Aliases
# =============================================================================

ObjectMapping = Callable[[object], ObjectSymbol]
"""Maps Python objects to their corresponding ObjectSymbol representations."""

Partition = dict[Effects, list[AbstractTransition]]
"""Groups abstract transitions by their lifted effects signature."""


# =============================================================================
# SkillSignature
# =============================================================================


@dataclass(frozen=True)
class SkillSignature:
    """A skill signature with discrete (PDDL-style) typed parameters.

    This class bridges the gap between Python-typed Skill parameters and the
    discrete symbolic parameters used in PDDL-style operator learning.
    """

    name: str
    """Name of the skill (should match the source Skill's name)."""

    parameters: tuple[DiscreteParameter, ...]
    """Discrete parameters corresponding to the skill's arguments."""

    @classmethod
    def from_skill(cls, skill: Skill) -> SkillSignature:
        """Construct a SkillSignature from a Python-typed Skill.

        :param skill: Source skill with Python-typed parameters
        :return: Corresponding SkillSignature with discrete parameters
        """
        discrete_params = tuple(
            DiscreteParameter(p.name, p.type_name, p.semantics) for p in skill.parameters
        )
        return cls(skill.name, discrete_params)

    def get_argument_symbols(
        self,
        skill_instance: SkillInstance,
        obj_to_symbol: ObjectMapping,
    ) -> tuple[ObjectSymbol, ...]:
        """Convert a skill instance's Python arguments to ObjectSymbols.

        :param skill_instance: Skill instance whose arguments to convert
        :param obj_to_symbol: Mapping from Python objects to ObjectSymbols
        :return: Tuple of ObjectSymbols corresponding to the skill arguments
        """
        return tuple(obj_to_symbol(arg) for arg in skill_instance.arguments)


# =============================================================================
# LiftingContext
# =============================================================================


@dataclass
class LiftingContext:
    """Manages the state of the lifting process for a single transition.

    The lifting context tracks which concrete objects have been assigned to which
    lifted parameters, ensuring consistent parameter assignment across effects
    and preconditions for the same transition.
    """

    obj_to_param: dict[ObjectSymbol, DiscreteParameter] = field(default_factory=dict)
    """Maps concrete objects to their assigned lifted parameters."""

    used_param_names: set[str] = field(default_factory=set)
    """Set of parameter names already in use (to avoid collisions)."""

    type_to_next_idx: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    """Next unused parameter index for each object type."""

    @classmethod
    def from_skill_instance(
        cls,
        skill_signature: SkillSignature,
        skill_instance: SkillInstance,
        obj_to_symbol: ObjectMapping,
    ) -> LiftingContext:
        """Create a LiftingContext seeded with skill parameter bindings.

        The skill's parameters serve as the "anchor" parameters - objects bound
        to skill arguments are assigned the corresponding skill parameter names.

        :param skill_signature: Signature providing discrete parameter definitions
        :param skill_instance: Skill instance providing concrete argument bindings
        :param obj_to_symbol: Mapping from Python objects to ObjectSymbols
        :return: LiftingContext seeded with skill parameter mappings
        """
        arg_symbols = skill_signature.get_argument_symbols(skill_instance, obj_to_symbol)

        obj_to_param = dict(zip(arg_symbols, skill_signature.parameters, strict=True))
        used_param_names = {p.name for p in skill_signature.parameters}

        return cls(
            obj_to_param=obj_to_param,
            used_param_names=used_param_names,
            type_to_next_idx=defaultdict(int),
        )

    def lift_object(self, obj: ObjectSymbol) -> DiscreteParameter:
        """Lift a concrete object to a discrete parameter.

        If the object has already been assigned a parameter, returns that parameter.
        Otherwise, creates a new parameter with a canonical name based on object type.

        :param obj: Concrete object to lift
        :return: Discrete parameter representing the lifted object
        """
        if obj in self.obj_to_param:
            return self.obj_to_param[obj]

        # Create a new parameter with a canonical name: ?{type}{index}
        idx = self.type_to_next_idx[obj.type_]
        new_param_name = f"?{obj.type_}{idx}"

        # Ensure the name doesn't collide with existing parameters
        while new_param_name in self.used_param_names:
            idx += 1
            new_param_name = f"?{obj.type_}{idx}"

        new_param = DiscreteParameter(new_param_name, obj.type_)
        self.obj_to_param[obj] = new_param
        self.used_param_names.add(new_param_name)
        self.type_to_next_idx[obj.type_] = idx + 1

        return new_param

    def lift_ground_atom(self, ground_atom: GroundAtom) -> Predicate:
        """Lift a ground atom to a predicate with lifted parameters.

        Each argument object is replaced by its corresponding lifted parameter.
        The resulting Predicate has the same name as the original but with
        parameter bindings derived from the lifting context.

        :param ground_atom: Ground atom to lift
        :return: Predicate with lifted parameter bindings
        """
        lifted_params = tuple(self.lift_object(arg) for arg in ground_atom.arguments)
        return Predicate(ground_atom.predicate.name, lifted_params)


# =============================================================================
# Canonical Ordering for Ground Literals
# =============================================================================


def canonical_order_key(
    literal: GroundLiteral,
    skill_instance: SkillInstance,
    obj_to_symbol: ObjectMapping,
) -> tuple:
    """Construct a key defining a canonical order over ground literals.

    This key enables deterministic sorting of ground literals for lifting,
    ensuring that transitions with the same "lifted structure" produce
    identical lifted effects regardless of concrete object names.

    The key captures:
    - Predicate name and negation status
    - Parameter types (order-preserving)
    - Equality pattern (which argument positions share the same object)
    - Role-index vector (which arguments correspond to skill parameters)

    :param literal: Ground literal to create a key for
    :param skill_instance: Skill instance providing argument context
    :param obj_to_symbol: Mapping from Python objects to ObjectSymbols
    :return: Tuple key for sorting
    """
    atom = literal.ground_atom

    name = atom.predicate.name
    negated = literal.negated
    param_types = tuple(arg.type_ for arg in atom.arguments)

    # Equality pattern: maps each position to the first position with the same object
    first_pos: dict[ObjectSymbol, int] = {}
    eq_pattern = tuple(first_pos.setdefault(arg, i) for i, arg in enumerate(atom.arguments))

    # Role-index vector: maps each argument to its skill parameter index (or -1)
    skill_arg_symbols = tuple(obj_to_symbol(arg) for arg in skill_instance.arguments)
    skill_obj_to_idx = {obj: i for i, obj in enumerate(skill_arg_symbols)}
    role_idx_vec = tuple(skill_obj_to_idx.get(arg, -1) for arg in atom.arguments)

    return (name, negated, param_types, eq_pattern, role_idx_vec)


# =============================================================================
# Effects Lifting
# =============================================================================


def _compute_ground_effects(
    pre_facts: frozenset[GroundAtom],
    post_facts: frozenset[GroundAtom],
) -> tuple[frozenset[GroundAtom], frozenset[GroundAtom]]:
    """Compute the ground add and delete effects between two abstract states.

    :param pre_facts: Facts in the abstract state before the transition
    :param post_facts: Facts in the abstract state after the transition
    :return: Tuple of (add_effects, delete_effects) as frozensets of GroundAtoms
    """
    add_effects = post_facts - pre_facts
    delete_effects = pre_facts - post_facts
    return add_effects, delete_effects


def lift_effects(
    transition: AbstractTransition,
    skill_signature: SkillSignature,
    obj_to_symbol: ObjectMapping,
) -> tuple[Effects, LiftingContext]:
    """Lift the ground effects of a transition to a symbolic Effects signature.

    This function:
    1. Computes ground add/delete effects from the transition's pre/post states
    2. Wraps them as GroundLiterals (negated=False for add, negated=True for delete)
    3. Sorts all literals by canonical key for deterministic parameter assignment
    4. Lifts each literal using a LiftingContext seeded from skill arguments

    :param transition: Abstract transition with pre/post states
    :param skill_signature: Skill signature providing parameter schema
    :param obj_to_symbol: Mapping from Python objects to ObjectSymbols
    :return: Tuple of (lifted Effects, LiftingContext used for lifting)
    :raises ValueError: If transition is unsuccessful (no post state)
    """
    if not transition.success or transition.post is None:
        raise ValueError("Cannot lift effects of an unsuccessful transition.")

    # Compute ground effects
    add_atoms, delete_atoms = _compute_ground_effects(
        transition.pre.facts,
        transition.post.facts,
    )

    # Wrap as GroundLiterals for unified sorting
    add_literals = [GroundLiteral(atom, negated=False) for atom in add_atoms]
    delete_literals = [GroundLiteral(atom, negated=True) for atom in delete_atoms]
    all_literals = add_literals + delete_literals

    # Sort by canonical key for deterministic parameter assignment
    sorted_literals = sorted(
        all_literals,
        key=lambda lit: canonical_order_key(lit, transition.skill_instance, obj_to_symbol),
    )

    # Create lifting context seeded from skill arguments
    ctx = LiftingContext.from_skill_instance(
        skill_signature,
        transition.skill_instance,
        obj_to_symbol,
    )

    # Lift each literal, assigning parameters in canonical order
    for literal in sorted_literals:
        for arg in literal.ground_atom.arguments:
            ctx.lift_object(arg)  # Ensures consistent parameter assignment

    # Now create the lifted effects
    add_predicates = frozenset(
        ctx.lift_ground_atom(lit.ground_atom) for lit in sorted_literals if not lit.negated
    )
    delete_predicates = frozenset(
        ctx.lift_ground_atom(lit.ground_atom) for lit in sorted_literals if lit.negated
    )

    return Effects(add_predicates, delete_predicates), ctx


# =============================================================================
# Precondition Learning
# =============================================================================


def lift_state(
    facts: frozenset[GroundAtom],
    ctx: LiftingContext,
    skill_instance: SkillInstance,
    obj_to_symbol: ObjectMapping,
) -> frozenset[Predicate]:
    """Lift a set of ground atoms to predicates using an existing lifting context.

    Objects already in the context use their assigned parameters. New objects
    (not seen in effects) are assigned new parameters in canonical order.

    :param facts: Ground atoms to lift (e.g., from an abstract state)
    :param ctx: Lifting context (typically from effects lifting)
    :param skill_instance: Skill instance for canonical ordering
    :param obj_to_symbol: Mapping from Python objects to ObjectSymbols
    :return: Frozenset of lifted predicates
    """
    # Wrap facts as non-negated literals for sorting
    literals = [GroundLiteral(atom, negated=False) for atom in facts]

    # Sort for deterministic parameter assignment to new objects
    sorted_literals = sorted(
        literals,
        key=lambda lit: canonical_order_key(lit, skill_instance, obj_to_symbol),
    )

    # Lift, assigning new parameters as needed
    for literal in sorted_literals:
        for arg in literal.ground_atom.arguments:
            ctx.lift_object(arg)

    return frozenset(ctx.lift_ground_atom(lit.ground_atom) for lit in sorted_literals)


def _clone_lifting_context(ctx: LiftingContext) -> LiftingContext:
    """Create a deep copy of a LiftingContext for independent lifting."""
    return LiftingContext(
        obj_to_param=dict(ctx.obj_to_param),
        used_param_names=set(ctx.used_param_names),
        type_to_next_idx=defaultdict(int, ctx.type_to_next_idx),
    )


def learn_preconditions(
    transitions: list[AbstractTransition],
    effects_contexts: list[LiftingContext],
    obj_to_symbol: ObjectMapping,
    state_space: AbstractStateSpace | None = None,
) -> Preconditions:
    """Learn preconditions by intersecting lifted before-states across transitions.

    For each transition, the before-state is lifted using the partition's
    parameter schema (from effects lifting). The intersection of these lifted
    states gives the common positive preconditions.

    Negative preconditions are computed similarly: for each transition, compute
    the atoms that are absent (using closed-world assumption), lift them, and
    intersect across all transitions. This requires an AbstractStateSpace to
    define the universe of possible atoms.

    :param transitions: Transitions in the same partition (same lifted effects)
    :param effects_contexts: Lifting contexts from effects lifting (one per transition)
    :param obj_to_symbol: Mapping from Python objects to ObjectSymbols
    :param state_space: Optional state space defining all possible ground atoms.
        If provided, negative preconditions are computed; otherwise empty.
    :return: Learned preconditions (positive and negative)
    """
    if not transitions:
        return Preconditions(frozenset(), frozenset())

    # Lift each before-state (positive) using its corresponding context
    lifted_positive_states: list[frozenset[Predicate]] = []
    lifted_negative_states: list[frozenset[Predicate]] = []

    for transition, ctx in zip(transitions, effects_contexts, strict=True):
        # Clone context for positive preconditions
        ctx_pos = _clone_lifting_context(ctx)
        lifted_positive = lift_state(
            transition.pre.facts,
            ctx_pos,
            transition.skill_instance,
            obj_to_symbol,
        )
        lifted_positive_states.append(lifted_positive)

        # Compute negative preconditions if state space is provided
        if state_space is not None:
            # Atoms absent from before-state (closed-world assumption)
            absent_atoms = frozenset(state_space.possible_facts - transition.pre.facts)

            # Clone context for negative preconditions (independent lifting)
            ctx_neg = _clone_lifting_context(ctx)
            lifted_negative = lift_state(
                absent_atoms,
                ctx_neg,
                transition.skill_instance,
                obj_to_symbol,
            )
            lifted_negative_states.append(lifted_negative)

    # Compute intersection of all lifted positive before-states
    positive_preconditions = lifted_positive_states[0]
    for lifted_state in lifted_positive_states[1:]:
        positive_preconditions = positive_preconditions & lifted_state

    # Compute intersection of all lifted negative before-states
    if lifted_negative_states:
        negative_preconditions = lifted_negative_states[0]
        for lifted_state in lifted_negative_states[1:]:
            negative_preconditions = negative_preconditions & lifted_state
    else:
        negative_preconditions = frozenset()

    return Preconditions(positive_preconditions, negative_preconditions)


# =============================================================================
# Main Operator Learning Algorithm
# =============================================================================


def learn_operators(
    transitions: list[AbstractTransition],
    obj_to_symbol: ObjectMapping,
    state_space: AbstractStateSpace | None = None,
) -> set[Operator]:
    """Learn lifted operators from a collection of abstract transitions.

    Algorithm overview:
    1. Compute lifted effects for each successful transition
    2. Partition transitions by their lifted effects signature
    3. For each partition, learn preconditions via intersection
    4. Return the set of learned operators

    :param transitions: Collection of abstract transitions (may span multiple skills)
    :param obj_to_symbol: Mapping from Python objects to ObjectSymbols
    :param state_space: Optional state space defining all possible ground atoms.
        If provided, negative preconditions are learned; otherwise only positive.
    :return: Set of learned Operator instances
    """
    # Cache skill signatures to avoid repeated conversion
    skill_signatures: dict[str, SkillSignature] = {}

    def get_signature(skill_instance: SkillInstance) -> SkillSignature:
        skill = skill_instance.skill
        if skill.name not in skill_signatures:
            skill_signatures[skill.name] = SkillSignature.from_skill(skill)
        return skill_signatures[skill.name]

    # Step 1-2: Partition transitions by lifted effects
    partition: Partition = defaultdict(list)
    transition_contexts: dict[int, LiftingContext] = {}  # Maps transition id to context

    for transition in transitions:
        if not transition.success:
            continue  # Skip unsuccessful transitions for now

        signature = get_signature(transition.skill_instance)
        effects, ctx = lift_effects(transition, signature, obj_to_symbol)

        partition[effects].append(transition)
        transition_contexts[id(transition)] = ctx

    # Step 3-4: Learn preconditions and create operators
    learned_operators: set[Operator] = set()
    partition_counts: dict[str, int] = defaultdict(int)  # For naming: skill_name -> count

    for effects, partition_transitions in partition.items():
        if not partition_transitions:
            continue

        # Get contexts for this partition's transitions
        contexts = [transition_contexts[id(t)] for t in partition_transitions]

        # Learn preconditions
        preconditions = learn_preconditions(
            partition_transitions,
            contexts,
            obj_to_symbol,
            state_space,
        )

        # Determine operator name and parameters
        skill_name = partition_transitions[0].skill_instance.skill.name
        partition_idx = partition_counts[skill_name]
        partition_counts[skill_name] += 1
        operator_name = f"{skill_name}{partition_idx}"

        # Collect all parameters from preconditions and effects
        all_params: set[DiscreteParameter] = set()
        for pred in preconditions.positive | preconditions.negative:
            all_params.update(pred.parameters)
        for pred in effects.add | effects.delete:
            all_params.update(pred.parameters)

        # Sort parameters for deterministic ordering
        # Skill parameters first (by original order), then auxiliary parameters by name
        skill_signature = get_signature(partition_transitions[0].skill_instance)
        skill_param_names = {p.name for p in skill_signature.parameters}

        skill_params = [p for p in skill_signature.parameters if p in all_params]
        aux_params = sorted(
            (p for p in all_params if p.name not in skill_param_names),
            key=lambda p: p.name,
        )
        parameters = tuple(skill_params + aux_params)

        operator = Operator(
            name=operator_name,
            parameters=parameters,
            preconditions=preconditions,
            effects=effects,
        )
        learned_operators.add(operator)

    return learned_operators
