"""Define a function to specify a canonical order over ground literals."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_utils.abstractions.symbols.ground_atom import GroundLiteral
    from robotics_utils.abstractions.symbols.objects import ObjectSymbol


def canonical_order_key(g_literal: GroundLiteral, skill_args: tuple[ObjectSymbol, ...]) -> tuple:
    """Construct a key defining a canonical order over ground literals.

    This key enables deterministic sorting of ground literals during lifting,
    ensuring that transitions with the same "lifted structure" produce
    identical lifted effects regardless of concrete object names.

    The key captures:
    - Predicate name and negation status
    - Parameter types (ordered)
    - Equality pattern (which argument positions share the same object)
    - Role-index vector (which arguments correspond to skill instance arguments)

    :param g_literal: Ground literal to create a key for
    :param skill_args: Object arguments of the transition's skill instance
    :return: Tuple key for sorting ground literals
    """
    g_atom = g_literal.ground_atom

    name = g_atom.name
    negated = g_literal.negated
    param_types = tuple(arg.type_ for arg in g_atom.arguments)

    # Equality pattern: map each each index to the first index it's equal to
    first_pos: dict[ObjectSymbol, int] = {}  # Index where each argument first appears
    eq_pattern = tuple(first_pos.setdefault(arg, i) for i, arg in enumerate(g_atom.arguments))

    # Role-index vector: map each ground atom argument to its skill argument index (else -1)
    skill_obj_to_idx = {obj: i for i, obj in enumerate(skill_args)}
    role_idx_vec = tuple(skill_obj_to_idx.get(arg, -1) for arg in g_atom.arguments)

    return (name, negated, param_types, eq_pattern, role_idx_vec)
