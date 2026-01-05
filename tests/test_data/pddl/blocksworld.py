"""Implement the classic Blocksworld PDDL domain.

Reference: https://github.com/AI-Planning/classical-domains/blob/main/classical/blocks/domain.pddl
"""

from robotics_utils.abstractions.pddl import PDDLDomain
from robotics_utils.abstractions.symbols import (
    DiscreteParameter,
    Effects,
    Operator,
    Preconditions,
    Predicate,
)


def blocksworld_domain() -> PDDLDomain:
    """Construct and return the Blocksworld PDDL domain."""
    object_types = {"block"}  # All objects are blocks
    block_x = DiscreteParameter(name="x", type_="block")
    block_y = DiscreteParameter(name="y", type_="block")

    on_xy = Predicate("on", (block_x, block_y))
    ontable_x = Predicate("ontable", (block_x,))
    clear_x = Predicate("clear", (block_x,))
    handempty = Predicate("handempty", ())
    holding_x = Predicate("holding", (block_x,))

    predicates = {on_xy, ontable_x, clear_x, handempty, holding_x}

    pick_up = Operator(
        "pick-up",
        parameters=(block_x,),
        preconditions=Preconditions(positive=frozenset({clear_x, ontable_x, handempty})),
        effects=Effects(
            add=frozenset({holding_x}),
            delete=frozenset({ontable_x, clear_x, handempty}),
        ),
    )

    put_down = Operator(
        "put-down",
        parameters=(block_x,),
        preconditions=Preconditions(positive=frozenset({holding_x})),
        effects=Effects(
            add=frozenset({clear_x, handempty, ontable_x}),
            delete=frozenset({holding_x}),
        ),
    )

    clear_y = Predicate("clear", (block_y,))

    stack = Operator(
        "stack",
        parameters=(block_x, block_y),
        preconditions=Preconditions(positive=frozenset({holding_x, clear_y})),
        effects=Effects(
            add=frozenset({clear_x, handempty, on_xy}),
            delete=frozenset({holding_x, clear_y}),
        ),
    )

    unstack = Operator(
        "unstack",
        parameters=(block_x, block_y),
        preconditions=Preconditions(positive=frozenset({on_xy, clear_x, handempty})),
        effects=Effects(
            add=frozenset({holding_x, clear_y}),
            delete=frozenset({clear_x, handempty, on_xy}),
        ),
    )

    return PDDLDomain(
        name="blocksworld",
        requirements={":strips"},
        types=object_types,
        predicates=predicates,
        operators={pick_up, put_down, stack, unstack},
    )
