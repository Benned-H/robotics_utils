"""Implement the unification algorithm for first-order logic expressions.

Reference: Figure 9.1, Section 9.2, pg. 285 of AIMA (4th Ed.) by Stuart Russell and Peter Norvig
"""

from __future__ import annotations

from collections.abc import Sequence

from robotics_utils.classical_planning import (
    DiscreteParameter,
    PartialAtom,
    Predicate,
    PredicateInstance,
)
from robotics_utils.classical_planning.parameters import ObjectT, is_unbound

ParametersT = tuple[DiscreteParameter | ObjectT, ...]
"""Represents any tuple of parameters or (partially) bound arguments."""

PredicateT = Predicate | PredicateInstance | PartialAtom
"""Represents any "predicate-like" expression."""


def get_parameters(predicate: PredicateT) -> ParametersT:
    """Retrieve the parameters or arguments of a predicate-like expression."""
    if isinstance(predicate, Predicate):
        return predicate.parameters
    if isinstance(predicate, PredicateInstance):
        return predicate.arguments
    if isinstance(predicate, PartialAtom):  # Replace unbound values with corresponding parameters
        args_list = list(predicate.arguments)
        for idx, bound_value in enumerate(predicate.arguments):
            if is_unbound(bound_value):
                args_list[idx] = predicate.predicate.parameters[idx]

        return tuple(args_list)

    raise RuntimeError(f"Unexpected predicate type: {predicate} (type {type(predicate)}).")


Unifiable = DiscreteParameter | ParametersT | PredicateT | Sequence[PredicateT]

UnifierBindings = dict[str, Unifiable]
"""A mapping from parameter (i.e., variable) names to bound values."""


def unify(x: Unifiable, y: Unifiable, bindings: UnifierBindings | None) -> UnifierBindings | None:
    """Find parameter bindings to unify the given first-order logic (FOL) expressions.

    Reference: Figure 9.1, Section 9.2, pg. 285 of AIMA (4th Ed.) by Russell and Norvig.

    :param x: First of the two FOL expressions to be unified
    :param y: Second of the two FOL expressions to be unified
    :param bindings: Current set of parameter-value bindings (or None upon failure)
    :return: Substitution (i.e., bindings) that unifies the expressions, or None upon failure
    """
    if bindings is None:
        return None

    if type(x) is type(y) and x == y:
        return bindings

    if isinstance(x, DiscreteParameter):
        return unify_parameter(x, y, bindings)
    if isinstance(y, DiscreteParameter):
        return unify_parameter(y, x, bindings)

    if isinstance(x, PredicateT) and isinstance(y, PredicateT):
        return unify_predicates(x, y, bindings)

    if isinstance(x, Sequence) and isinstance(y, Sequence):
        return unify_sequences(x, y, bindings)

    return None  # Otherwise, the expressions cannot be unified; return None to indicate failure


def unify_parameter(
    param: DiscreteParameter,
    x: Unifiable,
    bindings: UnifierBindings,
) -> UnifierBindings | None:
    """Find a parameter binding to unify with the given first-order logic (FOL) expression.

    Reference: "Unify-Var" function in Figure 9.1, Section 9.2, pg. 285 of AIMA (Fourth Edition).

    :param param: Parameter for which a binding is found
    :param x: FOL expression to unify with
    :param bindings: Current set of parameter-value bindings
    :return: Updated bindings to unify with the expression, or None upon failure
    """
    if param.name in bindings:  # If the parameter already has a binding, use it
        bound_value = bindings[param.name]
        return unify(bound_value, x, bindings)

    if isinstance(x, DiscreteParameter) and x.name in bindings:
        x_bound_value = bindings[x.name]
        return unify(param, x_bound_value, bindings)

    if occurs_in(param, x):
        return None  # If the parameter occurs in the expression, unification is impossible

    bindings[param.name] = x
    return bindings


def occurs_in(param: DiscreteParameter, x: Unifiable) -> bool:
    """Check whether the given parameter occurs in a first-order logic (FOL) expression.

    :param param: Parameter searched for in the expression
    :param x: FOL expression searched for the parameter
    :return: True if the parameter occurs in the expression, else False
    """
    if isinstance(x, PredicateInstance):  # All PredicateInstance parameters are already bound
        return False

    if isinstance(x, str):  # Any string is already a bound value
        return False

    if isinstance(x, DiscreteParameter):
        return param == x

    if isinstance(x, Predicate):
        return param in x.parameters

    if isinstance(x, PartialAtom):
        return param in x.unbound_params

    if isinstance(x, Sequence):
        return any(occurs_in(param, item) for item in x)

    return False  # Otherwise, x is a ground expression or concrete argument


def unify_predicates(
    x: PredicateT,
    y: PredicateT,
    bindings: UnifierBindings,
) -> UnifierBindings | None:
    """Find parameter bindings to unify the given predicates.

    :param x: First predicate-like expression to unify
    :param y: Second predicate-like expression to unify
    :param bindings: Current set of parameter-value bindings
    :return: Updated bindings to unify the predicates, or None upon failure
    """
    if x.name != y.name:  # Cannot unify predicates with different names
        return None

    # If the predicates' names match, all that's left to unify are their parameters
    return unify_sequences(get_parameters(x), get_parameters(y), bindings)


def unify_sequences(
    x: Sequence[Unifiable],
    y: Sequence[Unifiable],
    bindings: UnifierBindings,
) -> UnifierBindings | None:
    """Find parameter bindings to unify the given sequences.

    :param x: First sequence of elements to be unified
    :param y: Second sequence of elements to be unified
    :param bindings: Current set of parameter-value bindings
    :return: Updated bindings to unify the sequences, or None upon failure
    """
    if len(x) != len(y):  # Cannot unify sequences with different lengths
        return None

    if not x:  # Empty sequences -> Return the bindings as given
        return bindings

    if len(x) == 1:
        (x_only,) = x
        (y_only,) = y
        return unify(x_only, y_only, bindings)

    # Otherwise, attempt to unify the first element in each sequence
    x_first, *x_rest = x
    y_first, *y_rest = y

    updated_bindings = unify(x_first, y_first, bindings)
    if updated_bindings is None:
        return None

    return unify_sequences(x_rest, y_rest, updated_bindings)
