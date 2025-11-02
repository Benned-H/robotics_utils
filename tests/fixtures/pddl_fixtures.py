"""Define test fixtures providing example PDDL snippets."""

from textwrap import dedent

import pytest


@pytest.fixture
def briefcase_world_domain_partial() -> str:
    """Return a string containing an example `briefcase-world` PDDL (partial) domain.

    Reference: Section 2 (pg. 2) of Ghallab et al. (1998).
    """
    return dedent("""\
                  (define (domain briefcase-world)
                    (:requirements :strips :equality :typing :conditional-effects)
                    (:types location physob)
                    (:constants (B - physob))
                    (:predicates (at ?x - physob ?l - location)
                                 (in ?x ?y - physob))
                  )""")


@pytest.fixture
def mov_b_action() -> str:
    """Return a string containing an example `mov-b` PDDL action.

    Reference: Section 2 (pg. 2) of Ghallab et al. (1998).
    """
    return dedent("""\
                  (:action mov-b
                    :parameters (?m ?l - location)
                    :precondition (and (at B ?m) (not (= ?m ?l)))
                    :effect (and (at B ?l) (not (at B ? m))
                        (forall (?z)
                            (when (and (in ?z) (not (= ?z B)))
                                  (and (at ?z ?l) (not (at ?z ?m)))))) )""")


@pytest.fixture
def put_in_action() -> str:
    """Return a string containing an example `put-in` PDDL action.

    Reference: Section 2 (pg. 3) of Ghallab et al. (1998).
    """
    return dedent("""\
                  (:action put-in
                    :parameters (?x - physob ?l - location)
                    :precondition (not (= ?x B))
                    :effect (when (and (at ?x ?l) (at B ?l))
                        (in?x)) )""")


@pytest.fixture
def take_out_action() -> str:
    """Return a string containing an example `take-out` PDDL action.

    Reference: Section 2 (pg. 3) of Ghallab et al. (1998).
    """
    return dedent("""\
                  (:action take-out
                    :parameters (?x - physob)
                    :precondition (not (= ?x B))
                    :effect (not (in ?x)) )""")


@pytest.fixture
def get_paid_problem() -> str:
    """Return a string containing an example `get-paid` PDDL problem.

    Reference: Section 2 (pg. 3) of Ghallab et al. (1998).
    """
    return dedent("""\
                  (define (problem get-paid)
                    (:domain briefcase-world)
                    (:init (place home) (place office)
                        (object p) (object d) (object b)
                        (at B home) (at P home) (at D home) (in P))
                    (:goal (and (at B office) (at D office) (at P home))))""")
