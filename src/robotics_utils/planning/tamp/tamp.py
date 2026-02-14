"""Implement the task and motion planning (TAMP) algorithm of Srivastava et al. (ICRA 2014).

Reference:
    S. Srivastava, E. Fang, L. Riano, R. Chitnis, S. Russell, and P. Abbeel,
    “Combined task and motion planning through an extensible planner-independent
    interface layer,” in 2014 IEEE International Conference on Robotics and
    Automation (ICRA), May 2014, pp. 639–646. doi: 10.1109/ICRA.2014.6906922.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, Protocol, TypeVar

StateT = TypeVar("StateT")
"""Type variable representing a low-level environment state."""

FactT = TypeVar("FactT")
"""Type variable representing a grounded predicate that holds in a low-level state."""

AbstractState = set[FactT]
"""An abstract state is the set of grounded predicates (i.e., facts) that hold in a state."""

AbstractActionT = TypeVar("AbstractActionT")
"""Type variable representing grounded symbolic actions (i.e., grounded operators)."""

TaskPlan = list[AbstractActionT]
"""A task plan is a sequence of planned abstract actions."""

KinematicState = bool  # TODO: Implement class
Trajectory = bool  # TODO: Implement real type
MPErrors = list[str]  # TODO: Implement real type


class TaskPlanner(Protocol):
    """Define a general interface for task planners."""

    def plan(self, abstract_state: AbstractState) -> TaskPlan | None:
        """Plan from the given abstract state to the stored problem's goal state."""
        ...


class MotionPlanner(Protocol):
    """Define a general interface for motion planners."""

    def plan(self, start: StateT, goal: StateT) -> Trajectory:
        """Plan motions from the start state to the goal state."""
        ...


class RefinementMode(Enum):
    """Enumeration of modes during refinement of abstract actions."""

    ERROR_FREE = 0
    PARTIAL_TRAJ = 1


@dataclass
class RefinementNode(Generic[StateT]):
    """A node representing the state of refinement during task and motion planning."""

    task_plan: TaskPlan
    """Task plan of abstract actions to be refined into concrete actions."""

    abstract_refine_from: AbstractState
    """Abstract state beyond which high-level actions still need to be refined."""

    refine_from: StateT
    """Low-level state beyond which high-level actions still need to be refined."""

    refine_idx: int = 0  # TODO: Replace `step`, which was initialized to 1 instead
    """Index (in the task plan) of the next abstract action to be refined."""

    partial_traj: Trajectory | None = None
    """Low-level actions corresponding to successfully refined high-level actions."""


@dataclass
class RefinementResult(Generic[StateT, FactT]):  # TODO: Combine these two classes soon
    """The outcome of an attempt at partial refinement of a task-level plan."""

    partial_traj: Trajectory
    """Low-level actions corresponding to successfully refined high-level actions."""

    refine_from: StateT
    """Low-level state beyond which high-level actions still need to be refined."""

    refine_idx: int
    """Index (in the task plan) of the next abstract action to be refined."""

    failure_causes: set[FactT] | None = None
    """Set of motion planning errors (stated as facts) that prevented refinement."""


class TAMP:
    """Implement the task and motion planning (TAMP) algorithm of Srivastava et al. (ICRA 2014)."""

    def __init__(self, tp: TaskPlanner, mp: MotionPlanner, max_traj_count: int = 5000) -> None:
        """Initialize the task and motion planner.

        :param tp: Task planner for high-level symbolic planning
        :param mp: Motion planner for low-level motion planning
        :param max_traj_count: Maximum number of refinement attempts before reset (default: 5000)
        """
        self.task_planner = tp
        self.motion_planner = mp

        self.max_traj_count = max_traj_count
        """Maximum number of refinements attempted before planning is reset."""

    def resource_limit_reached(self) -> bool:
        """Check whether the resource limit for planning has been reached."""
        return False  # TODO: Implement actual logic

    def tamp(self, initial_ll_state: StateT, initial_hl_state: AbstractState) -> Trajectory:
        """Implement Algorithm 1 of Srivastava et al. (ICRA 2014)."""
        task_plan = self.task_planner.plan(initial_hl_state)
        node1 = RefinementNode(task_plan, initial_hl_state, initial_ll_state)

        while not self.resource_limit_reached():
            error_free_result = self.try_refine(node1, RefinementMode.ERROR_FREE)
            if error_free_result is not None:
                return error_free_result.partial_traj

            traj_count = 0  # TODO: Rename to refine_attempts
            while True:
                # partial_traj, s2, fail_step, fail_cause

                partial_result = self.try_refine(node1, RefinementMode.PARTIAL_TRAJ)
                # TODO: Assumes that the failure causes were added into the abstract state already
                traj_count += 1

                new_hl_state = partial_result.hl_state
                new_plan_suffix = self.task_planner.plan(new_hl_state)

                if new_plan_suffix is not None:
                    node1 = RefinementNode(
                        task_plan=node1.task_plan[: partial_result.refine_idx] + new_plan_suffix,
                        abstract_refine_from=partial_result.abstract_refine_from,
                        refine_from=partial_result.refine_from,
                        refine_idx=partial_result.refine_idx,
                    )  # TODO: If all of this is the same, shouldn't we just combine the classes?

                if new_plan_suffix is not None or traj_count >= self.max_traj_count:
                    break

            if traj_count >= self.max_traj_count:
                # TODO: Reset pose generators with new random seed

                node1 = RefinementNode(task_plan, initial_hl_state, initial_ll_state)

    def try_refine(self, node: RefinementNode, mode: RefinementMode) -> RefinementResult | None:
        """Implement Algorithm 2 of Srivastava et al. (ICRA 2014)."""
        # TODO: Local variables, pose generators persist across calls
        if first_invocation or is_new(task_plan):
            refine_node = copy(node)
            # TODO: Initialize pose generators

        while node.refine_idx <= refine_node.refine_idx < len(task_plan):
            action = task_plan[refine_node.refine_idx]
            next_action = task_plan[refine_node.refine_idx + 1]
            s2 = action.pose_generator.next()
            if s2 is None:
                next_action.pose_generator.reset()

                # Backtrack the refinement node one step in the task plan
                refine_node.refine_from = action.pose_generator.next()
                refine_node.refine_idx -= 1
                refine_node.partial_traj.delete_suffix_for(action)
                continue
            plan_result = self.motion_planner.plan(refine_node.s1, s2)
            if plan is not None:
                if refine_node.refine_idx == len(node.task_plan):
                    return plan_result.trajectory
                refine_node.partial_traj += plan_result.computed_path
                refine_node.refine_idx += 1
                refine_node.refine_from = s2
            elif mode == RefinementMode.PARTIAL_TRAJ:
                return RefinementResult(
                    partial_traj=refine_node.partial_traj,
                    refine_from=refine_node.refine_from,
                    refine_idx=refine_node.refine_idx,
                    failure_causes=plan_result.mp_errors,
                )
