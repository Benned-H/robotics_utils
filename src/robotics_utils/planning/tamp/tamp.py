"""Implement the task and motion planning (TAMP) algorithm of Srivastava et al. (ICRA 2014).

Reference:
    S. Srivastava, E. Fang, L. Riano, R. Chitnis, S. Russell, and P. Abbeel,
    “Combined task and motion planning through an extensible planner-independent
    interface layer,” in 2014 IEEE International Conference on Robotics and
    Automation (ICRA), May 2014, pp. 639–646. doi: 10.1109/ICRA.2014.6906922.
"""

from __future__ import annotations

from dataclasses import dataclass
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
ErrFreeMode = bool  # TODO: Should be enum
PartialTrajMode = bool  # TODO: Should be enum
Trajectory = bool  # TODO: Implement real type
Mode = bool  # TODO: Implement real type (enum or bool?)
MPErrors = list[str]  # TODO: Implement real type


class TaskPlanner(Protocol):
    """Define a general interface for task planners."""

    def plan(self, abstract_state: AbstractState) -> TaskPlan:
        """Plan from the given abstract state to the stored problem's goal state."""
        ...


class MotionPlanner(Protocol):
    """Define a general interface for motion planners."""

    def plan(self, start: StateT, goal: StateT) -> Trajectory:
        """Plan motions from the start state to the goal state."""
        ...


@dataclass
class RefinementNode:
    """A node representing the state of refinement during task and motion planning."""

    task_plan: TaskPlan
    """Task plan of abstract actions to be refined into concrete actions."""

    step: int = 1
    """TODO: Exactly what does this integer represent?"""

    partial_traj: Trajectory | None = None
    """TODO: Exactly what does the partial trajectory mean?"""


@dataclass
class PartialRefinement(Generic[StateT]):
    """The outcome of an attempt at partial refinement of a task-level plan."""

    partial_traj: Trajectory
    """Low-level actions corresponding to successfully refined high-level actions."""

    refine_from: StateT
    """Low-level state beyond which high-level actions still need to be refined."""

    refine_idx: int
    """Index (in the task plan) of the next abstract action to be refined."""

    errors: MPErrors | None = None
    """List of motion planning errors that prevented refinement."""


class TAMP:
    """Implement the task and motion planning (TAMP) algorithm of Srivastava et al. (ICRA 2014)."""

    def __init__(self, task_planner: TaskPlanner, motion_planner: MotionPlanner) -> None:
        """Initialize the task and motion planner.

        :param task_planner: Used for high-level task planning
        :param motion_planner: Used for low-level motion planning
        """
        self.task_planner = task_planner
        self.motion_planner = motion_planner

    def resource_limit_reached(self) -> bool:
        """Check whether the resource limit for planning has been reached."""
        return False  # TODO: Implement actual logic

    def tamp(self, initial_ll_state: StateT, initial_hl_state: AbstractState) -> None:
        """Implement Algorithm 1 of Srivastava et al. (ICRA 2014)."""
        task_plan = self.task_planner.plan(initial_hl_state)
        step = 1
        partial_traj = None
        s1 = initial_ll_state

        while not self.resource_limit_reached():
            refine_result = self.try_refine(s1, task_plan, step, partial_traj, ErrFreeMode)
            if refine_result is not None:
                return refine_result

            while True:
                partial_traj, s2, fail_step, fail_cause = try_refine(
                    s1,
                    task_plan,
                    step,
                    partial_traj,
                    PartialTrajMode,
                )
                hl_state = hl_state.update(fail_cause, fail_step)
                new_plan = self.task_planner.plan(hl_state)
                if new_plan is not None:
                    task_plan = task_plan[:fail_step] + new_plan
                    s1 = s2
                    step = fail_step

                if new_plan is not None or traj_count >= max_traj_count:
                    break

            if traj_count >= max_traj_count:
                # TODO: Clear all learned facts from the initial state
                hl_state = initial_hl_state

                # TODO: Reset pose generators with new random seed

                step = 1
                partial_traj = None
                s1 = initial_ll_state

    def try_refine(
        self,
        initial_state: StateT,
        task_plan: TaskPlan,
        step: int,
        traj_prefix: Trajectory,
        mode: Mode,
    ) -> None:
        """Implement Algorithm 2 of Srivastava et al. (ICRA 2014)."""
        # TODO: Local variables, pose generators persist across calls
        if first_invocation or is_new(task_plan):
            index = step - 1
            traj = traj_prefix
            # TODO: Initialize pose generators
            s1 = initial_state
        while step - 1 <= index <= len(task_plan):
            action = task_plan[index]
            next_action = task_plan[index + 1]
            s2 = action.pose_generator.next()
            if s2 is None:
                next_action.pose_generator.reset()
                s1 = action.pose_generator.next()
                index -= 1
                traj = traj.delete_suffix_for(action)
                continue
            plan = motion_planner.plan(s1, s2)
            if plan is not None:
                if index == len(task_plan) + 1:
                    return traj
                traj = traj + plan.computed_path
                index += 1
                s1 = s2
            elif mode == PartialTrajMode:
                return (s1, traj, index + 1, plan.mp_errors)
