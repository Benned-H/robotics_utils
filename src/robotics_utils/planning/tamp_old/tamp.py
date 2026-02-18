"""Implement the task and motion planning (TAMP) algorithm of Srivastava et al. (ICRA 2014).

Reference:
    S. Srivastava, E. Fang, L. Riano, R. Chitnis, S. Russell, and P. Abbeel,
    “Combined task and motion planning through an extensible planner-independent
    interface layer,” in 2014 IEEE International Conference on Robotics and
    Automation (ICRA), May 2014, pp. 639–646. doi: 10.1109/ICRA.2014.6906922.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Generic, Protocol, TypeVar

StateT = TypeVar("StateT")
"""Type variable representing a low-level environment state."""

Fact = object  # TODO: Implement actual class
"""A fact is a grounded predicate that holds in a low-level state."""

AbstractState = set[Fact]
"""An abstract state is the set of grounded predicates (i.e., facts) that hold in a state."""

# TODO: Needs `apply_effects` and `revert_effects`
AbstractAction = object  # TODO: Implement actual class
"""An abstract action is a grounded symbolic action (i.e., grounded operator)."""

TaskPlan = list[AbstractAction]
"""A task plan is a sequence of planned abstract actions."""


class TaskPlanner(Protocol):
    """Define a general interface for task planners."""

    def plan(self, abstract_state: AbstractState) -> TaskPlan | None:
        """Plan from the given abstract state to the stored problem's goal state."""
        ...


class RobotAction(Protocol, Generic[StateT]):
    """A general interface for any low-level action to be executed on a robot."""

    def apply(self, curr_state: StateT) -> StateT:
        """Directly apply the effects of the robot action onto the low-level state.

        :param curr_state: Current low-level state (not to be modified)
        :return: Updated low-level state in which the action's effects have occurred
        """
        ...

    def execute(self, curr_state: StateT) -> None:
        """Execute the full action on the robot.

        :param curr_state: Current low-level state of the environment
        """
        ...


Trajectory = list[RobotAction[StateT]]
"""A trajectory is a sequence of low-level actions to be executed on a robot."""


@dataclass(frozen=True)
class MotionPlanningResult:
    """The result of a motion planning query."""

    trajectory: Trajectory | None = None
    """Solution trajectory found by the motion planner (or None if no solution was found)."""

    failure_causes: set[Fact] | None = None
    """Facts specifying the causes for motion planning failure (or None if planning succeeded)."""


class MotionPlanner(Protocol):
    """Define a general interface for motion planners."""

    def plan(self, start: StateT, goal: StateT) -> MotionPlanningResult:
        """Plan motions from the start state to the goal state."""
        ...


class RefineMode(Enum):
    """Enumeration of modes during refinement of abstract actions."""

    ERROR_FREE = 0
    PARTIAL_TRAJ = 1


@dataclass(frozen=True)
class RefinementNode(Generic[StateT]):
    """A node representing the state of refinement during task and motion planning."""

    task_plan: TaskPlan
    """Task plan of abstract actions to be refined into concrete actions."""

    curr_abstract_state: AbstractState
    """Current abstract state beyond which high-level actions still need to be refined."""

    curr_state: StateT
    """Current low-level state beyond which high-level actions still need to be refined."""

    refine_idx: int = 0
    """Index (in the task plan) of the next abstract action to be refined."""

    partial_traj: Trajectory = field(default_factory=list)
    """Low-level actions corresponding to successfully refined high-level actions."""

    def copy(self) -> RefinementNode[StateT]:
        """Create and return a deep copy of the refinement node."""
        return deepcopy(self)

    def backtrack_last(self, prev_state: StateT) -> RefinementNode[StateT]:
        """Backtrack one step and return the resulting refinement node.

        :param prev_state: TODO - What is this supposed to mean? How do we select it?
        :return: Refinement state after reversing the last refined action
        """
        last_refined_idx = self.refine_idx - 1  # Index of the last action we've refined
        prev_action = self.task_plan[last_refined_idx]

        reverted_abstract_state = prev_action.revert_effects(self.curr_abstract_state)
        reverted_partial_traj = self.partial_traj[:-1]  # TODO: Assumes 1:1 actions/trajectories

        return RefinementNode(
            task_plan=self.task_plan,
            curr_abstract_state=reverted_abstract_state,
            curr_state=prev_state,
            refine_idx=last_refined_idx,
            partial_traj=reverted_partial_traj,
        )

    def refine_next(self, refined_traj: Trajectory) -> RefinementNode[StateT]:
        """Advance the refinement node by one action using the given trajectory.

        :param refined_traj: Trajectory resulting from refining the current abstract action
        :return: Refinement state after the trajectory is appended and state is advanced
        """
        refined_action = self.task_plan[self.refine_idx]  # Abstract action that is now refined
        next_abstract_state = refined_action.apply_effects(self.curr_abstract_state)
        next_state = self.curr_state
        for traj_action in refined_traj:
            next_state = traj_action.apply(next_state)

        # TODO: This can easily be smarter: Include the abstract action index in any
        #   refined actions, use those tuples when reverting multi-trajectory abstract actions
        updated_traj = self.partial_traj + refined_traj

        return RefinementNode(
            task_plan=self.task_plan,
            curr_abstract_state=next_abstract_state,
            curr_state=next_state,
            refine_idx=self.refine_idx + 1,
            partial_traj=updated_traj,
        )


class TAMP(Generic[StateT]):
    """Implement the task and motion planning (TAMP) algorithm of Srivastava et al. (ICRA 2014)."""

    def __init__(self, tp: TaskPlanner, mp: MotionPlanner, max_attempts: int = 5000) -> None:
        """Initialize the task and motion planner.

        :param tp: Task planner for high-level symbolic planning
        :param mp: Motion planner for low-level motion planning
        :param max_attempts: Maximum number of refinement attempts before reset (default: 5000)
        """
        self.task_planner = tp
        self.motion_planner = mp

        self.max_refine_attempts = max_attempts
        """Maximum number of refinements attempted before planning is reset."""

    def resource_limit_reached(self) -> bool:
        """Check whether the resource limit for planning has been reached."""
        return False  # TODO: Implement actual logic

    def tamp(self, initial_state: StateT, initial_abstract: AbstractState) -> Trajectory | None:
        """Run task and motion planning from the given initial state.

        This method implements Algorithm 1 (pg. 643) of Srivastava et al. (ICRA 2014).

        :param initial_state: Initial low-level environment state in the problem
        :param initial_abstract: Abstract state corresponding to the initial low-level state
        :return: Low-level motion plan solving the problem, or None if no plan is found
        """
        task_plan = self.task_planner.plan(initial_abstract)
        if task_plan is None:
            return None

        # TODO: Initialize pose generators once

        curr_node = RefinementNode(task_plan, initial_abstract, initial_state)

        while not self.resource_limit_reached():
            error_free_result = self.try_refine(curr_node, RefineMode.ERROR_FREE)
            if error_free_result is not None:
                return error_free_result.partial_traj

            refine_attempts = 0
            while True:
                refined_node = self.try_refine(curr_node, RefineMode.PARTIAL_TRAJ)
                # TODO: Assumes that the failure causes were added into the abstract state already
                refine_attempts += 1

                assert refined_node is not None

                plan_suffix = self.task_planner.plan(refined_node.curr_abstract_state)

                if plan_suffix is not None:
                    new_task_plan = curr_node.task_plan[: refined_node.refine_idx] + plan_suffix
                    curr_node = replace(refined_node, task_plan=new_task_plan)

                    # TODO: Re-initialize pose generators (was in try_refine)

                if plan_suffix is not None or refine_attempts >= self.max_refine_attempts:
                    break

            if refine_attempts >= self.max_refine_attempts:
                # TODO: Reset pose generators with new random seed

                curr_node = RefinementNode(task_plan, initial_abstract, initial_state)

        return None

    def try_refine(self, curr_node: RefinementNode, mode: RefineMode) -> RefinementNode | None:
        """Attempt to refine the remaining abstract actions in the given node.

        This method implements Algorithm 2 (pg. 643) of Srivastava et al. (ICRA 2014).

        :param curr_node: Captures the current state of partial refinement during TAMP
        :param mode: Mode defining refinement behavior (i.e., error-free or partial trajectory)
        :return: Result of successful error-free or failed partial refinement, or None if
            error-free mode failed
        """
        # TODO: Local variables, pose generators persist across calls
        node = curr_node.copy()

        while curr_node.refine_idx <= node.refine_idx < len(node.task_plan):
            action = node.task_plan[node.refine_idx]
            next_action = node.task_plan[node.refine_idx + 1]
            target_state = next_action.pose_generator.next()
            if target_state is None:  # If generator runs out of possible poses, backtrack
                next_action.pose_generator.reset()

                # Backtrack one step of refinement in the task plan
                prev_state = action.pose_generator.next()
                node = node.backtrack_last(prev_state)
                continue

            plan_result = self.motion_planner.plan(node.curr_state, target_state)
            if plan_result.trajectory is not None:
                # Append the successful motion plan to the partial refinement
                node = node.refine_next(refined_traj=plan_result.trajectory)

                if node.refine_idx == len(node.task_plan):
                    return node  # If we've refined the final abstract action, return the result

            elif mode == RefineMode.PARTIAL_TRAJ:
                assert plan_result.failure_causes is not None
                node.curr_abstract_state.update(plan_result.failure_causes)

                return node

        return None
