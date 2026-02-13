"""Implement the task and motion planning (TAMP) algorithm of Srivastava et al. (ICRA 2014).

Reference:
    S. Srivastava, E. Fang, L. Riano, R. Chitnis, S. Russell, and P. Abbeel,
    “Combined task and motion planning through an extensible planner-independent
    interface layer,” in 2014 IEEE International Conference on Robotics and
    Automation (ICRA), May 2014, pp. 639–646. doi: 10.1109/ICRA.2014.6906922.
"""

State = bool  # TODO: Implement real type
Pose = bool  # TODO: Implement real type
ErrFreeMode = bool  # TODO: Should be enum
PartialTrajMode = bool  # TODO: Should be enum
TaskPlan = bool  # TODO: Implement real type
Trajectory = bool  # TODO: Implement real type
Mode = bool  # TODO: Implement real type (enum or bool?)


def tamp(state: State, initial_pose: Pose) -> None:
    """Implement Algorithm 1 of Srivastava et al. (ICRA 2014)."""
    if task_plan is None:
        task_plan = task_planner.plan(state)
        step = 1
        partial_traj = None
        pose1 = initial_pose

    while not resource_limit_reached():
        refine_result = try_refine(pose1, task_plan, step, partial_traj, ErrFreeMode)
        if refine_result is not None:
            return refine_result

        while True:
            partial_traj, pose2, fail_step, fail_cause = try_refine(
                pose1,
                task_plan,
                step,
                partial_traj,
                PartialTrajMode,
            )
            state = state.update(fail_cause, fail_step)
            new_plan = task_planner.plan(state)
            if new_plan is not None:
                task_plan = task_plan[:fail_step] + new_plan
                pose1 = pose2
                step = fail_step

            if new_plan is not None or traj_count > max_traj_count:
                break

        if traj_count > max_traj_count:
            # Clear all learned facts from the initial state
            # Reset pose generators with new random seed
            # Reset step, partial_traj, pose1 to initial values
            pass


def try_refine(
    initial_pose: Pose,
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
        pose1 = initial_pose
    while step - 1 <= index <= len(task_plan):
        action = task_plan[index]
        next_action = task_plan[index + 1]
        pose2 = action.pose_generator.next()
        if pose2 is None:
            next_action.pose_generator.reset()
            pose1 = action.pose_generator.next()
            index -= 1
            traj = traj.delete_suffix_for(action)
            continue
        plan = motion_planner.plan(pose1, pose2)
        if plan is not None:
            if index == len(task_plan) + 1:
                return traj
            traj = traj + plan.computed_path
            index += 1
            pose1 = pose2
        elif mode == PartialTrajMode:
            return (pose1, traj, index + 1, plan.mp_errors)
