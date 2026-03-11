"""Define a class to synchronize the MoveIt planning scene with an external kinematic state."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Iterator

import rospy
from moveit_commander import PlanningSceneInterface
from moveit_msgs.msg import AllowedCollisionEntry as AllowedCollisionEntryMsg
from moveit_msgs.msg import AllowedCollisionMatrix as AllowedCollisionMatrixMsg
from moveit_msgs.msg import CollisionObject as CollisionObjectMsg
from moveit_msgs.msg import PlanningSceneComponents as PlanningSceneComponentsMsg
from moveit_msgs.srv import (
    ApplyPlanningScene,
    ApplyPlanningSceneRequest,
    GetPlanningScene,
    GetPlanningSceneRequest,
)

from robotics_utils.collision_models.primitive_shapes import get_shape_center_pose_wrt_primitive
from robotics_utils.ros.msg_conversion import (
    pose_to_msg,
    primitive_shape_to_msg,
    trimesh_to_msg,
)
from robotics_utils.ros.transform_manager import TransformManager
from robotics_utils.spatial import Pose3D
from robotics_utils.states import GraspAttachment

if TYPE_CHECKING:
    from robotics_utils.motion_planning import MotionPlanningQuery
    from robotics_utils.states import ObjectCentricState, ObjectKinematicState


class PlanningSceneManager:
    """Synchronize MoveIt's planning scene while keeping collision-ignore logic explicit.

    Key ideas in this implementation:
    - World geometry state and collision-ignore policy are handled separately.
    - Object state synchronization still happens through `set_state` and `set_object_pose`.
    - Temporary collision ignores are applied via the allowed-collision matrix (ACM),
      scoped with `ignored_collisions(query)` so ignores are always reverted.
    - ACM updates are reference-counted, which makes nested ignore scopes safe.
    - Held (attached) objects are handled naturally by ACM updates, avoiding the
      remove/re-add hiding pattern that tends to cause synchronization drift.
    """

    def __init__(self, *, planning_frame: str) -> None:
        """Initialize an interface for the MoveIt planning scene.

        :param planning_frame: Reference frame used by MoveIt for motion planning
        """
        self.planning_frame = planning_frame
        self.planning_scene = PlanningSceneInterface()
        rospy.sleep(3)  # Allow time for the scene to initialize

        self._added_objects: set[str] = set()
        """Names of world objects that this manager has added to the planning scene."""

        self._ignored_object_counts: defaultdict[str, int] = defaultdict(int)
        """Reference count of objects currently ignored via ACM overrides."""

        self._baseline_acm: AllowedCollisionMatrixMsg | None = None
        """Saved ACM snapshot from before the current collision-ignore scope started."""

        self._grasp_attachments: dict[str, GraspAttachment] = {}
        """Map each attached object's name to its grasp attachment metadata."""

        self._get_scene_srv = rospy.ServiceProxy("/get_planning_scene", GetPlanningScene)
        self._apply_scene_srv = rospy.ServiceProxy("/apply_planning_scene", ApplyPlanningScene)

    def _set_object_state(self, obj_state: ObjectKinematicState) -> bool:
        """Set the kinematic state of an object in the MoveIt planning scene.

        :param obj_state: Kinematic state of the object to be updated
        :return: True if the object state was successfully set, else False
        """
        msg = self._make_collision_object_msg(obj_state)
        return self._add_object_msg(msg)

    def _add_object_msg(self, collision_obj_msg: CollisionObjectMsg) -> bool:
        """Add a moveit_msgs/CollisionObject message to the MoveIt planning scene.

        :param collision_obj_msg: ROS message representing an object's collision geometry
        :return: True if the object was successfully added, else False
        """
        self.planning_scene.add_object(collision_obj_msg)

        object_exists = self._wait_until_object_exists(collision_obj_msg.id)

        if object_exists:
            self._added_objects.add(collision_obj_msg.id)
        else:
            rospy.logerr(f"Failed to add '{collision_obj_msg.id}' to the MoveIt planning scene.")

        return object_exists

    def _remove_object(self, obj_name: str) -> bool:
        """Remove the named object from the MoveIt planning scene."""
        if obj_name not in self._added_objects:
            return False

        self.planning_scene.remove_world_object(obj_name)
        removed = self._wait_until_object_removed(obj_name)

        if removed:
            self._added_objects.discard(obj_name)

        return removed

    @contextmanager
    def ignored_collisions(self, query: MotionPlanningQuery) -> Iterator[None]:
        """Temporarily ignore object collisions while executing code inside the context."""
        ignored_objects = self._resolve_ignored_objects(query)
        if not ignored_objects:
            yield
            return

        if not self._acquire_ignored_collisions(ignored_objects):
            raise RuntimeError(f"Unable to apply collision ignores for query: {query}")

        try:
            yield
        finally:
            if not self._release_ignored_collisions(ignored_objects):
                rospy.logerr(f"Failed to restore ACM after collision-ignored query: {query}")

    def attach_object(
        self,
        *,
        obj_name: str,
        robot_name: str,
        ee_link_name: str,
        touch_links: list[str],
    ) -> bool:
        """Attach the named object to the given robot's specified end-effector.

        :param obj_name: Name of the object to be attached
        :param robot_name: Name of a robot
        :param ee_link_name: Name of the robot's relevant end-effector link
        :param touch_links: Names of the robot links allowed to touch the attached object
        :return: True if the object was successfully attached, else False
        """
        self.planning_scene.attach_object(obj_name, link=ee_link_name, touch_links=touch_links)
        is_attached = self._wait_until_object_attached(obj_name)

        if is_attached:
            pose_ee_o = TransformManager.lookup_transform(obj_name, ee_link_name)
            if pose_ee_o is None:
                rospy.logwarn(
                    "Unable to look up object pose while recording grasp attachment for "
                    f"'{obj_name}'; using identity pose relative to '{ee_link_name}' frame.",
                )
                pose_ee_o = Pose3D.identity(ee_link_name)

            self._grasp_attachments[obj_name] = GraspAttachment(
                obj_name=obj_name,
                robot_name=robot_name,
                ee_link_name=ee_link_name,
                pose_ee_o=pose_ee_o,
                touching_link_names=set(touch_links),
            )
        return is_attached

    def detach_object(self, *, obj_name: str, robot_name: str, ee_link_name: str) -> bool:
        """Detach an object from the specified robot end-effector.

        :param obj_name: Name of the object to be detached
        :param robot_name: Name of the robot that the object is attached to
        :param ee_link_name: Name of the robot's relevant end-effector link
        :return: True if the object was successfully detached, else False
        """
        grasp = self._grasp_attachments.get(obj_name)
        if grasp is None:
            rospy.logwarn(f"Cannot detach unattached object '{obj_name}' from '{robot_name}'.")
            return False

        if grasp.robot_name != robot_name or grasp.ee_link_name != ee_link_name:
            rospy.logwarn(
                f"Cannot detach '{obj_name}' from '{robot_name}:{ee_link_name}' because it is "
                f"attached to '{grasp.robot_name}:{grasp.ee_link_name}'.",
            )
            return False

        self.planning_scene.remove_attached_object(link=ee_link_name, name=obj_name)

        is_detached = self._wait_until_object_detached(obj_name)

        if is_detached:
            self._grasp_attachments.pop(obj_name, None)
        return is_detached

    def detach_all(self) -> bool:
        """Detach all objects in the planning scene.

        :return: True if all objects were successfully detached, else False
        """
        all_attached_objects = deepcopy(set(self._grasp_attachments.keys()))

        all_detached = True
        for obj_name in all_attached_objects:
            attachment = self._grasp_attachments.get(obj_name)
            if attachment is None:
                continue

            detached = self.detach_object(
                obj_name=obj_name,
                robot_name=attachment.robot_name,
                ee_link_name=attachment.ee_link_name,
            )
            all_detached = all_detached and detached

        return all_detached

    def get_attached_objects(self, robot_name: str) -> set[str]:
        """Retrieve the names of objects attached to the named robot (defaults to empty set)."""
        return {
            grasp.obj_name
            for grasp in self._grasp_attachments.values()
            if grasp.robot_name == robot_name
        }

    def _resolve_ignored_objects(self, query: MotionPlanningQuery) -> set[str]:
        """Resolve which object names should be ACM-ignored for a query."""
        if query.ignore_all_collisions:
            names = set(self.planning_scene.get_known_object_names())
            names.update(self._grasp_attachments.keys())
            return names
        return set(query.ignored_objects)

    def _acquire_ignored_collisions(self, ignored_objects: set[str]) -> bool:
        """Enter an ACM ignore scope for the given object names."""
        if not ignored_objects:
            return True

        if not self._validate_ignorable_objects(ignored_objects):
            return False

        if self._baseline_acm is None:
            self._baseline_acm = self._get_current_allowed_collision_matrix()
            if self._baseline_acm is None:
                return False

        for obj_name in ignored_objects:
            self._ignored_object_counts[obj_name] += 1

        if self._apply_current_ignored_collision_matrix():
            return True

        # Roll back local state if ACM application fails
        for obj_name in ignored_objects:
            count = self._ignored_object_counts.get(obj_name, 0)
            if count <= 1:
                self._ignored_object_counts.pop(obj_name, None)
            else:
                self._ignored_object_counts[obj_name] = count - 1

        if not self._ignored_object_counts:
            self._baseline_acm = None

        return False

    def _release_ignored_collisions(self, ignored_objects: set[str]) -> bool:
        """Exit an ACM ignore scope for the given object names."""
        if not ignored_objects:
            return True

        for obj_name in ignored_objects:
            count = self._ignored_object_counts.get(obj_name, 0)
            if count <= 1:
                self._ignored_object_counts.pop(obj_name, None)
            else:
                self._ignored_object_counts[obj_name] = count - 1

        # If there isn't a baseline ACM, expect that no objects are currently ignored
        if self._baseline_acm is None:
            return not self._ignored_object_counts

        if self._ignored_object_counts:
            return self._apply_current_ignored_collision_matrix()

        # No ignored objects remain: restore the baseline ACM
        success = self._apply_allowed_collision_matrix(self._baseline_acm)
        if success:
            self._baseline_acm = None
        return success

    def _validate_ignorable_objects(self, ignored_objects: set[str]) -> bool:
        """Check that each ignored object is either a known world object or an attached object."""
        known_world_objects = set(self.planning_scene.get_known_object_names())
        all_attached = set(self._grasp_attachments.keys())
        all_attached.update(self.planning_scene.get_attached_objects().keys())

        for obj_name in ignored_objects:
            if obj_name not in known_world_objects and obj_name not in all_attached:
                rospy.logwarn(
                    f"Cannot ignore unknown object '{obj_name}' in planning scene ACM.",
                )
                return False

        return True

    def _get_current_allowed_collision_matrix(self) -> AllowedCollisionMatrixMsg | None:
        """Retrieve MoveIt's current allowed-collision matrix."""
        request = GetPlanningSceneRequest()
        request.components.components = PlanningSceneComponentsMsg.ALLOWED_COLLISION_MATRIX

        try:
            response = self._get_scene_srv(request)
        except rospy.ServiceException as exc:
            rospy.logerr(f"Failed to retrieve planning scene ACM: {exc}")
            return None

        return deepcopy(response.scene.allowed_collision_matrix)

    def _apply_allowed_collision_matrix(self, acm: AllowedCollisionMatrixMsg) -> bool:
        """Apply the given allowed-collision matrix as a diff to the planning scene."""
        request = ApplyPlanningSceneRequest()
        request.scene.is_diff = True
        request.scene.allowed_collision_matrix = deepcopy(acm)

        try:
            response = self._apply_scene_srv(request)
        except rospy.ServiceException as exc:
            rospy.logerr(f"Failed to apply planning scene ACM: {exc}")
            return False

        if not response.success:
            rospy.logerr("MoveIt rejected ACM update via /apply_planning_scene.")
            return False

        return True

    def _apply_current_ignored_collision_matrix(self) -> bool:
        """Rebuild and apply an ACM from the stored baseline + current ignored object counts."""
        if self._baseline_acm is None:
            rospy.logerr("Cannot apply ACM ignores because baseline ACM is missing.")
            return False

        acm = deepcopy(self._baseline_acm)
        for obj_name in self._ignored_object_counts:
            self._set_object_ignored_in_acm(acm, obj_name=obj_name, ignored=True)

        return self._apply_allowed_collision_matrix(acm)

    def _set_object_ignored_in_acm(
        self,
        acm: AllowedCollisionMatrixMsg,
        *,
        obj_name: str,
        ignored: bool,
    ) -> None:
        """Set whether the given object should collide with any other entry in the ACM."""
        obj_idx = self._ensure_acm_entry(acm, obj_name)
        matrix_size = len(acm.entry_names)

        # Set the object's allowed collisions symmetrically
        for entry_idx in range(matrix_size):
            acm.entry_values[obj_idx].enabled[entry_idx] = ignored
            acm.entry_values[entry_idx].enabled[obj_idx] = ignored

        # Set default behavior for collisions against names not explicitly listed in the ACM
        if obj_name in acm.default_entry_names:
            default_idx = acm.default_entry_names.index(obj_name)
            acm.default_entry_values[default_idx] = ignored
        else:
            acm.default_entry_names.append(obj_name)
            acm.default_entry_values.append(ignored)

    def _ensure_acm_entry(self, acm: AllowedCollisionMatrixMsg, obj_name: str) -> int:
        """Ensure that the ACM has a matrix row and column for the object and return its index."""
        self._normalize_acm(acm)

        if obj_name in acm.entry_names:
            return acm.entry_names.index(obj_name)

        new_idx = len(acm.entry_names)
        acm.entry_names.append(obj_name)

        for row in acm.entry_values:
            row.enabled.append(False)  # Add a column for this new object to each row

        new_row = AllowedCollisionEntryMsg()
        new_row.enabled = [False] * (new_idx + 1)  # Also add a new row with an entry per object
        acm.entry_values.append(new_row)
        return new_idx

    def _normalize_acm(self, acm: AllowedCollisionMatrixMsg) -> None:
        """Normalize ACM dimensions so matrix operations are safe."""
        matrix_size = len(acm.entry_names)

        while len(acm.entry_values) < matrix_size:
            row = AllowedCollisionEntryMsg()
            row.enabled = [False] * matrix_size
            acm.entry_values.append(row)

        if len(acm.entry_values) > matrix_size:
            del acm.entry_values[matrix_size:]

        for row in acm.entry_values:
            if len(row.enabled) < matrix_size:
                row.enabled.extend([False] * (matrix_size - len(row.enabled)))
            elif len(row.enabled) > matrix_size:
                del row.enabled[matrix_size:]

    def set_object_pose(self, obj_name: str, pose: Pose3D) -> bool:
        """Update the pose of the named object in the MoveIt planning scene.

        :return: True if the object pose was successfully set, else False
        """
        pose_p_o = TransformManager.convert_to_frame(pose, self.planning_frame)

        move_object_msg = CollisionObjectMsg()
        move_object_msg.id = obj_name
        move_object_msg.operation = CollisionObjectMsg.MOVE
        move_object_msg.pose = pose_to_msg(pose_p_o)
        move_object_msg.header.frame_id = self.planning_frame
        move_object_msg.header.stamp = rospy.Time.now()

        return self._add_object_msg(move_object_msg)

    def set_state(self, state: ObjectCentricState, attempts_per_obj: int = 3) -> bool:
        """Update the MoveIt planning scene to reflect the given environment state."""
        # Skip objects currently attached to robot end-effectors
        all_attached: set[str] = set(self._grasp_attachments.keys())

        kinematic_states = state.available_kinematic_states
        known_objects = set(kinematic_states.keys())
        unknown_objects = set(state.object_names).difference(known_objects)

        # Remove any unattached objects in the planning scene that don't have a specified state
        to_be_removed = self._added_objects.difference(known_objects)
        to_be_removed.update(unknown_objects)
        to_be_removed.difference_update(all_attached)  # Don't remove attached objects

        all_removed = True
        for obj_name in to_be_removed:
            if obj_name in self._added_objects:
                removed = self._remove_object(obj_name)
                all_removed = all_removed and removed

        # Add updated versions of all unattached objects with known kinematic states
        all_added = True
        for kin_state in kinematic_states.values():
            if kin_state.name in all_attached:  # Skip attached objects, which move with the robot
                continue

            attempts_left = attempts_per_obj
            object_added = False
            while attempts_left and not object_added:
                object_added = self._set_object_state(kin_state)
                attempts_left -= 1

            all_added = all_added and object_added

        return all_added

    def _wait_until_object_exists(self, name: str, timeout_s: float = 10.0) -> bool:
        """Wait until the MoveIt planning scene contains the named object.

        :param name: Name of the object to find in the planning scene
        :param timeout_s: Timeout duration (seconds)
        :returns: True if the object appears in time, otherwise False
        """
        end_time = time.time() + timeout_s
        while time.time() < end_time:
            if name in self.planning_scene.get_known_object_names():
                return True
            time.sleep(0.1)

        return False

    def _wait_until_object_removed(self, name: str, timeout_s: float = 10.0) -> bool:
        """Wait until the named object is removed from the MoveIt planning scene.

        :param name: Name of the object to be removed from the planning scene
        :param timeout_s: Timeout duration (seconds)
        :returns: True if the object is removed in time, otherwise False
        """
        end_time = time.time() + timeout_s
        while time.time() < end_time:
            if name not in self.planning_scene.get_known_object_names():
                return True
            time.sleep(0.1)

        return False

    def _wait_until_object_attached(self, name: str, timeout_s: float = 10.0) -> bool:
        """Wait until the named object is attached in the MoveIt planning scene.

        :param name: Name of the object to check for attachment
        :param timeout_s: Timeout duration (seconds)
        :returns: True if the object is attached in time, otherwise False
        """
        end_time = time.time() + timeout_s
        while time.time() < end_time:
            if name in self.planning_scene.get_attached_objects():
                return True
            time.sleep(0.1)

        return False

    def _wait_until_object_detached(self, name: str, timeout_s: float = 10.0) -> bool:
        """Wait until the named object is detached in the MoveIt planning scene.

        :param name: Name of the object to check for detachment
        :param timeout_s: Timeout duration (seconds)
        :returns: True if the object is detached in time, otherwise False
        """
        end_time = time.time() + timeout_s
        while time.time() < end_time:
            if name not in self.planning_scene.get_attached_objects():
                return True
            time.sleep(0.1)

        return False

    def _make_collision_object_msg(
        self,
        object_state: ObjectKinematicState,
        object_type: str | None = None,
    ) -> CollisionObjectMsg:
        """Construct a moveit_msgs/CollisionObject message using the given data.

        :param object_state: Kinematic state of an object (i.e., its pose and collision model)
        :param object_type: Type of the object (e.g., `"box"`)
        :return: Constructed moveit_msgs/CollisionObject message
        """
        # Convert object pose into the target frame (i.e., planning frame)
        pose_t_o = TransformManager.convert_to_frame(object_state.pose, self.planning_frame)

        msg = CollisionObjectMsg()
        msg.id = object_state.name
        msg.header.frame_id = self.planning_frame
        msg.header.stamp = rospy.Time.now()
        msg.operation = CollisionObjectMsg.ADD

        if object_type is not None:
            msg.type.key = object_type  # Ignore 'db' field of message

        # Mesh and primitive poses are defined relative to msg.pose
        msg.pose = pose_to_msg(pose_t_o)

        msg.meshes = [trimesh_to_msg(mesh) for mesh in object_state.collision_model.meshes]
        msg.mesh_poses = [pose_to_msg(Pose3D.identity(object_state.name)) for _ in msg.meshes]

        msg.primitives = [
            primitive_shape_to_msg(ps) for ps in object_state.collision_model.primitives
        ]
        shape_local_poses = []
        for primitive, pose_o_p in zip(
            object_state.collision_model.primitives,
            object_state.collision_model.primitive_poses,
        ):
            transform_o_p = pose_o_p.to_homogeneous_matrix()
            transform_p_s = get_shape_center_pose_wrt_primitive(primitive).to_homogeneous_matrix()
            pose_o_s = Pose3D.from_homogeneous_matrix(
                transform_o_p @ transform_p_s,
                ref_frame=object_state.name,
            )
            shape_local_poses.append(pose_o_s)

        # Primitive poses are object-relative; msg.pose converts from object to planning frame
        msg.primitive_poses = [pose_to_msg(pose_o_s) for pose_o_s in shape_local_poses]

        return msg
