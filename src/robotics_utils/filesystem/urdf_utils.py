"""Define classes to import robot models from URDF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree.ElementTree import Element

from defusedxml.ElementTree import parse

from robotics_utils.kinematics.collision_models import (
    Box,
    CollisionModel,
    Cylinder,
    MeshData,
    PrimitiveShape,
    Rotate,
    Scale,
    Sphere,
    Translate,
)
from robotics_utils.kinematics.point3d import Point3D
from robotics_utils.kinematics.poses import Pose3D
from robotics_utils.kinematics.robot_model import Joint, JointType, Link, RobotModel


@dataclass(frozen=True)
class JointData:
    """A temporary data structure for joint data before the full URDF has been processed."""

    name: str
    joint_type: JointType
    parent_link: str
    child_link: str
    origin: Pose3D
    axis: Point3D
    effort: float
    lower_limit: float
    upper_limit: float


class URDFParser:
    """A parser for URDF files specifying robot models."""

    def __init__(self, urdf_path: Path) -> None:
        """Parse the given URDF file."""
        if not urdf_path.exists():
            raise FileNotFoundError(f"Cannot parse nonexistent URDF file: {urdf_path}")

        self.joint_data_map: dict[str, JointData] = {}  # Map joint names to their data
        self.child_to_parent: dict[str, str] = {}  # Map link names to parent frame names

        # self.joints: set[Joint] = set()
        # self.links: set[Link] = set()

        tree = parse(urdf_path)
        self.parse_robot(tree.getroot())

    def parse_robot(self, robot_elem: Element) -> RobotModel:
        """Parse a robot model from the given URDF XML element."""
        if robot_elem.tag != "robot":
            raise ValueError(f"Expected URDF element to have tag 'robot', not {robot_elem.tag}")

        # Parse all joints before links, so that links' parent frames are defined
        joint_elems = robot_elem.findall("joint")
        for joint_elem in joint_elems:
            joint_data = self.parse_joint_data(joint_elem)
            self.joint_data_map[joint_data.name] = joint_data
            self.child_to_parent[joint_data.child_link] = joint_data.parent_link

        link_elems = robot_elem.findall("link")
        for link_elem in link_elems:
            link = self.parse_link(link_elem)
            self.links.add(link)

        return RobotModel(joints=self.joints, links=self.links)

    def parse_joint_data(self, joint_elem: Element) -> JointData:
        """Parse data describing a joint from the given URDF XML element."""
        if joint_elem.tag != "joint":
            raise ValueError(f"Expected URDF element to have tag 'joint', not {joint_elem.tag}")

        joint_name = joint_elem.get("name")
        if joint_name is None:
            raise ValueError(f"Joint element is missing required attribute 'name': {joint_elem}")

        joint_type_string = joint_elem.get("type")
        if joint_type_string is None:
            raise ValueError(f"Joint element is missing required attribute 'type': {joint_elem}")
        joint_type = self.parse_joint_type(joint_type_string)

        parent_elem = joint_elem.find("parent")
        if parent_elem is None:
            raise ValueError(f"Joint element is missing parent link: {joint_elem}")
        parent_link = parent_elem.get("link")
        if parent_link is None:
            raise ValueError(f"Parent link element is missing attribute 'link': {parent_elem}")

        child_elem = joint_elem.find("child")
        if child_elem is None:
            raise ValueError(f"Joint element is missing child link: {joint_elem}")
        child_link = child_elem.get("link")
        if child_link is None:
            raise ValueError(f"Child link element is missing attribute 'link': {child_elem}")

        origin_elem = joint_elem.find("origin")  # Optional; default to identity if unspecified
        if origin_elem is None:
            origin_pose = Pose3D.identity(ref_frame=parent_link)
        else:
            origin_pose = self.parse_pose_3d(origin_elem, parent_frame=parent_link)

        axis_elem = joint_elem.find("axis")  # Optional; default to (1,0,0)
        if axis_elem is None:
            axis = Point3D(1, 0, 0)
        else:
            axis_string = axis_elem.get("xyz")
            if axis_string is None:
                raise ValueError(f"Axis element is missing attribute 'xyz': {axis_elem}")
            axis_list = self.parse_floats(axis_string, expected_len=3, description="axis")
            axis = Point3D.from_sequence(axis_list)

        limit_elem = joint_elem.find("limit")
        if limit_elem is None:  # Limits are required for revolute and prismatic joints
            raise ValueError(f"Joint element is missing required limits: {joint_elem}")

        effort_str = limit_elem.get("effort")
        if effort_str is None:
            raise ValueError(f"Limit element is missing attribute 'effort': {limit_elem}")
        effort = float(effort_str)

        lower_limit_str = limit_elem.get("lower")
        lower_limit = 0.0 if lower_limit_str is None else float(lower_limit_str)  # Default to 0
        upper_limit_str = limit_elem.get("upper")
        upper_limit = 0.0 if upper_limit_str is None else float(upper_limit_str)  # Default to 0

        return JointData(
            name=joint_name,
            joint_type=joint_type,
            parent_link=parent_link,
            child_link=child_link,
            origin=origin_pose,
            axis=axis,
            effort=effort,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )

    def parse_floats(self, string: str, expected_len: int, description: str) -> list[float]:
        """Parse space-separated floats from the given string.

        :param string: String split into a list of floats
        :param expected_len: Expected length of the resulting list
        :param description: Description of the expected values (for use in error message)
        """
        result = [float(f.strip()) for f in string.split()]
        if len(result) != expected_len:
            raise ValueError(
                f"Expected {expected_len} floats from {description} string, got {len(result)}",
            )
        return result

    def parse_pose_3d(self, origin_elem: Element, parent_frame: str) -> Pose3D:
        """Parse a 3D pose from the given URDF XML element."""
        if origin_elem.tag != "origin":
            raise ValueError(f"Expected URDF element to have tag 'origin', not {origin_elem.tag}")

        xyz_string = origin_elem.get("xyz")
        if xyz_string is None:
            raise ValueError(f"Origin element is missing attribute 'xyz': {origin_elem}")

        rpy_string = origin_elem.get("rpy")
        if rpy_string is None:
            raise ValueError(f"Origin element is missing attribute 'rpy': {origin_elem}")

        xyz_list = self.parse_floats(xyz_string, expected_len=3, description="xyz")
        rpy_list = self.parse_floats(rpy_string, expected_len=3, description="rpy")

        return Pose3D.from_list(xyz_list + rpy_list, ref_frame=parent_frame)

    def parse_joint_type(self, string: str) -> JointType:
        """Parse a joint type from the given string."""
        joint_type_map = {
            "revolute": JointType.REVOLUTE,
            "prismatic": JointType.PRISMATIC,
        }
        if string not in joint_type_map:
            raise ValueError(f"Unrecognized joint type: {string}")
        return joint_type_map[string]

    def parse_link(self, link_elem: Element) -> Link:
        """Parse a link from the given URDF XML element."""
        if link_elem.tag != "link":
            raise ValueError(f"Expected URDF element to have tag 'link', not {link_elem.tag}")

        link_name = link_elem.get("name")
        if link_name is None:
            raise ValueError(f"Link element is missing attribute 'name': {link_elem}")

        if link_name not in self.child_to_parent:
            raise ValueError(f"Link '{link_name}' did not specify a parent frame")
        parent_frame = self.child_to_parent[link_name]

        # Require links to define either a collision (preferred) or visual model
        collision_elem = link_elem.find("collision")
        visual_elem = link_elem.find("visual")
        preferred_model_elem = collision_elem if collision_elem is not None else visual_elem
        if preferred_model_elem is None:
            raise ValueError(f"Link element had neither a collision nor visual model: {link_elem}")

        # TODO: Multiple collision and visual are possible! Maybe only use one collection of two?

        return None  # TODO

    def parse_geometry_model(self, elem: Element, parent_frame: str) -> MeshData | PrimitiveShape:
        """Parse a geometry model (i.e., collision or visual) from the given URDF XML element."""
        if elem.tag not in {"collision", "visual"}:
            raise ValueError(f"Cannot parse geometry model from element: {elem.tag}")

        origin_elem = elem.find("origin")  # Optional; default to identity if unspecified
        origin = None if origin_elem is None else self.parse_pose_3d(origin_elem, parent_frame)

        geometry_elem = elem.find("geometry")
        if geometry_elem is None:
            raise ValueError(f"Element with tag '{elem.tag}' didn't define a geometry model")
        loaded_geometry: list[PrimitiveShape | MeshData] = []

        box_elem = geometry_elem.find("box")
        if box_elem is not None:
            box_size_str = box_elem.get("size")
            if box_size_str is None:
                raise ValueError(f"Box element is missing attribute 'size': {box_elem}")
            box_size = self.parse_floats(box_size_str, expected_len=3, description="box size")
            loaded_geometry.append(Box(*box_size))

        cylinder_elem = geometry_elem.find("cylinder")
        if cylinder_elem is not None:
            cylinder_radius_str = cylinder_elem.get("radius")
            if cylinder_radius_str is None:
                raise ValueError(f"Cylinder element didn't define a radius: {cylinder_elem}")
            cylinder_radius = self.parse_float(cylinder_radius_str, description="cylinder radius")

            length_string = cylinder_elem.get("length")
            if length_string is None:
                raise ValueError(f"Cylinder element didn't define a length: {cylinder_elem}")
            length = self.parse_float(length_string, description="cylinder length")
            loaded_geometry.append(Cylinder(height_m=length, radius_m=cylinder_radius))

        sphere_elem = geometry_elem.find("sphere")
        if sphere_elem is not None:
            sphere_radius_string = sphere_elem.get("radius")
            if sphere_radius_string is None:
                raise ValueError(f"Sphere element didn't define a radius: {sphere_elem}")
            sphere_radius = self.parse_float(sphere_radius_string, description="sphere radius")
            loaded_geometry.append(Sphere(radius_m=sphere_radius))

        mesh_elem = geometry_elem.find("mesh")
        if mesh_elem is not None:
            mesh_filename = mesh_elem.get("filename")
            if mesh_filename is None:
                raise ValueError(f"Mesh element didn't define a filename: {mesh_elem}")
            mesh_data = MeshData.from_mesh_path(Path(mesh_filename))  # TODO: Package-relative path

            scale_str = mesh_elem.get("scale")
            if scale_str is not None:
                s_x, s_y, s_z = self.parse_floats(scale_str, 3, description="mesh scale")
                scale = Scale((s_x, s_y, s_z))
                scale.apply(mesh_data.mesh)  # TODO: Cannot modify frozen dataclass in-place?

            loaded_geometry.append(mesh_data)

        if not loaded_geometry:
            raise ValueError(f"Geometry element didn't define any geometry: {geometry_elem}")
        if len(loaded_geometry) > 1:
            raise ValueError(f"Expected one loaded geometry model, got {len(loaded_geometry)}")

        # Apply the transform to account for the model's origin before returning
        if origin is not None:
            rotate = Rotate(origin.orientation.to_euler_rpy())
            translate = Translate(origin.position)

        return loaded_geometry[0]

    def parse_float(self, string: str, description: str) -> float:
        """Parse the given string into a float.

        :param string: String converted into a float
        :param description: Description of the expected value (for use in error message)
        """
        try:
            return float(string.strip())
        except ValueError as exc:
            raise ValueError(f"Couldn't parse {description} string into a float") from exc


#     def parse_collision_model(self, visual_elem: Element, urdf_path: Path) -> CollisionModel:
#         """Parse a collision model from the given URDF XML element."""
#         mesh_path_names = [mesh_e.attrib["filename"] for mesh_e in visual_elem.findall("mesh")]
#         full_mesh_paths = [urdf_path.parent / rel_path for rel_path in mesh_path_names]
#         meshes = [MeshData.from_mesh_path(p) for p in full_mesh_paths]

#         origin = self.parse_pose_3d(visual_elem.find("origin"))

#         return CollisionModel(meshes=meshes, primitives=[])  # TODO: Handle primitive shapes?

#     def parse_link(self, link_elem: Element) -> Link:
#         """Parse a link from the given URDF ML element."""
#         name = link_elem.get("name")
#         if name is None:
#             raise ValueError(f"Link element didn't specify a name: {link_elem}")

#         visual_elem = link_elem.find("visual")  # TODO: Handle collision and/or inertial

#         return Link(name, geometry=self.parse_collision_model())


# def main() -> None:
#     """Try loading a simple robot from URDF."""
#     urdf_path = "examples/spot_base_urdf/model.urdf"
#     tree = parse(urdf_path)

#     # Begin from the root of the tree, representing the full robot
#     robot_elem = tree.getroot()
#     if robot_elem.tag != "robot":  # Sanity-check our place in the URDF
#         raise ValueError(f"Expected root of URDF to have tag 'robot', not {robot_elem.tag}")

#     robot_name = robot_elem.attrib["name"]
#     # for child_elem in robot_elem:
#     #     # Expect links and joints
#     #     if child_elem.tag == "link"

#     root = tree.getroot()
#     for child in root:
#         print(child.tag, child.attrib)


# if __name__ == "__main__":
#     main()
