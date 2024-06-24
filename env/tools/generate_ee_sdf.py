import os
import numpy as np
from numpy.typing import NDArray
import open3d as o3d
from omegaconf import OmegaConf
import code

from .sdf import gen_sdf
from .convex_decompose import convex_decompose
from ..utils.robot_kinematics import RobotKinematics, RobotKinematicsConfig

def load_collision_mesh(robot_kinematics: RobotKinematics, robot_dir: str, link_name: str, transform: NDArray[np.float64]=None) -> o3d.geometry.TriangleMesh:
    link = robot_kinematics.link_map[link_name]
    mesh_path = os.path.join(robot_dir, link.collision_file_path)
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if transform is not None:
        mesh.transform(transform)
    return mesh

def main():
    env_path = os.path.dirname(os.path.dirname(__file__))
    robot_dir = os.path.join(env_path, "data", "assets", "franka_panda")

    robot_kinematics_cfg: RobotKinematicsConfig = OmegaConf.to_object(OmegaConf.structured(RobotKinematicsConfig))
    robot_kinematics = RobotKinematics(robot_kinematics_cfg)
    hand_to_left_finger = robot_kinematics.joint_map["panda_finger_joint1"].get_parent_to_child(0.04)
    hand_to_right_finger = robot_kinematics.joint_map["panda_finger_joint2"].get_parent_to_child(0.04)
    hand_to_camera = robot_kinematics.joint_map["panda_hand_camera_joint"].parent_to_child_default

    hand_mesh = load_collision_mesh(robot_kinematics, robot_dir, "panda_hand")
    left_finger_mesh = load_collision_mesh(robot_kinematics, robot_dir, "panda_leftfinger")
    right_finger_mesh = load_collision_mesh(robot_kinematics, robot_dir, "panda_rightfinger", transform=np.diag(np.array([-1., -1., 1., 1.]))) # rotate 180 around z, as specified by link.collision.origin rpy="0 0 3.14159265359" xyz="0 0 0" in urdf
    camera_mesh = load_collision_mesh(robot_kinematics, robot_dir, "panda_hand_camera")

    left_finger_mesh.transform(hand_to_left_finger)
    right_finger_mesh.transform(hand_to_right_finger)
    camera_mesh.transform(hand_to_camera)

    ee_mesh = hand_mesh+left_finger_mesh+right_finger_mesh+camera_mesh
    ee_mesh_path = os.path.join(robot_dir, "ee", "model.obj")
    os.makedirs(os.path.dirname(ee_mesh_path), exist_ok=True)
    o3d.io.write_triangle_mesh(ee_mesh_path, ee_mesh)

    convex_decompose(ee_mesh_path)

    gen_sdf(os.path.join(os.path.dirname(ee_mesh_path), "decomp.obj"))

if __name__ == "__main__":
    main()

"""
python -m env.tools.generate_ee_sdf
"""