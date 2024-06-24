import numpy as np
from numpy.typing import NDArray
import os
import open3d as o3d
import code

def generate_collision_points(mesh_path: str, point_cloud_path: str, num_points: int=1000, transform: NDArray[np.float64]=None): # point_cloud_path should be ${body_dir}/${link_name}.xyz
    if mesh_path is not None:
        assert os.path.exists(mesh_path)
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if transform is not None:
            mesh.transform(transform)
        o3d.utility.random.seed(0)
        pcd = mesh.sample_points_uniformly(num_points)
    else:
        pcd = o3d.geometry.PointCloud()
    os.makedirs(os.path.dirname(point_cloud_path), exist_ok=True)
    o3d.io.write_point_cloud(point_cloud_path, pcd)


if __name__ == "__main__":
    galbot_robot_mesh_directory = "env/data/assets/galbot_one_simplified/meshes"
    galbot_robot_collision_directory = "env/data/assets/galbot_one_simplified/collision_points"
    files_stl = []
    files_xyz = []
    for root, dirs, files in os.walk(galbot_robot_mesh_directory):
        for file in files:
            files_stl.append(os.path.join(root, file))
            files_xyz.append(os.path.join(galbot_robot_collision_directory, file.replace('stl', 'xyz')))

        # code.interact(local=dict(globals(), **locals()))
    # root, dirs, files_stl  = os.walk(galbot_robot_mesh_directory)
            # items.append(os.path.join(galbot_robot_mesh_directory, file))
    # code.interact(local=dict(globals(), **locals()))
    # files_xyz = [file.replace('.stl', '.xyz') for file in files_stl]
    
    for i in range(len(files_xyz)):
        generate_collision_points(files_stl[i], files_xyz[i])
    # code.interact(local=dict(globals(), **locals()))

    generate_collision_points(None, os.path.join(galbot_robot_collision_directory, 'mobile_base.xyz'))
    generate_collision_points(None, os.path.join(galbot_robot_collision_directory, 'base_link_x.xyz'))
    generate_collision_points(None, os.path.join(galbot_robot_collision_directory, 'base_link_y.xyz'))
    generate_collision_points(None, os.path.join(galbot_robot_collision_directory, 'base_link_z.xyz'))
    generate_collision_points(None, os.path.join(galbot_robot_collision_directory, 'base_link.xyz'))
    generate_collision_points(None, os.path.join(galbot_robot_collision_directory, 'omni_chassis_leg_mount_link.xyz'))
    generate_collision_points(None, os.path.join(galbot_robot_collision_directory, 'leg_torso_mount_link.xyz'))
    generate_collision_points(None, os.path.join(galbot_robot_collision_directory, 'torso_head_mount_link.xyz'))
    generate_collision_points(None, os.path.join(galbot_robot_collision_directory, 'torso_right_arm_mount_link.xyz'))
    generate_collision_points(None, os.path.join(galbot_robot_collision_directory, 'torso_left_arm_mount_link.xyz'))
    generate_collision_points(None, os.path.join(galbot_robot_collision_directory, 'right_arm_end_effector_mount_link.xyz'))
    generate_collision_points(os.path.join(galbot_robot_mesh_directory, 'flange/flange_base_link.stl'), os.path.join(galbot_robot_collision_directory, 'right_flange_base_link.xyz'))
    generate_collision_points(None, os.path.join(galbot_robot_collision_directory, 'right_flange_mount_link.xyz'))
    generate_collision_points(None, os.path.join(galbot_robot_collision_directory, 'right_arm_camera_flange_link.xyz'))
    generate_collision_points(None, os.path.join(galbot_robot_collision_directory, 'right_suction_cup_base_link.xyz'))
    generate_collision_points(None, os.path.join(galbot_robot_collision_directory, 'right_suction_cup_link1.xyz'))
    generate_collision_points(None, os.path.join(galbot_robot_collision_directory, 'right_suction_cup_tcp_link.xyz'))
    generate_collision_points(None, os.path.join(galbot_robot_collision_directory, 'left_arm_end_effector_mount_link.xyz'))
    generate_collision_points(os.path.join(galbot_robot_mesh_directory, 'flange/flange_base_link.stl'), os.path.join(galbot_robot_collision_directory, 'left_flange_base_link.xyz'))
    generate_collision_points(None, os.path.join(galbot_robot_collision_directory, 'left_flange_mount_link.xyz'))
    generate_collision_points(os.path.join(galbot_robot_mesh_directory, 'camera/arm_camera.stl'), os.path.join(galbot_robot_collision_directory, 'left_arm_camera_flange_link.xyz'))
    generate_collision_points(os.path.join(galbot_robot_mesh_directory, 'gripper/gripper_hitbot_base_link.stl'), os.path.join(galbot_robot_collision_directory, 'left_gripper_base_link.xyz'))
    generate_collision_points(os.path.join(galbot_robot_mesh_directory, 'gripper/gripper_hitbot_left_link.stl'), os.path.join(galbot_robot_collision_directory, 'left_gripper_left_link.xyz'))
    generate_collision_points(os.path.join(galbot_robot_mesh_directory, 'gripper/gripper_hitbot_right_link.stl'), os.path.join(galbot_robot_collision_directory, 'left_gripper_right_link.xyz'))
    generate_collision_points(None, os.path.join(galbot_robot_collision_directory, 'left_gripper_tcp_link.xyz'))