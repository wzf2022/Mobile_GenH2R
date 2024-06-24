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
    # for link_id in range(8):
    #     generate_collision_points(f"env/data/assets/franka_panda/meshes/collision/link{link_id}.obj", f"env/data/assets/franka_panda/collision_points/panda_link{link_id}.xyz")
    # generate_collision_points(f"env/data/assets/franka_panda/meshes/collision/hand.obj", f"env/data/assets/franka_panda/collision_points/panda_hand.xyz")
    # generate_collision_points(f"env/data/assets/franka_panda/meshes/collision/finger.obj", f"env/data/assets/franka_panda/collision_points/panda_leftfinger.xyz")
    # generate_collision_points(f"env/data/assets/franka_panda/meshes/collision/finger.obj", f"env/data/assets/franka_panda/collision_points/panda_rightfinger.xyz", transform=np.diag(np.array([-1., -1., 1., 1.]))) # rotate 180 around z, as specified by link.collision.origin rpy="0 0 3.14159265359" xyz="0 0 0" in urdf
    # generate_collision_points(f"env/data/assets/franka_panda/meshes/collision/camera.stl", f"env/data/assets/franka_panda/collision_points/panda_hand_camera.xyz")

    generate_collision_points(None, "env/data/assets/galbot_zero_lefthand/collision_points/mobile_base.xyz")
    generate_collision_points(None, "env/data/assets/galbot_zero_lefthand/collision_points/base_link_x.xyz")
    generate_collision_points(None, "env/data/assets/galbot_zero_lefthand/collision_points/base_link_y.xyz")
    generate_collision_points(None, "env/data/assets/galbot_zero_lefthand/collision_points/base_link_z.xyz")
    generate_collision_points(None, "env/data/assets/galbot_zero_lefthand/collision_points/base_link.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/chassis/hexman_chassis_base_link.stl", "env/data/assets/galbot_zero_lefthand/collision_points/hexman_chassis_base_link.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/chassis/hexman_chassis_right_wheel_link.stl", "env/data/assets/galbot_zero_lefthand/collision_points/hexman_chassis_right_wheel_link.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/chassis/hexman_chassis_left_wheel_link.stl", "env/data/assets/galbot_zero_lefthand/collision_points/hexman_chassis_left_wheel_link.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/chassis/hexman_chassis_caster_swivel_link1.stl", "env/data/assets/galbot_zero_lefthand/collision_points/hexman_chassis_caster_swivel_link1.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/chassis/hexman_chassis_caster_wheel_link1.stl", "env/data/assets/galbot_zero_lefthand/collision_points/hexman_chassis_caster_wheel_link1.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/chassis/hexman_chassis_caster_swivel_link2.stl", "env/data/assets/galbot_zero_lefthand/collision_points/hexman_chassis_caster_swivel_link2.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/chassis/hexman_chassis_caster_wheel_link2.stl", "env/data/assets/galbot_zero_lefthand/collision_points/hexman_chassis_caster_wheel_link2.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/chassis/hexman_chassis_caster_swivel_link3.stl", "env/data/assets/galbot_zero_lefthand/collision_points/hexman_chassis_caster_swivel_link3.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/chassis/hexman_chassis_caster_wheel_link3.stl", "env/data/assets/galbot_zero_lefthand/collision_points/hexman_chassis_caster_wheel_link3.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/chassis/hexman_chassis_caster_swivel_link4.stl", "env/data/assets/galbot_zero_lefthand/collision_points/hexman_chassis_caster_swivel_link4.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/chassis/hexman_chassis_caster_wheel_link4.stl", "env/data/assets/galbot_zero_lefthand/collision_points/hexman_chassis_caster_wheel_link4.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/chassis/hexman_chassis_lidar_link.stl", "env/data/assets/galbot_zero_lefthand/collision_points/hexman_chassis_lidar_link.xyz")
    generate_collision_points(None, "env/data/assets/galbot_zero_lefthand/collision_points/hexman_chassis_lift_fix_point_link.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/body/lift_base_link.stl", "env/data/assets/galbot_zero_lefthand/collision_points/lift_base_link.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/body/body_lift_link.stl", "env/data/assets/galbot_zero_lefthand/collision_points/body_lift_link.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/head/head_link.stl", "env/data/assets/galbot_zero_lefthand/collision_points/head_link.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/head/head_yaw_link.stl", "env/data/assets/galbot_zero_lefthand/collision_points/head_yaw_link.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/head/head_pitch_link.stl", "env/data/assets/galbot_zero_lefthand/collision_points/head_pitch_link.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/head/head_camera_link.stl", "env/data/assets/galbot_zero_lefthand/collision_points/head_camera_normal_frame.xyz")
    generate_collision_points(None, "env/data/assets/galbot_zero_lefthand/collision_points/head_camera_optical_frame.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/arm/realman_rm75_base_link.stl", "env/data/assets/galbot_zero_lefthand/collision_points/left_arm_base_link.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/arm/realman_rm75_link1.stl", "env/data/assets/galbot_zero_lefthand/collision_points/left_arm_link1.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/arm/realman_rm75_link2.stl", "env/data/assets/galbot_zero_lefthand/collision_points/left_arm_link2.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/arm/realman_rm75_link3.stl", "env/data/assets/galbot_zero_lefthand/collision_points/left_arm_link3.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/arm/realman_rm75_link4.stl", "env/data/assets/galbot_zero_lefthand/collision_points/left_arm_link4.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/arm/realman_rm75_link5.stl", "env/data/assets/galbot_zero_lefthand/collision_points/left_arm_link5.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/arm/realman_rm75_link6.stl", "env/data/assets/galbot_zero_lefthand/collision_points/left_arm_link6.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/arm/realman_rm75_link7.stl", "env/data/assets/galbot_zero_lefthand/collision_points/left_arm_link7.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/adapter/arm_end_effector_flange.stl", "env/data/assets/galbot_zero_lefthand/collision_points/left_arm_end_effector_flange_link.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/adapter/camera_d415_flange.stl", "env/data/assets/galbot_zero_lefthand/collision_points/left_arm_camera_d415_flange_link.xyz")
    generate_collision_points(None, "env/data/assets/galbot_zero_lefthand/collision_points/left_arm_camera_bottom_screw_frame.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/camera/camera_d415.stl", "env/data/assets/galbot_zero_lefthand/collision_points/left_arm_camera_link.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/adapter/gripper_inspire_flange.stl", "env/data/assets/galbot_zero_lefthand/collision_points/gripper_inspire_flange_link.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/gripper/gripper_inspire_body_link.stl", "env/data/assets/galbot_zero_lefthand/collision_points/gripper_inspire_body_link.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/gripper/gripper_inspire_left_link_1.stl", "env/data/assets/galbot_zero_lefthand/collision_points/gripper_inspire_left_link_1.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/gripper/gripper_inspire_right_link_1.stl", "env/data/assets/galbot_zero_lefthand/collision_points/gripper_inspire_right_link_1.xyz")
    generate_collision_points(None, "env/data/assets/galbot_zero_lefthand/collision_points/gripper_inspire_tcp_frame.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/arm/realman_rm75_6f_base_link.stl", "env/data/assets/galbot_zero_lefthand/collision_points/right_arm_base_link.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/arm/realman_rm75_6f_link1.stl", "env/data/assets/galbot_zero_lefthand/collision_points/right_arm_link1.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/arm/realman_rm75_6f_link2.stl", "env/data/assets/galbot_zero_lefthand/collision_points/right_arm_link2.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/arm/realman_rm75_6f_link3.stl", "env/data/assets/galbot_zero_lefthand/collision_points/right_arm_link3.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/arm/realman_rm75_6f_link4.stl", "env/data/assets/galbot_zero_lefthand/collision_points/right_arm_link4.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/arm/realman_rm75_6f_link5.stl", "env/data/assets/galbot_zero_lefthand/collision_points/right_arm_link5.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/arm/realman_rm75_6f_link6.stl", "env/data/assets/galbot_zero_lefthand/collision_points/right_arm_link6.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/arm/realman_rm75_6f_link7.stl", "env/data/assets/galbot_zero_lefthand/collision_points/right_arm_link7.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/adapter/arm_end_effector_flange.stl", "env/data/assets/galbot_zero_lefthand/collision_points/right_arm_end_effector_flange_link.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/adapter/camera_d415_flange.stl", "env/data/assets/galbot_zero_lefthand/collision_points/right_arm_camera_d415_flange_link.xyz")
    generate_collision_points(None, "env/data/assets/galbot_zero_lefthand/collision_points/right_arm_camera_bottom_screw_frame.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/camera/camera_d415.stl", "env/data/assets/galbot_zero_lefthand/collision_points/right_arm_camera_link.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/camera/camera_d415_plug_collision.stl", "env/data/assets/galbot_zero_lefthand/collision_points/right_arm_camera_usb_plug_link.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/sucker/long_sucker_base_link.stl", "env/data/assets/galbot_zero_lefthand/collision_points/long_sucker_base_link.xyz")
    generate_collision_points("env/data/assets/galbot_zero_lefthand/meshes/sucker/long_sucker_tool_link.stl", "env/data/assets/galbot_zero_lefthand/collision_points/long_sucker_tool_link.xyz")
    generate_collision_points(None, "env/data/assets/galbot_zero_lefthand/collision_points/long_sucker_tcp_link.xyz")
    code.interact(local=dict(globals(), **locals()))

"""
python -m env.tools.generate_collision_points
"""