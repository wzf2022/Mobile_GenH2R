import os
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List
from dataclasses import dataclass, field
from omegaconf import MISSING
from pybullet_utils.bullet_client import BulletClient
from scipy.spatial.transform import Rotation as Rt
import ipdb
import code

from .body import Body, BodyConfig
from .camera import Camera, CameraConfig
from .utils.transform import pos_ros_quat_to_mat, pos_euler_to_mat, mat_to_pos_ros_quat, se3_inverse, se3_transform_pc
from .utils.robot_kinematics import RobotKinematics, RobotKinematicsConfig
import  matplotlib.pyplot as plt

@dataclass
class GalbotConfig(BodyConfig):
    name: str = "galbot"
    # urdf_file: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "assets", "galbot_zero_lefthand", "galbot_zero_lefthand.urdf")
    urdf_file: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "assets", "galbot_one_simplified/galbot_one_10_DoF.urdf")
    urdf_file_7DoF: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "assets", "galbot_one_simplified/galbot_one_7_DoF.urdf")
    # collision
    use_self_collision: bool = True
    collision_mask: int = -1
    # base
    use_fixed_base: bool = True
    base_position: Tuple[float] = (0.0, -0.0, 0.)
    base_orientation: Tuple[float] = (0., 0., 0., 1.)
    # links
    num_dofs: int = 12
    # dof_default_position: Tuple[float] = (0., 0., 0., 0., np.pi/3, -np.pi/2, -2*np.pi/3, 0., np.pi/3, 0., 0.01, -0.01)
    # dof_default_position: Tuple[float] = (0., 0., 0., 
    #                                       0., -1.571, -1.571, -1.571, -3.0, 0., 0.,
    #                                         0.01, -0.01)
    dof_default_position: Tuple[float] = (0., 0., 0., 
                                          2.972, 1.422, -0.551, 2.00, 0.059, -0.667, -0.052,
                                            0.05, 0.05)
    # dof_default_position: Tuple[float] = (0.95581754, 0.26486026, -0.42207467,
    #                                       0.28342464, 1.24447524, 0.5107008, 1.1955415, -1.28844111, 1.47269412, -0.90335175, 
    #                                       0.05, 0.05)
    # dof_default_position: Tuple[float] = (0.99573455, 0.34089191, -0.5448275, -0.42518432, 1.34477468, 2.21821546, 0.83843199, 0.37696668, -1.08891333, -0.80624996, 0.05, 0.05)
    
    
    dof_position: Optional[Tuple[float]] = None
    dof_velocity: Tuple[float] = (0.0,)*12
    dof_max_force: Tuple[float] = (250.0,)*12
    dof_position_gain: Tuple[float] = (0.01,)*12
    dof_velocity_gain: Tuple[float] = (1.0,)*12

    # IK
    IK_solver: str = "PyKDL" # "pybullet"
    IK_solver_max_iter: int = 100
    IK_solver_eps: float = 1e-6
    # camera
    step_time: float = MISSING
    camera: CameraConfig = field(default_factory=lambda: CameraConfig(width=480, height=480, fov=90., near=0.001, far=10.0, step_time="${..step_time}")) # the format of nested structured configs must be like this https://omegaconf.readthedocs.io/en/2.3_branch/structured_config.html#nesting-structured-configs
    robot_kinematics: RobotKinematicsConfig = field(default_factory=lambda: RobotKinematicsConfig(
        urdf_file="${..urdf_file_7DoF}",
        IK_solver_max_iter="${..IK_solver_max_iter}", 
        IK_solver_eps="${..IK_solver_eps}",
        chain_tip="left_gripper_acronym_link",               # left_arm_link7(31),  gripper_inspire_tcp_frame(40)
        # left_gripper_tcp_link   48
        # left_gripper_base_link   45
        # left_arm_link7        41
    ))

class Galbot(Body):
    def __init__(self, bullet_client: BulletClient, cfg: GalbotConfig):
        super().__init__(bullet_client, cfg)
        self.cfg: GalbotConfig
        # self.ee_link_id = 31              # 40
        self.ee_link_id = 50
        self.torso_link_id = 12
        self.robot_base_id = 3
        # self.fingers_link_id = (38, 39)
        self.fingers_link_id = (47, 48)
        # self.head_camera_link_id = 22                 # head_camera_normal_frame 22     head_camera_optical_frame 23 
        # self.wrist_camera_link_id = 35
        self.head_camera_link_id = 33                 
        self.wrist_camera_link_id = 52
        self.world_to_base = pos_ros_quat_to_mat(cfg.base_position, cfg.base_orientation)
        self.base_to_world = se3_inverse(self.world_to_base)

        self.head_camera = Camera(bullet_client, cfg.camera)
        self.wrist_camera = Camera(bullet_client, cfg.camera)

        self.hand_to_camera = np.eye(4)
        self.hand_to_camera[:3, 3] = (0.036, 0.0, 0.036)
        self.hand_to_camera[:3, :3] = Rt.from_euler("XYZ", (0.0, 0.0, np.pi/2)).as_matrix()

        if cfg.IK_solver == "PyKDL":
            self.robot_kinematics = RobotKinematics(cfg.robot_kinematics)

    def reset(self, joint_state: Optional[Tuple[float]]=None):
        self.base_reset()
        if joint_state is not None:
            self.cfg.dof_position = joint_state
        else:
            self.cfg.dof_position = self.cfg.dof_default_position
        self.load()
        # for j in range(self.bullet_client.getNumJoints(self.body_id)):
        #     print(j, self.bullet_client.getJointInfo(self.body_id, j)[12].decode())
        self.set_dof_target(self.cfg.dof_position)

        camera_link_state = self.bullet_client.getLinkState(self.body_id, self.head_camera_link_id, computeForwardKinematics=1)
        self.head_camera.update_pose(camera_link_state[4], camera_link_state[5])
        # rotation = Rt.from_euler('z', 90, degrees=True)
        # new_quat = (rotation * Rt.from_quat(np.array(camera_link_state[5]))).as_quat()
        # self.head_camera.update_pose(camera_link_state[4], new_quat)

        camera_link_state = self.bullet_client.getLinkState(self.body_id, self.wrist_camera_link_id, computeForwardKinematics=1)
        self.head_camera.update_pose(camera_link_state[4], camera_link_state[5])
        # rotation = Rt.from_euler('z', 90, degrees=True)
        # new_quat = (rotation * Rt.from_quat(np.array(camera_link_state[5]))).as_quat()
        # self.wrist_camera.update_pose(camera_link_state[4], new_quat)


    def pre_step(self, dof_target_position):
        # print(abs(self.get_joint_positions() - dof_target_position))
        self.set_dof_target(dof_target_position)
        self.head_camera.pre_step()
        self.wrist_camera.pre_step()
    
    def get_wrist_camera_pos_orn(self):
        camera_link_state = self.bullet_client.getLinkState(self.body_id, self.wrist_camera_link_id, computeForwardKinematics=1)
        return camera_link_state[4], camera_link_state[5]
        rotation = Rt.from_euler('X', 180, degrees=True)
        # new_quat = (rotation * Rt.from_quat(np.array(camera_link_state[5]))).as_quat()
        new_quat = (Rt.from_quat(np.array(camera_link_state[5])) * rotation).as_quat()
        return camera_link_state[4], new_quat
    
        return np.array(camera_link_state[4]), np.array(camera_link_state[5])
    
    def get_head_camera_pos_orn(self):
        camera_link_state = self.bullet_client.getLinkState(self.body_id, self.head_camera_link_id, computeForwardKinematics=1)
        return camera_link_state[4], camera_link_state[5]
        rotation = Rt.from_euler('X', 180, degrees=True)
        # new_quat = (rotation * Rt.from_quat(np.array(camera_link_state[5]))).as_quat()
        new_quat = (Rt.from_quat(np.array(camera_link_state[5])) * rotation).as_quat()
        return camera_link_state[4], new_quat
    
    def get_world_to_head_camera(self):
        pos, orn = self.get_head_camera_pos_orn()
        pose = pos_ros_quat_to_mat(pos, orn)
        return pose

    def get_world_to_wrist_camera(self):
        pos, orn = self.get_wrist_camera_pos_orn()
        pose = pos_ros_quat_to_mat(pos, orn)
        return pose

    def post_step(self):
        head_pos, head_orn = self.get_head_camera_pos_orn()
        self.head_camera.update_pose(head_pos, head_orn)
        self.head_camera.post_step()

        wrist_pos, wrist_orn = self.get_wrist_camera_pos_orn()
        self.wrist_camera.update_pose(wrist_pos, wrist_orn)
        self.wrist_camera.post_step()
        # self.head_camera.update_pose((camera_link_state[4][0], camera_link_state[4][1], camera_link_state[4][2] - 0.3), (q_rotated[0], q_rotated[1], q_rotated[2], q_rotated[3]))

        # print('pos', self.)
        # current_position, current_orientation = self.bullet_client.getBasePositionAndOrientation(self.body_id) 
        # print(current_position, current_orientation)
        # pos, orn = camera_link_state[0], camera_link_state[1]
        # pos, orn = np.array(pos), np.array(orn)
        # # code.interact(local=dict(globals(), **locals()))
        # unit_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # rot_matrix = np.array(self.bullet_client.getMatrixFromQuaternion(orn))
        # rotated_axes = np.dot(unit_axes, rot_matrix.reshape(3, 3))
        # axis_length = 0.3
        # self.bullet_client.addUserDebugLine(pos, pos + np.array(rotated_axes[0]) * axis_length, lineColorRGB=[1.0, 0.0, 0.0], lineWidth=2)
        # self.bullet_client.addUserDebugLine(pos, pos + np.array(rotated_axes[1]) * axis_length, lineColorRGB=[0.0, 1.0, 0.0], lineWidth=2)
        # self.bullet_client.addUserDebugLine(pos, pos + np.array(rotated_axes[2]) * axis_length, lineColorRGB=[0.0, 0.0, 1.0], lineWidth=2)
        
        # self.wrist_camera.post_step()
    
    def get_world_to_ee(self):
        world_to_ee = self.get_link_pose(self.ee_link_id)
        return world_to_ee
    
    def robot_base_to_ee(self):
        return se3_inverse(self.get_world_to_robot_base()) @ self.get_world_to_ee()

    def get_world_to_torso(self):
        world_to_torso = self.get_link_pose(self.torso_link_id)
        return world_to_torso
    
    def get_world_to_robot_base(self):
        world_to_robot_base = self.get_link_pose(self.robot_base_id)
        return world_to_robot_base

    def get_tip_pos(self):
        world_to_ee = self.get_world_to_ee()
        tip_pos = world_to_ee[:3, 3]+0.108*world_to_ee[:3, 2]
        return tip_pos

    def ego_cartesian_action_to_dof_target_position(self, pos: NDArray[np.float64], orn: NDArray[np.float64], width: NDArray[np.float64], orn_type="euler") -> NDArray[np.float64]:
        world_to_ee = self.get_world_to_ee()
        if orn_type == "euler":
            ee_to_new_ee = pos_euler_to_mat(pos, orn)
        else:
            raise NotImplementedError
        world_to_new_ee = world_to_ee@ee_to_new_ee
        
        if self.cfg.IK_solver == "PyKDL":
            base_to_new_ee = self.base_to_world@world_to_new_ee
            joint_positions = self.get_joint_positions()
            dof_target_position, info = self.robot_kinematics.cartesian_to_joint(base_to_new_ee, seed=joint_positions[-2])
            if self.cfg.verbose and info<0: print(f"PyKDL IK error: {info}")
            dof_target_position = np.append(dof_target_position, width)
        elif self.cfg.IK_solver == "pybullet":
            world_to_new_ee_pos, world_to_new_ee_ros_quat = mat_to_pos_ros_quat(world_to_new_ee)
            dof_target_position = self.bullet_client.calculateInverseKinematics(self.body_id, self.ee_link_id, world_to_new_ee_pos, world_to_new_ee_ros_quat, maxNumIterations=self.cfg.IK_solver_max_iter, residualThreshold=self.cfg.IK_solver_eps)
            dof_target_position = np.array(dof_target_position)
            dof_target_position[-2:] = width
        else:
            raise NotImplementedError
        return dof_target_position

    def robot_base_ego_cartesian_action_to_dof_target_position(self, chassis_action: NDArray[np.float64], pos: NDArray[np.float64], orn: NDArray[np.float64], width: NDArray[np.float64], orn_type="euler") -> NDArray[np.float64]:
        robot_base_to_ee = self.robot_base_to_ee()
        if orn_type == "euler":
            ee_to_new_ee = pos_euler_to_mat(pos, orn)
        else:
            raise NotImplementedError
        robot_base_to_new_ee = robot_base_to_ee @ ee_to_new_ee
        
        if self.cfg.IK_solver == "PyKDL":
            # base_to_new_ee = self.base_to_world@world_to_new_ee
            joint_positions = self.get_joint_positions()
            # chassis_target_position = joint_positions[:3] + chassis_action
            
            x_y_theta_matrix = np.array([[np.cos(joint_positions[2]), -np.sin(joint_positions[2]), 0], 
                                    [np.sin(joint_positions[2]), np.cos(joint_positions[2]), 0], 
                                    [0, 0, 1]])

            chassis_target_position = joint_positions[:3] + x_y_theta_matrix.dot(chassis_action)


            arm_target_position, info = self.robot_kinematics.cartesian_to_joint(robot_base_to_new_ee, seed=joint_positions[3:-2])
            if self.cfg.verbose and info<0: print(f"PyKDL IK error: {info}")
            dof_target_position = np.concatenate([chassis_target_position, arm_target_position, width])
        elif self.cfg.IK_solver == "pybullet":
            raise NotImplementedError
            world_to_new_ee_pos, world_to_new_ee_ros_quat = mat_to_pos_ros_quat(world_to_new_ee)
            dof_target_position = self.bullet_client.calculateInverseKinematics(self.body_id, self.ee_link_id, world_to_new_ee_pos, world_to_new_ee_ros_quat, maxNumIterations=self.cfg.IK_solver_max_iter, residualThreshold=self.cfg.IK_solver_eps)
            dof_target_position = np.array(dof_target_position)
            dof_target_position[-2:] = width
        else:
            raise NotImplementedError
        return dof_target_position

    def world_pos_action_to_dof_target_position(self, pos, width):
        world_to_ee = self.get_world_to_ee()
        world_to_new_ee_pos = world_to_ee[:3, 3]+pos
        # if self.cfg.IK_solver == "PyKDL":
        #     base_to_new_ee_pos = se3_transform_pc(self.base_to_world, world_to_new_ee_pos)
        #     joint_positions = self.get_joint_positions()
        #     dof_target_position, info = self.robot_kinematics.inverse_kinematics(position=base_to_new_ee_pos, seed=joint_positions[:7])
        #     if self.root_cfg.env.verbose and info<0: print(f"PyKDL IK error: {info}")
        #     dof_target_position = np.append(dof_target_position, [width, width])
        # elif self.cfg.IK_solver == "pybullet":
        dof_target_position = self.bullet_client.calculateInverseKinematics(self.body_id, self.ee_link_id, world_to_new_ee_pos)
        dof_target_position = np.array(dof_target_position)
        dof_target_position[-2:] = width
        return dof_target_position

    def get_visual_observation(self, segmentation_ids: List[int]=[]):
        # camera_link_state = self.bullet_client.getLinkState(self.body_id, self.head_camera_link_id, computeForwardKinematics=1)
        # rotation = Rt.from_euler('z', 90, degrees=True)
        # new_quat = (rotation * Rt.from_quat(np.array(camera_link_state[5]))).as_quat()

        head_pos, head_orn = self.get_head_camera_pos_orn()
        self.head_camera.update_pose(head_pos, head_orn)

        # self.head_camera.update_pose(camera_link_state[4], camera_link_state[5])
        color, depth, segmentation, points = self.head_camera.render(segmentation_ids)
        # for i in range(len(points)):
        #     points[i] = se3_transform_pc(self.hand_to_camera, points[i])
        # code.interact(local=dict(globals(), **locals()))

        # plt.imshow(color)
        # plt.savefig('tmp.png')
        # plt.close()
        # print(np.unique(segmentation))
        # print(points[0].shape)
        return color, depth, segmentation, points
    
    def get_wrist_visual_observation(self, segmentation_ids: List[int]=[]):
        head_pos, head_orn = self.get_wrist_camera_pos_orn()
        self.wrist_camera.update_pose(head_pos, head_orn)

        color, depth, segmentation, points = self.wrist_camera.render(segmentation_ids)
    
        return color, depth, segmentation, points


def debug():
    from omegaconf import OmegaConf
    import pybullet
    from contextlib import contextmanager
    import copy
    import time

    @contextmanager
    def disable_rendering(bullet_client: BulletClient): # speedup setting up scene
        rendering_enabled = bullet_client.rendering_enabled
        if rendering_enabled:
            bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
            bullet_client.rendering_enabled = False
        yield
        if rendering_enabled:
            bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
            bullet_client.rendering_enabled = True

    default_cfg = OmegaConf.structured(GalbotConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg: GalbotConfig = OmegaConf.to_object(OmegaConf.merge(default_cfg, cli_cfg))
    bullet_client = BulletClient(connection_mode=pybullet.GUI) # or pybullet.DIRECT
    bullet_client.resetDebugVisualizerCamera(cameraDistance=2.4, cameraPitch=-58, cameraYaw=102, cameraTargetPosition=[0, 0, 0])
    bullet_client.rendering_enabled = True
    with disable_rendering(bullet_client):
        galbot = Galbot(bullet_client, cfg)
        galbot.reset()
        # galbot_id = bullet_client.loadURDF(cfg.urdf_file)
    dof_default_position = np.array(cfg.dof_default_position)
    
    for j in range(galbot.bullet_client.getNumJoints(galbot.body_id)):
        print(j, galbot.bullet_client.getJointInfo(galbot.body_id, j)[12].decode())
    
    # link_poses = galbot.get_link_poses()
    # sphere_id = bullet_client.loadURDF("/share1/junyu/HRI/genh2r_mobile/env/data/assets/sphere/sphere.urdf", globalScaling=1.)
    # bullet_client.resetBasePositionAndOrientation(sphere_id, (0.5, 0., 0.), (0., 0., 0., 1.))

    def move_to(dof_target_position: NDArray[np.float64], steps=1) -> None:
        dof_current_position = galbot.get_joint_positions()
        for i in range(steps):
            dof_target_position_i = (dof_target_position-dof_current_position)/steps*(i+1)+dof_current_position
            for _ in range(130):
                galbot.pre_step(dof_target_position_i)
                bullet_client.stepSimulation()
                galbot.post_step()
                # time.sleep(0.003)
            galbot.get_visual_observation()
    # code.interact(local=dict(globals(), **locals()))
    
    # for i in range(len(dof_default_position)-2): # arm links
    #     print(f"moving link {i}")
    #     dof_target_position = dof_default_position.copy()
    #     dof_target_position[i] += np.pi/2
    #     move_to(dof_target_position, 100)
    #     print(i, abs(dof_target_position[i] - galbot.get_joint_positions()[i]))
    #     move_to(dof_default_position, 100)
    #     print(i, abs(dof_default_position[i] - galbot.get_joint_positions()[i]))
    print('defalut dof_default_position', dof_default_position)

    for i in range(len(dof_default_position)-2): # arm links
        print(f"\nmoving link {i}")
        dof_target_position = galbot.get_joint_positions().copy()
        print('current Dof', dof_target_position[i])
        dof_target_position[i] += 0.6
        print('expected Dof', dof_target_position[i])
        move_to(dof_target_position, 10)
        print('going to expected Dof, error:', i, abs(dof_target_position[i] - galbot.get_joint_positions()[i]))
        move_to(dof_default_position, 10)
        print('going back, error:', i, abs(dof_default_position[i] - galbot.get_joint_positions()[i]))


    dof_target_position = dof_default_position.copy()
    dof_target_position[7:] = 0.
    move_to(dof_target_position) # gripper
    move_to(dof_default_position)

    for i in range(6): # cartesian action
        pos, orn = np.array([0., 0., 0.]), np.array([0., 0., 0.])
        if i<3:
            pos[i] = 0.1
        else:
            orn[i-3] = np.pi/2
        dof_target_position = galbot.ego_cartesian_action_to_dof_target_position(pos, orn, 0.04)
        move_to(dof_target_position, 10) # forward
        move_to(dof_default_position, 10) # backward
    code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    debug()

"""
DISPLAY="localhost:11.0" python -m env.galbot step_time=0.001 base_position=[0.,0.,0.]
DISPLAY="localhost:12.0" python -m env.galbot step_time=0.001 IK_solver=pybullet
"""

    # right
    # 0 base_link_x
    # 1 base_link_y
    # 2 base_link_z
    # 3 base_link
    # 4 omni_chassis_base_link
    # 5 omni_chassis_leg_mount_link
    # 6 leg_base_link
    # 7 leg_link1
    # 8 leg_link2
    # 9 leg_link3
    # 10 leg_link4
    # 11 leg_torso_mount_link
    # 12 torso_base_link
    # 13 torso_head_mount_link
    # 14 head_base_link
    # 15 head_link1
    # 16 head_link2
    # 17 torso_right_arm_mount_link
    # 18 right_arm_base_link
    # 19 right_arm_link1
    # 20 right_arm_link2
    # 21 right_arm_link3
    # 22 right_arm_link4
    # 23 right_arm_link5
    # 24 right_arm_link6
    # 25 right_arm_link7
    # 26 right_arm_end_effector_mount_link
    # 27 right_flange_base_link
    # 28 right_flange_mount_link
    # 29 right_suction_cup_base_link
    # 30 right_suction_cup_link1
    # 31 right_suction_cup_tcp_link
    # 32 right_arm_camera_flange_link
    # 33 front_head_camera_color_optical_frame
    # 34 torso_left_arm_mount_link
    # 35 left_arm_base_link
    # 36 left_arm_link1
    # 37 left_arm_link2
    # 38 left_arm_link3
    # 39 left_arm_link4
    # 40 left_arm_link5
    # 41 left_arm_link6
    # 42 left_arm_link7
    # 43 left_arm_end_effector_mount_link
    # 44 left_flange_base_link
    # 45 left_flange_mount_link
    # 46 left_gripper_base_link
    # 47 left_gripper_left_link
    # 48 left_gripper_right_link
    # 49 left_gripper_tcp_link
    # 50 left_gripper_acronym_link
    # 51 left_arm_camera_flange_link
    # 52 left_arm_camera_color_optical_frame

    # 7DoF
    #     0 base_link_x
    # 1 base_link_y
    # 2 base_link_z
    # 3 base_link
    # 4 omni_chassis_base_link
    # 5 omni_chassis_leg_mount_link
    # 6 leg_base_link
    # 7 leg_link1
    # 8 leg_link2
    # 9 leg_link3
    # 10 leg_link4
    # 11 leg_torso_mount_link
    # 12 torso_base_link
    # 13 torso_head_mount_link
    # 14 head_base_link
    # 15 head_link1
    # 16 head_link2
    # 17 torso_right_arm_mount_link
    # 18 right_arm_base_link
    # 19 right_arm_link1
    # 20 right_arm_link2
    # 21 right_arm_link3
    # 22 right_arm_link4
    # 23 right_arm_link5
    # 24 right_arm_link6
    # 25 right_arm_link7
    # 26 right_arm_end_effector_mount_link
    # 27 right_flange_base_link
    # 28 right_flange_mount_link
    # 29 right_suction_cup_base_link
    # 30 right_suction_cup_link1
    # 31 right_suction_cup_tcp_link
    # 32 right_arm_camera_flange_link
    # 33 torso_left_arm_mount_link
    # 34 left_arm_base_link
    # 35 left_arm_link1
    # 36 left_arm_link2
    # 37 left_arm_link3
    #     38 left_arm_link4
    # 39 left_arm_link5
    # 40 left_arm_link6
    # 41 left_arm_link7
    # 42 left_arm_end_effector_mount_link
    # 43 left_flange_base_link
    # 44 left_flange_mount_link
    # 45 left_gripper_base_link
    # 46 left_gripper_left_link
    # 47 left_gripper_right_link
    # 48 left_gripper_tcp_link
    # 49 left_gripper_acronym_link
    # 50 left_arm_camera_flange_link

    # kinematics
    #     1 mobile_base
    # 2 base_link_x
    # 3 base_link_y
    # 4 base_link_z
    # 5 base_link
    # 6 omni_chassis_base_link
    # 7 omni_chassis_leg_mount_link
    # 8 leg_base_link
    # 9 leg_link1
    # 10 leg_link2
    # 11 leg_link3
    # 12 leg_link4
    # 13 leg_torso_mount_link
    # 14 torso_base_link
    # 15 torso_head_mount_link
    # 16 torso_right_arm_mount_link
    # 17 torso_left_arm_mount_link
    # 18 head_base_link
    # 19 head_link1
    # 20 head_link2
    # 21 right_arm_base_link
    # 22 right_arm_link1
    # 23 right_arm_link2
    # 24 right_arm_link3
    # 25 right_arm_link4
    # 26 right_arm_link5
    # 27 right_arm_link6
    # 28 right_arm_link7
    # 29 right_arm_end_effector_mount_link
    # 30 right_flange_base_link
    # 31 right_flange_mount_link
    # 32 right_arm_camera_flange_link
    # 33 right_suction_cup_base_link
    # 34 right_suction_cup_link1
    # 35 right_suction_cup_tcp_link
    # 36 left_arm_base_link
    # 37 left_arm_link1
    # 38 left_arm_link2
    # 39 left_arm_link3
    # 40 left_arm_link4
    # 41 left_arm_link5
    # 42 left_arm_link6
    #     43 left_arm_link7
    # 44 left_arm_end_effector_mount_link
    # 45 left_flange_base_link
    # 46 left_flange_mount_link
    # 47 left_arm_camera_flange_link
    # 48 left_gripper_base_link
    # 49 left_gripper_left_link
    # 50 left_gripper_right_link
    # 51 left_gripper_tcp_link
    # 52 left_gripper_acronym_link

    # old
    # 0 base_link_x
    # 1 base_link_y
    # 2 base_link_z
    # 3 base_link
    # 4 hexman_chassis_base_link
    # 5 hexman_chassis_right_wheel_link
    # 6 hexman_chassis_left_wheel_link
    # 7 hexman_chassis_caster_swivel_link1
    # 8 hexman_chassis_caster_wheel_link1
    # 9 hexman_chassis_caster_swivel_link2
    # 10 hexman_chassis_caster_wheel_link2
    # 11 hexman_chassis_caster_swivel_link3
    # 12 hexman_chassis_caster_wheel_link3
    # 13 hexman_chassis_caster_swivel_link4
    # 14 hexman_chassis_caster_wheel_link4
    # 15 hexman_chassis_lidar_link
    # 16 hexman_chassis_lift_fix_point_link
    # 17 lift_base_link
    # 18 body_lift_link
    # 19 head_link
    # 20 head_yaw_link
    # 21 head_pitch_link
    # 22 head_camera_normal_frame    camera_frame
    # 23 head_camera_optical_frame
    # 24 left_arm_base_link
    # 25 left_arm_link1
    # 26 left_arm_link2
    # 27 left_arm_link3
    # 28 left_arm_link4
    # 29 left_arm_link5
    # 30 left_arm_link6
    # 31 left_arm_link7
    # 32 left_arm_end_effector_flange_link
    # 33 left_arm_camera_d415_flange_link
    # 34 left_arm_camera_bottom_screw_frame
    # 35 left_arm_camera_link
    # 36 gripper_inspire_flange_link
    # 37 gripper_inspire_body_link
    # 38 gripper_inspire_left_link_1
    # 39 gripper_inspire_right_link_1
    # 40 gripper_inspire_tcp_frame
    # 41 right_arm_base_link
    # 42 right_arm_link1
    # 43 right_arm_link2
    # 44 right_arm_link3
    # 45 right_arm_link4
    # 46 right_arm_link5
    # 47 right_arm_link6
    # 48 right_arm_link7
    # 49 right_arm_end_effector_flange_link
    # 50 right_arm_camera_d415_flange_link
    # 51 right_arm_camera_bottom_screw_frame
    # 52 right_arm_camera_link
    # 53 right_arm_camera_usb_plug_link
    # 54 long_sucker_base_link
    # 55 long_sucker_tool_link
    # 56 long_sucker_tcp_link