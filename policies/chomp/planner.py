import os
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, List, Union
import torch
from torch.optim import Adam
from dataclasses import dataclass, field
import copy
import open3d as o3d
import ipdb
import code
from .trajectory import TrajectoryConfig, Trajectory
from env.utils.sdf_loss import SDFData, SDFDataTensor, compute_sdf_loss
from env.utils.robot_kinematics import RobotKinematicsConfig, RobotKinematics

def trans_distance(p1: Union[NDArray[np.float64], torch.DoubleTensor], p2: Union[NDArray[np.float64], torch.DoubleTensor]) -> Union[NDArray[np.float64], torch.DoubleTensor]:
    "p1, p2: (..., 3)"
    if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
        return np.linalg.norm(p1-p2, axis=-1)
    elif isinstance(p1, torch.Tensor) and isinstance(p2, torch.Tensor):
        return (p1-p2).norm(dim=-1)
    else:
        raise ValueError(f"trans distance not supported for {type(p1)} and {type(p2)}")

def rot_distance(R1: Union[NDArray[np.float64], torch.DoubleTensor], R2: Union[NDArray[np.float64], torch.DoubleTensor]) -> Union[NDArray[np.float64], torch.DoubleTensor]:
    "R1, R2: (..., 3, 3)"
    if isinstance(R1, np.ndarray) and isinstance(R2, np.ndarray):
        return np.arccos(np.clip(((R1*R2).sum(axis=(-1, -2))-1)/2, -1, 1))
    elif isinstance(R1, torch.Tensor) and isinstance(R2, torch.Tensor):
        return (((R1*R2).sum(dim=(-1, -2))-1)/2).clamp(-0.99999, 0.99999).arccos()
    else:
        raise ValueError(f"rot distance not supported for {type(R1)} and {type(R2)}")

def se3_distance(T1: Union[NDArray[np.float64], torch.DoubleTensor], T2: Union[NDArray[np.float64], torch.DoubleTensor]) -> Union[Tuple[NDArray[np.float64], NDArray[np.float64]], Tuple[torch.DoubleTensor, torch.DoubleTensor]]:
    "T1, T2: (..., 4, 4)"
    return trans_distance(T1[..., :3, 3], T2[..., :3, 3]), rot_distance(T1[..., :3, :3], T2[..., :3, :3])

class ObstacleLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, points: torch.DoubleTensor, sdf_loss: torch.DoubleTensor, sdf_grad: torch.DoubleTensor, time_interval: float) -> torch.DoubleTensor:
        """
        Input:
            points: (length, num_points, 3)
            sdf_loss: (length-2, num_points)
            sdf_grad: (length-2, num_points, 3)
        """
        velocity = (points[1:-1]-points[:-2])/time_interval # (length-2, num_points, 3)
        acceleration = (points[:-2]+points[2:]-2*points[1:-1])/time_interval**2 # (length-2, num_points, 3)
        velocity_norm = velocity.norm(dim=2) # (length-2, num_points)
        obstacle_loss = sdf_loss*velocity_norm # (length-2, num_points)
        ctx.save_for_backward(sdf_loss, sdf_grad, velocity, velocity_norm, acceleration)
        return obstacle_loss.sum()

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, output_grad: torch.DoubleTensor) -> torch.DoubleTensor:
        """
        Input:
            grad_output: scalar
        """
        sdf_loss, sdf_grad, velocity, velocity_norm, acceleration = ctx.saved_tensors
        device = sdf_grad.device
        normalized_velocity = velocity/(velocity_norm[..., None]+1e-8) # (length-2, num_points, 3)
        projection_matrix = torch.eye(3, dtype=torch.float64, device=device)[None, None]-normalized_velocity[:, :, None]*normalized_velocity[:, :, :, None] # (length-2, num_points, 3, 3)
        curvature = (projection_matrix@acceleration[..., None]).squeeze(-1)/(velocity_norm[..., None]**2+1e-8) # (length-2, num_points, 3)
        projected_sdf_grad = (projection_matrix@sdf_grad[..., None]).squeeze(-1) # (length-2, num_points, 3)
        points_grad = output_grad*velocity_norm[..., None]*(projected_sdf_grad-sdf_loss[..., None]*curvature) # (length-2, num_points, 3)
        points_grad = torch.cat([torch.zeros_like(points_grad[[0]]), points_grad, torch.zeros_like(points_grad[[0]])])
        return points_grad, None, None, None

@dataclass
class CHOMPPlannerConfig:
    optimize_steps: int = 100
    device: str = "cuda"

    standoff_steps: int = 5
    standoff_dist: float = 0.064

    traj: TrajectoryConfig = field(default_factory=lambda: TrajectoryConfig(fix_end="${..fix_end}", device="${..device}"))
    robot_kinematics: RobotKinematicsConfig = field(default_factory=lambda: RobotKinematicsConfig(urdf_file="env/data/assets/galbot_one_simplified/galbot_one_7_DoF.urdf", chain_tip="left_gripper_acronym_link"))
    # robot_kinematics: RobotKinematicsConfig = field(default_factory=lambda: RobotKinematicsConfig(urdf_file="env/data/assets/galbot_one_simplified/galbot_one_7_DoF.urdf", chain_tip="left_gripper_base_link"))
    # robot_kinematics: RobotKinematicsConfig = field(default_factory=lambda: RobotKinematicsConfig(urdf_file="env/data/assets/galbot_zero_lefthand/galbot_zero_lefthand_static.urdf", chain_tip="left_arm_link7"))

    fix_end: bool = False
    end_loss_coef: float = 1.

    # smooth loss
    smooth_loss_coef: float = 1.

    # obstacle loss
    obstacle_loss_coef: float = 1.
    num_collision_points_per_link: int = 15

    # orient loss
    ee_orient_loss_coef: float = 0.
    camera_orient_loss_coef: float = 0.
    object_orient_loss_coef: float = 10.0

    camera_name: str = "base_link_z"
    camera_look_at_idx: int = 0

    verbose: bool = False

class CHOMPPlanner:
    def __init__(self, cfg: CHOMPPlannerConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.robot_kinematics = RobotKinematics(cfg.robot_kinematics)
        print('ChmopPlanner Robot:', cfg.robot_kinematics.urdf_file)
        self.ee_link_idx = self.robot_kinematics.tip_link.idx
        if self.cfg.camera_orient_loss_coef > 0.:
            self.camera_link_idx = self.robot_kinematics.link_map[cfg.camera_name].idx
        
        self.grasp_to_standoff_grasps = np.tile(np.eye(4), (cfg.standoff_steps, 1, 1)) # (cfg.standoff_steps, 4, 4)
        self.grasp_to_standoff_grasps[:, 2, 3] = -cfg.standoff_dist*np.linspace(0, 1, cfg.standoff_steps)[::-1]

        self.min_joint_values = torch.tensor([-100, -100, -6] + self.robot_kinematics.q_min+[-1., -1.], dtype=torch.float64, device=self.device)
        self.max_joint_values = torch.tensor([100, 100, 6] + self.robot_kinematics.q_max+[ 1.,  1.], dtype=torch.float64, device=self.device)

        collision_points_dir = os.path.join(os.path.dirname(cfg.robot_kinematics.urdf_file), "collision_points")
        collision_points_list = []
        collision_points_exist_list = []
        for link in self.robot_kinematics.links:
            collision_points_path = os.path.join(collision_points_dir, link.name+".xyz")
            pcd = o3d.io.read_point_cloud(collision_points_path)
            collision_points = np.array(pcd.points)[:cfg.num_collision_points_per_link] # num_collision_points_per_link should not exceed 1000 (only 1000 points are generated for each link)
            if collision_points.shape[0] > 0:
                collision_points_tensor = torch.tensor(collision_points, device=self.device)
                collision_points_list.append(collision_points_tensor)
            collision_points_exist_list.append(collision_points.shape[0] > 0)
            
        self.collision_points: torch.DoubleTensor = torch.stack(collision_points_list, dim=0) # (num_links, num_points_per_link, 3)
        self.collision_points_exist: NDArray[np.bool] = np.array(collision_points_exist_list)
    
    def compute_A_and_A_inv(self, traj_cfg: TrajectoryConfig) -> Tuple[torch.DoubleTensor, torch.DoubleTensor]:
        diff_matrix = torch.zeros((traj_cfg.length+1, traj_cfg.length), dtype=torch.float64, device=self.device)
        diff_matrix[:-1] += torch.eye(traj_cfg.length, dtype=torch.float64, device=self.device)/traj_cfg.time_interval
        diff_matrix[1:] -= torch.eye(traj_cfg.length, dtype=torch.float64, device=self.device)/traj_cfg.time_interval
        # if self.cfg.fix_end == False:
        #     diff_matrix[-1, -1] = 0.
        A = diff_matrix.T@diff_matrix # (length, length)
        A_inv = torch.linalg.inv(A)
        return A, A_inv

    def compute_end_loss(self, current_base_to_standoff_grasps: torch.DoubleTensor, target_base_to_standoff_grasps: torch.DoubleTensor) -> torch.DoubleTensor:
        trans_dist, rot_dist = se3_distance(current_base_to_standoff_grasps, target_base_to_standoff_grasps)
        return (trans_dist+0.25*rot_dist).sum()

    def compute_smooth_loss(self, traj: Trajectory) -> torch.DoubleTensor:
        if self.cfg.fix_end:
            velocity = (torch.cat([traj.traj, traj.end[None]])-torch.cat([traj.start[None], traj.traj]))/self.cfg.traj.time_interval
        else:
            velocity = (traj.traj-torch.cat([traj.start[None], traj.traj[:-1]]))/self.cfg.traj.time_interval
        speed = velocity.norm(dim=1)
        smooth_loss = (0.5*speed**2).mean()
        return smooth_loss

    def compute_obstacle_loss(self, base_to_links: torch.DoubleTensor, sdf_data_list: List[SDFDataTensor], body_to_base_list: List[torch.DoubleTensor]) -> torch.DoubleTensor:
        " base_to_links: (length, num_links, 4, 4), body_to_world_list: list of (4, 4) "
        base_to_links = base_to_links[:, self.collision_points_exist]
        length, num_points = base_to_links.shape[0], base_to_links.shape[1]*self.collision_points.shape[1]
        base_to_points = self.collision_points[None]@base_to_links[:, :, :3, :3].transpose(-2, -1)+base_to_links[:, :, None, :3, 3] # (1, num_links, num_points_per_link, 3)*(length, num_links, 3, 3)+(length, num_links, 1, 3) -> (length, num_links, num_points_per_link, 3)
        base_to_points = base_to_points.view(length, num_points, 3)
        sdf_loss, sdf_grad = compute_sdf_loss(base_to_points[1:-1].view(-1, 3), sdf_data_list, body_to_base_list) # ((length-2)*num_points), ((length-2)*num_points, 3)
        sdf_loss, sdf_grad = sdf_loss.view(length-2, num_points), sdf_grad.view(length-2, num_points, 3)
        obstacle_loss = ObstacleLoss.apply(base_to_points, sdf_loss, sdf_grad, self.cfg.traj.time_interval)
        return obstacle_loss

    def compute_ee_orient_loss(self, traj: torch.DoubleTensor, base_to_ee: torch.DoubleTensor, base_to_target_grasp: NDArray[np.float64], traj_smooth_step_size: Optional[float], action_repeat_frames: int, look_at_idx: int=2, project_to: Optional[str]=None) -> torch.DoubleTensor:
        """
        Input:
            traj: (length, dof)
            base_to_ee: (length, 4, 4)
            base_to_target_grasp: (num_frames, 4, 4)
        """
        if traj_smooth_step_size is not None:
            step_lengths = (traj[1:]-traj[:-1]).norm(dim=1).detach().cpu().numpy() # (length-1,)
            accumulated_lengths = np.concatenate([[0], np.cumsum(step_lengths)]) # (length, )
            frame_idxs = (accumulated_lengths/traj_smooth_step_size*action_repeat_frames).astype(np.int64) # (length, )
        else:
            frame_idxs = np.arange(traj.shape[0])*action_repeat_frames # (length, )
        frame_idxs = np.clip(frame_idxs, a_min=0, a_max=base_to_target_grasp.shape[0]-1)
        base_to_target_grasp_at_each_step = base_to_target_grasp[frame_idxs]

        ee_look_at = base_to_ee[:, :3, look_at_idx] # (length, 3)
        # ipdb.set_trace()
        ee_to_target_grasp_trans = torch.tensor(base_to_target_grasp_at_each_step[:, :3, 3], device=base_to_ee.device)-base_to_ee[:, :3, 3] # (length, 3)
        ee_to_target_grasp_dir = ee_to_target_grasp_trans/(ee_to_target_grasp_trans.norm(dim=1, keepdim=True)+1e-6) # (length, 3)
        if project_to == "xy":
            ee_to_target_grasp_dir[:, 2] = 0
            ee_look_at[:, 2] = 0
        ee_orient_loss = torch.cross(ee_look_at, ee_to_target_grasp_dir).norm(dim=1).mean() # (length, 3) -> ()
        return ee_orient_loss
    
    def compute_robot_orient_loss(self, traj: torch.DoubleTensor, traj_smooth_step_size: Optional[float], action_repeat_frames: int, base_to_global_link: torch.DoubleTensor, base_to_target_object: NDArray[np.float64]):
        """
        Input:
            base_to_global_link: (length, 4, 4)
            base_to_target_object: (num_frames, 4, 4)
        """
        if traj_smooth_step_size is not None:
            step_lengths = (traj[1:]-traj[:-1]).norm(dim=1).detach().cpu().numpy() # (length-1,)
            accumulated_lengths = np.concatenate([[0], np.cumsum(step_lengths)]) # (length, )
            frame_idxs = (accumulated_lengths/traj_smooth_step_size*action_repeat_frames).astype(np.int64) # (length, )
        else:
            frame_idxs = np.arange(traj.shape[0])*action_repeat_frames # (length, )
        frame_idxs = np.clip(frame_idxs, a_min=0, a_max=base_to_target_object.shape[0]-1)
        base_to_target_object_at_each_step = base_to_target_object[frame_idxs]  # (length, 4, 4)

        object_position = torch.tensor(base_to_target_object_at_each_step[:, :2, 3], device = traj.device)   # (length, 2)
        robot_position = base_to_global_link[:, :2, 3]    # (length, 2)
        direction_robot_to_object = object_position - robot_position  # (length, 2)
        direction_robot_head =  base_to_global_link[:, :2, 0]    #(length, 2)

        # code.interact(local=dict(globals(), **locals()))
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cosine_similarity = cos(direction_robot_to_object, direction_robot_head)
        cosine_distance = (1 - cosine_similarity) / 2

        orientation_loss = torch.mean(cosine_distance)
        # orientation_loss = torch.mean(cosine_distance ** 2)
        # cross_product = torch.cross(direction_robot_head, direction_robot_to_object, dim = 1)    #(length,)
        # dot_product = torch.sum(direction_robot_head * direction_robot_to_object, dim = 1)    #(length,)
        # theta_error = torch.atan2(cross_product, dot_product)
        # orientation_loss = torch.mean(theta_error ** 2)

        # code.interact(local=dict(globals(), **locals()))
        return orientation_loss

        

        
    def compute_loss(self, traj: Trajectory, base_to_target_grasp: NDArray[np.float64], sdf_data_list: List[SDFDataTensor], body_to_base_list: List[torch.DoubleTensor], step: int, traj_smooth_step_size: Optional[float], action_repeat_frames: int, target_base_to_standoff_grasps: torch.DoubleTensor, base_to_target_object = None) -> torch.DoubleTensor:
        if self.cfg.fix_end:
            complete_traj = torch.cat([traj.start[None], traj.traj, traj.end[None]]) # (length, dof)
        else:
            complete_traj = torch.cat([traj.start[None], traj.traj]) # (length, dof)
        base_to_links = self.robot_kinematics.joint_to_cartesian_for_all_links(complete_traj) # (length, num_links, 4, 4)
        # code.interact(local=dict(globals(), **locals()))   vase_to_links[:, 3]   
        # code.interact(local=dict(globals(), **locals()))       
        loss = 0.

        # end loss
        if not self.cfg.fix_end and self.cfg.end_loss_coef > 0.:
            end_loss = self.compute_end_loss(base_to_links[-target_base_to_standoff_grasps.shape[0]:, self.ee_link_idx], target_base_to_standoff_grasps)
            loss = loss+self.cfg.end_loss_coef*end_loss
            if self.cfg.verbose:
                print(f"end_loss: {end_loss}")

        # smooth loss
        if self.cfg.smooth_loss_coef > 0.:
            smooth_loss = self.compute_smooth_loss(traj)
            loss = loss+self.cfg.smooth_loss_coef*smooth_loss
            if self.cfg.verbose:
                print(f"smooth_loss: {smooth_loss}")

        # obstacle loss
        if self.cfg.obstacle_loss_coef > 0.:
            obstacle_loss = self.compute_obstacle_loss(base_to_links, sdf_data_list, body_to_base_list)
            loss = loss+self.cfg.obstacle_loss_coef*obstacle_loss

        # ee orient loss
        if self.cfg.ee_orient_loss_coef > 0.:
            ee_orient_loss = self.compute_ee_orient_loss(complete_traj, base_to_links[:, self.ee_link_idx], base_to_target_grasp, traj_smooth_step_size, action_repeat_frames)
            loss = loss+self.cfg.ee_orient_loss_coef*ee_orient_loss
        
        if self.cfg.camera_orient_loss_coef > 0.:
            camera_orient_loss = self.compute_ee_orient_loss(complete_traj, base_to_links[:, self.camera_link_idx], base_to_target_grasp, traj_smooth_step_size, action_repeat_frames, self.cfg.camera_look_at_idx, "xy")
            if self.cfg.verbose:
                print(f"camera orient loss {camera_orient_loss}")
            loss = loss+self.cfg.camera_orient_loss_coef*camera_orient_loss

        if self.cfg.object_orient_loss_coef > 0:
            robot_orient_loss = self.compute_robot_orient_loss(complete_traj, traj_smooth_step_size, action_repeat_frames, base_to_links[:, 3], base_to_target_object)
            # code.interact(local=dict(globals(), **locals()))
            # if self.cfg.verbose:
            # print(f"robot orient loss {robot_orient_loss};  loss {loss}")
            loss = loss + self.cfg.object_orient_loss_coef * robot_orient_loss
            # code.interact(local=dict(globals(), **locals()))

        if self.cfg.verbose:
            print(f"traj {traj.traj.square().mean().item()}")

        return loss
    
    def compute_traj_v(self, traj: torch.DoubleTensor) -> torch.DoubleTensor:
        " traj: (length, dof) "
        low_mask = traj < self.min_joint_values
        high_mask = traj > self.max_joint_values
        traj_v = low_mask*(self.min_joint_values-traj)+high_mask*(self.max_joint_values-traj)
        return traj_v

    def handle_joint_limit(self, traj: Trajectory, A_inv: torch.DoubleTensor):
        " A_inv: (length, length) "
        " reference: https://github.com/ros-planning/moveit/blob/master/moveit_planners/chomp/chomp_motion_planner/src/chomp_optimizer.cpp: ChompOptimizer::handleJointLimits() "
        traj.traj.data.clamp_(min=self.min_joint_values, max=self.max_joint_values)
        # length, dof = traj.traj.shape
        # traj_v = self.compute_traj_v(traj.traj) # (length, dof)
        # cnt = 0
        # while traj_v.abs().max() > 1e-6 and cnt < 10:
        #     abs_v_max, p_max = traj_v.abs().max(dim=0) # (dof,) (dof,)
        #     v_max = traj_v[p_max, torch.arange(dof)] # (dof,)
        #     multiplier = v_max/A_inv[p_max, p_max] # (dof,)
        #     traj.traj.data += multiplier*A_inv[:, p_max] # (length, dof)
        #     cnt += 1
        #     traj_v = self.compute_traj_v(traj.traj)

    def optimize_one_step(self, traj: Trajectory, base_to_target_grasp: NDArray[np.float64], sdf_data_list: List[SDFDataTensor], body_to_base_list: List[torch.DoubleTensor], A_inv: torch.DoubleTensor, step: int, traj_smooth_step_size: Optional[float], action_repeat_frames: int, target_base_to_standoff_grasps: torch.DoubleTensor, base_to_target_object) -> None:
        loss = self.compute_loss(traj, base_to_target_grasp, sdf_data_list, body_to_base_list, step, traj_smooth_step_size, action_repeat_frames, target_base_to_standoff_grasps, base_to_target_object)
        grad = torch.autograd.grad(loss, traj.traj)[0]
        # ipdb.set_trace()
        # traj.update(self.cfg.lr*(A_inv@grad))
        traj.update(grad)
        # assert traj.traj.grad is None
        self.handle_joint_limit(traj, A_inv)
    
    def standoff_cartesian_to_joint(self, base_to_grasp: NDArray[np.float64], current_joint_values: NDArray[np.float64], orientation = None) -> Tuple[NDArray[np.float64], bool, NDArray[np.float64]]:
        base_to_standoff_grasps = base_to_grasp@self.grasp_to_standoff_grasps # (cfg.standoff_steps, 4, 4)
        ik_success = True
        standoff_joint_values = []
        assert self.robot_kinematics.num_joints == 7
        joint_values = current_joint_values[3:10] # seed
        for i in range(self.cfg.standoff_steps):
            base_to_standoff_grasp = base_to_standoff_grasps[i]
            # joint_values, ik_info = self.robot_kinematics.cartesian_to_joint(base_to_standoff_grasp, seed=joint_values) # Use the solved ik in the previous step as the seed in the next step. From the farthest standoff to the actual grasp pose.
            # joint_values, ik_info = self.robot_kinematics.cartesian_to_joint_with_orientation(base_to_standoff_grasp, seed=joint_values, orientation = orientation)
            joint_values, ik_info = self.robot_kinematics.cartesian_to_joint_without_orientation(base_to_standoff_grasp, seed=joint_values)

            standoff_joint_values.append(np.concatenate([current_joint_values[0:3], joint_values, current_joint_values[10:12]]))
            if ik_info != 0:
                ik_success = False
                break
        standoff_joint_values = np.stack(standoff_joint_values, axis=0)
        return standoff_joint_values, ik_success, base_to_standoff_grasps
    
    def get_best_grasp(self, base_to_target_grasps: NDArray[np.float64], current_joint_values: NDArray[np.float64], current_base_to_ee: NDArray[np.float64], 
                       base_to_target_object: NDArray[np.float64], world_to_hand_position: NDArray[np.float64], world_to_humanoid_position: NDArray[np.float64], base_to_hand_point:  NDArray[np.float64]) -> Tuple[NDArray[np.float64], float, int, NDArray[np.float64]]:
        """
        Input:
            base_to_target_grasps: (num_grasps, 4, 4)
            current_joint_values: (dof)
            current_base_to_ee: (4, 4)
            base_to_target_object: (4, 4)
            world_to_hand_position: (3)
            world_to_humanoid_position (3)
        """
        object_to_humanoid = world_to_humanoid_position[:2] - base_to_target_object[:2, 3]
        object_to_hand = world_to_hand_position[:2] - base_to_target_object[:2, 3]
        expect_robot_to_object = (object_to_humanoid + object_to_hand) / 2
        # expect_robot_to_object = object_to_hand
        expect_object_to_robot_angle = np.arctan2(-expect_robot_to_object[1], -expect_robot_to_object[0])

        robot_pos_angle_range = np.deg2rad(40)
        sample_robot_angle_num = 30
        sample_robot_radius = 0.6
        candidate_object_to_robot_angle = np.random.uniform(-robot_pos_angle_range, robot_pos_angle_range, sample_robot_angle_num) + expect_object_to_robot_angle    # (sample_angle_num)
        candidate_base_to_robot_position = sample_robot_radius * np.column_stack((np.cos(candidate_object_to_robot_angle), np.sin(candidate_object_to_robot_angle))) + base_to_target_object[:2, 3] # (sample_angle_num, 2)
        candidate_robot_rotation = candidate_object_to_robot_angle + np.pi        # (sample_angle_num)
        candidate_robot_rotation[candidate_robot_rotation > np.pi] -= 2 * np.pi
        candidate_robot_rotation[candidate_robot_rotation < -np.pi] += 2 * np.pi

        #  T = np.array([
        #     [1, 0, 0, -x],
        #     [0, 1, 0, -y],
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1]
        # ])
        # R = np.array([
        #     [cos(-theta), -sin_(-theta), 0, 0],
        #     [sin(-theta), cos(-theta), 0, 0],
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1]
        # ])
        # M = R @ T
        robot_T = np.eye(4)[np.newaxis, :, :] * np.ones((sample_robot_angle_num, 1, 1))    # (sample_angle_num, 4, 4)
        robot_R = np.eye(4)[np.newaxis, :, :] * np.ones((sample_robot_angle_num, 1, 1))
        robot_T[:, 0:2, 3] = - candidate_base_to_robot_position
        robot_R[:, 0, 0] = robot_R[:, 1, 1] = np.cos(candidate_robot_rotation)
        robot_R[:, 0, 1] = np.sin(candidate_robot_rotation)
        robot_R[:, 1, 0] = -np.sin(candidate_robot_rotation)     # (sample_angle_num, 4, 4)
        robot_transformation_matrix = robot_R @ robot_T    # (sample_angle_num, 4, 4)
        # robot_transformation_matrix_inv =  np.linalg.inv(robot_transformation_matrix)

        # code.interact(local=dict(globals(), **locals()))
        target_standoff_joint_values, min_dist, target_grasp_idx, target_base_to_standoff_grasps = None, np.inf, -1, None
        min_angle = 180.0
        target_degree = None
        candidate_joint_value = copy.deepcopy(current_joint_values)
        base_to_target_grasps = base_to_target_grasps[:50]
        for grasp_idx, base_to_grasp in enumerate(base_to_target_grasps):
            for robot_pos_sample_index in range(sample_robot_angle_num):

                grasp_to_object = base_to_target_object[:2, 3] - base_to_grasp[:2, 3]
                grasp_to_object_angle = np.arctan2(grasp_to_object[1], grasp_to_object[0])
                cos_theta = np.clip(np.dot(expect_robot_to_object, grasp_to_object) / (np.linalg.norm(expect_robot_to_object) * np.linalg.norm(grasp_to_object)), -1.0, 1.0)
                angle_degrees = np.degrees(np.arccos(cos_theta))
                # print('angle_degrees', angle_degrees)
                # code.interact(local=dict(globals(), **locals()))
                # if abs(angle_degrees) > 30.0:
                #     continue
                
                candidate_joint_value[:3] = candidate_base_to_robot_position[robot_pos_sample_index][0], candidate_base_to_robot_position[robot_pos_sample_index][1], candidate_robot_rotation[robot_pos_sample_index]
                # standoff_joint_values, ik_success, base_to_standoff_grasps = self.standoff_cartesian_to_joint(base_to_grasp, current_joint_values, expect_robot_to_object_angle)
                standoff_joint_values, ik_success, base_to_standoff_grasps = self.standoff_cartesian_to_joint(robot_transformation_matrix[robot_pos_sample_index] @ base_to_grasp, candidate_joint_value, grasp_to_object_angle)
                # print('ik_success', ik_success)
                if ik_success:
                    base_to_standoff_grasps = base_to_grasp@self.grasp_to_standoff_grasps

                    joint_values = standoff_joint_values[0] # this is the joint values for the farthest standoff grasp
                    # current_dist = ((joint_values-current_joint_values)**2).sum()

                    robot_to_object = np.arctan2(base_to_target_object[1,3] - standoff_joint_values[-1, 1], base_to_target_object[0,3] - standoff_joint_values[-1, 0])
                    robot_head = standoff_joint_values[-1, 2]
                    angle_diff = abs(np.arctan2(np.sin(robot_head - robot_to_object), np.cos(robot_head - robot_to_object))) * 180 / np.pi
                    # min_angle = min(angle_diff, min_angle)
                    # code.interact(local=dict(globals(), **locals()))
                    
                    trans_dist, rot_dist = se3_distance(current_base_to_ee, base_to_grasp)
                    current_dist = trans_dist+0.25*rot_dist
                    # print(f"dist: {current_dist}")
                    # if current_dist < min_dist:
                    #     target_standoff_joint_values = standoff_joint_values
                    #     min_dist = current_dist
                    #     target_grasp_idx = grasp_idx
                    #     target_base_to_standoff_grasps = base_to_standoff_grasps

                    if angle_diff < min_angle:
                        target_standoff_joint_values = standoff_joint_values
                        min_angle = angle_diff
                        min_dist = current_dist
                        target_grasp_idx = grasp_idx
                        target_base_to_standoff_grasps = base_to_standoff_grasps
                        target_degree = angle_degrees
        if target_degree is not None:
            print('min_angle: ', min_angle, 'target_degree', target_degree)
        # code.interact(local=dict(globals(), **locals()))
        
        return target_standoff_joint_values, min_dist, target_grasp_idx, target_base_to_standoff_grasps

    def plan(self, base_to_target_object: NDArray[np.float64], object_to_target_grasps: NDArray[np.float64], current_joint_values: NDArray[np.float64], current_base_to_ee: NDArray[np.float64], sdf_data_list: List[SDFData], 
             body_to_base_list: List[NDArray[np.float64]], traj_smooth_step_size: Optional[float], action_repeat_frames: int, world_to_hand_position: NDArray[np.float64], world_to_humanoid_position: NDArray[np.float64], base_to_hand_point: NDArray[np.float64]) -> Tuple[Optional[NDArray[np.float64]], Optional[int]]:
        """
        Input:
            base_to_target_object: (num_frames, 4, 4)
            object_to_target_grasps: (num_grasps, 4, 4)
            current_joint_values: (dof)
            current_base_to_ee: (4, 4)
            sdf_data_list: list of SDFData
            body_to_base_list: list of (4, 4)
            world_to_hand_position (3)
            world_to_humanoid_position (3)
        """
        # code.interact(local=dict(globals(), **locals()))
        base_to_target_grasps = base_to_target_object[-1]@object_to_target_grasps # (num_grasps, 4, 4)
        target_standoff_joint_values, min_dist, target_grasp_idx, target_base_to_standoff_grasps = self.get_best_grasp(base_to_target_grasps, current_joint_values, current_base_to_ee, base_to_target_object[-1], world_to_hand_position, world_to_humanoid_position, base_to_hand_point)
        if min_dist == np.inf:
            return None, None
        target_base_to_standoff_grasps = torch.from_numpy(target_base_to_standoff_grasps).to(self.device)
        base_to_target_grasp = base_to_target_object@object_to_target_grasps[target_grasp_idx] # (num_frames, 4, 4)
        traj_cfg = copy.deepcopy(self.cfg.traj)
        traj_cfg.start = current_joint_values.tolist()
        if self.cfg.fix_end:
            traj_cfg.end = target_standoff_joint_values[0].tolist()
        else:
            traj_cfg.end = target_standoff_joint_values[-1].tolist()
        traj = Trajectory(traj_cfg)
        A, A_inv = self.compute_A_and_A_inv(traj_cfg)
        sdf_data_list: List[SDFDataTensor] = [SDFDataTensor(sdf_data, self.device) for sdf_data in sdf_data_list]
        body_to_base_list: List[torch.DoubleTensor] = [torch.tensor(body_to_base, device=self.device) for body_to_base in body_to_base_list]
        for i in range(self.cfg.optimize_steps):
            self.optimize_one_step(traj, base_to_target_grasp, sdf_data_list, body_to_base_list, A_inv, i, traj_smooth_step_size, action_repeat_frames, target_base_to_standoff_grasps, base_to_target_object)
        if self.cfg.fix_end:
            traj_numpy = np.concatenate([traj.traj.detach().cpu().numpy(), target_standoff_joint_values]) # ignore start, and end is the first element of target_standoff_joint_values
        else:
            traj_numpy = traj.traj.detach().cpu().numpy()
        # ipdb.set_trace()
        # print('traj_numpy', traj_numpy[-1])
        # for i in range(traj_numpy.shape[0]):
        #     print(traj_numpy[i])
        # code.interact(local=dict(globals(), **locals()))
        return traj_numpy, target_grasp_idx

def debug():
    from omegaconf import OmegaConf
    from env.utils.transform import pos_ros_quat_to_mat
    import code

    np.set_printoptions(suppress=True) # no scientific notation

    default_cfg = OmegaConf.structured(CHOMPPlannerConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg: CHOMPPlannerConfig = OmegaConf.to_object(OmegaConf.merge(default_cfg, cli_cfg))

    chomp_planner = CHOMPPlanner(cfg)
    pos = np.array([ 0.1439894 , -0.00910749,  0.71072687])
    ros_quat = np.array([ 0.96438653,  0.03465594,  0.2612568 , -0.02241564])
    base_to_ee = pos_ros_quat_to_mat(pos, ros_quat)
    base_to_ees = base_to_ee[None]
    seed = np.array([ 0.   , -1.285,  0.   , -2.356,  0.   ,  1.571,  0.785, 0.04, 0.04])
    current_base_to_ee = chomp_planner.robot_kinematics.joint_to_cartesian(seed[:7])
    traj = chomp_planner.plan(base_to_ees, seed, current_base_to_ee, [], [])
    code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    debug()

"""
python -m policies.chomp.planner traj.init_mode=random optimize_steps=3000
"""