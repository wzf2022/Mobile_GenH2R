import numpy as np
from numpy.typing import NDArray
import torch
from typing import Dict, Any, List, Tuple
import os
from typing import Optional
from scipy.spatial.transform import Rotation as Rt
import ipdb

from .base_policy import BasePolicy
from .chomp_policy_config import CHOMPPolicyConfig
from .chomp.planner import CHOMPPlanner, se3_distance
from env.mobile_handover_env import Observation, MobileH2RSim
from env.body import SDFData, Body
from env.objects import Object
from env.hand import Hand
from env.utils.transform import se3_inverse, pos_ros_quat_to_mat, mat_to_pos_euler
from env.utils.scene import scene_id_to_hierarchical_dir
from env.utils.hand_collision_filter import HandCollisionFilter
import code

class CHOMPPolicy(BasePolicy):
    def __init__(self, cfg: CHOMPPolicyConfig):
        super().__init__(cfg)
        self.cfg: CHOMPPolicyConfig
        self.device = torch.device(cfg.device)
        self.planner = CHOMPPlanner(cfg.planner)
        if cfg.filter_hand_collision:
            self.hand_collision_filter = HandCollisionFilter(cfg.hand_collision_filter)
        self.world_to_base = np.eye(4)
        self.world_to_base[:3, :3] = Rt.from_quat(np.array(cfg.base_orientation)).as_matrix()
        self.world_to_base[:3, 3] = np.array(cfg.base_position)
        self.base_to_world = se3_inverse(self.world_to_base)
        self.grasp_augmentation_matrix: NDArray[np.float64] = np.diag(np.array([-1., -1., 1., 1.], dtype=float))
        self.scene_id = -1

    def reset(self, scene_id: Optional[int]=None):
        self.base_reset()
        self.traj: Optional[NDArray[np.float64]] = None
        self.step: int = 0

        self.last_plan_frame = -np.inf
        self.object_stopped = False
        self.world_to_target_object = None
        self.object_to_target_grasp = None
        if scene_id is not None and self.scene_id != scene_id:
            self.scene_id = scene_id
            self.target_object_grasps: Optional[NDArray[np.float64]] = None
            self.filter_results: Optional[NDArray[np.bool_]] = None
            self.endpoints = None

    def get_foresee_frames(self, env: MobileH2RSim) -> int:
        foresee_time = self.cfg.foresee_time
        if env.cfg.stop_moving_dist is not None:
            if self.cfg.trajectory_smoothing:
                remain_steps = max((env.get_panda_object_dist()-env.cfg.stop_moving_dist)/(self.cfg.trajectory_smoothing_step_size*0.625), 0) # 0.08 -> 0.05, 0.12 -> 0.075
            else:
                remain_steps = max((env.get_panda_object_dist()-env.cfg.stop_moving_dist)/0.05, 0) # assume every step is 5 cm
            # print(f"remain steps: {remain_steps}")
            foresee_time = min(foresee_time, remain_steps*self.cfg.action_repeat_time)
        foresee_frames = int(foresee_time/env.cfg.step_time)
        if self.cfg.use_endpoints:
            if self.endpoints is None:
                if env.scene_data["endpoints"] is not None:
                    self.endpoints = env.scene_data["endpoints"]
                else:
                    data_path = os.path.join(self.cfg.endpoints_dir, f"{env.scene_id}.npz")
                    data = np.load(data_path)
                    trans_dist, rot_dist = data["trans_dist"], data["rot_dist"]
                    endpoint_idxs = np.where(trans_dist.mean(axis=1)*50+rot_dist.mean(axis=1)*4>1)[0]+1
                    endpoint_idxs = np.concatenate([[0], endpoint_idxs])
                    self.endpoints = endpoint_idxs*130

            current_target_object_frame = env.objects.target_object.frame
            next_endpoint_idx = np.searchsorted(self.endpoints, current_target_object_frame)
            if next_endpoint_idx >= self.endpoints.shape[0]:
                next_endpoint = 1000000000
            else:
                next_endpoint = self.endpoints[next_endpoint_idx]
            foresee_frames = min(foresee_frames, next_endpoint-next_endpoint_idx)
            # if self._cfg.OMG.SHOW_LANDMARKS and self.get_next_landmark(obs["env"].scene_id, obs["env"].ycb._frame) < obs["env"].ycb._frame + self._steps_action_repeat:
            #     sphere_id = pybullet.loadURDF("/share/haoran/HRI/GA-DDPG/data/sphere_model/sphere.urdf", basePosition=obs["env"].ycb._pose[obs["env"].ycb._frame, 0, :3], useMaximalCoordinates=True, globalScaling=0.1)
            #     pybullet.setCollisionFilterGroupMask(sphere_id, 0, 0, 0)
            #     # input("encounter a landmark, press to continue")
        return foresee_frames

    def get_world_to_target_object_for_plan(self, env: MobileH2RSim) -> Tuple[NDArray[np.float64], int]:
        target_obj: Object = env.objects.target_object
        if self.cfg.know_destination:
            start_frame = target_obj.num_frames-1
            foreseed_frame = target_obj.num_frames-1
        elif self.cfg.foresee_time > 0:
            start_frame = target_obj.frame
            foreseed_frame = target_obj.frame+self.get_foresee_frames(env)
            foreseed_frame = min(foreseed_frame, target_obj.num_frames-1)
        else:
            start_frame = target_obj.frame
            foreseed_frame = target_obj.frame

        pose = target_obj.pose[start_frame:foreseed_frame+1] # (num_frames, 6)
        pos = pose[:, :3] # (num_frames, 3)
        ros_quat = Rt.from_euler("XYZ", pose[:, 3:]).as_quat() # (num_frames, 4)
        world_to_target_object = pos_ros_quat_to_mat(pos, ros_quat) # (num_frames, 4, 4)
        return world_to_target_object, foreseed_frame

    def check_replan(self, observation: Observation) -> bool:
        '''
        input
        observation: robot state and visual input
        output
        whether chomp should replan
        basically it check whether the object has changed significantly or the object has just stopped
        '''
        env = observation.env
        object_stopped = env.target_object_stopped_because_of_dist
        object_just_stopped = object_stopped and not self.object_stopped
        self.object_stopped = object_stopped
        if env.frame < self.last_plan_frame+self.cfg.replan_period/env.cfg.step_time and not object_just_stopped:
            return False
        world_to_target_object, foreseed_frame = self.get_world_to_target_object_for_plan(env)
        if self.world_to_target_object is None:
            target_object_pose_changed = True
        else:
            trans_distance, rot_distance = se3_distance(world_to_target_object[-1], self.world_to_target_object)
            target_object_pose_changed = trans_distance > self.cfg.object_pose_change_threshold or rot_distance > 4*self.cfg.object_pose_change_threshold
        in_standoff_phase = self.traj is not None and self.step >= self.traj.shape[0]+1-self.cfg.planner.standoff_steps
        # suppose traj.shape[0] is 10, standoff_steps is 5, then when step in [6, 7, 8, 9], is in standoff phase
        return target_object_pose_changed

    def get_target_grasps(self, env: MobileH2RSim) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        target_object = env.objects.target_object
        hand = env.hand
        world_to_target_object, foreseed_frame = self.get_world_to_target_object_for_plan(env)
        if self.target_object_grasps is None:
            self.target_object_grasps: NDArray[np.float64] = np.load(os.path.join(os.path.dirname(target_object.cfg.urdf_file), "grasps.npy"))@np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., -0.05], [0., 0., 0., 1.]])
            if self.cfg.augment_grasps:
                self.target_object_grasps: NDArray[np.float64] = np.concatenate([self.target_object_grasps, self.target_object_grasps@self.grasp_augmentation_matrix])
        # code.interact(local=dict(globals(), **locals()))
        world_to_target_grasps: NDArray[np.float64] = world_to_target_object[-1]@self.target_object_grasps
        if self.cfg.filter_hand_collision:
            if self.cfg.hand_collision_free_mask_dir is not None:
                if self.filter_results is None:
                    self.filter_results = np.load(os.path.join(self.cfg.hand_collision_free_mask_dir, scene_id_to_hierarchical_dir(env.scene_id), f"{env.scene_id:08d}.npy")).astype(bool)
                filter_results = self.filter_results
            else:
                if foreseed_frame == hand.frame:
                    hand_joint_values = hand.get_joint_positions()
                else:
                    hand_joint_values = hand.pose[foreseed_frame].astype(np.float64)
                filter_results = self.hand_collision_filter.filter_hand_collision(hand.cfg.urdf_file, hand_joint_values, world_to_target_grasps)
            if self.cfg.show_filter_results:
                env.clear_grasps()
                with env.disable_rendering():
                    for world_to_target_grasp, filter_result in zip(world_to_target_grasps, filter_results):
                        env.load_grasp(world_to_target_grasp, color=(0., 1., 0., 1.) if filter_result else (1., 0., 0., 1.))
                # input("press to continue...")
                # env.clear_grasps()
            world_to_target_grasps: NDArray[np.float64] = world_to_target_grasps[filter_results]
        # exit()
        return world_to_target_object, self.target_object_grasps[filter_results], env.hand.pose[foreseed_frame, :3], env.humanoid.pose[foreseed_frame, :3]

    def get_sdf_info(self, observation: Observation) -> Tuple[List[SDFData], List[NDArray[np.float64]]]:
        env = observation.env
        sdf_data_list: List[SDFData] = []
        body_to_base_list: List[torch.DoubleTensor] = []
        for idx, object in enumerate(env.objects.objects):
            # if idx != env.objects.grasp_id:
            sdf_data_list.append(object.get_sdf())
            world_to_object = object.get_world_to_obj()
            object_to_base = se3_inverse(world_to_object)@self.world_to_base
            body_to_base_list.append(object_to_base)
        return sdf_data_list, body_to_base_list

    def trajectory_smoothing(self, traj: NDArray[np.float64], current_joint_values: NDArray[np.float64], env: MobileH2RSim) -> NDArray[np.float64]:
        num_steps = traj.shape[0] # 30
        traj = np.concatenate([current_joint_values[None], traj], axis=0) # (31, 9)
        step_lengths = np.linalg.norm(traj[1:]-traj[:-1], axis=1) # (30, )
        accumulate_lengths = np.concatenate([[0], np.cumsum(step_lengths)]) # (31, )
        total_length: float = accumulate_lengths[-1]
        smoothed_traj, smoothed_traj_idxs = [], []
        # last_conf = start_conf
        smoothing_step_size = self.cfg.trajectory_smoothing_step_size
        if self.cfg.know_destination:
            target_object = env.objects.target_object
            smoothing_step_size = min(smoothing_step_size, total_length/((target_object.num_frames-target_object.frame)/self.action_repeat_steps+5))

        idx: int = 0
        cur_length: float = 0.
        while cur_length < total_length:
            cur_length = min(cur_length+smoothing_step_size, total_length)
            while idx < num_steps-1 and cur_length >= accumulate_lengths[idx+1]:
                idx += 1
            # accumulate_lengths[idx] <= cur_length <= accumulate_lengths[idx+1]
            ratio = (cur_length-accumulate_lengths[idx])/(accumulate_lengths[idx+1]-accumulate_lengths[idx])
            smoothed_traj.append(traj[idx]*(1-ratio)+traj[idx+1]*ratio)
            smoothed_traj_idxs.append(num_steps-idx) # traj[num_steps, num_steps-1, num_steps-2, num_steps-3, num_steps-4, num_steps-5] are standoff poses. if num_steps-idx <= 5, then already enter the standoff phase
        
        smoothed_traj, smoothed_traj_idxs = np.stack(smoothed_traj, axis=0), np.array(smoothed_traj_idxs)
        # print(f"new traj step lengths: {np.linalg.norm(smoothed_traj[1:]-smoothed_traj[:-1], axis=1)}")
        return smoothed_traj

    def plan(self, observation: Observation):
        info: Dict[str, Any] = {}
        if self.check_replan(observation):
            
            

            env = observation.env
            current_joint_values = observation.joint_positions
            current_base_to_ee = self.base_to_world@observation.world_to_ee
            # code.interact(local=dict(globals(), **locals()))
            world_to_target_object, object_to_target_grasps, world_to_hand_position, world_to_humanoid_position = self.get_target_grasps(env)
            self.world_to_target_object = world_to_target_object[-1]
            base_to_target_object = self.base_to_world@world_to_target_object
            sdf_data_list, body_to_base_list = self.get_sdf_info(observation)
            base_to_hand_point = env.objects.target_object.get_world_to_object_pc()
            self.traj, target_grasp_idx = self.planner.plan(base_to_target_object, object_to_target_grasps, current_joint_values, current_base_to_ee, sdf_data_list, body_to_base_list, 
                                                            self.cfg.trajectory_smoothing_step_size if self.cfg.trajectory_smoothing else None, self.action_repeat_steps, world_to_hand_position, world_to_humanoid_position,
                                                            base_to_hand_point)
            # code.interact(local=dict(globals(), **locals()))
            if self.traj is not None:
                self.last_plan_frame = observation.frame
                self.object_to_target_grasp = object_to_target_grasps[target_grasp_idx]
                if self.cfg.show_target_grasp:
                    world_to_target_grasp = self.world_to_target_object@self.object_to_target_grasp
                    with env.disable_rendering():
                        env.load_grasp(world_to_target_grasp, color=(1., 0., 1., 1.))
                # code.interact(local=dict(globals(), **locals()))
                if self.cfg.trajectory_smoothing:
                    # approach_traj = self.traj[:self.traj.shape[0]+1-self.cfg.planner.standoff_steps]
                    # standoff_traj = self.traj[self.traj.shape[0]+1-self.cfg.planner.standoff_steps:]
                    # self.traj = np.concatenate([self.trajectory_smoothing(approach_traj, current_joint_values, env), standoff_traj])
                    self.traj = self.trajectory_smoothing(self.traj, current_joint_values, env)
                # ipdb.set_trace()
            else:
                self.object_to_target_grasp = None
            self.step = 0
        # code.interact(local=dict(globals(), **locals()))
        if self.traj is None:
            info["no plan"], reached = True, False
            if self.cfg.action_type == "joint":
                action, action_type = observation.joint_positions, "joint"
            elif self.cfg.action_type == "ego_cartesian":
                action = np.zeros(7)
                action_type = "ego_cartesian"
            else:
                raise ValueError(f"action type {self.cfg.action_type} is not supported")
        elif self.step == self.traj.shape[0]:
            action, action_type, reached = None, None, True
        else:
            if self.cfg.action_type == "joint":
                action = self.traj[self.step]
                action_type, reached = "joint", False
            elif self.cfg.action_type == "ego_cartesian":
                base_to_ee = self.base_to_world@observation.world_to_ee
                base_to_target_ee = self.planner.robot_kinematics.joint_to_cartesian(self.traj[self.step, :10])
                ee_to_target_ee = se3_inverse(base_to_ee)@base_to_target_ee
                action = np.concatenate([*mat_to_pos_euler(ee_to_target_ee), [0.]])
                # print(f"action step length: {np.linalg.norm(action[:3])}")
                action_type, reached = "ego_cartesian", False
            else:
                raise ValueError(f"action type {self.cfg.action_type} is not supported")
            self.step += 1
            info["world_to_target_grasp"] = observation.env.objects.target_object.get_world_to_obj()@self.object_to_target_grasp
        return action, action_type, reached, info

"""
DISPLAY="localhost:11.0" python -m evaluate use_ray=False env.panda.IK_solver=PyKDL setup=s0 split=train policy=chomp chomp.wait_time=3. chomp.planner.ee_orient_loss_coef=1. chomp.filter_hand_collision=True chomp.show_filter_results=True env.visualize=True env.show_camera=True

CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=32 env.panda.IK_solver=PyKDL setup=s0 split=train policy=chomp chomp.wait_time=3 chomp.filter_hand_collision=True print_failure_ids=True
success rate: 624/720=0.8666666666666667
contact rate: 41/720=0.05694444444444444
   drop rate: 34/720=0.04722222222222222
timeout rate: 21/720=0.029166666666666667
average done frame        : 8533.795833333334
average success done frame: 8587.778846153846
average success num steps : 41.65224358974359
average success           : 0.29421474358974364
contact indices: [166 415 561 117 418 558 408 421 838 487 125 458 328  17 906 126 216  18 155 350 218 981  68 253 122 542 765 361 665 753 850 627 658 653 183 225 625 890 118 515 550]
   drop indices: [370 490 982 453 680 782 842 983  80 513 292 381 108 480 607 123 923 308  71 392 781  91 211 950 871 157 105 860 250 306 851 153 451 482]
timeout indices: [ 50 826  11 783 552 383 352 180 628 827  15 780  57  81  98 553 683  53 555 380  62]
evaluting uses 561.2112858295441 seconds

CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=32 env.panda.IK_solver=PyKDL setup=s0 split=train policy=chomp chomp.wait_time=3 chomp.filter_hand_collision=True demo_dir=tmp/debug_chomp/failure demo_structure=flat record_third_person_video=True scene_ids=[166,415,561,117,418,558,408,421,838,487,125,458,328,17,906,126,216,18,155,350,218,981,68,253,122,542,765,361,665,753,850,627,658,653,183,225,625,890,118,515,550,370,490,982,453,680,782,842,983,80,513,292,381,108,480,607,123,923,308,71,392,781,91,211,950,871,157,105,860,250,306,851,153,451,482,50,826,11,783,552,383,352,180,628,827,15,780,57,81,98,553,683,53,555,380,62]

CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=32 env.panda.IK_solver=PyKDL setup=s0 split=train policy=chomp chomp.wait_time=3 chomp.filter_hand_collision=True chomp.retreat_steps=2 chomp.augment_grasps=False print_failure_ids=True
success rate: 621/720=0.8625
contact rate: 23/720=0.03194444444444444
   drop rate: 46/720=0.06388888888888888
timeout rate: 30/720=0.041666666666666664
average done frame        : 8802.398611111112
average success done frame: 8838.141706924316
average success num steps : 43.589371980676326
average success           : 0.27618963675213676
contact indices: [216 988 218 415  17 753 850 837  18 135 542 890 417 196 527 118 561 121
 558 350 741 981 480]
   drop indices: [157 871 250 763 860 282 115 782 461 105 453 283 451 920  33 842 211 592
 681 370 983 996 160 643 851  91 381 861 290 408 490 633 513 640  71 680
 950 392 781  96 492 123 306 153 272 515]
timeout indices: [ 15 352 552 783  13  11 628  98 827  50 380  80 212 553  81 458 582 411
 251 550  53 683 780  68 555  57 826  62 383 180]
evaluting uses 463.26043462753296 seconds

CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=32 env.panda.IK_solver=PyKDL setup=s0 split=train policy=chomp chomp.wait_time=3 chomp.filter_hand_collision=True chomp.retreat_steps=2 chomp.augment_grasps=False print_failure_ids=True demo_dir=tmp/debug_chomp/retreat_no_augment_failure demo_structure=flat record_third_person_video=True scene_ids=[216,988,218,415,17,753,850,837,18,135,542,890,417,196,527,118,561,121,558,350,741,981,480,157,871,250,763,860,282,115,782,461,105,453,283,451,920,33,842,211,592,681,370,983,996,160,643,851,91,381,861,290,408,490,633,513,640,71,680,950,392,781,96,492,123,306,153,272,515,15,352,552,783,13,11,628,98,827,50,380,80,212,553,81,458,582,411,251,550,53,683,780,68,555,57,826,62,383,180]

CUDA_VISIBLE_DEVICES=0,1,3,4 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=32 env.panda.IK_solver=PyKDL setup=s0 split=train policy=chomp chomp.wait_time=3 chomp.filter_hand_collision=True chomp.retreat_steps=2 print_failure_ids=True
success rate: 638/720=0.8861111111111111
contact rate: 23/720=0.03194444444444444
   drop rate: 39/720=0.05416666666666667
timeout rate: 20/720=0.027777777777777776
average done frame        : 8737.502777777778
average success done frame: 8811.7131661442
average success num steps : 43.39811912225705
average success           : 0.28555181623931625
contact indices: [17, 166, 216, 218, 253, 328, 350, 358, 361, 408, 415, 418, 421, 515, 542, 550, 558, 561, 665, 753, 765, 850, 890]
   drop indices: [15, 71, 80, 91, 105, 108, 123, 153, 157, 211, 250, 282, 283, 292, 306, 308, 370, 381, 392, 441, 451, 453, 480, 482, 490, 513, 607, 643, 680, 781, 782, 842, 851, 860, 871, 923, 950, 982, 983]
timeout indices: [11, 50, 53, 57, 62, 81, 98, 180, 352, 380, 383, 552, 553, 555, 628, 683, 780, 783, 826, 827]
evaluting uses 531.239711523056 seconds

CUDA_VISIBLE_DEVICES=0,1,3,4 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=32 env.panda.IK_solver=PyKDL setup=s0 split=train policy=chomp chomp.wait_time=3 chomp.filter_hand_collision=True chomp.retreat_steps=2 print_failure_ids=True demo_dir=tmp/debug_chomp/retreat_failure_drop demo_structure=flat record_third_person_video=True scene_ids=[786,481,607,451,441,108,860,480,392,842,123,153,292,370,761,940,981,306,583,482,985,80,490,871,211,453,283,991,15,288,381,91,643,105,308,681,851,157,966,982,923,742,71,282,781,950,250,115,580,740,782,513,983,680]

CUDA_VISIBLE_DEVICES=6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=10 env.panda.IK_solver=PyKDL policy=chomp env.stop_moving_time=3 chomp.wait_time=3 chomp.filter_hand_collision=True chomp.retreat_steps=2 print_failure_ids=True demo_dir=tmp/debug_scene_generation/ demo_structure=flat record_third_person_video=True scene_ids=[1000000,1008836,1017672,1026508,1035344,1044180] chomp.show_filter_results=True chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0.01

CUDA_VISIBLE_DEVICES=0 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=10 env.panda.IK_solver=PyKDL policy=chomp env.stop_moving_time=8 env.stop_moving_dist=0.1 chomp.filter_hand_collision=True chomp.retreat_steps=2 print_failure_ids=True demo_dir=tmp/debug_chomp/simultaneous demo_structure=flat record_third_person_video=True scene_ids=[1000000,1008836,1017672,1026508,1035344,1044180] chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0.01

CUDA_VISIBLE_DEVICES=0 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=10 env.panda.IK_solver=PyKDL policy=chomp env.stop_moving_time=8 env.stop_moving_dist=0.1 chomp.filter_hand_collision=True chomp.retreat_steps=2 print_failure_ids=True demo_dir=tmp/debug_chomp/simultaneous demo_structure=flat record_third_person_video=True scene_ids=[1035344] chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0.01 env.status_checker.verbose=True verbose=True
"""