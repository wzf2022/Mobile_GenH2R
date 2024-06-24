import os
import sys
import numpy as np
import pybullet
from pybullet_utils.bullet_client import BulletClient
import random
from transforms3d.euler import euler2quat, euler2mat, mat2euler
from transforms3d.quaternions import quat2mat, mat2quat
from scipy.spatial.transform import Rotation as Rt
import time
import code

from env.handover_env import Observation
from env.hand import Hand
from env.objects import Object, Objects
from env.utils.transform import tf_quat, ros_quat, se3_inverse
from env.utils.robot_kinematics import RobotKinematics
from .base_policy import BasePolicy
from .omg_planner_policy_config import OMGPlannerPolicyConfig

assert "OMG_PLANNER_DIR" in os.environ, "Environment variable 'OMG_PLANNER_DIR' is not set"
sys.path.append(os.environ["OMG_PLANNER_DIR"])
from omg.config import cfg as cfg_planner
from omg.core import PlanningScene

def rotZ(rotz):
    RotZ = np.array(
        [
            [np.cos(rotz), -np.sin(rotz), 0, 0],
            [np.sin(rotz), np.cos(rotz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return RotZ

def safemat2quat(mat):
    quat = np.array([1,0,0,0])
    try:
        quat = mat2quat(mat)
    except:
        pass
    quat[np.isnan(quat)] = 0
    return quat

def pack_pose(pose, rot_first=False):
    packed = np.zeros(7)
    if rot_first:
        packed[4:] = pose[:3, 3]
        packed[:4] = safemat2quat(pose[:3, :3])
    else:
        packed[:3] = pose[:3, 3]
        packed[3:] = safemat2quat(pose[:3, :3])
    return packed

def unpack_pose(pose, rot_first=False):
    unpacked = np.eye(4)
    if rot_first:
        unpacked[:3, :3] = quat2mat(pose[:4])
        unpacked[:3, 3] = pose[4:]
    else:
        unpacked[:3, :3] = quat2mat(pose[3:])
        unpacked[:3, 3] = pose[:3]
    return unpacked

def compact2mat(pose_compact):
    "pose_compact: (7, ) [*trans, x, y, z, w]"
    pose_mat = np.eye(4)
    pose_mat[:3, :3] = Rt.from_quat(pose_compact[3:]).as_matrix()
    pose_mat[:3, 3] = pose_compact[:3]
    return pose_mat

def six_d_to_mat(six_d):
    " (..., 6) "
    shape_prefix = six_d.shape[:-1]
    mat = np.zeros(shape_prefix + (4, 4), dtype=six_d.dtype)
    mat[..., :3, 3] = six_d[..., :3]
    mat[..., :3, :3] = Rt.from_euler("XYZ", six_d[..., 3:].reshape(-1, 3)).as_matrix().reshape(shape_prefix + (3, 3))
    mat[..., 3, 3] = 1
    return mat

def ycb_special_case(pose_grasp, name):
    if name == '037_scissors': # only accept top down for edge cases
        z_constraint = np.where((np.abs(pose_grasp[:, 2, 3]) > 0.09) * \
                 (np.abs(pose_grasp[:, 1, 3]) > 0.02) * (np.abs(pose_grasp[:, 0, 3]) < 0.05)) 
        pose_grasp = pose_grasp[z_constraint[0]]
        top_down = []
        
        for pose in pose_grasp:
            top_down.append(mat2euler(pose[:3, :3]))
        
        top_down = np.array(top_down)[:,1]
        rot_constraint = np.where(np.abs(top_down) > 0.06) 
        pose_grasp = pose_grasp[rot_constraint[0]]
    
    elif name == '024_bowl' or name == '025_mug':
        if name == '024_bowl':
            angle = 30
        else:
            angle = 15
        top_down = []
        for pose in pose_grasp:
            top_down.append(mat2euler(pose[:3, :3]))
        top_down = np.array(top_down)[:,1]
        rot_constraint = np.where(np.abs(top_down) > angle * np.pi / 180)
        pose_grasp = pose_grasp[rot_constraint[0]]
    return pose_grasp

def trans_distance(p1, p2):
    "p1, p2: (..., 3)"
    return np.linalg.norm(p1-p2, axis=-1)

def rot_distance(R1, R2):
    "R1, R2: (..., 3, 3)"
    return np.arccos(np.clip(((R1*R2).sum(axis=(-1, -2))-1)/2, -1, 1))

def se3_distance(T1, T2):
    "T1, T2: (..., 4, 4)"
    return trans_distance(T1[..., :3, 3], T2[..., :3, 3]), rot_distance(T1[..., :3, :3], T2[..., :3, :3])

class OMGPlanner:
    def __init__(self, silent=True, seed=0):
        cfg_planner.ik_parallel = False
        cfg_planner.vis = False
        cfg_planner.scene_file = ""
        cfg_planner.cam_V = None
        self.seed = seed

        # Enforce determinism. This accounts for the call of random.sample() in
        # `Robot.load_collision_points()` in `OMG-Planner/omg/core.py`.
        random.seed(self.seed)

        self._scene = PlanningScene(None)
        self.goal_ids = []
        self.default_timesteps = cfg_planner.timesteps
        self.silent = silent
        self._scene_id = None

    def reset_scene(self, names, poses, scene_id):
        if self._scene_id != scene_id:
            self._scene_id = scene_id
            for name in list(self._scene.env.names):
                self._scene.env.remove_object(name)
            assert len(self._scene.env.objects) == 0

            for name, pose in zip(names, poses):
                self._scene.env.add_object(name, pose[:3], pose[3:], compute_grasp=False)
        else:
            for name, pose in zip(names, poses):
                self._scene.env.update_pose(name, pose)
        self._scene.env.combine_sdfs()

    def show_goals(self, goals):
        # show_goals_num = int(input(f"show goals num (total {len(goals)}):"))
        # show_goals_num = min(show_goals_num, len(goals))
        # print(f"showing {show_goals_num} goals out of {len(goals)}")
        # selected_indices = np.random.choice(len(goals), show_goals_num, replace=False)
        # selected_goals = goals[selected_indices]

        print(f"{len(goals)} goals in total")
        code.interact(local=dict(globals(), **locals()))
        for i, goal in enumerate(goals):
            self.panda.body.dof_target_position = goal
            self.simulate_panda(0.0)
            input(f"({i}/{len(goals)}) press to continue...")
    
    def clear_goals(self):
        for goal_id in self.goal_ids:
            print(f"removing {pybullet.getBodyInfo(goal_id)}")
            pybullet.removeBody(goal_id)
        self.goal_ids = []

    def plan_to_target(self, start_conf, target_name, grasps_no_contact_mask=None, grasps=None, step=-1):
        # Enfore determinism. This accounts for the call of `np.random.choice()` in
        # `Planner.setup_goal_set()` in `OMG-Planner/omg/planner.py`.
        # print(f"start_conf {start_conf}")
        random_state = np.random.get_state()
        np.random.seed(self.seed)

        self._scene.traj.start = start_conf
        self._scene.env.set_target(target_name)

        if step == -1 or cfg_planner.timesteps == step:
            if not hasattr(self._scene, "planner"):
                self._scene.reset(grasps_no_contact_mask=grasps_no_contact_mask, grasps=grasps)
            else:
                self._scene.update_planner(grasps_no_contact_mask=grasps_no_contact_mask, grasps=grasps)
            
            target_obj_index = self._scene.env.names.index(target_name)

            # code.interact(local=dict(globals(), **locals()))
            # self.show_goals(self._scene.env.objects[target_obj_index].grasps) # (3, 9)
            # self.show_goals(self._scene.env.objects[target_obj_index].reach_grasps) # (3, 5, 9)
            # self.show_goals(self._scene.traj.goal_set) # (3, 9)
            # self.show_goals(self._scene.env.objects[target_obj_index].reach_grasps_after_ik) # list of (5, 9), len=119
            # self.show_goals(self._scene.env.objects[target_obj_index].grasps_after_ik) # list of (9,), len=119

            # self._scene.planner.grasp_init(self._scene.planner.env)

            info = self._scene.step()
        else:
            # self._scene.env.objects[self._scene.env.target_idx].compute_grasp = False
            cfg_planner.timesteps = step # 20
            cfg_planner.get_global_param(cfg_planner.timesteps)
            self._scene.reset(grasps_no_contact_mask=grasps_no_contact_mask, grasps=grasps)
            info = self._scene.step()
            cfg_planner.timesteps = self.default_timesteps # 20
            cfg_planner.get_global_param(cfg_planner.timesteps)

        traj = self._scene.planner.history_trajectories[-1]
        if len(info) == 0:
            traj = None

        # print(f"random state {np.random.get_state()}")
        np.random.set_state(random_state)
        return traj, info

class OMGPlannerPolicy(BasePolicy):
    def __init__(self, cfg: OMGPlannerPolicyConfig, silent=True):
        super().__init__(cfg)
        self.cfg: OMGPlannerPolicyConfig
        cfg_planner.one_trial = self.cfg.ONE_TRIAL
        self._panda_base_invert_transform = pybullet.invertTransform(cfg.base_position, cfg.base_orientation)
        self.world_to_base = np.eye(4)
        self.world_to_base[:3, :3] = Rt.from_quat(np.array(cfg.base_orientation)).as_matrix()
        self.world_to_base[:3, 3] = np.array(cfg.base_position)
        self.base_to_world = se3_inverse(self.world_to_base)
        self._omg_planner = OMGPlanner(silent=silent, seed=self.cfg.seed)
        self.grasp_ids = []
        self.hand_finger_points = np.array([ [ 0.,  0.,  0.   , -0.   ,  0.   , -0.   ],
                                            [ 0.,  0.,  0.053, -0.053,  0.053, -0.053],
                                            [ 0.,  0.,  0.075,  0.075,  0.105,  0.105]])
        self.cur_target_grasp = np.eye(4)
        self.robot = RobotKinematics(cfg.robot_kinematics)
        self._start_conf = np.array(cfg.init_joint)
        self._contact_mask_scene_id = None
        self._grasp_set_obj_name = None
    
    def reset(self, remain_timesteps=None):
        self.base_reset()

        self._traj = None
        self.all_grasps = None
        self.all_grasps_world = None
        self.grasps_no_contact_mask = None
        self.obj_pose = None
        self._last_plan_frame = -np.inf
        self._last_action_frame = -np.inf
        self._target_grasp = None
        if remain_timesteps is None:
            self.remain_timesteps = self.cfg.TIMESTEPS
        else:
            self.remain_timesteps = remain_timesteps
        self._reach_standoff_phase = False
        self._landmark_scene_id = None
        self._obj_stopped = False

    def get_target_obj_pose_for_plan(self, obs):
        target_obj: Object = obs.env.objects.target_object
        if (self.cfg.SIMULTANEOUS and self.cfg.KNOW_DESTINATION) or self.cfg.USE_RAW_TARGET_OBJ_POSE:
            pos = target_obj.pose[-1, :3]
            orn = Rt.from_euler("XYZ", target_obj.pose[-1, 3:]).as_quat()
        elif self.cfg.SIMULTANEOUS and self.cfg.FORESEE_FRAMES > 0:
            foresee_frame = target_obj.frame + self.get_foresee_frames(obs)
            foresee_frame_obj = min(foresee_frame, target_obj.pose.shape[0] - 1)
            if foresee_frame_obj == target_obj.frame and self.cfg.USE_REAL_POSE_WHEN_CLOSE:
                pos, orn = target_obj.get_link_pos_orn(5)
                # pos = obs["ycb_bodies"][target_obj_idx].link_state[0, 6, 0:3]
                # orn = obs["ycb_bodies"][target_obj_idx].link_state[0, 6, 3:7]
            else:
                pos = target_obj.pose[foresee_frame_obj, :3]
                orn = Rt.from_euler("XYZ", target_obj.pose[foresee_frame_obj, 3:]).as_quat()
            if self.cfg.stop_moving_dist is not None and obs.env.min_dist <= self.cfg.stop_moving_dist:
                print("object stop moving")
            print(f"target pos {target_obj.pose[target_obj.frame, :3]}")
            print(f"current pos orn {target_obj.get_link_pos_orn(5)}")
        else:
            pos, orn = target_obj.get_link_pos_orn(5)
            # pos = obs["ycb_bodies"][target_obj_idx].link_state[0, 6, 0:3]
            # orn = obs["ycb_bodies"][target_obj_idx].link_state[0, 6, 3:7]
        return pos, orn

    def load_grasp_set(self, obj_name):
        if self._grasp_set_obj_name == obj_name:
            return self._grasp_set
        self._grasp_set_obj_name = obj_name

        grasp_path = os.path.join(self.cfg.GRASP_DIR, f"{obj_name}.npy")
        if "acronym" in self.cfg.GRASP_DIR:
            grasp_pose = np.load(grasp_path)
            offset_pose = np.array(rotZ(np.pi / 2))  # and
            grasp_pose = np.matmul(grasp_pose, offset_pose)  # flip x, y
            self._grasp_set = grasp_pose
            rot_mat = np.diag([-1, -1, 1, 1]) # rotate 180 around z
            self._grasp_set = np.concatenate([self._grasp_set, self._grasp_set @ rot_mat])
            # /share/haoran/HRI/handover-sim/data/acronym/Truck_40597e0ca3a8eca9a1cba95c26616410_0.0004911644164552468.npy
        else: # ycb
            simulator_grasp = np.load(
                grasp_path,
                allow_pickle=True,
                fix_imports=True,
                encoding="bytes",
            )
            pose_grasp = simulator_grasp.item()[b"transforms"]
            offset_pose = np.array(rotZ(np.pi / 2))  # and
            pose_grasp = np.matmul(pose_grasp, offset_pose)  # flip x, y
            # print(f"load {pose_grasp.shape[0]} grasps")
            pose_grasp = ycb_special_case(pose_grasp, obj_name)
            self._grasp_set = pose_grasp

            processed_pose_grasp = np.load(os.path.join("env", "data", "assets", "objects", "ycb", obj_name, "grasps.npy"))
            print(f"{np.abs(pose_grasp-processed_pose_grasp).max()}")
        # print(f"load {pose_grasp.shape[0]} grasps after ycb special case")
        return self._grasp_set

    def filter_grasps(self, grasps, hand: Hand, hand_pose=None):
        if hand_pose is None:
            if hand.body_id is None:
                return np.ones(grasps.shape[0], dtype=bool)
            hand_pose = hand.get_joint_positions()

        if self.cfg.SHOW_FILTERING:
            bullet_client = BulletClient(pybullet.GUI)
            bullet_client.resetDebugVisualizerCamera(cameraDistance=2.4, cameraPitch=-58, cameraYaw=102, cameraTargetPosition=[0, 0, 0])
        else:
            bullet_client = BulletClient(pybullet.DIRECT)
        hand_urdf_file = os.path.join("env", "data", "assets", "hand", hand.cfg.name, "mano.urdf")
        hand_id = bullet_client.loadURDF(hand_urdf_file)

        dof_indices = []
        for j in range(bullet_client.getNumJoints(hand_id)):
            joint_info = bullet_client.getJointInfo(hand_id, j)
            if joint_info[2] != pybullet.JOINT_FIXED:
                dof_indices.append(j)
        dof_indices = np.asanyarray(dof_indices, dtype=np.int64)
        for i, idx in enumerate(dof_indices):
            bullet_client.resetJointState(hand_id, idx, hand_pose[i])

        if self.cfg.SHOW_FILTERING:
            for i in range(52):
                bullet_client.changeVisualShape(hand_id, i, rgbaColor=[0, 1, 0, 1])

        # pybullet.setJointMotorControlArray(hand_id, hand_joint_indices, pybullet.POSITION_CONTROL, hand_pose, physicsClientId=cid)
        # pre_joint_states = None
        # while True:
        #     pybullet.stepSimulation(physicsClientId=cid)
        #     joint_states = np.stack(pybullet.getJointStates(hand_id, hand_joint_indices, physicsClientId=cid), axis=1)[0]
        #     # print(f"len(joint_states)={len(joint_states)}") # 51
        #     if pre_joint_states is not None and np.abs(joint_states-pre_joint_states).max() < 1e-3:
        #         break
        #     pre_joint_states = joint_states
        
        grasp_id = bullet_client.loadURDF("/share/haoran/HRI/handover-sim/data/gripper_model/gripper_collision.urdf")
        # code.interact(local=dict(globals(), **locals()))

        mask = np.zeros(grasps.shape[0], dtype=bool)
        for i, grasp in enumerate(grasps):
            grasp_pack_pose = pack_pose(grasp)
            if self.cfg.SHOW_FILTERING:
                bullet_client.changeVisualShape(grasp_id, -1, rgbaColor=[0, 1, 0, 1])
            bullet_client.resetBasePositionAndOrientation(grasp_id, grasp_pack_pose[:3], ros_quat(grasp_pack_pose[3:]))
            bullet_client.performCollisionDetection()
            contact_points = bullet_client.getContactPoints(hand_id, grasp_id)
            if len(contact_points)==0: 
                mask[i] = 1
            else:
                if self.cfg.SHOW_FILTERING:
                    bullet_client.changeVisualShape(grasp_id, -1, rgbaColor=[1, 0, 0, 1])
                pass
            if self.cfg.SHOW_FILTERING:
                time.sleep(0.1)
        # code.interact(local=dict(globals(), **locals()))
        bullet_client.disconnect()
        # print(f"mask {mask.sum()}")
        return mask

    def get_grasps_no_contact_mask(self, obs: Observation, target_obj_grasps_world): # depend on filter_grasps
        if self.cfg.ONLY_FILTER_ONCE:
            assert self.cfg.GRASP_DIR != "/share/haoran/HRI/OMG-Planner/data/grasps/simulated" # not compatible with dexycb
            if self._contact_mask_scene_id != obs.env.scene_id:
                self._contact_mask_scene_id = obs.env.scene_id
                mask_path = os.path.join("/share1/haoran/HRI/data/grasps_no_contact_mask", f"{obs.env.scene_id}.npy")
                if self.cfg.LOAD_GRASPS_NO_CONTACT_MASK and os.path.exists(mask_path):
                    self._grasps_no_contact_mask = np.load(mask_path)
                else:
                    obj_pose = six_d_to_mat(obs.env.ycb._pose[0, 0])
                    target_name = obs.env.objects.target_object.cfg.name
                    target_obj_grasps_world = obj_pose @ self.load_grasp_set(target_name)
                    self._grasps_no_contact_mask = self.filter_grasps(target_obj_grasps_world, obs.env.hand) # should be frame 0
                    assert obs.frame == 0
                if self.cfg.SAVE_GRASPS_NO_CONTACT_MASK and not os.path.exists(mask_path):
                    np.save(mask_path, self._grasps_no_contact_mask)
            return self._grasps_no_contact_mask
        else:
            if self.cfg.SIMULTANEOUS and self.cfg.KNOW_DESTINATION:
                hand_pose = obs.env.hand.pose[-1]
            elif self.cfg.SIMULTANEOUS and self.cfg.FORESEE_FRAMES > 0:
                foresee_frame = obs.env.hand.frame + self.get_foresee_frames(obs)
                foresee_frame_hand = min(foresee_frame, obs.env.hand.pose.shape[0]-1)
                hand = obs.env.hand
                if foresee_frame_hand == obs.env.hand.frame and self.cfg.USE_REAL_POSE_WHEN_CLOSE:
                    hand_pose = None
                else:
                    hand_pose = obs.env.hand.pose[foresee_frame_hand]
            else:
                hand_pose = None
            grasps_no_contact_mask = self.filter_grasps(target_obj_grasps_world, obs.env.hand, hand_pose=hand_pose)
            return grasps_no_contact_mask

    def compute_distance(self, T1, T2):
        "T1: (..., 4, 4) T2: (..., 4, 4)"
        trans_dist, rot_dist = se3_distance(T1, T2)
        return trans_dist + 0.25 * rot_dist

    def trajectory_smoothing(self, traj, start_conf):
        num_steps = traj.shape[0] # 30
        traj = np.concatenate([start_conf[None], traj], axis=0) # (31, 9)
        step_lengths = np.linalg.norm(traj[1:]-traj[:-1], axis=1) # (30, )
        # print(f"traj step lengths: {np.linalg.norm(traj[1:]-traj[:-1], axis=1)}")
        accumulate_lengths = np.concatenate([[0], np.cumsum(step_lengths)]) # (31, )
        total_length = accumulate_lengths[-1]
        new_traj, new_traj_idxs = [], []
        idx = 0
        # last_conf = start_conf
        smoothing_step_size = self.cfg.TRAJECTORY_SMOOTHING_STEP_SIZE
        if self.cfg.stop_moving_frame is not None and self.cfg.KNOW_DESTINATION:
            smoothing_step_size = min(smoothing_step_size, total_length / (self.cfg.stop_moving_frame / self.action_repeat_steps + 5))
        if self.cfg.SLOW_DOWN_RATIO == 1:
            cur_length = 0
            while cur_length < total_length:
                cur_length = min(cur_length+smoothing_step_size, total_length)
                while idx < num_steps-1 and cur_length >= accumulate_lengths[idx+1]:
                    idx += 1
                # accumulate_lengths[idx] <= cur_length <= accumulate_lengths[idx+1]
                ratio = (cur_length-accumulate_lengths[idx])/(accumulate_lengths[idx+1]-accumulate_lengths[idx])
                new_traj.append(traj[idx]*(1-ratio)+traj[idx+1]*ratio)
                new_traj_idxs.append(num_steps-idx) # traj[num_steps, num_steps-1, num_steps-2, num_steps-3, num_steps-4, num_steps-5] are standoff poses. if num_steps-idx <= 5, then already enter the standoff phase
                # print(f"step length {np.linalg.norm(new_traj[-1]-last_conf)}")
                # last_conf = new_traj[-1]
                # code.interact(local=dict(globals(), **locals()))
        else:
            step_lengths = []
            slow_down_steps, slow_down_ratio, slow_down_final_steps = self.cfg.SLOW_DOWN_STEPS, self.cfg.SLOW_DOWN_RATIO, self.cfg.SLOW_DOWN_FINAL_STEPS
            remain_length = total_length

            final_step_size = smoothing_step_size*slow_down_ratio
            final_steps = min(slow_down_final_steps, int(remain_length/final_step_size))
            step_lengths.extend([final_step_size]*final_steps)
            remain_length -= final_step_size*final_steps
            if final_steps < slow_down_final_steps and remain_length > 0:
                step_lengths.append(remain_length)
                remain_length = 0

            if remain_length > 0:
                for i in range(slow_down_steps):
                    cur_step_length = smoothing_step_size*(slow_down_ratio+i/slow_down_steps*(1-slow_down_ratio))
                    if cur_step_length <= remain_length:
                        step_lengths.append(cur_step_length)
                        remain_length -= cur_step_length
                    else:
                        step_lengths.append(remain_length)
                        remain_length = 0
                        break
            if remain_length > 0:
                step_lengths.extend([smoothing_step_size]*int(remain_length/smoothing_step_size))
                step_lengths.append(remain_length%smoothing_step_size)
            step_lengths = sorted(step_lengths, reverse=True)
            # print(step_lengths)

            for cur_length in np.cumsum(step_lengths):
                while idx < num_steps-1 and cur_length >= accumulate_lengths[idx+1]:
                    idx += 1
                # accumulate_lengths[idx] <= cur_length <= accumulate_lengths[idx+1]
                ratio = (cur_length-accumulate_lengths[idx])/(accumulate_lengths[idx+1]-accumulate_lengths[idx])
                new_traj.append(traj[idx]*(1-ratio)+traj[idx+1]*ratio)
                new_traj_idxs.append(num_steps-idx)
        new_traj, new_traj_idxs = np.stack(new_traj, axis=0), np.array(new_traj_idxs)
        # print(f"new traj step lengths: {np.linalg.norm(new_traj[1:]-new_traj[:-1], axis=1)}")
        return new_traj, new_traj_idxs

    def _run_omg_planner(self, obs: Observation, step=-1, cur_start_conf=None): # depend on get_target_obj_pose_for_plan, load_grasp_set, get_grasps_no_contact_mask, show_grasps, compute_distance, trajectory_smoothing
        info = {}
        if cur_start_conf is None:
            cur_start_conf = self._start_conf
        print("running omg planner...")
        # pybullet.resetDebugVisualizerCamera(cameraDistance=2.5, cameraPitch=-45, cameraYaw=90, cameraTargetPosition=[0, 0, 0])
        objects: Objects = obs.env.objects
        raw_poses = [self.get_target_obj_pose_for_plan(obs)]
        names = [objects.target_object.cfg.name]
        for i, obj in enumerate(objects.objects):
            if i != objects.grasp_id:
                pos, orn = obj.get_link_pos_orn(5)
                # pos = obs["ycb_bodies"][i].link_state[0, 6, 0:3]
                # orn = obs["ycb_bodies"][i].link_state[0, 6, 3:7]
                raw_poses.append((pos, orn))
                names.append(obj.cfg.name)
        poses = []
        poses_handoversim = []
        for pos, orn in raw_poses:
            poses_handoversim.append(np.concatenate([pos, tf_quat(orn)]))
            pos, orn = pybullet.multiplyTransforms(*self._panda_base_invert_transform, pos, orn)
            poses += [orn + pos]
        poses = [p[4:] + (p[3], p[0], p[1], p[2]) for p in poses]
        self._omg_planner.reset_scene(names, poses, obs.env.scene_id)

        target_name = names[0]

        target_obj_index = self._omg_planner._scene.env.names.index(target_name)
        target_obj_pose = unpack_pose(poses_handoversim[target_obj_index])
        target_obj_grasps = self.load_grasp_set(target_name)
        self.all_grasps = target_obj_grasps

        target_obj_grasps_world = target_obj_pose@target_obj_grasps
        self.all_grasps_world = target_obj_grasps_world

        if self.cfg.FILTER_HAND_COLLISION_GRASPS:
            grasps_no_contact_mask = self.get_grasps_no_contact_mask(obs, target_obj_grasps_world)

            target_obj_grasps_world_filtered = target_obj_grasps_world[grasps_no_contact_mask]

            if self.cfg.SHOW_GRASPS:
                self.show_grasps(target_obj_grasps_world_filtered)

            # self._omg_planner._scene.reset_traj()
            if self.cfg.ONLY_NEAREST_GRASP:
                ef_pose = obs.world_to_ee
                dists = self.compute_distance(ef_pose, target_obj_grasps_world_filtered)
                idxs = dists.argsort()
                idxs = np.where(grasps_no_contact_mask)[0][idxs]
                num_trials, traj = 0, None
                for idx in idxs:
                    # print(f"grasp idx={idx}")
                    target_obj_grasp_world = target_obj_grasps_world[idx]
                    traj, _ = self._omg_planner.plan_to_target(cur_start_conf, target_name, grasps_no_contact_mask=np.arange(target_obj_grasps_world.shape[0])==idx, grasps=target_obj_grasps[[idx]], step=step)
                    num_trials += 1
                    if traj is not None:
                        break
                    self._omg_planner._scene.env.objects[target_obj_index].grasps_poses = []
                if self.cfg.SHOW_GRASPS:
                    grasp_id = self.show_grasp(target_obj_grasp_world, [1, 0, 0])
                    pybullet.changeVisualShape(grasp_id, -1, rgbaColor=[0, 0, 0, 0])
                # self.show_grasp(target_obj_grasp_world)
                # print(f"find a trajectory after {num_trials} trials")
                # return traj, info
            else:
                self.grasps_no_contact_mask = grasps_no_contact_mask
                grasps = target_obj_grasps[grasps_no_contact_mask]
                if grasps.shape[0] > 100:
                    grasps = grasps[np.random.choice(np.arange(grasps.shape[0]), 100, replace=False)]
                traj, _ = self._omg_planner.plan_to_target(cur_start_conf, target_name, grasps=grasps, step=step)
        else:
            traj, _ = self._omg_planner.plan_to_target(cur_start_conf, target_name, step=step)

        # self._omg_planner._scene.env.objects[target_obj_index].grasps_poses = ori_grasps_poses

        # if self.cfg.SHOW_GRASPS:
        #     target_obj_grasps_world_filtered_1 = target_obj_pose@self._omg_planner._scene.env.objects[target_obj_index].grasps_poses
        #     self.show_grasps(target_obj_grasps_world_filtered_1)
        # code.interact(local=dict(globals(), **locals()))
        # self.clear_grasps()
        # print_body_info()
        if traj is not None:
            target_grasp_base = self.robot.joint_to_cartesian(traj[-1][:7])
            target_grasp_world = self.world_to_base@target_grasp_base
            # print("target_grasp_world", target_grasp_world)
            target_grasp = se3_inverse(target_obj_pose)@target_grasp_world
            # target_grasp_inv = se3_inverse(target_grasp)
            # closest_pose, dist = None, np.inf
            # for grasp in self.all_grasps[grasps_no_contact_mask]:
            #     delta_pose = target_grasp_inv@grasp
            #     if np.abs(delta_pose-np.eye(4)).sum() < dist:
            #         dist = np.abs(delta_pose-np.eye(4)).sum()
            #         closest_pose = delta_pose
            # print("closest delta pose", closest_pose)
            # print("target_grasp_base", target_grasp_base)
            if self.cfg.TRAJECTORY_SMOOTHING:
                traj, info["traj_idxs"] = self.trajectory_smoothing(traj, cur_start_conf)
            else:
                info["traj_idxs"] = np.arange(len(traj)-1, -1, -1)
        else:
            target_grasp = None
        return traj, info, target_grasp

    def plan(self, obs: Observation): # depend on get_target_obj_pose_for_plan, _run_omg_planner
        info = {}
        if self._traj is not None and len(self._traj) > 0 and obs.frame == self.traj_start_frame + self.action_repeat_steps * len(self._traj):
            action, action_type, reached = None, None, True
            return action, action_type, reached, info

        self._last_action_frame = obs.frame
        obj_stopped = self.cfg.stop_moving_dist is not None and obs.env.min_dist <= self.cfg.stop_moving_dist
        obj_just_stopped = obj_stopped and not self._obj_stopped
        self._obj_stopped = obj_stopped
        if self._traj is None or obs.frame >= self._last_plan_frame + self.cfg.REPLAN_PERIOD*self.action_repeat_steps or (obj_just_stopped and self.cfg.REPLAN_WHEN_OBJ_STOP):
            self._last_plan_frame = obs.frame
            cur_obj_pose = np.concatenate(self.get_target_obj_pose_for_plan(obs)) # obs["ycb_bodies"][list(obs["ycb_bodies"].keys())[0]].link_state[0, 6, :7]
            if self.obj_pose is None:
                obj_pose_changed = True
            else:
                obj_pose_changed = np.abs(cur_obj_pose[:3]-self.obj_pose[:3]).max()>self.cfg.OBJ_POSE_CHANGE_THRES or np.abs(cur_obj_pose[3:]-self.obj_pose[3:]).max()>4*self.cfg.OBJ_POSE_CHANGE_THRES
            if obj_pose_changed:
                self.obj_pose = cur_obj_pose

            if (self._traj is None or (self.cfg.SIMULTANEOUS and not self.cfg.KNOW_DESTINATION and obj_pose_changed) or (self.cfg.ENFORCE_REPLAN and self.remain_timesteps >= 6)) and (obj_pose_changed or not self._reach_standoff_phase):
                cur_start_conf = obs.env.panda.get_joint_positions()
                traj, omg_info, target_grasp = self._run_omg_planner(obs, step=self.remain_timesteps, cur_start_conf=cur_start_conf)
                if traj is None:
                    print("Planning not run due to empty goal set. Stay in start conf.")
                    self._traj = []
                    self._target_grasp = None
                else:
                    assert len(traj) > 0
                    self._traj = traj
                    self._target_grasp = target_grasp
                    self._traj_idxs = omg_info["traj_idxs"]
                    self._reach_standoff_phase = False
                    # if self.cfg.SHOW_TRAJECTORY:
                    #     self.show_trajectory(traj)
                    if not self.cfg.TRAJECTORY_SMOOTHING:
                        self.remain_timesteps = max(self.remain_timesteps-self.cfg.REPLAN_PERIOD, 5)
                self.traj_start_frame = obs.frame

        if len(self._traj) == 0:
            info["no plan"], reached = True, False
            if self.cfg.ACTION_TYPE == "joint":
                action, action_type = obs.joint_positions, "joint"
            else:
                action = np.zeros(7)
                action[6] = 0.04
                action_type = "ego_cartesian"
        else:
            i = (obs.frame - self.traj_start_frame) // self.action_repeat_steps
            self._reach_standoff_phase = self._traj_idxs[i] <= 5
            if self.cfg.ACTION_TYPE == "joint":
                action, action_type, reached = self._traj[i].copy(), "joint", False
            else: # "cartesian"
                joint_action, action_type, reached = self._traj[i].copy(), "ego_cartesian", False
                world_to_ee = obs.env.panda.get_world_to_ee()
                base_to_ee = self.base_to_world@world_to_ee
                base_to_target_ee = self.robot.joint_to_cartesian(joint_action[:7]) # (4, 4)
                ee_to_target_ee = se3_inverse(base_to_ee)@base_to_target_ee
                action = np.concatenate([ee_to_target_ee[:3, 3], mat2euler(ee_to_target_ee[:3, :3]), np.array([0.04])])
        if self._target_grasp is not None:
            info["world_to_target_grasp"] = obs.env.objects.target_object.get_world_to_obj() @ self._target_grasp
        return action, action_type, reached, info

"""
DISPLAY="localhost:11.0" OMG_PLANNER_DIR=/share1/junyu/HRI/OMG-Planner python -m evaluate setup=s0 split=train use_ray=False policy=omg_planner omg_planner.wait_time=3. env.visualize=True env.verbose=True
scene_id 5, step 47, status 1, reward 1.0, reached frame 6900, done frame 9271

OMG_PLANNER_DIR=/share1/junyu/HRI/OMG-Planner CUDA_VISIBLE_DEVICES=0,1,4,6 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=32 env.panda.IK_solver=PyKDL setup=s0 split=train policy=omg_planner omg_planner.wait_time=3.
success rate: 500/720=0.6944444444444444
contact rate: 183/720=0.25416666666666665
   drop rate: 20/720=0.027777777777777776
timeout rate: 17/720=0.02361111111111111
average done frame        : 8277.027777777777
average success done frame: 8703.858
average success num steps : 42.522
average success           : 0.22954818376068375
evaluting uses 492.7538249492645 seconds

OMG_PLANNER_DIR=/share1/junyu/HRI/OMG-Planner CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=32 env.panda.IK_solver=PyKDL setup=s0 split=train policy=omg_planner omg_planner.wait_time=3. omg_planner.FILTER_HAND_COLLISION_GRASPS=True
success rate: 603/720=0.8375
contact rate: 45/720=0.0625
   drop rate: 35/720=0.04861111111111111
timeout rate: 37/720=0.05138888888888889
average done frame        : 8716.9875
average success done frame: 8674.021558872306
average success num steps : 42.3150912106136
average success           : 0.27875726495726494
evaluting uses 417.3503336906433 seconds

OMG_PLANNER_DIR=/share/haoran/HRI/OMG-Planner CUDA_VISIBLE_DEVICES=0,1,4,6 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=32 env.panda.IK_solver=PyKDL setup=s0 split=train policy=omg_planner omg_planner.wait_time=3.
"""