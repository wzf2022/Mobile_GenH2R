import os
import numpy as np
from scipy.spatial.transform import Rotation as Rt
from transforms3d.euler import euler2quat, euler2mat, mat2euler
from transforms3d.quaternions import quat2mat, mat2quat
import pybullet
import time
from yacs.config import CfgNode as CN

from env.handover_env import Observation
from env.hand import Hand
from utils.robot_kinematics import RobotKinematics
from .base_policy import BasePolicy
from utils.transform import ros_quat, tf_quat, se3_inverse

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

def se3_distance(T1, T2): # depend on trans_distance and rot_distance
    "T1, T2: (..., 4, 4)"
    return trans_distance(T1[..., :3, 3], T2[..., :3, 3]), rot_distance(T1[..., :3, :3], T2[..., :3, :3])

def get_rotation_to_align_vectors(cur, dest):
    cur, dest = cur / np.linalg.norm(cur), dest / np.linalg.norm(dest)
    rot_axis = np.cross(cur, dest)
    if np.linalg.norm(rot_axis) < 0.01:
        return np.zeros(3), 0
    rot_axis /= np.linalg.norm(rot_axis)
    total_rot = np.arccos(np.clip(np.dot(cur, dest), -1, 1))
    return rot_axis, total_rot

def get_rotation_around_axis(rot_axis, theta, center=None):
    transform = np.eye(4)
    rot_mat = Rt.from_rotvec(theta*rot_axis).as_matrix()
    transform[:3, :3] = rot_mat
    if center is not None:
        transform[:3, 3] = center - rot_mat @ center
    return transform

def compact2mat(pose_compact):
    "pose_compact: (7, ) [*trans, x, y, z, w]"
    pose_mat = np.eye(4)
    pose_mat[:3, :3] = Rt.from_quat(pose_compact[3:]).as_matrix()
    pose_mat[:3, 3] = pose_compact[:3]
    return pose_mat

def mat2compact(pose_mat):
    "pose_compact: (7, ) [*trans, x, y, z, w]"
    pose_compact = np.concatenate([pose_mat[:3, 3], Rt.from_matrix(pose_mat[:3, :3]).as_quat()])
    return pose_compact

def load_grasp(pose_mat, color=None):
    pose_compact = mat2compact(pose_mat)
    grasp_id = pybullet.loadURDF("/share/haoran/HRI/GA-DDPG/data/gripper_model/gripper_simplified.urdf", basePosition=pose_compact[:3], baseOrientation=pose_compact[3:], useMaximalCoordinates=True)
    pybullet.setCollisionFilterGroupMask(grasp_id, 0, 0, 0)
    if color is not None:
        pybullet.changeVisualShape(grasp_id, -1, rgbaColor=[*color, 1])
    return grasp_id

def regularize_pc_point_count(pc, npoints, np_random):
    """
    If point cloud pc has less points than npoints, it oversamples.
    Otherwise, it downsample the input pc to have npoint points.
    use_farthest_point: indicates whether to use farthest point sampling
    to downsample the points. Farthest point sampling version runs slower.
    """
    if pc.shape[0] > npoints:
        center_indexes = np_random.choice(
            range(pc.shape[0]), size=npoints, replace=False
        )
        pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np_random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc

class CartesianPlannerPolicy(BasePolicy):
    def __init__(self, root_cfg: CN):
        super().__init__(root_cfg)

        self._base_pose = np.eye(4)
        self._base_pose[:3, :3] = Rt.from_quat(np.array(self.root_cfg.env.panda.base_orientation)).as_matrix()
        self._base_pose[:3, 3] = np.array(self.root_cfg.env.panda.base_position)
        self.robot = RobotKinematics(root_cfg)
        self._start_conf = np.array(root_cfg.env.panda.initial_dof_position)
        cfg_cartesian = root_cfg.policy.cartesian
        self._trans_close_threshold = cfg_cartesian.trans_close_threshold
        self._rot_close_threshold = cfg_cartesian.rot_close_threshold
        self._rot_weight = cfg_cartesian.rot_weight
        self._trans_step_size = cfg_cartesian.trans_step_size
        self._rot_step_size = cfg_cartesian.rot_step_size
        self._step_size_eps_ratio = cfg_cartesian.step_size_eps_ratio
        self._show_grasps = cfg_cartesian.show_grasps
        self._standoff_dist = cfg_cartesian.standoff_dist
        self._staged = cfg_cartesian.staged
        self._stay_dist_max = cfg_cartesian.stay_dist_max
        self._stay_dist_ratio = cfg_cartesian.stay_dist_ratio
        self._foresee_frames = cfg_cartesian.foresee_frames
        self._load_collision_mask = cfg_cartesian.load_collision_mask
        self._grasp_dir = cfg_cartesian.grasp_dir
        self._show_filtering = cfg_cartesian.show_filtering
        self._grasp_dist_measure = cfg_cartesian.grasp_dist_measure
        self._change_target_thres = cfg_cartesian.change_target_thres
        self._stay_hinge_steps = cfg_cartesian.stay_hinge_steps
        self._check_object_stopped = cfg_cartesian.check_object_stopped
        self._use_stop_heuristics = cfg_cartesian.use_stop_heuristics
        self._only_filter_once = cfg_cartesian.only_filter_once
        self._target_grasp_for_current_obj_pose = cfg_cartesian.target_grasp_for_current_obj_pose
        self._always_towards_cur_target = cfg_cartesian.always_towards_cur_target
        self._landmarks_dir = cfg_cartesian.landmarks_dir
        self._verbose = cfg_cartesian.verbose

        self.default_cartesian_action = np.zeros(7)
        self.default_cartesian_action[6] = 0.04

    def reset(self):
        self.base_reset()
        self._target_joint = self._start_conf
        self._cur_action = np.zeros(6)
        # self._target_ef_pose = self.robot.convert_position_from_joint_to_cartesian(self._start_conf)
        self._last_plan_frame = -np.inf
        self._see_object = False
        self._target_grasp_pose = None
        self._target_grasp_idx = None
        self._target_standoff_grasp_pose = None
        self._all_obj_to_grasp_poses = None
        self._grasps_no_contact_mask = None
        self._all_obj_to_grasp_poses = None
        self._stage = 0
        self._show_grasps_cur_frame = False
        self._last_obj_pose = None
        self._landmark_scene_id = None

    def show_grasps(self, grasps):
        show_grasps_num = int(input(f"show grasps num (total {len(grasps)}):"))
        show_grasps_num = min(show_grasps_num, len(grasps))
        if self._verbose:
            print(f"showing {show_grasps_num} grasps out of {len(grasps)}")
        selected_indices = np.random.choice(len(grasps), show_grasps_num, replace=False)
        selected_grasps = grasps[selected_indices]

        for grasp in selected_grasps:
            grasp_pack_pose = pack_pose(grasp)
            grasp_id = pybullet.loadURDF("/share/haoran/HRI/GA-DDPG/data/gripper_model/gripper_simplified.urdf", basePosition=grasp_pack_pose[:3], baseOrientation=ros_quat(grasp_pack_pose[3:]), useMaximalCoordinates=True)
            pybullet.setCollisionFilterGroupMask(grasp_id, -1, 0, 0)
            pybullet.resetBasePositionAndOrientation(grasp_id, grasp_pack_pose[:3], ros_quat(grasp_pack_pose[3:]))
            pybullet.changeVisualShape(grasp_id, -1, rgbaColor=[*np.random.random(3), 1])
            self.grasp_ids.append(grasp_id)
        input("press to continue...")
    
    def clear_grasps(self):
        for grasp_id in self.grasp_ids:
            if self._verbose:
                print(f"removing {pybullet.getBodyInfo(grasp_id)}")
            pybullet.removeBody(grasp_id)
        self.grasp_ids = []

    def load_grasp_set(self, obj_name, scene_id):
        grasp_path = os.path.join(self._grasp_dir, f"{obj_name}.npy")
        if "s4g" in self._grasp_dir: # load_collision_mask must be False
            grasp_pose = np.load(grasp_path, allow_pickle=True).item()["gripper_grasp"]
            if self._verbose:
                print(f"load {grasp_pose.shape[0]} grasp poses")
            self._all_obj_to_grasp_poses = grasp_pose
            rot_mat = np.diag([-1, -1, 1, 1]) # rotate 180 around z
            self._all_obj_to_grasp_poses = np.concatenate([self._all_obj_to_grasp_poses, self._all_obj_to_grasp_poses @ rot_mat])
            assert self._load_collision_mask == False
        elif "acronym" in self._grasp_dir:
            grasp_pose = np.load(grasp_path)
            offset_pose = np.array(rotZ(np.pi / 2))  # and
            grasp_pose = np.matmul(grasp_pose, offset_pose)  # flip x, y
            self._all_obj_to_grasp_poses = grasp_pose
            rot_mat = np.diag([-1, -1, 1, 1]) # rotate 180 around z
            self._all_obj_to_grasp_poses = np.concatenate([self._all_obj_to_grasp_poses, self._all_obj_to_grasp_poses @ rot_mat])
            # /share/haoran/HRI/handover-sim/data/acronym/Truck_40597e0ca3a8eca9a1cba95c26616410_0.0004911644164552468.npy
            assert self._load_collision_mask == False
        else: # simulated
            simulator_grasp = np.load(grasp_path, allow_pickle=True, fix_imports=True, encoding="bytes")
            pose_grasp = simulator_grasp.item()[b"transforms"]
            offset_pose = np.array(rotZ(np.pi / 2))  # and
            pose_grasp = np.matmul(pose_grasp, offset_pose)  # flip x, y
            # print(f"load {pose_grasp.shape[0]} grasps")
            pose_grasp = ycb_special_case(pose_grasp, obj_name)
            # print(f"load {pose_grasp.shape[0]} grasps after ycb special case")
            self._all_obj_to_grasp_poses = pose_grasp
            rot_mat = np.diag([-1, -1, 1, 1]) # rotate 180 around z
            self._all_obj_to_grasp_poses = np.concatenate([self._all_obj_to_grasp_poses, self._all_obj_to_grasp_poses @ rot_mat])
            if self._load_collision_mask:
                self._grasps_no_contact_mask = np.load(os.path.join("grasps_no_contact_mask", f"{scene_id}.npy"))
                self._grasps_no_contact_mask = np.concatenate([self._grasps_no_contact_mask, self._grasps_no_contact_mask])
                self._all_obj_to_grasp_poses = self._all_obj_to_grasp_poses[self._grasps_no_contact_mask]

        # standoff_mat = np.eye(4) # pose_to_standoff_pose
        # standoff_mat[2, 3] = -self._standoff_dist
        # self._filtered_obj_to_standoff_grasp_poses = self._all_obj_to_grasp_poses @ standoff_mat

    def filter_grasps(self, grasps, hand: Hand, hand_pose=None):
        start_time = time.time()
        if hand_pose is None:
            if hand.body_id is None:
                return np.ones(grasps.shape[0], dtype=bool)
            hand_pose = hand.get_joint_positions()

        if self._show_filtering:
            cid = pybullet.connect(pybullet.GUI)
            pybullet.resetDebugVisualizerCamera(cameraDistance=2.4, cameraPitch=-58, cameraYaw=102, cameraTargetPosition=[0, 0, 0], physicsClientId=cid)
        else:
            cid = pybullet.connect(pybullet.DIRECT)
        hand_urdf_file = os.path.join("data", "assets", "hand", hand.name, "mano.urdf")
        hand_id = pybullet.loadURDF(hand_urdf_file, physicsClientId=cid)

        dof_indices = []
        for j in range(pybullet.getNumJoints(hand_id, physicsClientId=cid)):
            joint_info = pybullet.getJointInfo(hand_id, j, physicsClientId=cid)
            if joint_info[2] != pybullet.JOINT_FIXED:
                dof_indices.append(j)
        dof_indices = np.asanyarray(dof_indices, dtype=np.int64)
        for i, idx in enumerate(dof_indices):
            pybullet.resetJointState(hand_id, idx, hand_pose[i], physicsClientId=cid)

        if self._show_filtering:
            for i in range(52):
                pybullet.changeVisualShape(hand_id, i, rgbaColor=[0, 1, 0, 1], physicsClientId=cid)
        
        grasp_id = pybullet.loadURDF("/share/haoran/HRI/handover-sim/data/gripper_model/gripper_collision.urdf", physicsClientId=cid)

        mask = np.zeros(grasps.shape[0], dtype=bool)
        for i, grasp in enumerate(grasps):
            grasp_pack_pose = pack_pose(grasp)
            if self._show_filtering:
                pybullet.changeVisualShape(grasp_id, -1, rgbaColor=[0, 1, 0, 1], physicsClientId=cid)
            pybullet.resetBasePositionAndOrientation(grasp_id, grasp_pack_pose[:3], ros_quat(grasp_pack_pose[3:]), physicsClientId=cid)
            pybullet.performCollisionDetection(physicsClientId=cid)
            contact_points = pybullet.getContactPoints(hand_id, grasp_id, physicsClientId=cid)
            if len(contact_points)==0: 
                mask[i] = 1
            else:
                if self._show_filtering:
                    pybullet.changeVisualShape(grasp_id, -1, rgbaColor=[1, 0, 0, 1], physicsClientId=cid)
                pass
            if self._show_filtering:
                time.sleep(0.1)
        pybullet.disconnect(physicsClientId=cid)
        if self._verbose:
            print(f"filter grasps uses {time.time() - start_time:.2f} seconds, remaining {mask.sum()} grasps")
        return mask

    def check_close(self, T1, T2):
        trans_dist, rot_dist = se3_distance(T1, T2)
        return trans_dist <= self._trans_close_threshold and rot_dist <= self._rot_close_threshold

    def compute_distance(self, T1, T2):
        "T1: (..., 4, 4) T2: (..., 4, 4)"
        trans_dist, rot_dist = se3_distance(T1, T2)
        return trans_dist + self._rot_weight * rot_dist

    def get_current_grasp_poses(self, obs, obj_pose, hand_pose):
        if self._load_collision_mask:
            return obj_pose @ self._all_obj_to_grasp_poses, np.arange(self._all_obj_to_grasp_poses.shape[0])
        else:
            grasp_poses = obj_pose @ self._all_obj_to_grasp_poses
            if not self._only_filter_once or self._grasps_no_contact_mask is None:
                self._grasps_no_contact_mask = self.filter_grasps(grasp_poses, obs["env"].hand, hand_pose)
            return grasp_poses[self._grasps_no_contact_mask], np.where(self._grasps_no_contact_mask)[0]

    def convert_action_from_cartesian_to_joint(self, action, ef_pose, cur_joint, env): # reduce generation & test gap
        action_mat = np.eye(4)
        action_mat[:3, :3] = Rt.from_euler("xyz", action[3:]).as_matrix()
        action_mat[:3, 3] = action[:3]
        target_ef_pose = ef_pose@action_mat
        target_joint = self.robot.convert_position_from_cartesian_to_joint(target_ef_pose, cur_joint[:7])
        if target_joint is None:
            target_joint = env.egocentric_to_joint_action(action)
        return target_joint
    
    def plan_stages(self, ef_pose, target_grasp_pose, staged=False):
        trans_dist = trans_distance(ef_pose[:3, 3], target_grasp_pose[:3, 3])
        # stage 0: look towards target grasp
        start_ef_pose = ef_pose
        stage_0_rot_axis, stage_0_total_rot = get_rotation_to_align_vectors(ef_pose[:3, 2], target_grasp_pose[:3, 3] - ef_pose[:3, 3]) # cur, forward
        stage_0_step_rot = self._rot_step_size
        target_grasp_pose_stage_0 = get_rotation_around_axis(stage_0_rot_axis, stage_0_total_rot, ef_pose[:3, 3]) @ ef_pose
        # stage 1: move along the ball surfache
        stage_1_rot_axis, stage_1_total_rot = get_rotation_to_align_vectors(target_grasp_pose[:3, 3] - ef_pose[:3, 3], target_grasp_pose[:3, 2]) # forward, final forward
        stage_1_step_rot = min(self._trans_step_size / trans_dist, self._rot_step_size) # theta * r = length
        target_grasp_pose_stage_1 = get_rotation_around_axis(stage_1_rot_axis, stage_1_total_rot, target_grasp_pose[:3, 3]) @ target_grasp_pose_stage_0
        # stage 2
        stage_2_rot_axis, stage_2_total_rot = get_rotation_to_align_vectors(target_grasp_pose_stage_1[:3, 0], target_grasp_pose[:3, 0]) # current upward, target upward
        if np.sum(stage_2_rot_axis == 0) == 3 and np.dot(target_grasp_pose_stage_1[:3, 0], target_grasp_pose[:3, 0]) < 0:
            stage_2_rot_axis = target_grasp_pose[:3, 2] # z axis
            stage_2_total_rot = np.pi
        stage_2_rot_dir = 1 if np.dot(stage_2_rot_axis, target_grasp_pose[:3, 2]) > 0 else -1
        # print(f"stage 2 rot dir: {self._stage_2_rot_dir}")
        stage_2_step_rot = self._rot_step_size
        target_grasp_pose_stage_2 = get_rotation_around_axis(stage_2_rot_axis, stage_2_total_rot, target_grasp_pose_stage_1[:3, 3]) @ target_grasp_pose_stage_1
        # stage 3
        target_grasp_pose_stage_3 = target_grasp_pose
        stage_3_total_trans = trans_dist
        stage_3_step_trans = self._trans_step_size
        return start_ef_pose, \
            stage_0_rot_axis, stage_0_total_rot, stage_0_step_rot, target_grasp_pose_stage_0, \
            stage_1_rot_axis, stage_1_total_rot, stage_1_step_rot, target_grasp_pose_stage_1, \
            stage_2_rot_axis, stage_2_total_rot, stage_2_step_rot, target_grasp_pose_stage_2, stage_2_rot_dir, \
            stage_3_total_trans, stage_3_step_trans, target_grasp_pose_stage_3

    def get_nearest_grasp_pose(self, ef_pose, grasp_poses, grasp_idxs):
        "ef_pose: (4, 4) grasp_poses: (N, 4, 4)"
        if len(grasp_poses) == 0:
            if self._verbose:
                print("no available grasp poses")
            return None, None
        if self._grasp_dist_measure == "se3":
            dists = self.compute_distance(ef_pose, grasp_poses)
        elif self._grasp_dist_measure == "steps":
            dists = np.zeros(grasp_poses.shape[0])
            min_dist, target_grasp_pose, target_grasp_idx = np.inf, np.eye(4), -1
            for i, grasp_pose in enumerate(grasp_poses):
                start_ef_pose, stage_0_rot_axis, stage_0_total_rot, stage_0_step_rot, target_grasp_pose_stage_0, stage_1_rot_axis, stage_1_total_rot, stage_1_step_rot, target_grasp_pose_stage_1, stage_2_rot_axis, stage_2_total_rot, stage_2_step_rot, target_grasp_pose_stage_2, stage_2_rot_dir, stage_3_total_trans, stage_3_step_trans, target_grasp_pose_stage_3 = self.plan_stages(ef_pose, grasp_pose)
                total_steps = stage_0_total_rot / stage_0_step_rot + stage_1_total_rot / stage_1_step_rot + stage_2_total_rot / stage_2_step_rot + stage_3_total_trans / stage_3_step_trans
                dists[i] = total_steps
                if total_steps < min_dist:
                    min_dist, target_grasp_pose, target_grasp_idx = total_steps, grasp_pose, i
            
            # print(f"target_grasp_idx {target_grasp_idx}, {dists.argmin()} {min_dist} {dists.min()}")
            # assert target_grasp_idx == dists.argmin()
            # return target_grasp_pose, grasp_idxs[target_grasp_idx]
        nearest_idx = np.argmin(dists)
        # assert target_grasp_idx == nearest_idx
        # print(f"min dist: {dists.min()}")
        if self._target_grasp_idx is None or self._target_grasp_idx not in grasp_idxs:
            if self._verbose:
                print("use new target")
            return grasp_poses[nearest_idx], grasp_idxs[nearest_idx]
        cur_idx = grasp_idxs.tolist().index(self._target_grasp_idx)
        if dists[cur_idx] > dists[nearest_idx] + self._change_target_thres:
            if self._verbose:
                print("switch to new target")
            return grasp_poses[nearest_idx], grasp_idxs[nearest_idx]
        else:
            if self._verbose:
                print("use previous target")
            return grasp_poses[cur_idx], self._target_grasp_idx

    def termination_heuristics(self, point_state):
        if point_state.shape[1] == 0:
            return False
        cage_points_mask = (
            (point_state[2] > +0.06)
            & (point_state[2] < +0.11)
            & (point_state[1] > -0.05)
            & (point_state[1] < +0.05)
            & (point_state[0] > -0.02)
            & (point_state[0] < +0.02)
        )
        cage_points_mask_reg = regularize_pc_point_count(cage_points_mask[:, None], 1024, self.np_random)
        return np.sum(cage_points_mask_reg) > 50

    def check_done(self, obs: Observation):
        if self._target_grasp_pose is None: return False
        if self._use_stop_heuristics:
            return self.termination_heuristics(obs["get_visual_observation"]()[3][0].T)
        else:
            ef_pose = obs["env"].panda.get_world_to_ee()
            return self.check_close(ef_pose, self._target_grasp_pose)

    def get_target_grasp_pose(self, obs: Observation):
        if self._target_grasp_pose is None:
            return None
        if self._target_grasp_for_current_obj_pose:
            obj_pose = obs["env"].objects.target_object.get_world_to_obj()
            return obj_pose @ self._all_obj_to_grasp_poses[self._target_grasp_idx]
        else:
            return self._target_grasp_pose

    def get_next_landmark(self, scene_id, frame):
        if self._landmark_scene_id != scene_id:
            self._landmark_scene_id = scene_id
            data = np.load(os.path.join(self._landmarks_dir, f"{scene_id}.npz"))
            trans_error, rot_error = data["trans_error"], data["rot_error"]
            landmark_idxs = np.where(trans_error * 23 + rot_error * 4 > 1)[0] + 1
            landmark_idxs = np.concatenate([[0], landmark_idxs])
            self._landmarks = landmark_idxs * 130
        next_landmark_idx = np.searchsorted(self._landmarks, frame)
        if next_landmark_idx >= len(self._landmarks):
            return np.inf
        else:
            # if self._landmarks[next_landmark_idx] == frame:
            #     input("encounter a landmark")
            return self._landmarks[next_landmark_idx]
        # code.interact(local=dict(globals(), **locals()))

    def plan(self, obs: Observation): # depend on get_next_landmark, load_grasp_set, get_current_grasp_poses, get_nearest_grasp_pose, plan_stages
        if not self._see_object:
            object_points, hand_points = obs["get_visual_observation"]()[3]
            if object_points.shape[0] > 0:
                self._see_object = True

        info = {}
        if not self._see_object:
            action, action_type, reached = self.default_cartesian_action, "ego_cartesian", False
            return action, action_type, reached, info
        
        if self.check_done(obs): # target grasp pose is determined by last planning
            action, action_type, reached = None, None, True
            return action, action_type, reached, info
        
        if self._verbose:
            print(f"planning at frame {obs['frame']}")
        if self.root_cfg.env.visualize:
            object_points, hand_points = obs["get_visual_observation"]()[3] # to activate the ego camera
        self._last_plan_frame = obs["frame"]
        # preparation
        ef_pose = obs["env"].panda.get_world_to_ee()
        target_object = obs["env"].objects.target_object
        hand = obs["env"].hand
        if self._foresee_frames == 0:
            obj_pose = target_object.get_world_to_obj()
            if hand.body_id is None:
                hand_pose = np.zeros(51)
            else:
                hand_pose = hand.get_joint_positions()
        else:
            obj_pose = np.eye(4)
            foresee_frame = obs["frame"] + self._foresee_frames
            if self._landmarks_dir is not None:
                foresee_frame = min(foresee_frame, self.get_next_landmark(obs["env"].scene_id, obs["frame"]))
            foresee_frame_obj = min(foresee_frame, target_object.pose.shape[0] - 1)
            obj_pose[:3, 3] = target_object.pose[foresee_frame_obj, :3]
            obj_pose[:3, :3] = Rt.from_euler("XYZ", target_object.pose[foresee_frame_obj, 3:]).as_matrix()
            foresee_frame_hand = min(foresee_frame, hand.pose.shape[0] - 1)
            hand_pose = hand.pose[foresee_frame_hand]
        joint = obs["env"].panda.get_joint_positions()
        if self._verbose:
            print(f"joint_state {joint.sum()} {joint}")
        # ipdb.set_trace()
        if self._all_obj_to_grasp_poses is None:
            obj_name = target_object.name
            self.load_grasp_set(obj_name, obs["env"].scene_id)
        # if self._show_grasps:
        #     ef_id = load_grasp(ef_pose, color=[0, 0, 1])
        if self._staged:
            if self._target_grasp_pose is None:
                grasp_poses, grasp_idxs = self.get_current_grasp_poses(obs, obj_pose, hand_pose)
                self._target_grasp_pose, self._target_grasp_idx = self.get_nearest_grasp_pose(ef_pose, grasp_poses, grasp_idxs)
                if self._target_grasp_pose is None:
                    action, action_type, reached = self.default_cartesian_action, "ego_cartesian", False
                    return action, action_type, reached, info
                self._start_ef_pose, self._stage_0_rot_axis, self._stage_0_total_rot, self._stage_0_step_rot, self._target_grasp_pose_stage_0, self._stage_1_rot_axis, self._stage_1_total_rot, self._stage_1_step_rot, self._target_grasp_pose_stage_1, self._stage_2_rot_axis, self._stage_2_total_rot, self._stage_2_step_rot, self._target_grasp_pose_stage_2, self._stage_2_rot_dir, self._stage_3_total_trans, self._stage_3_step_trans, self._target_grasp_pose_stage_3 = self.plan_stages(ef_pose, self._target_grasp_pose, staged=True)
                self._stage_0_cur_rot, self._stage_1_cur_rot, self._stage_2_cur_rot, self._stage_3_cur_trans = 0, 0, 0, 0
                self._stage_steps = 0
                if self._show_grasps:
                    load_grasp(self._target_grasp_pose_stage_0, color=[0, 1, 0])
                    load_grasp(self._target_grasp_pose_stage_1, color=[0, 0, 1])
                    load_grasp(self._target_grasp_pose_stage_2, color=[1, 1, 0])
                    load_grasp(self._target_grasp_pose_stage_3, color=[1, 0, 1])
            target_ef_pose = np.eye(4)
            self._stage_steps += 1
            if self._stage == 0:
                if self._stage_0_cur_rot + self._stage_0_step_rot >= self._stage_0_total_rot:
                    self._stage_0_cur_rot = self._stage_0_total_rot
                    self._stage += 1
                    if self._verbose:
                        print(f"stage 0 steps {self._stage_steps}")
                    self._stage_steps = 0
                else:
                    self._stage_0_cur_rot += self._stage_0_step_rot
                target_ef_pose = get_rotation_around_axis(self._stage_0_rot_axis, self._stage_0_cur_rot, self._start_ef_pose[:3, 3]) @ self._start_ef_pose
                if self._show_grasps:
                    load_grasp(target_ef_pose, [0, 1, 0])
            elif self._stage == 1:
                if self._stage_1_cur_rot + self._stage_1_step_rot >= self._stage_1_total_rot:
                    self._stage_1_cur_rot = self._stage_1_total_rot
                    self._stage += 1
                    if self._verbose:
                        print(f"stage 1 steps {self._stage_steps}")
                    self._stage_steps = 0
                else:
                    self._stage_1_cur_rot += self._stage_1_step_rot
                target_ef_pose = get_rotation_around_axis(self._stage_1_rot_axis, self._stage_1_cur_rot, self._target_grasp_pose[:3, 3]) @ self._target_grasp_pose_stage_0
                if self._show_grasps:
                    load_grasp(target_ef_pose, [0, 0, 1])
            elif self._stage == 2:
                if self._stage_2_cur_rot + self._stage_2_step_rot >= self._stage_2_total_rot:
                    self._stage_2_cur_rot = self._stage_2_total_rot
                    self._stage += 1
                    if self._verbose:
                        print(f"stage 2 steps {self._stage_steps}")
                    self._stage_steps = 0
                else:
                    self._stage_2_cur_rot += self._stage_2_step_rot
                target_ef_pose = get_rotation_around_axis(self._stage_2_rot_axis, self._stage_2_cur_rot, self._target_grasp_pose_stage_1[:3, 3]) @ self._target_grasp_pose_stage_1
                if self._show_grasps:
                    load_grasp(target_ef_pose, [1, 1, 0])
            else:
                if self._stage_3_cur_trans + self._stage_3_step_trans >= self._stage_3_total_trans:
                    self._stage_3_cur_trans = self._stage_3_total_trans
                    if self._verbose:
                        print(f"stage 3 steps {self._stage_steps}")
                else:
                    self._stage_3_cur_trans += self._stage_3_step_trans
                # delta_ef_pose = np.eye(4)
                # delta_ef_pose[2, 3] = self._stage_3_cur_trans
                # target_ef_pose = self._target_grasp_pose_stage_2 @ delta_ef_pose
                forward = self._target_grasp_pose[:3, 3] - self._target_grasp_pose_stage_2[:3, 3]
                forward /= np.linalg.norm(forward)
                delta_ef_pose = np.eye(4)
                delta_ef_pose[:3, 3] = self._stage_3_cur_trans * forward
                target_ef_pose = delta_ef_pose @ self._target_grasp_pose_stage_2
                if self._show_grasps:
                    load_grasp(target_ef_pose, [1, 0, 1])
                # code.interact(local=dict(globals(), **locals()))
        else:
            if self._last_obj_pose is not None:
                obj_stopped = trans_distance(obj_pose[:3, 3], self._last_obj_pose[:3, 3]) < 0.01 and rot_distance(obj_pose[:3, :3], self._last_obj_pose[:3, :3]) < 0.01
            else:
                obj_stopped = False
            # if self._show_grasps:
            #     self._show_grasps_cur_frame = input("show grasps?")
            if not self._check_object_stopped or not obj_stopped or self._target_grasp_pose is None:
                grasp_poses, grasp_idxs = self.get_current_grasp_poses(obs, obj_pose, hand_pose)
                if self._show_grasps:
                    for grasp_pose in grasp_poses[np.random.choice(np.arange(grasp_poses.shape[0]), 20)]:
                        load_grasp(grasp_pose, np.random.random(3))
                self._target_grasp_pose, self._target_grasp_idx = self.get_nearest_grasp_pose(ef_pose, grasp_poses, grasp_idxs)
            if self._verbose:
                print(f"target_grasp_pose {self._target_grasp_pose}, target_grasp_idx {self._target_grasp_idx}")
            if self._target_grasp_pose is None:
                action, action_type, reached = self.default_cartesian_action, "ego_cartesian", False
                return action, action_type, reached, info
            if self._show_grasps:
                load_grasp(self._target_grasp_pose, [1, 0, 0])
            self._start_ef_pose, self._stage_0_rot_axis, self._stage_0_total_rot, self._stage_0_step_rot, self._target_grasp_pose_stage_0, self._stage_1_rot_axis, self._stage_1_total_rot, self._stage_1_step_rot, self._target_grasp_pose_stage_1, self._stage_2_rot_axis, self._stage_2_total_rot, self._stage_2_step_rot, self._target_grasp_pose_stage_2, self._stage_2_rot_dir, self._stage_3_total_trans, self._stage_3_step_trans, self._target_grasp_pose_stage_3 = self.plan_stages(ef_pose, self._target_grasp_pose)
            if self._verbose:
                print(f"stage 0 total rot {self._stage_0_total_rot}, step rot {self._stage_0_step_rot}")
                print(f"stage 1 total rot {self._stage_1_total_rot}, step rot {self._stage_1_step_rot}")
                print(f"stage 2 total rot {self._stage_2_total_rot}, step rot {self._stage_2_step_rot}")
                print(f"stage 3 total trans {self._stage_3_total_trans}, step trans {self._stage_3_step_trans}")

            target_ef_pose = ef_pose

            if self._always_towards_cur_target:
                cur_obj_pose = target_object.get_world_to_obj()
                cur_target_grasp_pose = cur_obj_pose @ self._all_obj_to_grasp_poses[self._target_grasp_idx]
                _, self._stage_0_rot_axis, self._stage_0_total_rot, self._stage_0_step_rot, _, _, _, _, _, _, _, _, _, _, _, _, _ = self.plan_stages(ef_pose, cur_target_grasp_pose)
            stage_0_rot = min(self._stage_0_step_rot, self._stage_0_total_rot)
            target_ef_pose = get_rotation_around_axis(self._stage_0_rot_axis, stage_0_rot, target_ef_pose[:3, 3]) @ target_ef_pose
            # if self._show_grasps_cur_frame:
            #     load_grasp(target_ef_pose, [0, 1, 0])
            
            stage_1_rot = min(self._stage_1_step_rot, self._stage_1_total_rot)
            target_ef_pose = get_rotation_around_axis(self._stage_1_rot_axis, stage_1_rot, self._target_grasp_pose[:3, 3]) @ target_ef_pose
            # if self._show_grasps_cur_frame:
            #     load_grasp(target_ef_pose, [0, 0, 1])
            
            stage_2_rot = min(self._stage_2_step_rot, self._stage_2_total_rot)
            target_ef_pose = get_rotation_around_axis(self._stage_2_rot_dir * target_ef_pose[:3, 2], stage_2_rot, target_ef_pose[:3, 3]) @ target_ef_pose
            # if self._show_grasps_cur_frame:
            #     load_grasp(target_ef_pose, [1, 1, 0])
                
            total_rot_steps = max(self._stage_0_total_rot / self._stage_0_step_rot - self._stay_hinge_steps, 0) + max(self._stage_1_total_rot / self._stage_1_step_rot - self._stay_hinge_steps, 0) + max(self._stage_2_total_rot / self._stage_2_step_rot - self._stay_hinge_steps, 0)
            if self._stage_3_total_trans < self._stay_dist_max and total_rot_steps * self._stay_dist_ratio > self._stage_3_total_trans and (not self._check_object_stopped or not obj_stopped):
                target_trans = min(self._stage_3_total_trans + self._trans_step_size, self._stay_dist_max, total_rot_steps * self._stay_dist_ratio)
                stage_3_trans = self._stage_3_total_trans - target_trans
            else:
                stage_3_trans = min(self._stage_3_step_trans, self._stage_3_total_trans)
            forward = self._target_grasp_pose[:3, 3] - target_ef_pose[:3, 3]
            forward /= np.linalg.norm(forward)
            delta_ef_pose = np.eye(4)
            delta_ef_pose[:3, 3] = stage_3_trans * forward
            target_ef_pose = delta_ef_pose @ target_ef_pose
            # if self._show_grasps_cur_frame:
            #     load_grasp(target_ef_pose, [1, 0, 1])
        
        self._last_obj_pose = obj_pose
        delta_ef_pose = se3_inverse(ef_pose) @ target_ef_pose
        cartesian_action = np.hstack([delta_ef_pose[:3, 3], Rt.from_matrix(delta_ef_pose[:3, :3]).as_euler("xyz")])
        action = np.append(cartesian_action, 0.04)
        action_type = "ego_cartesian"
        reached = False


        if self._target_grasp_idx is not None:
            info["world_to_target_grasp"] = obs["env"].objects.target_object.get_world_to_obj() @ self._all_obj_to_grasp_poses[self._target_grasp_idx]

        return action, action_type, reached, info
    
"""
conda activate genh2r
cd /share1/haoran/HRI/GenH2R
python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray False policy.name cartesian policy.wait_time 3. policy.cartesian.staged True policy.cartesian.verbose True env.visualize True
python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray False policy.name cartesian policy.wait_time 3. policy.cartesian.staged True policy.cartesian.check_object_stopped True policy.cartesian.verbose True env.visualize True
python -m evaluate evaluate.SCENE_IDS "[11]" evaluate.use_ray False policy.name cartesian policy.wait_time 3. env.visualize True policy.cartesian.staged True policy.cartesian.check_object_stopped True
"""