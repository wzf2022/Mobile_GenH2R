import os
import numpy as np
from numpy.typing import NDArray
from typing import List, Optional
import torch
from dataclasses import dataclass, field
from omegaconf import MISSING
import ray
from scipy.spatial.transform import Rotation as Rt
import code

from env.status_checker import EpisodeStatus
from policies.utils.point_cloud import PointCloudProcessor, PointCloudProcessorConfig
from policy_runner import DemoData, scene_id_to_demo_path, result_dtype
from env.utils.transform import se3_inverse, mat_to_pos_euler, mat_to_pos_ros_quat, to_ros_quat

def six_d_to_mat(six_d):
    " (..., 6) "
    shape_prefix = six_d.shape[:-1]
    mat = np.zeros(shape_prefix + (4, 4), dtype=six_d.dtype)
    mat[..., :3, 3] = six_d[..., :3]
    mat[..., :3, :3] = Rt.from_euler("XYZ", six_d[..., 3:].reshape(-1, 3)).as_matrix().reshape(shape_prefix + (3, 3))
    mat[..., 3, 3] = 1
    return mat

def mat_to_six_d(mat):
    shape_prefix = mat.shape[:-2]
    return np.concatenate([mat[..., :3, 3], Rt.from_matrix(mat[..., :3, :3].reshape(-1, 3, 3)).as_euler("XYZ").reshape(shape_prefix + (3, ))])

@dataclass
class DemoDataProcessorConfig:
    obj_pose_pred_frame_num: int = MISSING
    demo_source: str = "genh2r"
    pc: PointCloudProcessorConfig = field(default_factory=PointCloudProcessorConfig)

class DemoDataProcessor:
    def __init__(self, cfg: DemoDataProcessorConfig):
        self.cfg = cfg
        self.point_cloud_processor = PointCloudProcessor(cfg.pc)

        if self.cfg.demo_source == "handover-sim":
            self.world_to_base = np.eye(4)
            self.world_to_base[:3, :3] = Rt.from_quat(np.array((0.0, 0.0, 0.7071068, 0.7071068))).as_matrix()
            self.world_to_base[:3, 3] = np.array((0.61, -0.50, 0.875))
    
    def process(self, demo_data_path: str):
        self.point_cloud_processor.reset()
        processed_demo_data = []
        if self.cfg.demo_source == "genh2r":
            demo_data: DemoData = np.load(demo_data_path)
            num_steps = demo_data["num_steps"]
            assert demo_data["status"] == EpisodeStatus.SUCCESS
            actions = demo_data["action"][:, :6] # original is (num_steps, 7) with 0.04
            world_to_ees = demo_data["world_to_ee"]
            world_to_objects = demo_data["world_to_object"]
            world_to_target_grasps = demo_data["world_to_target_grasp"]
            for step in range(num_steps):
                object_points, hand_points = demo_data[f"object_points_{step}"], demo_data[f"hand_points_{step}"]
                ee_to_world = se3_inverse(world_to_ees[step])
                ee_to_object = ee_to_world@world_to_objects[step]
                object_to_ee = se3_inverse(ee_to_object)

                input_points = self.point_cloud_processor.process(object_points, hand_points, world_to_ees[step])
                if input_points is not None:
                    object_pred_pose = np.zeros((self.cfg.obj_pose_pred_frame_num, 6))

                    for j in range(step+1, min(step+1+self.cfg.obj_pose_pred_frame_num, num_steps)):
                        nxt_ee_to_object = ee_to_world@world_to_objects[j]
                        object_pred_pose[j-(step+1)] = mat_to_six_d(nxt_ee_to_object@object_to_ee)

                    if not (True in (np.isnan(world_to_target_grasps[step]))):
                        ee_to_target_grasp = se3_inverse(world_to_ees[step])@world_to_target_grasps[step]
                        pos, ros_quat = mat_to_pos_ros_quat(ee_to_target_grasp)
                        processed_demo_data.append((input_points, actions[step], np.concatenate([pos, ros_quat]), object_pred_pose.reshape(-1)))
        elif self.cfg.demo_source == "handover-sim":
            traj_data = np.load(demo_data_path)
            assert traj_data['status'] == EpisodeStatus.SUCCESS
            num_steps = traj_data['num_steps']
            expert_actions = traj_data['expert_action']
            goal_poses = traj_data['goal_pose']
            goal_masks = traj_data['goal_mask']
            joint_states = traj_data['joint_state']
            ef_poses = traj_data['ef_pose'] if 'ef_pose' in traj_data.keys() else None
            expert_mask = traj_data['expert_mask'] if 'expert_mask' in traj_data.keys() else None
            real_obj_pose = traj_data['real_obj_pose'] if 'real_obj_pose' in traj_data.keys() else None    #(steps, 6)
            for step in range(num_steps):
                ef_to_world = se3_inverse(self.world_to_base @ ef_poses[step])
                cur_pose_matrix = ef_to_world @ six_d_to_mat(real_obj_pose[step])
                inv_cur_pose_matrix = se3_inverse(cur_pose_matrix)

                object_pred_pose = np.zeros((self.cfg.obj_pose_pred_frame_num, 6))
                for j in range(step + 1, min(step + 1 + self.cfg.obj_pose_pred_frame_num, num_steps)):
                    nxt_pose_matrix = ef_to_world @ six_d_to_mat(real_obj_pose[j])
                    object_pred_pose[j - (step + 1)] = mat_to_six_d(nxt_pose_matrix @ inv_cur_pose_matrix)
                ef_pose = ef_poses[step]
                input_points = self.point_cloud_processor.process(traj_data[f'object_points_{step}'], traj_data[f'hand_points_{step}'], ef_pose)
                done = step == num_steps - 1
                goal_mask = goal_masks[step]
                if goal_mask:
                    goal_pose = goal_poses[step] # quat, pos
                    goal_pose = np.concatenate([goal_pose[4:], to_ros_quat(goal_pose[:4])]) # pos, quat
                else:
                    goal_pose = np.nan*np.ones(7)
                if expert_mask is None:
                    expert_flag = np.sum(np.abs(expert_actions[step]) > 0.001) > 0
                else:
                    expert_flag = expert_mask[step]
                assert expert_flag == goal_mask
                if input_points is not None and goal_mask:
                    processed_demo_data.append((input_points, expert_actions[step], goal_pose, object_pred_pose.reshape(-1)))
        else:
            raise ValueError(f"unknown demo source {self.cfg.demo_source}")

        return processed_demo_data
    
@ray.remote(num_cpus=1)
class DemoDataProcessorRemote(DemoDataProcessor):
    pass

@dataclass
class DataProviderConfig:
    demo_dir: str = MISSING
    demo_structure: str = "hierarchical" # "flat"
    demo_source: str = "genh2r" # "handover-sim"
    seed: int = MISSING
    batch_size: int = 256
    obj_pose_pred_frame_num: int = MISSING
    start_scene_id: Optional[int] = None
    end_scene_id: Optional[int] = None

    processor: DemoDataProcessorConfig = field(default_factory=lambda: DemoDataProcessorConfig(obj_pose_pred_frame_num="${..obj_pose_pred_frame_num}", demo_source="${..demo_source}"))
    num_processors: int = 20
    cache_all_data: bool = False
    buffer_size: int = 65536

class DataProvider:
    def __init__(self, cfg: DataProviderConfig):
        self.cfg = cfg
        ray.init()
        self.demo_data_processors = [DemoDataProcessorRemote.remote(cfg.processor) for _ in range(cfg.num_processors)]
        self.np_random = np.random.RandomState(cfg.seed)

        assert self.cfg.demo_dir is not None
        self.demo_results_list: List[NDArray[result_type]] = []
        self.demo_paths, self.demo_data_num = [], 0
        self.get_demo_paths(self.cfg.demo_dir)
        self.demo_paths.sort()
        self.remained_data = []
        assert cfg.buffer_size % cfg.batch_size == 0
        self.buffer = []
        self.process_calls = None
        self.reset()

        if self.cfg.cache_all_data:
            while self.demo_path_idx < len(self.demo_paths):
                process_calls = self.generate_process_calls()
                self.get_process_calls(process_calls)

        if self.cfg.demo_source == "genh2r":
            print(f"demo data number: {self.demo_data_num}", flush=True)
            results = np.concatenate(self.demo_results_list)
            num_scenes = results.shape[0]
            success_cnt = (results["status"] == EpisodeStatus.SUCCESS).sum()
            contact_cnt = (results["status"] == EpisodeStatus.FAILURE_HUMAN_CONTACT).sum()
            drop_cnt    = (results["status"] == EpisodeStatus.FAILURE_OBJECT_DROP).sum()
            timeout_cnt = (results["status"] == EpisodeStatus.FAILURE_TIMEOUT).sum()
            print(f"success rate: {success_cnt}/{num_scenes}={success_cnt/num_scenes}")
            print(f"contact rate: {contact_cnt}/{num_scenes}={contact_cnt/num_scenes}")
            print(f"   drop rate: {drop_cnt}/{num_scenes}={drop_cnt/num_scenes}")
            print(f"timeout rate: {timeout_cnt}/{num_scenes}={timeout_cnt/num_scenes}")
            success_mask = results["status"] == EpisodeStatus.SUCCESS
            # average_done_frame = results["done_frame"].mean()
            average_success_done_frame = results["done_frame"][success_mask].mean()
            average_success_reached_frame = results["reached_frame"][success_mask].mean()
            average_success_num_steps = results["num_steps"][success_mask].mean()
            print(f"average success done frame   : {average_success_done_frame}")
            print(f"average success reached frame: {average_success_reached_frame}")
            print(f"average success num steps    : {average_success_num_steps}")
    
    def get_demo_paths(self, demo_dir):
        if self.cfg.demo_source == "handover-sim" and os.path.exists(os.path.join(demo_dir, "success.npy")):
            success_scene_ids = np.load(os.path.join(demo_dir, "success.npy"))
            for scene_id in success_scene_ids:
                self.demo_paths.append(os.path.join(demo_dir, f"{scene_id}.npz"))
        elif self.cfg.demo_source == "genh2r" and any(filename.startswith("results") for filename in os.listdir(demo_dir)):
            for filename in os.listdir(demo_dir):
                if filename.startswith("results"):
                    demo_results: NDArray[result_dtype] = np.load(os.path.join(demo_dir, filename))
                    self.demo_results_list.append(demo_results)
                    for result in demo_results:
                        scene_id = result["scene_id"].item()
                        if result["status"].item() == EpisodeStatus.SUCCESS and (self.cfg.start_scene_id is None or scene_id >= self.cfg.start_scene_id) and (self.cfg.end_scene_id is None or scene_id < self.cfg.end_scene_id):
                            self.demo_paths.append(os.path.join(demo_dir, scene_id_to_demo_path(scene_id, self.cfg.demo_structure)))
                            self.demo_data_num += result["num_steps"].item()
        else:
            for file_name in os.listdir(demo_dir):
                file_path = os.path.join(demo_dir, file_name)
                if os.path.isdir(file_path):
                    self.get_demo_paths(file_path)
    
    def reset(self):
        self.np_random.shuffle(self.demo_paths)
        self.demo_path_idx = 0

    def generate_process_calls(self):
        process_calls = []
        for i in range(self.cfg.num_processors):
            if self.demo_path_idx == len(self.demo_paths):
                if self.cfg.cache_all_data:
                    break
                else:
                    self.reset()
            process_calls.append(self.demo_data_processors[i].process.remote(self.demo_paths[self.demo_path_idx]))
            self.demo_path_idx += 1
        return process_calls

    def get_process_calls(self, process_calls: List):
        processed_demo_data = ray.get(process_calls)
        for i in range(len(processed_demo_data)):
            self.remained_data.extend(processed_demo_data[i])

    def get_batch_data(self):
        if self.cfg.cache_all_data:
            batch_data_idxs = self.np_random.choice(len(self.remained_data), size=self.cfg.batch_size)
            batch_data_list = []
            for idx in batch_data_idxs:
                batch_data_list.append(self.remained_data[idx])
        else:
            # process previously generated background calls
            if self.process_calls is not None:
                ready_ids, remaining_ids = ray.wait(self.process_calls, timeout=0)
                if len(remaining_ids) == 0:
                    self.get_process_calls(self.process_calls)
                    self.process_calls = None

            while len(self.buffer) == 0 and len(self.remained_data) < self.cfg.buffer_size:
                # must first process previously generated background calls
                if self.process_calls is not None:
                    self.get_process_calls(self.process_calls)
                    self.process_calls = None
                else:
                    process_calls = self.generate_process_calls()
                    self.get_process_calls(process_calls)
            
            # generate background calls
            if self.process_calls is None and len(self.remained_data) < self.cfg.buffer_size:
                self.process_calls = self.generate_process_calls()

            if len(self.buffer) == 0:
                self.buffer = self.remained_data[:self.cfg.buffer_size]
                self.remained_data = self.remained_data[self.cfg.buffer_size:]
                self.np_random.shuffle(self.buffer)

            batch_data_list = self.buffer[:self.cfg.batch_size]
            self.buffer = self.buffer[self.cfg.batch_size:]
        
        # process batch data
        point_clouds, expert_actions, grasp_poses, object_pred_poses = [], [], [], []
        for point_cloud, expert_action, grasp_pose, object_pred_pose in batch_data_list:
            point_clouds.append(point_cloud)
            expert_actions.append(expert_action)
            grasp_poses.append(grasp_pose)
            object_pred_poses.append(object_pred_pose)

        point_clouds, expert_actions = torch.tensor(np.stack(point_clouds, axis=0), dtype=torch.float32), torch.tensor(np.stack(expert_actions, axis=0), dtype=torch.float32)
        grasp_poses, object_pred_poses = torch.tensor(np.stack(grasp_poses, axis=0), dtype=torch.float32), torch.tensor(np.stack(object_pred_poses, axis=0), dtype=torch.float32)
        # print(point_clouds.shape, expert_actions.shape, grasp_pose.shape, object_pred_pose.shape)
        # torch.Size([256, 1024, 14]) torch.Size([256, 6]) torch.Size([256, 2, 3]) torch.Size([256, 1, 6])
        batch_data = {
            "point_clouds": point_clouds,
            "expert_actions": expert_actions,
            "grasp_poses": grasp_poses,
            "object_pred_poses": object_pred_poses
        }
        return batch_data

def debug():
    from omegaconf import OmegaConf
    default_cfg = OmegaConf.structured(DataProviderConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.to_object(OmegaConf.merge(default_cfg, cli_cfg))
    data_provider = DataProvider(cfg)
    while True:
        batch_data = data_provider.get_batch_data()
        point_clouds = batch_data["point_clouds"]
        expert_actions = batch_data["expert_actions"]
        print(point_clouds.shape, expert_actions.shape, point_clouds.sum(), expert_actions.sum())
    # torch.Size([256, 1024, 5]) torch.Size([256, 6]) tensor(408378.4688) tensor(-318.6246)
    # torch.Size([256, 1024, 5]) torch.Size([256, 6]) tensor(372342.4688) tensor(-268.7459)
    # torch.Size([256, 1024, 5]) torch.Size([256, 6]) tensor(399093.6250) tensor(-248.0359)
    code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    debug()

"""
OMG_PLANNER_DIR=/share1/junyu/HRI/OMG-Planner CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=32 env.panda.IK_solver=PyKDL setup=s0 split=train policy=omg_planner omg_planner.wait_time=3. demo_dir=data/demo/s0/train/omg_sequential

python -m train.data_provider demo_dir=data/demo/s0/train/chomp/landmark_planning_dart seed=0 obj_pose_pred_frame_num=0
"""