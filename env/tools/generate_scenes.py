import os
from typing import Tuple, Optional, Dict, List
import numpy as np
from numpy.typing import NDArray
import json
import bezier
from tqdm import tqdm
import ray
from omegaconf import OmegaConf, MISSING
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as Rt
import code

from ..utils.scene import scene_id_to_scene_dir, six_d_to_mat, mat_to_six_d

def make_1d_rotation_continuous(r: NDArray[np.float64]) -> NDArray[np.float64]:
    " r: (N, ) "
    " The difference between two adjacent elements should be less than pi "
    overflow_mask = r[1:]-r[:-1]>np.pi
    underflow_mask = r[1:]-r[:-1]<-np.pi
    delta = np.concatenate([[0.], -2*np.pi*overflow_mask+2*np.pi*underflow_mask]) 
    new_r = r+np.cumsum(delta)
    return new_r

def mano_trans_to_pybullet_trans(rot_mat: NDArray[np.float64], mano_trans: NDArray[np.float64], center: NDArray[np.float64]) -> NDArray[np.float64]:
    pybullet_trans = mano_trans-rot_mat@center+center
    return pybullet_trans

@dataclass
class GenerateSceneConfig:
    num_remotes: int = 32

    trajs_per_object_per_side: int = 64

    start_scene_id: int = 1000000
    scale: float = 0.2
    speed: float = 0.2
    rot_speed: float = 0.2

    max_frames: int = 13000

    init_trans_center: Tuple[float] = (0.6, 0.1, 0.2)
    init_trans_extent: Tuple[float] = (0.6, 0.2, 0.2)
    trans_center: Tuple[float] = (0.6, -0.1, 0.4)
    trans_extent: Tuple[float] = (1.0, 0.4, 0.6)

    start_idx: int = 0
    end_idx: Optional[int] = None

    skip_existing_scenes: bool = True

env_dir = os.path.dirname(os.path.dirname(__file__))
acronym_dir = os.path.join(env_dir, "data", "assets", "objects", "acronym")

@ray.remote(num_cpus=1)
class Distributer:
    def __init__(self, cfg: GenerateSceneConfig):
        self.cfg = cfg

        acronym_list_path = os.path.join(acronym_dir, "acronym_list.txt")
        with open(acronym_list_path, "r") as f:
            self.object_dir_list = [os.path.join("acronym", line.rstrip()) for line in f.readlines()]
        self.num_objects = len(self.object_dir_list)
        self.idx = cfg.start_idx

    def get_next_task(self) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        if self.idx == self.num_objects or (self.cfg.end_idx is not None and self.idx == self.cfg.end_idx):
            return None, None, None
        idx = self.idx
        print(f"enumerate to {idx}", flush=True)
        self.idx += 1
        return self.object_dir_list[idx], self.cfg.start_scene_id+idx, self.num_objects

@ray.remote(num_cpus=1)
class SceneGenerator:
    def __init__(self, cfg: GenerateSceneConfig, distributer: Distributer):
        self.cfg = cfg
        self.distributer = distributer

        env_dir = os.path.dirname(os.path.dirname(__file__))
        self.hand_dir = os.path.join(env_dir, "data", "assets", "hand")
        self.objects_dir = os.path.join(env_dir, "data", "assets", "objects")

        self.init_trans_center = np.array(cfg.init_trans_center)
        self.init_trans_extent = np.array(cfg.init_trans_extent)
        self.init_trans_min = self.init_trans_center-self.init_trans_extent/2
        self.init_trans_max = self.init_trans_center+self.init_trans_extent/2
        self.trans_center = np.array(cfg.trans_center)
        self.trans_extent = np.array(cfg.trans_extent)
        self.trans_min = self.trans_center-self.trans_extent/2
        self.trans_max = self.trans_center+self.trans_extent/2
        self.hand_rot_center: Dict[str, NDArray[np.float64]] = {
            "right": np.array([0.09566990661621094, 0.006383429050445557, 0.00618630313873291]),
            "left": np.array([-0.09566990661621094, 0.006383429050445557, 0.00618630313873291]),
        } # this is hand_joints[0] generated from manolayer when trans=0
        print(f"init trans min {self.init_trans_min} init trans max {self.init_trans_max}")
        print(f"trans min {self.trans_min} trans max {self.trans_max}")

    def get_init_trans(self) -> NDArray[np.float64]:
        return np.random.random(3)*(self.init_trans_max-self.init_trans_min)+self.init_trans_min

    def get_init_rot(self) -> NDArray[np.float64]:
        quat = np.random.randn(4)
        quat /= np.linalg.norm(quat)
        return Rt.from_quat(quat).as_matrix()

    def get_trans(self) -> NDArray[np.float64]:
        return np.random.random(3)*(self.trans_max-self.trans_min)+self.trans_min
    
    def translation_bezier(self, start: NDArray[np.float64], end: NDArray[np.float64]) -> Tuple[bezier.Curve, int, NDArray[np.float64]]:
        key_points = np.array([start, 0.5*start+0.5*end+self.cfg.scale*np.random.randn(3), end]).T
        curve = bezier.Curve.from_nodes(key_points)
        speed = np.random.uniform(self.cfg.speed/2, self.cfg.speed)
        num_frames = int(1000*curve.length/speed)
        rot_vec = np.random.randn(3)
        rot_vec = rot_vec/np.linalg.norm(rot_vec)*np.random.uniform(self.cfg.rot_speed/2, self.cfg.rot_speed)
        return curve, num_frames, rot_vec

    def generate_object_pose(self, segments: List[Tuple[bezier.Curve, int, NDArray[np.float64]]]) -> NDArray[np.float64]:
        object_pose_list: List[NDArray[np.float64]] = []
        rot_mat = self.get_init_rot()
        remaining_frames = self.cfg.max_frames
        for curve, num_frames, rot_vec in segments:
            rot_mat_per_frame = Rt.from_rotvec(rot_vec/1000).as_matrix()
            for k in range(min(num_frames, remaining_frames)):
                cur_rot_mat = rot_mat_per_frame @ rot_mat
                cur_rot_euler = Rt.from_matrix(cur_rot_mat).as_euler("XYZ")
                object_pose_list.append(np.concatenate([curve.evaluate(k/num_frames).reshape(-1), cur_rot_euler]))
                rot_mat = cur_rot_mat
            remaining_frames = max(remaining_frames-num_frames, 0)
        object_pose = np.stack(object_pose_list, axis=0) # (N, 6)
        for i in range(3, 6):
            object_pose[:, i] = make_1d_rotation_continuous(object_pose[:, i])
        return object_pose

    def generate_scene(self, scene_id: int, scene_path: str, mano_grasp: NDArray[np.float32], hand_side: str, object_dir: str) -> None:
        np.random.seed(scene_id)
        os.makedirs(os.path.dirname(scene_path), exist_ok=True)

        location = self.get_init_trans()
        segments: List[Tuple[bezier.Curve, int, NDArray[np.float64]]] = []
        total_frames = 0
        endpoints: List[int] = []
        while total_frames < self.cfg.max_frames:
            dest = self.get_trans()
            curve, num_frames, rot_vec = self.translation_bezier(location, dest)
            segments.append((curve, num_frames, rot_vec))
            total_frames += num_frames
            location = dest
            endpoints.append(total_frames)
            print(f"a curve requires {num_frames} frames, total frames {total_frames}")
        endpoints = np.array(endpoints)

        object_pose = self.generate_object_pose(segments) # generate object trajectory, (N, 6)
        world_to_object = six_d_to_mat(object_pose) # (N, 4, 4)
        object_to_hand = np.eye(4)
        object_to_hand[:3, :3] = Rt.from_rotvec(mano_grasp[3:6]).as_matrix()
        object_to_hand[:3, 3] = mano_trans_to_pybullet_trans(object_to_hand[:3, :3], mano_grasp[:3], self.hand_rot_center[hand_side])
        # code.interact(local=dict(globals(), **locals()))
        world_to_hand = world_to_object@object_to_hand # (N, 4, 4)
        hand_pose = mat_to_six_d(world_to_hand) # (N, 6)
        for i in range(3, 6):
            hand_pose[:, i] = make_1d_rotation_continuous(hand_pose[:, i])

        hand_path = os.path.join(f"hand_0_{hand_side}", "mano.urdf")

        scene_data = {
            "hand_name": "hand_0",
            "hand_side": hand_side,
            "hand_path": hand_path,
            "hand_pose": hand_pose.astype(np.float32),
            "hand_theta": mano_grasp[6:].astype(np.float32),
            "object_names": [object_dir],
            "object_paths": [os.path.join(object_dir, "model.urdf")],
            "object_grasp_id": 0,
            "object_poses": object_pose[None].astype(np.float32),
            "source": "genh2r",
            "endpoints": endpoints,
        }
        np.savez(scene_path, **scene_data)

    def work(self):
        while True:
            object_dir, start_scene_id, scene_id_interval = ray.get(self.distributer.get_next_task.remote())
            if object_dir is None: break
            mano_grasps: Dict[str, NDArray[np.float32]] = {
                "right": np.load(os.path.join(self.objects_dir, object_dir, f"mano_grasps_right.npy")),
                "left": np.load(os.path.join(self.objects_dir, object_dir, f"mano_grasps_left.npy")),
            }
            traj_idx = 0
            for mano_grasp_idx in range(self.cfg.trajs_per_object_per_side):
                for side in ["right", "left"]:
                    scene_id = start_scene_id+scene_id_interval*traj_idx
                    scene_path = os.path.join(scene_id_to_scene_dir(scene_id), f"{scene_id:08d}.npz")
                    if not os.path.exists(scene_path) or not self.cfg.skip_existing_scenes:
                        if os.path.exists(scene_path):
                            print(f"warning: overwriting scene {scene_id}")
                        self.generate_scene(scene_id, scene_path, mano_grasps[side][mano_grasp_idx], side, object_dir)
                    traj_idx += 1

def main():
    default_cfg = OmegaConf.structured(GenerateSceneConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg: GenerateSceneConfig = OmegaConf.to_object(OmegaConf.merge(default_cfg, cli_cfg))

    ray.init()
    distributer = Distributer.remote(cfg)
    generators = [SceneGenerator.remote(cfg, distributer) for _ in range(cfg.num_remotes)]
    ray.get([generator.work.remote() for generator in generators])

if __name__ == "__main__":
    main()

"""
python -m env.tools.generate_scenes start_idx=6000 skip_existing_scenes=False
"""