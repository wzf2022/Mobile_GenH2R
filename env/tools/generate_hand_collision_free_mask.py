import os
import numpy as np
from numpy.typing import NDArray
import torch
import ray
from omegaconf import OmegaConf, MISSING
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

from ..utils.sdf_loss import SDFData
from ..utils.hand_collision_filter import HandCollisionFilterConfig, HandCollisionFilter
from ..utils.scene import scene_id_to_hierarchical_dir, load_scene_data, six_d_to_mat, mat_to_six_d
import code

env_dir = os.path.dirname(os.path.dirname(__file__))
objects_root_dir = os.path.join(env_dir, "data", "assets", "objects")
acronym_dir = os.path.join(objects_root_dir, "acronym")
hands_root_dir = os.path.join(env_dir, "data", "assets", "hand")

@dataclass
class GenerateHandCollisionFreeMaskConfig:
    device: str = "cuda"
    num_runners: int = 32

    start_scene_id: int = 1000000
    trajs_per_object: int = 128

    start_object_idx: int = 0
    end_object_idx: Optional[int] = None

    augment_grasps: bool = True
    frame: int = 0

    hand_collision_filter: HandCollisionFilterConfig = field(default_factory=lambda: HandCollisionFilterConfig(device="${..device}"))

    result_root_dir: str = os.path.join(env_dir, "data", "mobile_hand_collision_free_mask", "augment_True_threshold_0_use_bbox")

@ray.remote(num_cpus=1)
class Distributer:
    def __init__(self, cfg: GenerateHandCollisionFreeMaskConfig):
        self.cfg = cfg

        acronym_list_path = os.path.join(acronym_dir, "acronym_list.txt")
        with open(acronym_list_path, "r") as f:
            self.object_dir_list = [os.path.join("acronym", line.rstrip()) for line in f.readlines()]
        self.num_objects = len(self.object_dir_list)
        self.idx = cfg.start_object_idx

    def get_next_task(self) -> Tuple[Optional[str], Optional[List[int]]]:
        if self.idx == self.num_objects or (self.cfg.end_object_idx is not None and self.idx == self.cfg.end_object_idx):
            return None, None
        idx = self.idx
        print(f"enumerate to {idx}", flush=True)
        self.idx += 1
        return self.object_dir_list[idx], [self.cfg.start_scene_id+idx+i*self.num_objects for i in range(self.cfg.trajs_per_object)]

class HandCollisionFreeMaskGenerator:
    def __init__(self, cfg: GenerateHandCollisionFreeMaskConfig, distributer: Distributer):
        self.cfg = cfg
        self.distributer = distributer

        self.grasp_augmentation_matrix: NDArray[np.float64] = np.diag(np.array([-1., -1., 1., 1.], dtype=float))

        self.device = torch.device(cfg.device)
        self.hand_collision_filter = HandCollisionFilter(cfg.hand_collision_filter)

    def generate_hand_collision_free_mask(self, scene_id: int, object_grasps: NDArray[np.float64]):
        scene_data = load_scene_data(scene_id)
        world_to_object = six_d_to_mat(scene_data["object_poses"][scene_data["object_grasp_id"], self.cfg.frame])
        world_to_grasps = world_to_object@object_grasps

        code.interact(local=dict(globals(), **locals()))
        hand_collision_free_mask = self.hand_collision_filter.filter_hand_collision(os.path.join(hands_root_dir, scene_data["hand_path"]), scene_data["hand_pose"][self.cfg.frame].astype(np.float64), world_to_grasps)
        save_dir = os.path.join(self.cfg.result_root_dir, scene_id_to_hierarchical_dir(scene_id))
        save_path = os.path.join(save_dir, f"{scene_id:08d}.npy")
        os.makedirs(save_dir, exist_ok=True)
        np.save(save_path, hand_collision_free_mask)

    def work(self):
        while True:
            object_dir, scene_ids = ray.get(self.distributer.get_next_task.remote())
            if object_dir is None: break
            target_object_grasps: NDArray[np.float64] = np.load(os.path.join(objects_root_dir, object_dir, "grasps.npy"))
            if self.cfg.augment_grasps:
                target_object_grasps: NDArray[np.float64] = np.concatenate([target_object_grasps, target_object_grasps@self.grasp_augmentation_matrix])
            for scene_id in scene_ids:
                print(f"generate for scene {scene_id}")
                self.generate_hand_collision_free_mask(scene_id, target_object_grasps)

def main():
    default_cfg = OmegaConf.structured(GenerateHandCollisionFreeMaskConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg: GenerateHandCollisionFreeMaskConfig = OmegaConf.to_object(OmegaConf.merge(default_cfg, cli_cfg))
    num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    num_runners_per_gpu = (cfg.num_runners-1)//num_gpus+1

    @ray.remote(num_cpus=1, num_gpus=1/num_runners_per_gpu)
    class HandCollisionFreeMaskGeneratorRemote(HandCollisionFreeMaskGenerator):
        pass

    ray.init()
    distributer = Distributer.remote(cfg)
    generators = [HandCollisionFreeMaskGeneratorRemote.remote(cfg, distributer) for _ in range(cfg.num_runners)]
    ray.get([generator.work.remote() for generator in generators])

if __name__ == "__main__":
    main()

"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m env.tools.generate_hand_collision_free_mask start_object_idx=6000 num_runners=64 device=cpu
"""