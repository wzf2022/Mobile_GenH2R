import os
import ray
from typing import List, Optional
import numpy as np
from omegaconf import OmegaConf
from dataclasses import dataclass
import code

from ..third_party.DexGraspNet.generate import generate

@dataclass
class GenerateMANOGraspsConfig:
    object_start_idx: int = 0
    num_objects: Optional[int] = None
    num_objects_per_batch: int = 100

@ray.remote(num_cpus=1)
class Distributer:
    def __init__(self, cfg: GenerateMANOGraspsConfig):
        self.cfg = cfg
        env_dir = os.path.dirname(os.path.dirname(__file__))
        acronym_dir = os.path.join(env_dir, "data", "assets", "objects", "acronym")
        acronym_list_path = os.path.join(acronym_dir, "acronym_list.txt")
        with open(acronym_list_path, "r") as f:
            self.object_path_list = [os.path.join(acronym_dir, line.rstrip(), "coacd_decomposed.obj") for line in f.readlines()]
        self.object_path_list = self.object_path_list[cfg.object_start_idx:]
        if cfg.num_objects is not None:
            self.object_path_list = self.object_path_list[:cfg.num_objects]
        self.idx = 0

    def get_next_task(self) -> Optional[List[str]]:
        if self.idx >= len(self.object_path_list):
            return None
        object_path_list = self.object_path_list[self.idx:self.idx+self.cfg.num_objects_per_batch]
        self.idx += self.cfg.num_objects_per_batch
        return object_path_list

@ray.remote(num_cpus=10, num_gpus=1)
class Generator:
    def __init__(self, distributer: Distributer):
        self.distributer = distributer
    
    def work(self):
        while True:
            object_path_list = ray.get(self.distributer.get_next_task.remote())
            if object_path_list is None: break
            # print(f"generate mano grasps for {object_path_list}")
            right_hand_pose_list = generate(object_path_list, side="right")
            for object_path, right_hand_pose in zip(object_path_list, right_hand_pose_list):
                np.save(os.path.join(os.path.dirname(object_path), "mano_grasps_right.npy"), right_hand_pose)
            left_hand_pose_list = generate(object_path_list, side="left")
            for object_path, left_hand_pose in zip(object_path_list, left_hand_pose_list):
                np.save(os.path.join(os.path.dirname(object_path), "mano_grasps_left.npy"), left_hand_pose)
            # flipped_left_hand_pose_list = generate(object_path_list, side="left")
            # for object_path, flipped_left_hand_pose in zip(object_path_list, flipped_left_hand_pose_list):
            #     np.save(os.path.join(os.path.dirname(object_path), "mano_grasps_left_flipped.npy"), flipped_left_hand_pose)
            #     left_hand_pose = np.zeros_like(flipped_left_hand_pose)
            #     left_hand_pose[:, 6:] = flipped_left_hand_pose[:, 6:]*np.tile(np.array([1., -1., -1.]), 15)
            #     left_hand_pose[:, :3] = flipped_left_hand_pose[:, :3]*np.array([-1., 1., 1.])
            #     flipped_left_hand_rot_mat = Rt.from_rotvec(flipped_left_hand_pose[:, 3:6]).as_matrix()
            #     left_hand_rot_mat = flipped_left_hand_rot_mat*np.array([
            #         [ 1., -1., -1.],
            #         [-1.,  1.,  1.],
            #         [-1.,  1.,  1.]
            #     ])
            #     left_hand_pose[:, 3:6] = Rt.from_matrix(left_hand_rot_mat).as_rotvec()
            #     np.save(os.path.join(os.path.dirname(object_path), "mano_grasps_left.npy"), left_hand_pose)

def main():
    default_cfg = OmegaConf.structured(GenerateMANOGraspsConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg: GenerateMANOGraspsConfig = OmegaConf.to_object(OmegaConf.merge(default_cfg, cli_cfg))

    ray.init()
    distributer = Distributer.remote(cfg)
    num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    generators = [Generator.remote(distributer) for _ in range(num_gpus)] # 7 gpus
    ray.get([generator.work.remote() for generator in generators])

if __name__ == "__main__":
    main()

"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 python -m env.tools.generate_mano_grasps num_objects_per_batch=10
CUDA_VISIBLE_DEVICES=7 python -m env.tools.generate_mano_grasps num_objects_per_batch=10 num_objects=10 object_start_idx=50
"""

"""
(Generator pid=2082993) failed to process ['/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Chair/a09091780fcf3af2e9777a9dc292bbd2_0.0035264367437090466/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Chair/a14880ecec87a82bf9b9977a2406713a_0.006825321509110643/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Chair/a40e59ced8cf428a73a499fae25a1662_0.009511965320705815/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Chair/a6507d76e88e9d68c28a12683f5d3629_0.0019904626574298633/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Chair/a6bdde9da1322bd5116162acefe23592_0.0037115156805862027/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Chair/a6bdde9da1322bd5116162acefe23592_0.004160942451233466/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Chair/a724a8b2ea1ef63954a8e7cfdf35a7ed_0.006184090596276237/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Chair/a727426ed5a15fb0f7628281ecb18112_0.005373694647282868/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Chair/a8ba2f3d08993ddadbed71b890089f90_0.006874773697657984/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Chair/aa93247a7fa69991e074517a246f1e65_0.007378442947768858/coacd_decomposed.obj']
(Generator pid=2082995) failed to process ['/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/ChestOfDrawers/10f1b51f2207766ff11c3739edd52fa3_0.010587999127061765/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/ChestOfDrawers/152a850e5016fdd6e0771d4c079a0ec2_0.0029839043673442916/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/ChestOfDrawers/15e651b8b7a0c880ac13edc49a7166b9_0.012077822430538069/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/ChestOfDrawers/165c00d292773865ae6eaa32356aaf3b_0.002968165467274482/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/ChestOfDrawers/188daa867afa5eea43638dabe1eb5336_0.0060834938944589355/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/ChestOfDrawers/18d0a9703a1151475f4d9bb2d20be1fe_0.002572741857547399/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/ChestOfDrawers/18d84c10a5068315fe5c8ffd0f5eba47_0.006154941570634335/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/ChestOfDrawers/18d84c10a5068315fe5c8ffd0f5eba47_0.007413474952142491/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/ChestOfDrawers/1aaba7b87e5d53f05b8768d6ebff04cd_0.0053098025395948295/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/ChestOfDrawers/1aada3ab6bb4295635469b95109803c_0.005653658494948007/coacd_decomposed.obj']
(Generator pid=2082992) failed to process ['/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Desk/cdde9cb1f21676d12dd8e5d33382e245_0.004762778188139255/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Desk/cf07c06034a7b361e3f7a74e12a274ef_9.641563822352824e-07/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Desk/cffb55a559fa4f0ead8e5e47362bc281_0.003073883588892868/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Desk/d0b38b27495542461b02cde7e81f0fc3_0.0020447265960072294/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Desk/d0b38b27495542461b02cde7e81f0fc3_0.005194269991427889/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Desk/d2682afc21ae32584c51ad37c13d25a8_0.0015594301440103723/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Desk/d2b15ccec200fb004648b8db22f91655_0.003966635444003551/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Desk/d2b7b6ecf910ec6e60bd00cf60d56a5_0.0016671079030104092/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Desk/d3196bc71d782c3a39d9bff0bc23404e_0.002777071296876148/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Desk/d56f9f6d63b1fd0d812e3ecbeaae3980_0.0027863588473455883/coacd_decomposed.obj']
(Generator pid=2082995) failed to process ['/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Fireplace/83c33ab5e108f688657bf7ec906f4721_0.0004775916210184593/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Fireplace/b13dff579dd97f6168b24217426b180f_0.0022510213267168647/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Fireplace/b15402eb7b7843e2e3f7a74e12a274ef_0.003704614734882639/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Fireplace/b1bce627680b80b6e3f7a74e12a274ef_0.0010566871585161396/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Fireplace/b1bce627680b80b6e3f7a74e12a274ef_0.0031949826353021666/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Fireplace/b8c9733c8fe3f23cd201085e80edb26a_0.0012838227980578542/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Fireplace/bec1921aace45bdecbc6ff5546f4ec42_0.0018604775743442331/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Fireplace/bec1921aace45bdecbc6ff5546f4ec42_0.003884412225089298/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Fireplace/c55cd55a8c4d28a1ac51268fdb437a9e_0.003957724234942252/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Fireplace/ca96a54cb2eaf8d0d67a963c379eac15_0.003197893377961384/coacd_decomposed.obj']
(Generator pid=2082993) failed to process ['/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Laptop/367fbaea8743ec1cc98452c8fce6b43_0.0013906943609724922/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Laptop/39b54fd684a4ed3e6f0a9a10dc885e17_0.014869469389545946/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Laptop/39e61115ba9df4d3cf75707bd872e2b3_0.008557902338200871/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Laptop/3b2bddb0e8ff57c85831a0624cf5a945_0.02197555347943938/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Laptop/3e64b275e8ca0fd6c15df58910515c8b_0.006928242429014028/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Laptop/42d4c54e357eae5a48f6ba3cefa3ac08_0.018626420870191902/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Laptop/42fbd782557fc2a4a59beaee0a0099fc_0.00023200607973197952/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Laptop/42fbd782557fc2a4a59beaee0a0099fc_0.0004178121731394019/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Laptop/454a88f4b2fe4472f9cacf0b2853f40_0.01335769222405151/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Laptop/4618d2a74e83e471cd6c695b8c795149_0.006190032321941217/coacd_decomposed.obj']
(Generator pid=2082993) failed to process ['/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Loveseat/2298d06de1efec2a8ab33b80d7377fc3_0.002084012283293985/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Loveseat/26b4c8e753b7dd06a6fd617b2ff5d2d_0.006429270495365568/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Loveseat/4307657b2731fd392c37553db37ec752_0.004034309576845391/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Loveseat/4307657b2731fd392c37553db37ec752_0.005649057646339869/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Loveseat/49d6f351e82d186b366971e8a2cdf120_0.0045234513325064325/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Loveseat/73111c5505d7b5eae3db80a3cacc6e3_0.001573056262688191/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Loveseat/73111c5505d7b5eae3db80a3cacc6e3_0.0017858073363189873/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Loveseat/74490c45305da48a2b3e9b6eb52d35df_0.001213614261104857/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Loveseat/8c69c0bafdf91e85c05575120a46cd3b_0.0028823923636336515/coacd_decomposed.obj', '/share1/junyu/HRI/genh2r/env/data/assets/objects/acronym/Loveseat/9001c9558ffe9b5651b8631af720684f_0.005422012635315788/coacd_decomposed.obj']
"""