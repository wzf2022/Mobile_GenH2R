import os
import numpy as np
from tqdm import tqdm
import code

from ..utils.scene import scene_id_to_hierarchical_dir

def main():
    result_root_dir = "env/data/hand_collision_free_mask/augment_True_threshold_0.01"
    for scene_id in tqdm(range(1000000, 2000000)):
        save_dir = os.path.join(result_root_dir, scene_id_to_hierarchical_dir(scene_id))
        save_path = os.path.join(save_dir, f"{scene_id:08d}.npy")
        if os.path.exists(save_path):
            hand_collision_free_mask = np.load(save_path)
            if hand_collision_free_mask.dtype != bool:
                code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    main()

"""
python -m env.tools.check_hand_collision_free_mask
"""