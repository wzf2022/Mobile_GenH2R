import os
import numpy as np
import torch
import open3d as o3d
from dataclasses import dataclass
from omegaconf import OmegaConf
from manopth.manolayer import ManoLayer

@dataclass
class VisualizeMANOGraspsConfig:
    # object_dir: str = "env/data/assets/objects/acronym/1Shelves/12a64182bbaee7a12b2444829a3507de_0.00914554366969263"
    start_grasp_idx: int = 0
    end_grasp_idx: int = 64
    side: str = "right"
    start_object_idx: int = 0
    end_object_idx: int = 100
    # flip: bool = False

def visualize_mano_grasps(mano_layer: ManoLayer, object_dir: str, object_idx: int, cfg: VisualizeMANOGraspsConfig):
    object_mesh = o3d.io.read_triangle_mesh(os.path.join(object_dir, "coacd_decomposed.obj")).paint_uniform_color([1., 1., 1.])
    mano_grasps_path = os.path.join(object_dir, f"mano_grasps_{cfg.side}.npy")
    mano_grasps = np.load(mano_grasps_path)
    for grasp_idx in range(cfg.start_grasp_idx, cfg.end_grasp_idx):
        mano_grasp = mano_grasps[grasp_idx]

        theta = torch.FloatTensor(mano_grasp[3:]).unsqueeze(0)
        trans = torch.FloatTensor(mano_grasp[:3]).unsqueeze(0)
        hand_verts, hand_joints = mano_layer.forward(th_pose_coeffs=theta, th_trans=trans) # (1, 778, 3), (1, 21, 3)
        hand_transformed_verts = (hand_verts/1000.0)[0].detach().cpu().numpy()
        hand_mesh = o3d.geometry.TriangleMesh()
        hand_mesh.vertices = o3d.utility.Vector3dVector(hand_transformed_verts)
        hand_mesh.triangles = o3d.utility.Vector3iVector(mano_layer.th_faces)
        hand_mesh.paint_uniform_color([0., 1., 0.])

        mesh_store_path = os.path.join("tmp", "debug_dexgraspnet", f"{object_idx:04d}_{object_dir.split('/')[-1]}_{cfg.side}_{grasp_idx}.obj")
        os.makedirs(os.path.dirname(mesh_store_path), exist_ok=True)
        o3d.io.write_triangle_mesh(mesh_store_path, object_mesh+hand_mesh)
        # o3d.visualization.draw_geometries([object_mesh, hand_mesh], mesh_show_wireframe=True, mesh_show_back_face=True)

def main():
    default_cfg = OmegaConf.structured(VisualizeMANOGraspsConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg: VisualizeMANOGraspsConfig = OmegaConf.to_object(OmegaConf.merge(default_cfg, cli_cfg))

    env_dir = os.path.dirname(os.path.dirname(__file__))
    dexgraspnet_dir = os.path.join(env_dir, "third_party", "DexGraspNet")
    mano_dir = os.path.join(dexgraspnet_dir, "grasp_generation", "mano")
    mano_layer = ManoLayer(mano_root=mano_dir, side=cfg.side, flat_hand_mean=True, use_pca=False)

    acronym_dir = os.path.join(env_dir, "data", "assets", "objects", "acronym")
    acronym_list_path = os.path.join(acronym_dir, "acronym_list.txt")
    with open(acronym_list_path, "r") as f:
        object_dir_list = [os.path.join(acronym_dir, line.rstrip()) for line in f.readlines()]

    for object_idx in range(cfg.start_object_idx, cfg.end_object_idx):
        visualize_mano_grasps(mano_layer, object_dir_list[object_idx], object_idx, cfg)

if __name__ == "__main__":
    main()

"""
python -m env.tools.visualize_mano_grasps start_object_idx=4510 end_object_idx=4520 end_grasp_idx=1
python -m env.tools.visualize_mano_grasps start_object_idx=4720 end_object_idx=4730 end_grasp_idx=1
rm -r tmp/debug_dexgraspnet
"""