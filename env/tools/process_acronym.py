import os
from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
import open3d as o3d
import h5py
from tqdm import tqdm
from xml.etree.ElementTree import parse, ElementTree
import ray
from omegaconf import OmegaConf
from dataclasses import dataclass
import copy
import code

from .convex_decompose import convex_decompose_coacd
from .sdf import gen_sdf

env_dir = os.path.dirname(os.path.dirname(__file__))
shapenet_dir = os.path.join(env_dir, "data", "tmp", "models")
acronym_dir = os.path.join(env_dir, "data", "tmp", "grasps")
meshlab_root_dir = os.path.join(env_dir, "data", "tmp", "meshlab")

processed_object_root_dir = os.path.join(env_dir, "data", "assets", "objects", "acronym")

def parse_h5_data(data_path: str) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    data = h5py.File(data_path, "r")
    data_grasps = data["grasps"]
    grasps = np.array(data_grasps["transforms"]) # (2000, 4, 4)
    data_object = data["object"]
    obj_com, obj_density, obj_friction, obj_inertia, obj_mass, obj_scale, obj_volume = np.array(data_object["com"]), np.array(data_object["density"]), np.array(data_object["friction"]), np.array(data_object["inertia"]), np.array(data_object["mass"]), np.array(data_object["scale"]), np.array(data_object["volume"])
    success_mask = np.array(data_grasps["qualities/flex/object_in_gripper"]).astype(bool) # (2000,)
    grasps = grasps[success_mask]
    offset_pose = np.array([
        [0., -1., 0., 0.],
        [1., 0., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ])
    grasps = np.matmul(grasps, offset_pose)  # flip x, y
    return grasps.astype(np.float64), obj_com.astype(np.float64), obj_density.astype(np.float64), obj_friction.astype(np.float64), obj_inertia.astype(np.float64), obj_mass.astype(np.float64), obj_scale.astype(np.float64), obj_volume.astype(np.float64)

grasp_model_path = os.path.join(env_dir, "data", "assets", "grasp", "grasp_simplified.obj")
grasp_mesh = o3d.io.read_triangle_mesh(grasp_model_path)
frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
def vis_obj_grasps(obj_mesh, grasps=[], mesh_show_wireframe=True, mesh_show_back_face=True):
    meshes = [obj_mesh, frame_mesh]
    for grasp in grasps:
        grasp_mesh_transformed = copy.deepcopy(grasp_mesh).transform(grasp)
        meshes.append(grasp_mesh_transformed)
    o3d.visualization.draw_geometries(meshes, mesh_show_wireframe=mesh_show_wireframe, mesh_show_back_face=mesh_show_back_face)

tree_template: ElementTree = parse(os.path.join(os.path.dirname(__file__), "model_template.urdf"))
def generate_urdf(name: str, friction: float, mass: float, inertia: NDArray[np.float64], target_urdf_path):
    tree = copy.deepcopy(tree_template)
    root = tree.getroot()
    root.attrib["name"] = name
    link = root.findall("link")[-1]
    link.find("contact").find("lateral_friction").attrib['value'] = str(friction)
    inertial = link.find("inertial")
    inertial.find("mass").attrib['value'] = str(mass)
    for i in range(3):
        for j in range(i, 3):
            inertial.find("inertia").attrib[f'i{chr(120+i)}{chr(120+j)}'] = str(inertia[i, j])
    tree.write(target_urdf_path, xml_declaration=True, encoding="utf-8")

def process(obj_name: str, obj_code: str, obj_scale_str: str, acronym_file_name: str):
    target_obj_dir = os.path.join(processed_object_root_dir, obj_name, obj_code+"_"+obj_scale_str)
    os.makedirs(target_obj_dir, exist_ok=True)
    target_obj_path = os.path.join(target_obj_dir, "model.obj")
    
    # print(f"processing {obj_name} {obj_code} {acronym_file_name}")
    obj_path = os.path.join(shapenet_dir, obj_code+".obj")
    mtl_path = os.path.join(shapenet_dir, obj_code+".mtl")

    texture_images: List[str] = []
    with open(mtl_path, "r") as f:
        for line in f.readlines():
            line = line.rstrip()
            if line.endswith("jpg"):
                texture_images.append(line.split(" ")[-1])
    
    # load object
    # use meshlab to clean data first. https://github.com/isl-org/Open3D/issues/3037#issuecomment-981808333
    meshlab_dir = os.path.join(meshlab_root_dir, obj_name, obj_code+"_"+obj_scale_str)
    meshlab_obj_path = os.path.join(meshlab_dir, "model.obj")
    obj_mesh = o3d.io.read_triangle_mesh(meshlab_obj_path, enable_post_processing=True)

    # load grasp data
    grasp_data_path = os.path.join(acronym_dir, acronym_file_name)
    grasps, obj_com, obj_density, obj_friction, obj_inertia, obj_mass, obj_scale, obj_volume = parse_h5_data(grasp_data_path)
    
    # scale and center the object
    # save the object together with texture
    if not os.path.exists(target_obj_path):
        scaled_obj_mesh = copy.deepcopy(obj_mesh).scale(obj_scale, center=np.zeros(3))
        centered_scaled_obj_mesh = copy.deepcopy(scaled_obj_mesh).translate(-obj_com)
        centered_grasps = grasps.copy()
        centered_grasps[:, :3, 3] -= obj_com
        o3d.io.write_triangle_mesh(target_obj_path, centered_scaled_obj_mesh)
        # vis_obj_grasps(centered_scaled_obj_mesh, centered_grasps[np.random.choice(centered_grasps.shape[0], (100, ), replace=False)])
    # check if texture is processed
    if len(texture_images) > 0:
        assert os.path.exists(os.path.join(target_obj_dir, "model_0.png"))
    
    # save grasps
    target_obj_grasps_path = os.path.join(target_obj_dir, "grasps.npy")
    if not os.path.exists(target_obj_grasps_path):
        np.save(target_obj_grasps_path, centered_grasps)

    # convex decomposition
    target_convex_obj_path = os.path.join(target_obj_dir, "coacd_decomposed.obj")
    if not os.path.exists(target_convex_obj_path):
        convex_decompose_coacd(target_obj_path)

    # sdf
    if not os.path.exists(os.path.join(target_obj_dir, "sdf.npz")):
        gen_sdf(target_convex_obj_path)

    # urdf
    target_urdf_path = os.path.join(target_obj_dir, "model.urdf")
    if not os.path.exists(target_urdf_path):
        generate_urdf(f"{obj_name}_{obj_code}_{obj_scale_str}", obj_friction.item(), obj_mass.item(), obj_inertia, target_urdf_path)

@dataclass
class ProcessAcronymConfig:
    num_runners: int = 64

@ray.remote(num_cpus=1)
class Distributer:
    def __init__(self, cfg: ProcessAcronymConfig):
        self.cfg = cfg
        shapenet_files = os.listdir(shapenet_dir)
        shapenet_codes = [file_name[:-4] for file_name in shapenet_files if file_name.endswith(".obj")]

        acronym_files = sorted(os.listdir(acronym_dir))
        self.acronym_metas: List[Tuple[str, str, str, str]] = []
        for file_name in acronym_files:
            file_prefix = file_name[:-3] # remove .h5
            obj_name, obj_code, obj_scale_str = file_prefix.split("_")
            self.acronym_metas.append((obj_name, obj_code, obj_scale_str, file_name))
        
        os.makedirs(os.path.join(env_dir, "data", "assets", "objects", "acronym"), exist_ok=True)
        f = open(os.path.join(env_dir, "data", "assets", "objects", "acronym", "acronym_list.txt"), "w")
        for obj_name, obj_code, obj_scale_str, file_name in self.acronym_metas:
            assert obj_code in shapenet_codes
            target_obj_dir = os.path.join(obj_name, obj_code+"_"+obj_scale_str)
            f.write(target_obj_dir+"\n")
        f.close()
        self.idx = 0

    def get_next_task(self) -> Optional[Tuple[str, str, str, str]]:
        if self.idx >= len(self.acronym_metas):
            return None
        print(f"distribute idx {self.idx}/{len(self.acronym_metas)}")
        obj_name, obj_code, obj_scale_str, file_name = self.acronym_metas[self.idx]
        self.idx += 1
        return obj_name, obj_code, obj_scale_str, file_name

@ray.remote(num_cpus=1)
class Processor:
    def __init__(self, cfg: ProcessAcronymConfig, distributer: Distributer):
        self.cfg = cfg
        self.distributer = distributer
    
    def work(self):
        while True:
            acronym_meta = ray.get(self.distributer.get_next_task.remote())
            if acronym_meta is None: break
            obj_name, obj_code, obj_scale_str, file_name = acronym_meta
            process(obj_name, obj_code, obj_scale_str, file_name)

def main():
    default_cfg = OmegaConf.structured(ProcessAcronymConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg: ProcessAcronymConfig = OmegaConf.to_object(OmegaConf.merge(default_cfg, cli_cfg))

    ray.init()
    distributer = Distributer.remote(cfg)
    processors = [Processor.remote(cfg, distributer) for _ in range(cfg.num_runners)]
    ray.get([processor.work.remote() for processor in processors])

if __name__ == "__main__":
    main()

"""
rm -r env/data/assets/objects/acronym
CUDA_VISIBLE_DEVICES=-1 python -m env.tools.process_acronym
# set gpu=-1 to disable all gpus

QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms 
DISPLAY="localhost:11.0" meshlabserver -i tmp/debug_process_acronym/1cfdff88d43bc6f7ff6ace05b36a5.obj -o tmp/debug_process_acronym/meshlabserver/mesh.obj -m vn fc wt -l x
"""
