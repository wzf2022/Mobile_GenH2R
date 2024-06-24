import os
from typing import Tuple, List
import pymeshlab
from tqdm import tqdm

env_path = os.path.dirname(os.path.dirname(__file__))
shapenet_dir = os.path.join(env_path, "data", "tmp", "models")
acronym_dir = os.path.join(env_path, "data", "tmp", "grasps")
meshlab_root_dir = os.path.join(env_path, "data", "tmp", "meshlab")

# use meshlab to clean data first. https://github.com/isl-org/Open3D/issues/3037#issuecomment-981808333
def main():
    shapenet_files = os.listdir(shapenet_dir)
    shapenet_codes = [file_name[:-4] for file_name in shapenet_files if file_name.endswith(".obj")]

    acronym_files = sorted(os.listdir(acronym_dir))
    acronym_metas: List[Tuple[str, str, str, str]] = []
    for file_name in acronym_files:
        file_prefix = file_name[:-3] # remove .h5
        obj_name, obj_code, obj_scale_str = file_prefix.split("_")
        acronym_metas.append((obj_name, obj_code, obj_scale_str, file_name))
    
    for obj_name, obj_code, obj_scale_str, file_name in acronym_metas:
        assert obj_code in shapenet_codes
    
    for obj_name, obj_code, obj_scale_str, file_name in tqdm(acronym_metas, desc="preprocess with meshlab"):
        obj_path = os.path.join(shapenet_dir, obj_code+".obj")
        meshlab_dir = os.path.join(meshlab_root_dir, obj_name, obj_code+"_"+obj_scale_str)
        os.makedirs(meshlab_dir, exist_ok=True)
        meshlab_obj_path = os.path.join(meshlab_dir, "model.obj")

        if not os.path.exists(meshlab_obj_path):
            mesh = pymeshlab.MeshSet()
            mesh.load_new_mesh(obj_path)
            mesh.save_current_mesh(meshlab_obj_path)

if __name__ == "__main__":
    main()
