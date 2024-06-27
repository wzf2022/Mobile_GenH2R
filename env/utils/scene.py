import os
import numpy as np
from numpy.typing import NDArray
from typing import TypedDict, List, Optional
from scipy.spatial.transform import Rotation as Rt
from env.utils.transform import ros_quat_to_euler, ros_quat_to_rotvec
import shutil
import code

def scene_id_to_hierarchical_dir(scene_id: int) -> str:
    scene_id_str = f"{scene_id:08d}" # "12345678"
    hierarchical_dir = os.path.join(*(scene_id_str[i:i+2] for i in range(0, len(scene_id_str)-2, 2))) # "12/34/56/"
    return hierarchical_dir

def scene_id_to_dir(scene_id: int, demo_structure: str) -> str:
    if demo_structure == "hierarchical":
        scene_dir = scene_id_to_hierarchical_dir(scene_id)
    elif demo_structure == "flat":
        scene_dir = ""
    else:
        raise NotImplementedError
    return scene_dir

env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scene_root_dir = os.path.join(env_dir, "data", "mobile_scene")
def scene_id_to_scene_dir(scene_id: int) -> str:
    scene_path = os.path.join(scene_root_dir, scene_id_to_hierarchical_dir(scene_id))
    return scene_path

class SceneData(TypedDict):
    hand_name: str
    hand_side: str
    hand_path: str
    hand_pose: NDArray[np.float32]
    object_names: List[str]
    object_paths: List[str]
    object_grasp_id: int
    object_poses: NDArray[np.float32]
    endpoints: Optional[NDArray[np.int64]]

objects_dir = os.path.join(env_dir, "data", "assets", "objects")
hand_dir = os.path.join(env_dir, "data", "assets", "hand")
def load_scene_data(scene_id: int, table_height: float=0., stop_moving_frame: Optional[int]=None, frame_interval: int=1) -> SceneData:
    scene_dir = scene_id_to_scene_dir(scene_id)
    scene_data_path = os.path.join(scene_dir, f"scene_{scene_id:08d}", "pose.npy")
    scene_mobile_data = np.load(scene_data_path, allow_pickle = True).item()  
    # ict_keys(['body_pose', 'body_params', 'body_joints', 'hand_pose', 'hand_joints', 'hand_params', 'obj_pose', 'scene_meta'])
    # body_pose     (T, 7)
    # body_params   (T, 21, 3)
    # body_joints   (T, 144, 3)
    # hand_pose     (T, 7)
    # hand_joints   (T, 16, 3)
    # hand_params   (T, 15, 3)
    # obj_pose      (T, 60, 7)
    # scene_meta    {'dex_key': 'WineBottle_7746997b8cfd8d11f4718731863dd64d_0.0019632731498702683index001', 'locomotion': 'straight', 'index': 14, 'hand_side': 'right', 'gmd_length': 39, 'handover_length': 21}
    # code.interact(local=dict(globals(), **locals()))

    scene_data = {}
    scene_data["hand_name"] = "hand_0"
    scene_data["hand_side"] = scene_mobile_data["scene_meta"]["hand_side"]
    if scene_data["hand_side"] == "left":
        scene_data["hand_path"] = "env/data/assets/hand/hand_0_left/mano.urdf"
    else:
        scene_data["hand_path"] = "env/data/assets/hand/hand_0_right/mano.urdf"
    frame_num = scene_mobile_data["obj_pose"].shape[0]
    scene_data["hand_pose"] = np.concatenate((scene_mobile_data["hand_pose"][:, :3], ros_quat_to_euler(scene_mobile_data["hand_pose"][:, 3:]), scene_mobile_data["hand_params"].reshape(-1, 45)), axis = 1)  # (T, 51)
    scene_data["object_names"] = [scene_mobile_data["scene_meta"]["dex_key"]]
    scene_data["object_paths"] = [os.path.join("env/data/assets/objects/acronym", scene_mobile_data["scene_meta"]["dex_key"].replace('_', '/', 1).split("index")[0], "model.urdf")]
    scene_data["object_grasp_id"] = np.array(0)
    scene_data["object_poses"] = np.concatenate((scene_mobile_data["obj_pose"][:, :3], ros_quat_to_euler(scene_mobile_data["obj_pose"][:, 3:])), axis = 1).reshape(1, frame_num, 6)     # (1, T, 6)
    scene_data["body_pose"] = scene_mobile_data["body_pose"]  #(T, 7)
    scene_data['body_params'] = scene_mobile_data['body_params']   #(T, 21, 3)

    # the fps is 20            
    scene_data["hand_pose"] = scene_data["hand_pose"].repeat(50, axis = 0)
    scene_data["object_poses"] = scene_data["object_poses"].repeat(50, axis = 1)
    scene_data["body_pose"] = scene_data["body_pose"].repeat(50, axis = 0)
    scene_data["body_params"] = scene_data["body_params"].repeat(50, axis = 0)

    if "endpoints" in scene_data:
        scene_data["endpoints"] //= frame_interval
    else:
        scene_data["endpoints"] = None
    
    scene_data["source"] = "mobile_genh2r"
    # code.interact(local=dict(globals(), **locals()))

    # scene_data = dict(np.load(scene_data_path)) # "hand_name", "hand_side", "hand_path", "hand_pose", "object_names", "object_paths", "object_grasp_id", "object_poses", "endpoints", "source"
    # source = scene_data["source"]
    # scene_data["hand_name"] = scene_data["hand_name"].item()
    # scene_data["hand_side"] = scene_data["hand_side"].item()
    # scene_data["hand_path"] = os.path.join(hand_dir, scene_data["hand_path"].item())
    # if source == "dexycb":
    #     scene_data["hand_pose"] = scene_data["hand_pose"][::frame_interval] # (T, 51)
    # elif source == "genh2r":
    #     hand_pose = scene_data["hand_pose"][::frame_interval]
    #     scene_data["hand_pose"] = np.concatenate([hand_pose, np.tile(scene_data["hand_theta"], (hand_pose.shape[0], 1))], axis=1)
    # else:
    #     raise NotImplementedError
    # scene_data["object_names"] = scene_data["object_names"].tolist()
    # scene_data["object_paths"] = [os.path.join(objects_dir, object_path) for object_path in scene_data["object_paths"].tolist()]
    # scene_data["object_poses"] = scene_data["object_poses"][:, ::frame_interval] # (#objects, T, 6)

    # if stop_moving_frame is not None:
    #     scene_data["hand_pose"] = scene_data["hand_pose"][:stop_moving_frame]
    #     scene_data["object_poses"] = scene_data["object_poses"][:, :stop_moving_frame]
    # if table_height != 0.:
    #     hand_nonzero_mask = np.any(scene_data["hand_pose"]!=0, axis=1)
    #     hand_nonzeros = np.where(hand_nonzero_mask)[0]
    #     hand_start_frame = hand_nonzeros[0]
    #     hand_end_frame = hand_nonzeros[-1]+1
    #     scene_data["hand_pose"][hand_start_frame:hand_end_frame, 2] += table_height
    #     scene_data["object_poses"][:, :, 2] += table_height
    
    # if "endpoints" in scene_data:
    #     scene_data["endpoints"] //= frame_interval
    # else:
    #     scene_data["endpoints"] = None

    return scene_data

def six_d_to_mat(six_d: NDArray[np.float64]) -> NDArray[np.float64]:
    " (..., 6) "
    shape_prefix = six_d.shape[:-1]
    mat = np.zeros(shape_prefix+(4, 4))
    mat[..., :3, 3] = six_d[..., :3]
    mat[..., :3, :3] = Rt.from_euler("XYZ", six_d[..., 3:].reshape(-1, 3)).as_matrix().reshape(shape_prefix+(3, 3))
    mat[..., 3, 3] = 1
    return mat

def mat_to_six_d(mat: NDArray[np.float64]) -> NDArray[np.float64]:
    " (..., 4, 4) "
    shape_prefix = mat.shape[:-2]
    return np.concatenate([mat[..., :3, 3], Rt.from_matrix(mat[..., :3, :3].reshape(-1, 3, 3)).as_euler("XYZ").reshape(shape_prefix + (3, ))], axis=-1)