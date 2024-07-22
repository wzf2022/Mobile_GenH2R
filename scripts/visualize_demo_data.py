import open3d as o3d
import numpy as np
import code
from env.utils.transform import se3_inverse, pos_ros_quat_to_mat
from scipy.spatial.transform import Rotation as Rt
from env.utils.scene import scene_id_to_scene_dir
import os

# 加载npz文件
scene_id = 10007

scene_dir = scene_id_to_scene_dir(scene_id)
scene_data_path = os.path.join(scene_dir, f"scene_{scene_id:08d}", "pose.npy")
scene_mobile_data = np.load(scene_data_path, allow_pickle = True).item()
mesh_file_path = os.path.join("env/data/assets/objects/acronym", scene_mobile_data["scene_meta"]["dex_key"].replace('_', '/', 1).split("index")[0], "model.obj")

code.interact(local=dict(globals(), **locals()))

data = np.load(f'log/m0/static/00/01/00/{scene_id:08d}.npz')
visualize_step = 0


# 获取相机坐标系变换矩阵和点云数据
head_camera_matrix = data['world_to_head_camera'][visualize_step]  # 4x4 矩阵
wrist_camera_matrix = data['world_to_wrist_camera'][visualize_step]  # 4x4 矩阵
head_camera_object_point_cloud = data[f"object_points_{visualize_step}"]  # N x 3 点云
wrist_camera_object_point_cloud = data[f"wrist_object_points_{visualize_step}"]  # N x 3 点云

head_camera_hand_point_cloud = data[f"hand_points_{visualize_step}"]  # N x 3 点云
wrist_camera_hand_point_cloud = data[f"wrist_hand_points_{visualize_step}"]  # N x 3 点云

code.interact(local=dict(globals(), **locals()))

urdf_to_opengl = Rt.from_euler("XYZ", (-np.pi/2, 0.0, -np.pi)).as_matrix()
urdf_to_opengl = Rt.from_euler("XYZ", (np.pi, 0.0, 0.0)).as_matrix()
# head_camera_matrix[:3, :3] = head_camera_matrix[:3, :3] @ (urdf_to_opengl.T)
# wrist_camera_matrix[:3, :3] = wrist_camera_matrix[:3, :3] @ (urdf_to_opengl.T)
world_to_object = data['world_to_object'][visualize_step]
# code.interact(local=dict(globals(), **locals()))

# head_camera_point_cloud = (head_camera_matrix[:3, :3] @ head_camera_point_cloud.T).T + head_camera_matrix[:3, 3]
# wrist_camera_point_cloud = (wrist_camera_matrix[:3, :3] @ wrist_camera_point_cloud.T).T + wrist_camera_matrix[:3, 3]

code.interact(local=dict(globals(), **locals()))

# 定义一个函数来创建坐标轴
def create_coordinate_frame(size = 0.1):
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    return coordinate_frame

# 创建世界坐标系的坐标轴
world_coordinate_frame = create_coordinate_frame(0.3)

# 创建相机坐标系的坐标轴
head_camera_frame = create_coordinate_frame()
wrist_camera_frame = create_coordinate_frame()

# 将相机坐标系的坐标轴变换到相应的位置
head_camera_frame.transform(head_camera_matrix)
wrist_camera_frame.transform(wrist_camera_matrix)

# 创建点云对象并变换到世界坐标系
head_camera_object_pcd = o3d.geometry.PointCloud()
head_camera_object_pcd.points = o3d.utility.Vector3dVector(head_camera_object_point_cloud)
head_camera_object_pcd.transform(head_camera_matrix)

head_camera_hand_pcd = o3d.geometry.PointCloud()
head_camera_hand_pcd.points = o3d.utility.Vector3dVector(head_camera_hand_point_cloud)
head_camera_hand_pcd.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 1], (head_camera_hand_point_cloud.shape[0], 1)))  # 蓝色
head_camera_hand_pcd.transform(head_camera_matrix)


wrist_camera_object_pcd = o3d.geometry.PointCloud()
wrist_camera_object_pcd.points = o3d.utility.Vector3dVector(wrist_camera_object_point_cloud)
wrist_camera_object_pcd.transform(wrist_camera_matrix)

wrist_camera_hand_pcd = o3d.geometry.PointCloud()
wrist_camera_hand_pcd.points = o3d.utility.Vector3dVector(wrist_camera_hand_point_cloud)
wrist_camera_hand_pcd.colors = o3d.utility.Vector3dVector(np.tile([0, 1, 0], (wrist_camera_hand_point_cloud.shape[0], 1)))  # 蓝色
wrist_camera_hand_pcd.transform(wrist_camera_matrix)

### 可视化mesh
object_mesh = o3d.io.read_triangle_mesh(mesh_file_path)
object_mesh.transform(world_to_object)
object_mesh.compute_vertex_normals()

# code.interact(local=dict(globals(), **locals()))

# 可视化所有的对象
o3d.visualization.draw_geometries([world_coordinate_frame, head_camera_frame, wrist_camera_frame, 
head_camera_object_pcd, head_camera_hand_pcd, wrist_camera_object_pcd, wrist_camera_hand_pcd, object_mesh])