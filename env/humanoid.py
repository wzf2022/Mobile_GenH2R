import os
import numpy as np
from numpy.typing import NDArray
from pybullet_utils.bullet_client import BulletClient
from dataclasses import dataclass
from omegaconf import MISSING
from xml.etree.ElementTree import parse, ElementTree
import copy
from typing import Tuple, List, Optional
import open3d as o3d
import code
import pybullet
from .body import Body, BodyConfig
from .utils.transform import pos_ros_quat_to_mat, se3_transform_pc

@dataclass
class HumanoidConfig(BodyConfig):
    name: str = ""
    urdf_file: str = ""
    # collision
    collision_mask: Optional[int] = 2**8
    # base
    use_fixed_base: bool = True
    base_position: Tuple[float] = (0., 0., 0.)
    base_orientation: Tuple[float] = (0., 0., 0., 1.)
    # links
    num_dofs: int = 6
    dof_position: Optional[Tuple[float]] = None
    dof_velocity: Tuple[float] = (0.0,)*9

    translation_max_force: Tuple[float] = (50.0,)*3
    translation_position_gain: Tuple[float] = (0.2,)*3
    translation_velocity_gain: Tuple[float] = (1.0,)*3
    rotation_max_force: Tuple[float] = (5.0,)*3
    rotation_position_gain: Tuple[float] = (0.2,)*3
    rotation_velocity_gain: Tuple[float] = (1.0,)*3

    dof_max_force: Tuple[float] = "${concat_tuples:${.translation_max_force},${.rotation_max_force}}"
    dof_position_gain: Tuple[float] = "${concat_tuples:${.translation_position_gain},${.rotation_position_gain}}"
    dof_velocity_gain: Tuple[float] = "${concat_tuples:${.translation_velocity_gain},${.rotation_velocity_gain}}"

    use_cylider: bool = True
    radius: float = 0.2
    height: float = 2.0
    mass: float = 10.0
    # concat_tuples is defined in body.py
    # the "." is necessary in nested configs, see https://github.com/omry/omegaconf/issues/1099#issuecomment-1624496194

    compute_target_real_displacement: bool = False

max_target_real_displacement = 0.

class Humanoid(Body):
    def __init__(self, bullet_client: BulletClient, cfg: HumanoidConfig):
        super().__init__(bullet_client, cfg)
        self.cfg: HumanoidConfig

    def reset(self, pose: NDArray[np.float32]):
        self.base_reset()
        # config
        # self.cfg.name = name
        # self.cfg.urdf_file = path
        # self.cfg.collision_mask = collision_mask

        # process pose data
        self.pose = pose # (num_frames, 7)
        self.num_frames = self.pose.shape[0]

        if self.cfg.set_human_hand_obj_last_frame:
            self.frame = self.num_frames - 1
        else:
            self.frame = 0
        # self.frame = self.num_frames - 1

        # self.cfg.dof_position = self.pose[self.frame].tolist() # can not use tuple(self.pose[self.frame]), which keeps the type np.float32 for scalars
        # self.load()
        # self.set_dof_target(self.pose[self.frame])

        self.object_pc: Optional[NDArray[np.float64]] = None

        if self.cfg.use_cylider == True:
            self.pose[:, 2] = self.cfg.height / 2
            self.pose[:, 3:6] = 0
            self.pose[:, 6] = 1
            visualShapeId = self.bullet_client.createVisualShape(shapeType = pybullet.GEOM_CYLINDER, radius = self.cfg.radius, length = 0.01, rgbaColor=[0.8, 0.8, 0.8, 1], visualFramePosition=[0, 0, -self.cfg.height / 2])
            collisionShapeId = self.bullet_client.createCollisionShape(shapeType=pybullet.GEOM_CYLINDER, radius = self.cfg.radius, height = self.cfg.height)
            self.body_id = self.bullet_client.createMultiBody(baseMass = self.cfg.mass, baseVisualShapeIndex = visualShapeId, baseCollisionShapeIndex = collisionShapeId, 
                                                             basePosition = self.pose[self.frame, :3], baseOrientation = self.pose[self.frame, 3:])
            

        self.num_links = self.bullet_client.getNumJoints(self.body_id)+1
        self.set_collision_mask(self.cfg.collision_mask)
        # jointIndex = 0
        # jointRange = 0.01
        self.constraintId = self.bullet_client.createConstraint(self.body_id, -1, -1, -1, pybullet.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 1])


    def pre_step(self):
        self.frame += 1
        self.frame = min(self.frame, self.num_frames-1)
        # print("object_frame", self.pose[self.frame])
        # if self.frame == self.num_frames - 1:
        # code.interact(local=dict(globals(), **locals()))
        # self.set_dof_target(self.pose[self.frame])
        if self.cfg.use_cylider == True:
            # current_position, current_orientation = self.bullet_client.getBasePositionAndOrientation(self.body_id)
            # error_position = list(self.pose[self.frame, :3] - current_position)
            self.bullet_client.changeConstraint(self.constraintId, jointChildPivot = self.pose[self.frame, :3], jointChildFrameOrientation = self.pose[self.frame, 3:], maxForce = 50)
    
    def post_step(self):
        if self.cfg.compute_target_real_displacement:
            global max_target_real_displacement
            max_target_real_displacement = max(max_target_real_displacement, np.abs(self.pose[self.frame][:3]-self.get_joint_positions()[:3]).max())
            print(f"max_target_real_displacement={max_target_real_displacement}")

    def get_world_to_obj(self) -> NDArray[np.float64]:
        world_to_obj = self.get_link_pose(5)
        return world_to_obj
    
    def get_world_to_object_pc(self) -> NDArray[np.float64]:
        if self.object_pc is None:
            tree: ElementTree = parse(self.cfg.urdf_file)
            root = tree.getroot()
            collision_file_name: str = root.findall("link")[-1].find("collision").find("geometry").find("mesh").get("filename")
            collision_file_path = os.path.join(os.path.dirname(self.cfg.urdf_file), collision_file_name)
            object_mesh = o3d.io.read_triangle_mesh(collision_file_path)
            self.object_pc = np.array(object_mesh.vertices)
        world_to_object = self.get_world_to_obj()
        world_to_object_pc = se3_transform_pc(world_to_object, self.object_pc)
        return world_to_object_pc




def debug():
    from omegaconf import OmegaConf
    import pybullet
    import time
    from env.utils import load_scene_data
    objects_cfg = OmegaConf.structured(ObjectsConfig)
    object_cfg = OmegaConf.structured(ObjectConfig)
    bullet_client = BulletClient(connection_mode=pybullet.GUI) # or pybullet.DIRECT
    objects = Objects(bullet_client, objects_cfg, object_cfg)
    scene_data = load_scene_data(0)
    objects.reset(scene_data["object_names"], scene_data["object_paths"], scene_data["object_grasp_id"], scene_data["object_poses"])
    code.interact(local=dict(globals(), **locals()))
    while True:
        print(f"frame {objects.target_object.frame}")
        objects.pre_step()
        bullet_client.stepSimulation()
        objects.post_step()
        # time.sleep(0.01)

if __name__ == "__main__":
    debug()

"""
DISPLAY="localhost:11.0" python -m env.objects

CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=32 env.panda.IK_solver=PyKDL setup=s0 split=train policy=chomp chomp.wait_time=13 env.object.compute_target_real_displacement=True
"""