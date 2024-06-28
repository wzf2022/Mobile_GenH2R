import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from omegaconf import OmegaConf
import pybullet
from pybullet_utils.bullet_client import BulletClient
import pybullet_data
from functools import partial
from typing import TypedDict, Callable, List, Tuple, Optional
from scipy.spatial.transform import Rotation as Rt
from contextlib import contextmanager
import time
import code
import torch
from typing import Tuple, Optional, Dict, List
from .galbot import Galbot, GalbotConfig
from .camera import Camera, CameraConfig
from .table import Table, TableConfig
from .hand import Hand, HandConfig
from .humanoid import Humanoid, HumanoidConfig
from .objects import Objects, ObjectConfig, ObjectsConfig
from .status_checker import StatusChecker, StatusCheckerConfig, EpisodeStatus
from .bodies_for_visualization import Grasp, GraspConfig, get_grasp_config, Sphere, SphereConfig
from .utils.scene import load_scene_data
from manopth.manolayer import ManoLayer
from scipy.spatial.transform import Rotation as Rt

OmegaConf.register_new_resolver("divide_to_int", lambda x, y: int(x/y) if x is not None else None)

@dataclass
class MobileH2RSimConfig:
    gravity: Tuple[float] = (0.0, 0.0, -9.8)
    substeps: int = 1
    table_height: float = 0.8

    step_time: float = 0.001
    max_time: float = 15.0
    stop_moving_time: Optional[float] = None

    max_frames: int = "${divide_to_int:${.max_time},${.step_time}}"
    stop_moving_frame: Optional[int] = "${divide_to_int:${.stop_moving_time},${.step_time}}"
    frame_interval: int = "${divide_to_int:${.step_time},0.001}"
    # the "." is necessary in nested configs, see https://github.com/omry/omegaconf/issues/1099#issuecomment-1624496194

    stop_moving_dist: Optional[float] = None

    # GUI
    visualize: bool = False
    viewer_camera_distance: float = 2.4
    viewer_camera_yaw: float = 80.
    viewer_camera_pitch: float = -40.
    viewer_camera_target: Tuple[float] = (0., 0., 0.)
    # DRAW_VIEWER_AXES = True
    show_trajectory: bool = False
    show_camera: bool = False

    verbose: bool = False

    set_human_hand_obj_last_frame: bool = True


    table: TableConfig = field(default_factory=TableConfig)
    # hand: HandConfig = field(default_factory=HandConfig) # the format of nested structured configs must be like this https://omegaconf.readthedocs.io/en/2.3_branch/structured_config.html#nesting-structured-configs
    # object: ObjectConfig = field(default_factory=ObjectConfig)
    hand: HandConfig = field(default_factory=lambda: HandConfig(set_human_hand_obj_last_frame="${..set_human_hand_obj_last_frame}"))
    object: ObjectConfig = field(default_factory=lambda: ObjectConfig(set_human_hand_obj_last_frame="${..set_human_hand_obj_last_frame}"))
    objects: ObjectsConfig = field(default_factory=ObjectsConfig)
    robot: GalbotConfig = field(default_factory=lambda: GalbotConfig(step_time="${..step_time}"))
    # humanoid: HumanoidConfig = field(default_factory=HumanoidConfig)
    humanoid: HumanoidConfig = field(default_factory=lambda:HumanoidConfig(set_human_hand_obj_last_frame="${..set_human_hand_obj_last_frame}"))

    # third_person_camera: CameraConfig = field(default_factory=lambda: CameraConfig(width=1280, height=720, fov=60., near=0.1, far=10.0, pos=(1.5, -0.1, 1.8), target=(0.6, -0.1, 1.3), up_vector=(0., 0., 1.), step_time="${..step_time}"))
    third_person_camera: CameraConfig = field(default_factory=lambda: CameraConfig(width=1280, height=720, fov=90., near=0.1, far=10.0, pos=(2.0, 1.5, 2.0), target=(2.0, -0.1, 1.3), up_vector=(0., 0., 1.), step_time="${..step_time}"))
    status_checker: StatusCheckerConfig = field(default_factory=lambda: StatusCheckerConfig(table_height="${..table_height}", max_frames="${..max_frames}"))
    

class CartesianActionSpace: # not a hard constraint. only for policy learning
    def __init__(self):
        self.high = np.array([ 0.06,  0.06,  0.06,  np.pi/6,  np.pi/6,  np.pi/6]) #, np.pi/10
        self.low  = np.array([-0.06, -0.06, -0.06, -np.pi/6, -np.pi/6, -np.pi/6]) # , -np.pi/3
        self.shape = [6]
        self.bounds = np.vstack([self.low, self.high])

@dataclass
class Observation:
    frame: int
    world_to_ee: NDArray[np.float64]
    joint_positions: NDArray[np.float64]
    get_visual_observation: "MobileH2RSim.get_visual_observation"
    env: "MobileH2RSim"

class MobileH2RSim:
    def __init__(self, cfg: MobileH2RSimConfig):
        self.cfg = cfg

        if self.cfg.visualize:
            self.bullet_client = BulletClient(connection_mode=pybullet.GUI)
            self.bullet_client.resetDebugVisualizerCamera(cameraDistance=self.cfg.viewer_camera_distance, cameraYaw=self.cfg.viewer_camera_yaw, cameraPitch=self.cfg.viewer_camera_pitch, cameraTargetPosition=self.cfg.viewer_camera_target)
            self.bullet_client.rendering_enabled = True
        else:
            self.bullet_client = BulletClient(connection_mode=pybullet.DIRECT)
            self.bullet_client.rendering_enabled = False
        self.bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())

        # self.table = Table(self.bullet_client, cfg.table)
        self.robot = Galbot(self.bullet_client, cfg.robot)
        self.hand = Hand(self.bullet_client, cfg.hand)
        self.objects = Objects(self.bullet_client, cfg.objects, cfg.object)
        self.humanoid = Humanoid(self.bullet_client, cfg.humanoid)
        self.status_checker = StatusChecker(self.bullet_client, cfg.status_checker)
        self.third_person_camera = Camera(self.bullet_client, cfg.third_person_camera)
    
    @contextmanager
    def disable_rendering(self): # speedup setting up scene
        rendering_enabled = self.bullet_client.rendering_enabled
        if rendering_enabled:
            self.bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
            self.bullet_client.rendering_enabled = False
        yield
        if rendering_enabled:
            self.bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
            self.bullet_client.rendering_enabled = True

    def reset(self, scene_id):
        self.bullet_client.resetSimulation() # remove all objects from the scene
        self.bullet_client.setGravity(*self.cfg.gravity)
        self.bullet_client.setPhysicsEngineParameter(fixedTimeStep=self.cfg.step_time, numSubSteps=self.cfg.substeps, deterministicOverlappingPairs=1)

        self.scene_id = scene_id
        self.frame = 0
        self.scene_data = load_scene_data(scene_id, table_height=self.cfg.table_height, stop_moving_frame=self.cfg.stop_moving_frame, frame_interval=self.cfg.frame_interval)
        self.target_object_stopped_because_of_dist = False

        with self.disable_rendering():
            self.ground_id = self.bullet_client.loadURDF("plane_implicit.urdf", basePosition=(0., 0., 0.))
            self.robot.reset()
            # self.table.reset()
            # code.interact(local=dict(globals(), **locals()))

            ### convert mano trans to pybullet trans
            # mano_layer = ManoLayer(mano_root='env/third_party/DexGraspNet/grasp_generation/mano', side=self.scene_data["hand_side"], flat_hand_mean=True, use_pca=False)
            # code.interact(local=dict(globals(), **locals()))
            # _, hand_joints = mano_layer(torch.zeros(1, 48), torch.zeros(1, 10))
            # hand_rot_center = (hand_joints[0, 0]/1000).numpy()          # array([0.09254288, 0.00613023, 0.00574276], dtype=float32)
            # mano_rot_mat = Rt.from_rotvec(self.scene_data["hand_pose"][:, 3:6]).as_matrix()
            # mano_rot_mat = Rt.from_euler("XYZ", self.scene_data["hand_pose"][:, 3:6]).as_matrix()
            # pybullet_trans = self.scene_data["hand_pose"][:, :3] - mano_rot_mat @ hand_rot_center+ hand_rot_center
            # self.scene_data["hand_pose"][:, :3] = pybullet_trans

            if self.scene_data["hand_side"] == "left":
                hand_rot_center = np.array([-0.09566990661621094, 0.006383429050445557, 0.00618630313873291])
            else:
                hand_rot_center = np.array([0.09566990661621094, 0.006383429050445557, 0.00618630313873291])
            # mano_rot_mat = Rt.from_rotvec(self.scene_data["hand_pose"][:, 3:6]).as_matrix()
            # mano_rot_mat = Rt.from_quat(self.scene_data["hand_pose"][:, 3:7]).as_matrix()
            mano_rot_mat = Rt.from_euler("XYZ", self.scene_data["hand_pose"][:, 3:6]).as_matrix()

            pybullet_trans = self.scene_data["hand_pose"][:, :3] - mano_rot_mat @ hand_rot_center + hand_rot_center
            # self.scene_data["hand_pose"][:, :3] = pybullet_trans

            self.hand.reset(self.scene_data["hand_name"], self.scene_data["hand_side"], self.scene_data["hand_path"], self.scene_data["hand_pose"])
            # code.interact(local=dict(globals(), **locals()))

            self.objects.reset(self.scene_data["object_names"], self.scene_data["object_paths"], self.scene_data["object_grasp_id"], self.scene_data["object_poses"])
            self.humanoid.reset(self.scene_data["body_pose"], self.scene_data["body_params"])
            # code.interact(local=dict(globals(), **locals()))
            self.status_checker.reset()

        # bodies for visualization
        self.grasps: List[Grasp] = []
        self.spheres: List[Sphere] = []

        assert self.robot.body_id == 1
        assert self.hand.body_id == 2
        assert self.objects.target_object.body_id == 3
        assert self.humanoid.body_id == 4 
        
    def get_visual_observation(self):
        # code.interact(local=dict(globals(), **locals()))
        return self.robot.get_visual_observation([self.robot.body_id, self.hand.body_id, self.objects.target_object.body_id])

    def get_observation(self) -> Observation:
        # color, depth, segmentation, points = self.robot.get_visual_observation([self.objects.target_object.body_id, self.hand.body_id])
        # "color": color,
        # "depth": depth,
        # "segmentation": segmentation,
        # "object_points": points[0],
        # "hand_points": points[1]
        if self.cfg.show_camera:
            self.get_visual_observation() # to refresh the camera for visualization
        observation = Observation(
            frame=self.frame,
            world_to_ee=self.robot.get_world_to_ee(),
            joint_positions=self.robot.get_joint_positions(),
            get_visual_observation=self.get_visual_observation,
            env=self,
        )
        return observation

    # no disable_rendering because there are always multiple loadings together, so disable_rendering is placed there
    def load_grasp(self, pose_mat: NDArray[np.float64], color: Tuple[float]=[1., 0., 0., 1.]) -> Grasp:
        grasp_cfg: GraspConfig = OmegaConf.to_object(OmegaConf.structured(get_grasp_config(pose_mat=pose_mat, color=color)))
        grasp = Grasp(self.bullet_client, grasp_cfg)
        self.grasps.append(grasp)
        return grasp

    def clear_grasps(self):
        with self.disable_rendering():
            for grasp in self.grasps:
                grasp.clear()
        self.grasps = []

    # no disable_rendering because there are always multiple loadings together, so disable_rendering is placed there
    def load_sphere(self, pos: NDArray[np.float64], color: Tuple[float], scale: float) -> Sphere:
        sphere_cfg: SphereConfig = OmegaConf.to_object(OmegaConf.structured(SphereConfig(base_position=tuple(pos.tolist()), link_color=color, scale=scale)))
        sphere = Sphere(self.bullet_client, sphere_cfg)
        self.spheres.append(sphere)
        return sphere

    def clear_spheres(self):
        with self.disable_rendering():
            for sphere in self.spheres:
                sphere.clear()
        self.spheres = []
    
    def get_panda_object_dist(self) -> float:
        world_to_tip_pos = self.robot.get_tip_pos()
        world_to_object_pc = self.objects.target_object.get_world_to_object_pc()
        dists = np.square(world_to_object_pc-world_to_tip_pos).sum(axis=1)
        min_dist, min_dist_idx = np.sqrt(dists.min()), dists.argmin()
        return min_dist

    def sim_step(self, panda_dof_target_position, increase_frame=True):
        info = {}
        # pre step
        if increase_frame:
            self.frame += 1
        # self.table.pre_step()
        self.robot.pre_step(panda_dof_target_position)
        increase_hand_object_frame = increase_frame
        if self.cfg.stop_moving_dist is not None:
            panda_object_dist = self.get_panda_object_dist()
            if panda_object_dist < self.cfg.stop_moving_dist:
                increase_hand_object_frame = False
            self.target_object_stopped_because_of_dist = panda_object_dist < self.cfg.stop_moving_dist
            if self.cfg.verbose:
                print(f"frame: {self.frame}, panda object dist {panda_object_dist}")
        if increase_hand_object_frame:
            self.hand.pre_step(self.disable_rendering)
            self.objects.pre_step()
            self.humanoid.pre_step()
        # self.status_checker.pre_step()
        # self.third_person_camera.pre_step()
        self.bullet_client.stepSimulation()
        # post step
        # self.table.post_step()
        self.robot.post_step()
        # self.hand.post_step()
        self.objects.post_step()
        status, release = self.status_checker.post_step(self.robot, self.hand, self.objects, self.humanoid, self.frame)
        self.third_person_camera.post_step()
        if release:
            # code.interact(local=dict(globals(), **locals()))
            self.objects.release()
        reward = float(status == EpisodeStatus.SUCCESS)
        done = status != 0
        if self.cfg.verbose and status != 0: print(f"frame {self.frame}, status {status}, reward {reward}, done {done}")
        info["status"] = status
        return reward, done, info
    
    def joint_step(self, panda_dof_target_position, repeat, increase_frame=True):
        " panda_dof_target_position: (9, ) "
        if self.cfg.verbose:
            print(f"in joint_step, frame={self.frame}", end=" ")
            for position in panda_dof_target_position:
                print(position, end=" ")
            print("")
        if self.cfg.show_trajectory:
            with self.disable_rendering():
                self.load_sphere(self.objects.target_object.get_world_to_obj()[:3, 3], color=(1., 0.75, 0., 1.), scale=0.1)
                self.load_sphere(self.robot.get_tip_pos(), color=(1., 0., 0., 1.), scale=0.1)
                self.load_sphere(np.array(self.robot.get_wrist_camera_pos_orn()[0]), color=(0., 1., 0., 1.), scale=0.1)
                # camera_trans, camera_orn = self.robot.get_wrist_camera_pos_orn()
                # camera_orn_matrix = Rt.from_quat(camera_orn).as_matrix()
                # camera_trans = np.array(camera_trans)
                # code.interact(local=dict(globals(), **locals()))
                # self.load_sphere(camera_trans + 0.1 * camera_orn_matrix[:, 0], color=(1., 0., 0., 1.), scale=0.1)
                # self.load_sphere(camera_trans + 0.1 * camera_orn_matrix[:, 1], color=(0., 1., 0., 1.), scale=0.1)
                # self.load_sphere(camera_trans + 0.1 * camera_orn_matrix[:, 2], color=(0.5, 0.5, 0.5, 1.), scale=0.1)
                # self.load_sphere(camera_trans, color=(0., 0., 1., 1.), scale=0.1)
                self.load_grasp(self.robot.get_world_to_ee(), color=(1., 0., 0., 1.))
        for _ in range(repeat):
            reward, done, info = self.sim_step(panda_dof_target_position, increase_frame)
            if done: break
        return reward, done, info
    
    def ego_cartesian_step(self, action, repeat, increase_frame=True):
        " action: (7,) pos+euler+width "
        if self.cfg.verbose:
            print(f"in ego_cartesian_step, frame={self.frame}", end=" ")
            for action_i in action:
                print(action_i, end=" ")
            print("")
        panda_dof_target_position = self.robot.ego_cartesian_action_to_dof_target_position(pos=action[:3], orn=action[3:6], width=action[6:], orn_type="euler")
        reward, done, info = self.joint_step(panda_dof_target_position, repeat, increase_frame)
        return reward, done, info

    def world_pos_step(self, action, repeat):
        " action: (4,) pos+width "
        panda_dof_target_position = self.robot.world_pos_action_to_dof_target_position(pos=action[:3], width=action[3:])
        reward, done, info = self.joint_step(panda_dof_target_position, repeat)
        return reward, done, info

def debug():
    env_base_cfg = OmegaConf.structured(MobileH2RSimConfig)
    cli_cfg = OmegaConf.from_cli()
    env_cfg: MobileH2RSimConfig = OmegaConf.to_object(OmegaConf.merge(env_base_cfg, cli_cfg))
    env_cfg.robot.dof_default_position = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1., .0, .0])
    env = MobileH2RSim(env_cfg)
    env.reset(1000000)

    link_poses = env.robot.get_link_poses()
    world_to_ee = link_poses[31]
    world_to_ee = world_to_ee@np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.05], [0., 0., 0., 1.]])
    env.load_grasp(world_to_ee)
    code.interact(local=dict(globals(), **locals()))

    while True:
        scene_id = int(input("scene_id:"))
        env.reset(scene_id)
        traj_data = np.load(f"/data2/haoran/HRI/expert_demo/OMG/s0/know_dest_smooth_0.08/{scene_id}.npz")
        num_steps = traj_data["num_steps"]
        for step in range(num_steps):
            # env.joint_step(traj_data['joint_state'][step], 130)
            env.ego_cartesian_step(np.append(traj_data["expert_action"][step], 0.04), 130)
            env.get_observation().get_visual_observation()
        for step in range(num_steps-1, -1, -1):
            joint_state = traj_data['joint_state'][step]
            joint_state[7:] = 0
            env.joint_step(joint_state, 130)
            env.get_observation().get_visual_observation()
            # env.cartesian_step(np.append(traj_data["expert_action"][step], 0), 130)

        # ipdb.set_trace()
        # observation = env.get_observation()
        # code.interact(local=dict(globals(), **locals())) # env.panda.get_link_states()
        # while env.frame < 3000:
        #     env.sim_step(cfg.ENV.PANDA_INITIAL_POSITION)
        #     time.sleep(0.001)

if __name__ == "__main__":
    debug()

"""
DISPLAY="localhost:11.0" python -m env.mobile_handover_env verbose=True robot.IK_solver=pybullet visualize=True
pyinstrument -m env.handover_env verbose=True panda.IK_solver=PyKDL visualize=True
scene_id: 10

original:
pyinstrument -m env.handover_env env.verbose True env.panda.IK_solver PyKDL
"""
"""
# run handover-sim for reference
conda activate gaddpg
cd /share/haoran/HRI/handover-sim/examples
python demo_handover_env.py SIM.RENDER True ENV.ID HandoverHandCameraPointStateEnv-v1
"""