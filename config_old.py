import argparse
import numpy as np
from typing import List, Tuple
from yacs.config import CfgNode as CN
import os

_C = CN()

_C.env = CN()
_C.env.gravity = (0.0, 0.0, -9.8)
_C.env.substeps = 1
# _C.env.INIT_VIEWER_CAMERA_POSITION = (None, None, None)
# _C.env.INIT_VIEWER_CAMERA_TARGET = (None, None, None)
# _C.env.DRAW_VIEWER_AXES = True
_C.env.step_time = 0.001
_C.env.max_time = 13.0
_C.env.table_height = 0.92
_C.env.stop_moving_dist = None
_C.env.stop_moving_frame = None

_C.env.goal_center = (0.61, -0.20, 1.25)
_C.env.goal_radius = 0.15
# _C.env.draw_goal = False
_C.env.goal_color = (0.85, 0.19, 0.21, 0.5)
_C.env.success_time_thresh = 0.1

_C.env.contact_force_thresh = 0.0

_C.env.release_force_thresh = 0.0
_C.env.release_time_thresh = 0.1
_C.env.release_contact_region_range_x = (-0.0110, +0.0110)
_C.env.release_contact_region_range_y = (-0.0090, +0.0090)
_C.env.release_contact_region_range_z = (+0.0000, +0.0550)
# _C.env.draw_release_contact = False
# _C.env.release_contact_region_color = (0.85, 0.19, 0.21, 0.5)
# _C.env.release_contact_vertex_radius = 0.001
# _C.env.release_contact_vertex_color = (0.85, 0.19, 0.21, 1.0)

_C.env.visualize = False
_C.env.verbose = False
_C.env.show_trajectory: bool = False

_C.env.table = CN()
_C.env.table.base_position = (0.61, 0.28, 0.0)
_C.env.table.base_orientation = (0, 0, 0, 1)
_C.env.table.collision_mask = 2**0

_C.env.panda = CN()
_C.env.panda.base_position = (0.61, -0.50, 0.875)
_C.env.panda.base_orientation = (0.0, 0.0, 0.7071068, 0.7071068)
_C.env.panda.initial_dof_position = (0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04)
_C.env.panda.max_force = (250.0,) * 9
_C.env.panda.position_gain = (0.01,) * 9
_C.env.panda.velocity_gain = (1.0,) * 9
_C.env.panda.IK_solver = "PyKDL" # "pybullet"
_C.env.panda.IK_solver_max_iter = 100
_C.env.panda.IK_solver_eps = 1e-6
_C.env.panda.collision_mask = -1

_C.env.object = CN()
_C.env.object.translation_max_force = (50.0,) * 3
_C.env.object.translation_position_gain = (0.2,) * 3
_C.env.object.translation_velocity_gain = (1.0,) * 3
_C.env.object.rotation_max_force = (5.0,) * 3
_C.env.object.rotation_position_gain = (0.2,) * 3
_C.env.object.rotation_velocity_gain = (1.0,) * 3
_C.env.object.collision_mask = [2**2, 2**3, 2**4, 2**5, 2**6, 2**7] # allow 6 objects
_C.env.object.collision_mask_release = -1-2**1

_C.env.hand = CN()
_C.env.hand.translation_max_force = (50.0,) * 3
_C.env.hand.translation_position_gain = (0.2,) * 3
_C.env.hand.translation_velocity_gain = (1.0,) * 3
_C.env.hand.rotation_max_force = (5.0,) * 3
_C.env.hand.rotation_position_gain = (0.2,) * 3
_C.env.hand.rotation_velocity_gain = (1.0,) * 3
_C.env.hand.joint_max_force = (0.5,) * 45
_C.env.hand.joint_position_gain = (0.1,) * 45
_C.env.hand.joint_velocity_gain = (1.0,) * 45
_C.env.hand.collision_mask = 2**1

_C.env.third_person_camera = CN()
_C.env.third_person_camera.width: int = 1280
_C.env.third_person_camera.height: int = 720
_C.env.third_person_camera.vertical_fov: float = 60.0
_C.env.third_person_camera.near: float = 0.1
_C.env.third_person_camera.far: float = 10.0
_C.env.third_person_camera.position = (1.5, -0.1, 1.8) # (1.2127, -0.5531, 1.4090)
_C.env.third_person_camera.target = (0.6, -0.1, 1.3) # (0.08, 0.24, 1.19)

_C.policy = CN()
_C.policy.name = "offline"
_C.policy.wait_time = 0.
_C.policy.action_repeat_time = 0.13
_C.policy.close_gripper_time = 0.5
_C.policy.back_step_size = 0.03
_C.policy.seed = 0
_C.policy.demo_dir = None
_C.policy.demo_structure = "hierarchical" # "flat"
_C.policy.record_ego_video: bool = False
_C.policy.record_third_person_video: bool = False
_C.policy.dart: bool = False
_C.policy.dart_min_step: int = 0
_C.policy.dart_max_step: int = 30 # max is 30
_C.policy.dart_ratio: float = 0.5
_C.policy.save_state: bool = False
_C.policy.show_target_grasp: bool = False
_C.policy.verbose: bool = False

_C.policy.offline = CN()
_C.policy.offline.demo_dir = ""
_C.policy.offline.demo_structure = "hierarchical" # "flat"
_C.policy.offline.demo_source = "genh2r" # "handoversim"

_C.policy.OMG_ORIGINAL = CN()
_C.policy.OMG_ORIGINAL.SIMULTANEOUS = False
_C.policy.OMG_ORIGINAL.KNOW_DESTINATION = False
_C.policy.OMG_ORIGINAL.KNOW_ORIENTATION = False
_C.policy.OMG_ORIGINAL.FILTER_HAND_COLLISION_GRASPS = False
_C.policy.OMG_ORIGINAL.SHOW_GRASPS = False
_C.policy.OMG_ORIGINAL.REPLAN_PERIOD = 5
_C.policy.OMG_ORIGINAL.TIMESTEPS = 30
_C.policy.OMG_ORIGINAL.SHOW_TRAJECTORY = False
_C.policy.OMG_ORIGINAL.ONE_TRIAL = False
_C.policy.OMG_ORIGINAL.USE_RAW_TARGET_OBJ_POSE = False
_C.policy.OMG_ORIGINAL.TIME_ACTION_REPEAT = 0.13
_C.policy.OMG_ORIGINAL.TIME_CLOSE_GRIPPER = 0.5
_C.policy.OMG_ORIGINAL.BACK_STEP_SIZE = 0.03
_C.policy.OMG_ORIGINAL.SAVE_GRASPS_NO_CONTACT_MASK = False
_C.policy.OMG_ORIGINAL.SEED = 0
_C.policy.OMG_ORIGINAL.SHOW_FILTERING = False
_C.policy.OMG_ORIGINAL.FORESEE_FRAMES = 0
_C.policy.OMG_ORIGINAL.ONLY_NEAREST_GRASP = False
_C.policy.OMG_ORIGINAL.TRAJECTORY_SMOOTHING = False
_C.policy.OMG_ORIGINAL.TRAJECTORY_SMOOTHING_STEP_SIZE = 0.1
_C.policy.OMG_ORIGINAL.ENFORCE_REPLAN = False
_C.policy.OMG_ORIGINAL.GRASP_DIR = "/share/haoran/HRI/OMG-Planner/data/grasps/simulated"
_C.policy.OMG_ORIGINAL.ONLY_FILTER_ONCE = False
_C.policy.OMG_ORIGINAL.LOAD_GRASPS_NO_CONTACT_MASK = False
_C.policy.OMG_ORIGINAL.USE_REAL_POSE_WHEN_CLOSE = False
_C.policy.OMG_ORIGINAL.OBJ_POSE_CHANGE_THRES = 0.01
_C.policy.OMG_ORIGINAL.LANDMARKS_DIR = None
_C.policy.OMG_ORIGINAL.SHOW_LANDMARKS = False
_C.policy.OMG_ORIGINAL.REPLAN_WHEN_OBJ_STOP = False
_C.policy.OMG_ORIGINAL.SLOW_DOWN_RATIO = 1.0 # ratio
_C.policy.OMG_ORIGINAL.SLOW_DOWN_STEPS = 0
_C.policy.OMG_ORIGINAL.SLOW_DOWN_FINAL_STEPS = 0
_C.policy.OMG_ORIGINAL.ACTION_TYPE = "joint" # "cartesian"

_C.policy.cartesian = CN()
_C.policy.cartesian.trans_close_threshold = 0.02
_C.policy.cartesian.rot_close_threshold = 0.08
_C.policy.cartesian.rot_weight = 0.25
_C.policy.cartesian.trans_step_size = 0.05
_C.policy.cartesian.rot_step_size = 0.2
_C.policy.cartesian.step_size_eps_ratio = 0.1
_C.policy.cartesian.wait_time = 0.0
_C.policy.cartesian.time_close_gripper = 0.5
_C.policy.cartesian.show_grasps = False
_C.policy.cartesian.standoff_dist = 0.05
_C.policy.cartesian.staged = False
_C.policy.cartesian.stay_dist_max = 0.2
_C.policy.cartesian.stay_dist_ratio = 0.2
_C.policy.cartesian.stay_hinge_steps = 2.0
_C.policy.cartesian.foresee_frames = 0
_C.policy.cartesian.load_collision_mask = False
_C.policy.cartesian.grasp_dir = "/share/haoran/HRI/OMG-Planner/data/grasps/simulated"
_C.policy.cartesian.show_filtering = False
_C.policy.cartesian.grasp_dist_measure = "steps"
_C.policy.cartesian.change_target_thres = 0.0
_C.policy.cartesian.check_object_stopped = False
_C.policy.cartesian.use_stop_heuristics = False
_C.policy.cartesian.only_filter_once = False
_C.policy.cartesian.target_grasp_for_current_obj_pose = False
_C.policy.cartesian.always_towards_cur_target = False
_C.policy.cartesian.landmarks_dir = None
_C.policy.cartesian.verbose = False

_C.policy.pointnet2 = CN()
_C.policy.pointnet2.num_points: int = 1024
_C.policy.pointnet2.pretrained_dir: str = None
_C.policy.pointnet2.pretrained_suffix: str = "latest"
_C.policy.pointnet2.pretrained_source: str = "genh2r" # "handoversim"
_C.policy.pointnet2.use_hand: bool = True
_C.policy.pointnet2.flow_frame_num: int = 0
_C.policy.pointnet2.state_dim: int = 512
_C.policy.pointnet2.hidden_dim: int = 256
_C.policy.pointnet2.goal_pred: bool = True
_C.policy.pointnet2.obj_pose_pred_frame_num: int = 0

_C.evaluate = CN()
_C.evaluate.use_ray = True
_C.evaluate.num_runners = 32
_C.evaluate.setup = "s0"
_C.evaluate.split = "test"
_C.evaluate.scene_ids = None
# _C.BENCHMARK.POLICY_SETTING = ""
# _C.BENCHMARK.RAND_ORDER = False
# _C.BENCHMARK.START_SCENE_ID = None

_C.train = CN()
_C.train.run_dir: str = "data/models/tmp"
_C.train.save_model_period: int = 10000

_C.train.data = CN()
_C.train.data.demo_dir: str = None
_C.train.data.demo_structure: str = "hierarchical" # "flat"
_C.train.data.seed: int = 0
_C.train.data.batch_size: int = 256

_C.train.imitation = CN()
_C.train.imitation.debug: bool = False
_C.train.imitation.num_iters: int = 80000
_C.train.imitation.lr: float = 1e-3
_C.train.imitation.weight_decay: float = 0.
_C.train.imitation.milestones: Tuple[float] = (8000, 16000, 30000, 50000, 70000, 90000)
_C.train.imitation.gamma: float = 0.3

# _C.DDPG = CN()
# _C.DDPG.SIMULTANEOUS = False
# _C.DDPG.PRETRAINED = None
# _C.DDPG.TIME_ACTION_REPEAT = 0.15
# _C.DDPG.TIME_CLOSE_GRIPPER = 0.5
# _C.DDPG.SHOW_PRED_GRASPS = False

# _C.DIFFUSION = CN()
# _C.DIFFUSION.PRETRAINED = None
# _C.DIFFUSION.action_dim = 6
# _C.DIFFUSION.obs_horizon = 4
# _C.DIFFUSION.pred_horizon = 4
# _C.DIFFUSION.obs_dim = 512
# _C.DIFFUSION.add_joint_state = False


# _C.TRAIN_DDPG = CN()
# _C.TRAIN_DDPG.debug = False
# # _C.TRAIN_DDPG.env_name = "PandaYCBEnv"
# _C.TRAIN_DDPG.policy = "SAC"
# _C.TRAIN_DDPG.seed = 233
# _C.TRAIN_DDPG.save_model = False
# _C.TRAIN_DDPG.pretrained = None
# _C.TRAIN_DDPG.load_demo_state_feat = False
# _C.TRAIN_DDPG.max_epoch = 80000
# _C.TRAIN_DDPG.batch_size = 128
# _C.TRAIN_DDPG.num_remotes = 0
# _C.TRAIN_DDPG.rl_memory_size = 0
# # _C.TRAIN_DDPG.render = False
# _C.TRAIN_DDPG.index_file = None
# _C.TRAIN_DDPG.log = False
# _C.TRAIN_DDPG.model_surfix = "latest"
# _C.TRAIN_DDPG.save_buffer = False 
# _C.TRAIN_DDPG.save_online_buffer = False
# _C.TRAIN_DDPG.finetune = False
# _C.TRAIN_DDPG.config_file = None
# _C.TRAIN_DDPG.visdom = False
# # _C.TRAIN_DDPG.max_load_scene_num = -1
# _C.TRAIN_DDPG.load_buffer = False
# _C.TRAIN_DDPG.load_online_buffer = False
# _C.TRAIN_DDPG.fix_output_time = None
# # _C.TRAIN_DDPG.save_scene = False
# # _C.TRAIN_DDPG.load_scene = False
# # _C.TRAIN_DDPG.pretrained_policy_name = "BC"
# _C.TRAIN_DDPG.use_time = True
# _C.TRAIN_DDPG.use_hand_points = False
# _C.TRAIN_DDPG.hand_points_ratio = None
# _C.TRAIN_DDPG.accumulate_points = True
# _C.TRAIN_DDPG.use_flow_feature = False
# _C.TRAIN_DDPG.pre_flow_num = 0
# _C.TRAIN_DDPG.egocentric_flow = True
# _C.TRAIN_DDPG.simultaneous = False
# _C.TRAIN_DDPG.expert_know_dest = False
# _C.TRAIN_DDPG.expert_traj_len = 30
# _C.TRAIN_DDPG.time_action_repeat = 0.13
# _C.TRAIN_DDPG.time_close_gripper = 0.5
# _C.TRAIN_DDPG.back_step_size = 0.03
# _C.TRAIN_DDPG.load_traj_file = None
# _C.TRAIN_DDPG.fill_data_step = 10
# _C.TRAIN_DDPG.test = False
# _C.TRAIN_DDPG.on_policy = True
# _C.TRAIN_DDPG.test_period = 10000
# _C.TRAIN_DDPG.pointnet_nclusters = None
# _C.TRAIN_DDPG.pretrained_encoder_path = None
# _C.TRAIN_DDPG.train_feature = True
# _C.TRAIN_DDPG.fetch_frame = 1
# _C.TRAIN_DDPG.architecture = "concat"
# _C.TRAIN_DDPG.occupy = False
# _C.TRAIN_DDPG.pred_obj_pose_frame_num = 0
# _C.TRAIN_DDPG.pred_loss_type = "ef_delta"
# _C.TRAIN_DDPG.ignore_hand_label = False
# _C.TRAIN_DDPG.wd = 0.0
# _C.TRAIN_DDPG.pred_coff = 1.0

# _C.TEST = CN()
# _C.TEST.setting = None
# _C.TEST.split = None
# _C.TEST.show_pred_grasps = False
# _C.TEST.record_video = False
# _C.TEST.record_ego_video = False

# _C.GEN_DEMO = CN()
# _C.GEN_DEMO.PLANNER = "OMG"
# _C.GEN_DEMO.DATA_DIR = None
# _C.GEN_DEMO.NUM_REMOTES = 1
# _C.GEN_DEMO.SEED = 233
# _C.GEN_DEMO.TEST = False
# _C.GEN_DEMO.WAIT_FRAMES = None
# _C.GEN_DEMO.GET_RESULT = False
# _C.GEN_DEMO.USE_RAW_JOINT_ACTION = False
# _C.GEN_DEMO.DART = False
# _C.GEN_DEMO.DART_MIN_STEP = 0
# _C.GEN_DEMO.DART_MAX_STEP = 30 # max is 30
# _C.GEN_DEMO.DART_RATIO = 0.5
# _C.GEN_DEMO.DART_PERTURB = True
# _C.GEN_DEMO.RAND_INIT = False
# _C.GEN_DEMO.LOOK_AT_REGION_CENTER = (0.61, 0.28, 1) # TABLE_HEIGHT=0.92
# _C.GEN_DEMO.LOOK_AT_REGION_LENGTH = 0.1
# _C.GEN_DEMO.CAMERA_DIST_RANGE = (0.7, 0.8)
# _C.GEN_DEMO.PHI_RANGE = (0, np.pi/4)
# _C.GEN_DEMO.BACK_STEP_SIZE = 0.03
# _C.GEN_DEMO.TIME_ACTION_REPEAT = 0.13
# _C.GEN_DEMO.TIME_CLOSE_GRIPPER = 0.2
# _C.GEN_DEMO.SIMULTANEOUS = False
# _C.GEN_DEMO.ASYNCHRONOUS = False
# _C.GEN_DEMO.SAVE_VIDEO = False
# _C.GEN_DEMO.DISABLE_RAY = False
# _C.GEN_DEMO.START_IDX = None
# _C.GEN_DEMO.END_IDX = None

def get_config_from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-file", help="path to config file")
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="modify config options at the end of the command; use space-separated 'PATH.KEY VALUE' pairs;"
    )
    args = parser.parse_args()
    cfg = _C.clone()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    return cfg

def are_configs_different(config1, config2):
    if set(config1.keys()) != set(config2.keys()):
        return True
    for key in config1.keys():
        value1, value2 = config1[key], config2[key]
        if isinstance(value1, CN) and isinstance(value2, CN):
            if are_configs_different(value1, value2):
                return True
        elif isinstance(value1, list) or isinstance(value1, tuple):
            for x, y in zip(value1, value2):
                if x != y:
                    print(f"key {key} different with {value1} and {value2}")
                    return True
        else:
            if value1 != value2:
                print(f"key {key} different with {value1} and {value2}")
                return True
    return False

def merge_b_to_a(config1, config2):
    for key in config2.keys():
        if key not in config1.keys():
            config1[key] = config2[key]
        elif isinstance(config1[key], CN):
            assert isinstance(config2[key], CN)
            merge_b_to_a(config1[key], config2[key])

def cfg_from_file_handover(filename):
    cfg_handover = CN.load_cfg(open(filename, "r"))
    merge_b_to_a(cfg_handover, _C)
    return cfg_handover