from dataclasses import dataclass, field
from typing import Optional, Tuple
from omegaconf import MISSING

from .base_policy_config import BasePolicyConfig
from env.utils.robot_kinematics import RobotKinematicsConfig

@dataclass
class OMGPlannerPolicyConfig(BasePolicyConfig):
    name: str = "omg_planner"
    base_position: Tuple[float] = MISSING
    base_orientation: Tuple[float] = MISSING
    stop_moving_dist: Optional[float] = MISSING
    stop_moving_frame: Optional[int] = MISSING

    IK_solver_max_iter: int = 100
    IK_solver_eps: float = 1e-6

    robot_kinematics: RobotKinematicsConfig = field(default_factory=lambda: RobotKinematicsConfig(
        urdf_file="env/data/assets/franka_panda/panda_gripper_hand_camera.urdf",
        IK_solver_max_iter="${..IK_solver_max_iter}", 
        IK_solver_eps="${..IK_solver_eps}",
        chain_tip="panda_hand",
    ))

    SIMULTANEOUS: bool = False
    KNOW_DESTINATION: bool = False
    KNOW_ORIENTATION: bool = False
    FILTER_HAND_COLLISION_GRASPS: bool = False
    SHOW_GRASPS: bool = False
    REPLAN_PERIOD: int = 5
    TIMESTEPS: int = 30
    SHOW_TRAJECTORY: bool = False
    ONE_TRIAL: bool = False
    USE_RAW_TARGET_OBJ_POSE: bool = False
    SAVE_GRASPS_NO_CONTACT_MASK: bool = False
    SHOW_FILTERING: bool = False
    FORESEE_FRAMES: int = 0
    ONLY_NEAREST_GRASP: bool = False
    TRAJECTORY_SMOOTHING: bool = False
    TRAJECTORY_SMOOTHING_STEP_SIZE: float = 0.1
    ENFORCE_REPLAN: bool = False
    GRASP_DIR: str = "/share/haoran/HRI/OMG-Planner/data/grasps/simulated"
    ONLY_FILTER_ONCE: bool = False
    LOAD_GRASPS_NO_CONTACT_MASK: bool = False
    USE_REAL_POSE_WHEN_CLOSE: bool = False
    OBJ_POSE_CHANGE_THRES: float = 0.01
    LANDMARKS_DIR: Optional[str] = None
    SHOW_LANDMARKS: bool = False
    REPLAN_WHEN_OBJ_STOP: bool = False
    SLOW_DOWN_RATIO: float = 1.0 # ratio
    SLOW_DOWN_STEPS: int = 0
    SLOW_DOWN_FINAL_STEPS: int = 0
    ACTION_TYPE: str = "joint" # "cartesian"
