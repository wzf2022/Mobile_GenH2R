from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Optional, List

# from env.handover_env import GenH2RSimConfig
from env.mobile_handover_env import MobileH2RSimConfig
from policies.offline_policy_config import OfflinePolicyConfig
from policies.omg_planner_policy_config import OMGPlannerPolicyConfig
from policies.pointnet2_policy_config import PointNet2PolicyConfig
from policies.chomp_policy_config import CHOMPPolicyConfig

policy_default_kwargs = {
    "init_joint": "${..env.robot.dof_default_position}",
    "step_time": "${..env.step_time}",
    "goal_center": "${..env.status_checker.goal_center}",
}

@dataclass
class EvaluateConfig:
    use_ray: bool = True
    num_runners: int = 32
    setup: str = "s0"
    split: str = "test"
    scene_ids: Optional[List[int]] = None
    start_object_idx: Optional[int] = None
    end_object_idx: Optional[int] = None
    start_traj_idx: Optional[int] = None
    end_traj_idx: Optional[int] = None
    seed: int = 0

    # config for multiple seeds
    start_seed: Optional[int] = None
    end_seed: Optional[int] = None
    step_seed: Optional[int] = None

    policy: str = MISSING
    demo_dir: Optional[str] = None
    demo_structure: str = "hierarchical" # "flat"
    overwrite_demo: bool = False
    record_ego_video: bool = False
    record_third_person_video: bool = False
    dart: bool = False
    dart_min_step: int = 0
    dart_max_step: int = 30 # max is 30
    dart_ratio: float = 0.5
    print_failure_ids: bool = False
    save_state: bool = False
    show_target_grasp: bool = False
    verbose: bool = False
    mask_robot_pc: bool = True
    
    env: MobileH2RSimConfig = field(default_factory=MobileH2RSimConfig)
    offline: OfflinePolicyConfig = field(default_factory=lambda: OfflinePolicyConfig(**policy_default_kwargs))
    omg_planner: OMGPlannerPolicyConfig = field(default_factory=lambda: OMGPlannerPolicyConfig(
        base_position="${..env.robot.base_position}", 
        base_orientation="${..env.robot.base_orientation}", 
        stop_moving_dist="${..env.stop_moving_dist}", 
        stop_moving_frame="${..env.stop_moving_frame}", 
        **policy_default_kwargs
    ))
    pointnet2: PointNet2PolicyConfig = field(default_factory=lambda: PointNet2PolicyConfig(**policy_default_kwargs))
    chomp: CHOMPPolicyConfig = field(default_factory=lambda: CHOMPPolicyConfig(
        base_position="${..env.robot.base_position}", 
        base_orientation="${..env.robot.base_orientation}", 
        **policy_default_kwargs
    ))
