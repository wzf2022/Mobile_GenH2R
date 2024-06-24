from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Tuple, Optional

from .base_policy_config import BasePolicyConfig
from .chomp.planner import CHOMPPlannerConfig

from env.utils.hand_collision_filter import HandCollisionFilterConfig

@dataclass
class CHOMPPolicyConfig(BasePolicyConfig):
    name: str = "chomp"
    device: str = "cuda"
    planner: CHOMPPlannerConfig = field(default_factory=lambda: CHOMPPlannerConfig(device="${..device}"))
    base_position: Tuple[float] = MISSING
    base_orientation: Tuple[float] = MISSING
    
    augment_grasps: bool = True
    filter_hand_collision: bool = True
    hand_collision_filter: HandCollisionFilterConfig = field(default_factory=lambda: HandCollisionFilterConfig(device="${..device}"))
    hand_collision_free_mask_dir: Optional[str] = None
    show_filter_results: bool = False

    know_destination: bool = False
    foresee_time: float = 0.65
    replan_period: float = 0.65
    use_endpoints: bool = True
    endpoints_dir: Optional[str] = None
    object_pose_change_threshold: float = 0.01
    show_target_grasp: bool = False

    trajectory_smoothing: bool = True
    trajectory_smoothing_step_size: float = 0.08

    action_type: str = "joint"
