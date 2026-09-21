"""LLM waypoint evaluation configs kept separate from legacy MolmoBot configs.

Keeping this module isolated means an older installed MolmoSpaces wheel can
still import ``configure_molmo_spaces`` and run existing CuRobo evaluations.
"""

from typing import ClassVar

from molmo_spaces.configs.policy_configs import LLMWaypointPlannerPolicyConfig
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    RBY1PickAndPlaceDataGenConfig,
)


class MolmoBotRBY1LLMWaypointPickPnPEvalConfig(RBY1PickAndPlaceDataGenConfig):
    """RBY1 pick-and-place evaluation with API-generated, locally checked waypoints."""

    requires_task_bound_policy: ClassVar[bool] = True
    filter_for_successful_trajectories: bool = False
    end_on_success: bool = True
    use_wandb: bool = False
    policy_dt_ms: float = 100.0
    ctrl_dt_ms: float = 20.0
    sim_dt_ms: float = 4.0
    task_horizon: int = 600
    policy_config: LLMWaypointPlannerPolicyConfig | None = None

    def _init_policy_config(self) -> LLMWaypointPlannerPolicyConfig:
        return LLMWaypointPlannerPolicyConfig(
            # The policy replaces this with the selected left/right arm before IK.
            ik_unlocked_move_group_ids=["left_arm"],
            go_home_move_group_ids=[],
        )

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        from molmo_spaces.policy.solvers.object_manipulation.llm_waypoint_planner_policy import (
            validate_llm_environment,
        )

        if self.policy_config is None:
            self.policy_config = self._init_policy_config()
        validate_llm_environment(self.policy_config.api_timeout_s)
        self.robot_config.action_noise_config.enabled = False
