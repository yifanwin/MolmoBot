"""
Run SynthVLA evaluation using molmo_spaces.

This script is the entry point invoked by gantry for evaluation jobs.
It calls run_evaluation from molmo_spaces with the provided arguments.

Usage:
    python launch_scripts/run_eval.py \
        --benchmark_path /path/to/benchmark \
        --eval_config_cls olmo.eval.configure_molmo_spaces:SynthVLAFrankaBenchmarkEvalConfig

For learned policies, also pass ``--checkpoint_path``. Planner policies do not
require a checkpoint.
"""

import argparse
import importlib
import inspect
import logging
import types as _python_types
from pathlib import Path

# --- warp 版本兼容 shim（必须在 import molmo_spaces/curobo 之前执行）---
# curobo 0.7.8 依赖旧版 warp 的 warp.torch 桥接模块（warp 1.1 起已移除），
# 而 mujoco 3.4 内置的 mujoco_warp 后端需要新版 warp。为了在同一进程共存，
# 这里按旧版语义给 warp 补一个 warp.torch：device_from_torch = get_device(str(d))。
import warp as _warp

if not hasattr(_warp, "torch"):
    _warp_torch = _python_types.ModuleType("warp.torch")
    _warp_torch.device_from_torch = (
        lambda torch_device: _warp.get_device(str(torch_device))
    )
    _warp.torch = _warp_torch

import molmo_spaces.data_generation.pipeline as datagen_pipeline
import molmo_spaces.evaluation.eval_main as eval_main
from molmo_spaces.evaluation.eval_main import run_evaluation
from molmo_spaces.evaluation.json_eval_runner import JsonEvalRunner
from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler

# eval_main 顶层会用 policy_cls(exp_config, task_type 字符串) 构造 policy，
# 但 curobo planner policy 的签名是 policy_cls(exp_config, task: BaseMujocoTask)，
# 需要每个 episode 由 runner 创建的真实 task 对象。这里用哨兵短路顶层构造，
# 再让 setup_policy 忽略哨兵，恢复 pipeline 中按 episode 重建 policy 的路径。
_PLANNER_POLICY_SENTINEL = object()
_original_setup_policy = datagen_pipeline.setup_policy


def _planner_aware_setup_policy(exp_config, task, preloaded_policy, datagen_profiler):
    if preloaded_policy is _PLANNER_POLICY_SENTINEL:
        preloaded_policy = None
    return _original_setup_policy(exp_config, task, preloaded_policy, datagen_profiler)


class PolicyAwareJsonEvalTaskSampler(JsonEvalTaskSampler):
    """JSON sampler that also installs assets required by planner policies."""

    def add_auxiliary_objects(self, spec) -> None:
        super().add_auxiliary_objects(spec)
        self.config.policy_config.policy_cls.add_auxiliary_objects(self.config, spec)


class PolicyAwareJsonEvalRunner(JsonEvalRunner):
    """Use the policy-aware sampler for planner-backed JSON evaluations."""

    @staticmethod
    def get_episode_task_sampler(
        exp_config,
        episode_spec,
        shared_task_sampler,
        datagen_profiler,
    ) -> PolicyAwareJsonEvalTaskSampler:
        sampler = PolicyAwareJsonEvalTaskSampler(exp_config, episode_spec)
        if datagen_profiler is not None:
            sampler.set_datagen_profiler(datagen_profiler)
        return sampler

    @staticmethod
    def run_single_rollout(*args, **kwargs) -> bool:
        """Record expected planner/IK failures as failed benchmark episodes.

        The data-generation runner normally treats a planner ``ValueError`` as
        an invalid rollout and omits it from both the denominator and saved
        artifacts. For evaluation, those errors are genuine policy failures.
        """
        try:
            return JsonEvalRunner.run_single_rollout(*args, **kwargs)
        except ValueError:
            logging.getLogger(__name__).exception(
                "CuRobo/IK rollout failed; recording the episode as unsuccessful"
            )
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Run SynthVLA evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the model checkpoint to evaluate (not required for planner policies)",
    )
    parser.add_argument(
        "--benchmark_path",
        type=str,
        required=True,
        help="Path to the benchmark directory",
    )
    parser.add_argument(
        "--eval_config_cls",
        type=str,
        required=True,
        help="Evaluation config class (module:ClassName)",
    )
    parser.add_argument(
        "--task_horizon",
        type=int,
        default=600,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for eval results",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel eval workers",
    )
    parser.add_argument(
        "--episode_idx",
        type=int,
        default=None,
        help="Evaluate only one benchmark episode by its zero-based global index",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="mjthor-online-eval",
        help="WandB project name",
    )
    parser.add_argument(
        "--use_filament",
        action="store_true",
        help="Use filament renderer instead of legacy OpenGL",
    )
    parser.add_argument(
        "--environment_light_intensity",
        type=float,
        default=None,
        help="Intensity of default environmental light (filament only)",
    )
    args = parser.parse_args()

    # Resolve module:ClassName string to actual class so mujoco-thor uses __name__
    # (not the full "module:ClassName" string) when constructing the output directory.
    eval_config_cls = args.eval_config_cls
    if isinstance(eval_config_cls, str) and ":" in eval_config_cls:
        module_path, class_name = eval_config_cls.split(":")
        eval_config_cls = getattr(importlib.import_module(module_path), class_name)

    eval_kwargs = dict(
        eval_config_cls=eval_config_cls,
        benchmark_dir=Path(args.benchmark_path),
        checkpoint_path=Path(args.checkpoint_path) if args.checkpoint_path else None,
        task_horizon_steps=args.task_horizon,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        environment_light_intensity=args.environment_light_intensity,
        episode_idx=args.episode_idx,
    )
    # MolmoSpaces versions before the local evaluation API cleanup expose this
    # optional keyword; the current local checkout does not. Preserve the CLI
    # flag without forcing either version of MolmoSpaces.
    if "use_filament" in inspect.signature(run_evaluation).parameters:
        eval_kwargs["use_filament"] = args.use_filament

    # JsonEvalTaskSampler reproduces benchmark scene objects but, unlike the
    # data-generation samplers, does not ask a planner policy to add its own
    # MuJoCo helper bodies. CuRobo's batched grasp-collision filter needs those
    # bodies (grasp_collision_0, ...). Opt in to a runner that restores that
    # policy hook without modifying the local MolmoSpaces checkout.
    original_runner_cls = eval_main.JsonEvalRunner
    original_setup_policy = datagen_pipeline.setup_policy
    if getattr(eval_config_cls, "requires_policy_auxiliary_objects", False):
        eval_main.JsonEvalRunner = PolicyAwareJsonEvalRunner
        # 见文件顶部 _PLANNER_POLICY_SENTINEL 的说明：跳过 eval_main 顶层的
        # policy 构造（对 planner policy 会因缺少 task 对象而崩溃），改为
        # 在每个 episode 内用真实 task 重建。
        if "preloaded_policy" in inspect.signature(run_evaluation).parameters:
            eval_kwargs["preloaded_policy"] = _PLANNER_POLICY_SENTINEL
            datagen_pipeline.setup_policy = _planner_aware_setup_policy
    try:
        results = run_evaluation(**eval_kwargs)
    finally:
        eval_main.JsonEvalRunner = original_runner_cls
        datagen_pipeline.setup_policy = original_setup_policy

    print(f"Results saved to: {results.output_dir}")
    print(
        f"Success: {results.success_count}/{results.total_count} "
        f"({results.success_rate:.1%})"
    )
    for r in results.episode_results:
        print(f"{r.house_id}/ep{r.episode_idx}: {'pass' if r.success else 'fail'}")


if __name__ == "__main__":
    main()
