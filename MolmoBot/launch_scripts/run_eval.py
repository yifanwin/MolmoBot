"""
Run SynthVLA evaluation using molmo_spaces.

This script is the entry point invoked by gantry for evaluation jobs.
It calls run_evaluation from molmo_spaces with the provided arguments.

Usage:
    python launch_scripts/run_eval.py \
        --benchmark_path /path/to/benchmark \
        --eval_config_cls olmo.eval.configure_molmo_spaces:SynthVLAFrankaBenchmarkOriginalEvalConfig

本入口仅用于 MolmoBot learned policy；planner 请使用 molmo_spaces.evaluation.eval_main。
权重可通过 --checkpoint_path 或 policy config 指定。
"""

import argparse
import importlib
import inspect
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MOLMOBOT_ROOT = Path(__file__).resolve().parents[1]

# Executing ``python launch_scripts/run_eval.py`` places only the
# ``launch_scripts`` directory on sys.path. Dynamic imports of ``olmo.*`` must
# also work when MolmoBot has not been installed as an editable package.
if str(_MOLMOBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_MOLMOBOT_ROOT))


def _load_env_file(path: Path) -> None:
    """Load a simple dotenv file without logging values or adding a dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


from molmo_spaces.evaluation.eval_main import run_evaluation


def main():
    repo_env = _REPO_ROOT / ".env"
    _load_env_file(repo_env)
    parser = argparse.ArgumentParser(
        description="Run SynthVLA evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the model checkpoint to evaluate (or set it in the policy config)",
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
        "--house_index",
        type=int,
        default=None,
        help="Evaluate only episodes belonging to this benchmark house index",
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

    from olmo.eval.configure_molmo_spaces import SynthVLAPolicyConfig, SynthVLARBY1PolicyConfig

    policy_field = getattr(eval_config_cls, "model_fields", {}).get("policy_config")
    policy_config = policy_field.get_default(call_default_factory=True) if policy_field else None
    if not isinstance(policy_config, (SynthVLAPolicyConfig, SynthVLARBY1PolicyConfig)):
        parser.error("本入口仅支持 MolmoBot learned policy；planner/第三方策略请使用 molmo_spaces.evaluation.eval_main")

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
    if "house_index" in inspect.signature(run_evaluation).parameters:
        eval_kwargs["house_index"] = args.house_index
    elif args.house_index is not None:
        raise RuntimeError(
            "The active molmo_spaces installation does not support --house_index. "
            "Install the sibling checkout in editable mode or run with its environment."
        )
    # MolmoSpaces versions before the local evaluation API cleanup expose this
    # optional keyword; the current local checkout does not. Preserve the CLI
    # flag without forcing either version of MolmoSpaces.
    if "use_filament" in inspect.signature(run_evaluation).parameters:
        eval_kwargs["use_filament"] = args.use_filament

    results = run_evaluation(**eval_kwargs)

    print(f"Results saved to: {results.output_dir}")
    print(
        f"Success: {results.success_count}/{results.total_count} "
        f"({results.success_rate:.1%})"
    )
    for r in results.episode_results:
        print(f"{r.house_id}/ep{r.episode_idx}: {'pass' if r.success else 'fail'}")


if __name__ == "__main__":
    main()
