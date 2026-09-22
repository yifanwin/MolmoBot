"""MolmoBot-SPOC RBY1 Rigid pick-and-place 评测入口。

与 `python -m molmo_spaces.evaluation.eval_main` 相比，本脚本修掉了四个本地环境适配问题：

1. **checkpoint 从本地目录加载**：/nas 是 CIFS 网络挂载，不支持符号链接，
   无法承载 HF hub 缓存格式（snapshots/* -> blobs/*），因此用 local_dir 形式的
   本地副本，并把 hf_model_name 置空以跳过 snapshot_download。
2. **渲染分辨率**：benchmark JSON 里记录的 img_resolution 是旧版相机参数（640x480），
   而 SPOC 预处理要求 RBY1GoProD455CameraSystem 的 1024x576，需要覆盖。
3. **transformers 兼容**：venv 里是 transformers 5.x，移除了 T5Tokenizer.batch_encode_plus，
   而 open_clip 3.2.0 仍在调用；SPOC 官方 pin 的是 transformers==4.57.1。
4. **全局 episode 上限**：eval_main 的 --max_episodes 在 worker 路径下会被绕过
   （JsonEvalRunner 会重新全量加载 benchmark），本脚本改在 load_all_episodes 层截断。

用法：
    python launch_scripts/run_spoc_eval.py --num_episodes 5
    python launch_scripts/run_spoc_eval.py --idx 0
"""

import argparse
from pathlib import Path

# ---------------- 可修改的默认值 ----------------
DEFAULT_IMG_RESOLUTION = (1024, 576)  # RBY1GoProD455CameraSystem 的渲染分辨率
DEFAULT_TASK_HORIZON_STEPS = 200
DEFAULT_OUTPUT_DIR = "eval_output/spoc_rby1_rigid"
# ----------------------------------------------

# 全局 episode 上限，由 --num_episodes 设置；None 表示全量
GLOBAL_EPISODE_LIMIT = None

# 修正 3：transformers 5.x 移除了 T5Tokenizer.batch_encode_plus（open_clip 3.2.0 仍在调用）
from transformers import T5Tokenizer

if not hasattr(T5Tokenizer, "batch_encode_plus"):
    T5Tokenizer.batch_encode_plus = T5Tokenizer.__call__

# 先导入 evaluation.eval_main 让 molmo_spaces 包完整初始化，
# 否则直接切入 tasks 模块会触发 evaluation <-> tasks 的循环导入
import molmo_spaces.evaluation.eval_main as eval_main_module
import molmo_spaces.evaluation.json_eval_runner as json_eval_runner_module
from molmo_spaces.evaluation.eval_main import run_evaluation
from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler

# 修正 2：覆盖 benchmark JSON 里旧版相机参数记录的分辨率
_original_build_camera_config = JsonEvalTaskSampler._build_camera_config_from_spec


def _build_camera_config_with_gopro_resolution(self, episode_spec):
    camera_config = _original_build_camera_config(self, episode_spec)
    camera_config.img_resolution = DEFAULT_IMG_RESOLUTION
    return camera_config


JsonEvalTaskSampler._build_camera_config_from_spec = _build_camera_config_with_gopro_resolution

# 修正 4：在加载层截断 episode 数量。
# eval_main 与 json_eval_runner 都是 `from ... import load_all_episodes` 的形式，
# 各自持有独立引用，必须分别替换两个模块的属性才会全部生效。
_original_load_all_episodes = eval_main_module.load_all_episodes


def _load_all_episodes_limited(benchmark_dir):
    episodes = _original_load_all_episodes(benchmark_dir)
    if GLOBAL_EPISODE_LIMIT is not None:
        episodes = episodes[:GLOBAL_EPISODE_LIMIT]
    return episodes


eval_main_module.load_all_episodes = _load_all_episodes_limited
json_eval_runner_module.load_all_episodes = _load_all_episodes_limited

from molmobot_spoc.eval.config.rby1_eval_config import RBY1RigidManipEvalConfig
from molmobot_spoc.eval.config.spoc_policy_configs import SPOCRBY1RigidManipPolicyConfig


class RBY1RigidLocalEvalConfig(RBY1RigidManipEvalConfig):
    """RBY1 Rigid 评测配置：权重从本地目录加载，不触发联网下载。"""

    hf_model_name: str | None = None  # 置空以跳过 snapshot_download，权重由 --checkpoint_dir 指定
    policy_config: SPOCRBY1RigidManipPolicyConfig = SPOCRBY1RigidManipPolicyConfig()


def get_args():
    parser = argparse.ArgumentParser(
        description="Run MolmoBot-SPOC RBY1 Rigid evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark_dir", type=str, required=True, help="JSON benchmark 目录"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="SPOC 权重目录，需含 model.safetensors 与 preprocessor_config.json",
    )
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--task_horizon_steps",
        type=int,
        default=DEFAULT_TASK_HORIZON_STEPS,
        help="每个 episode 的最大步数",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="只评测 benchmark 中的前 N 个 episode；None 表示全量",
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--idx", type=int, default=None, help="只评测该全局下标对应的 episode")
    parser.add_argument("--use_wandb", action="store_true", help="启用 wandb 记录")
    return parser.parse_args()


def main():
    global GLOBAL_EPISODE_LIMIT

    args = get_args()
    benchmark_dir = Path(args.benchmark_dir).resolve()
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")

    GLOBAL_EPISODE_LIMIT = args.num_episodes

    # pydantic 在实例化时会深拷贝字段默认值，因此改默认值对象即可让 run_evaluation
    # 内部新建的 exp_config 采用命令行传入的权重目录
    RBY1RigidLocalEvalConfig.model_fields["policy_config"].default.checkpoint_dir = (
        args.checkpoint_dir
    )

    results = run_evaluation(
        eval_config_cls=RBY1RigidLocalEvalConfig,
        benchmark_dir=benchmark_dir,
        task_horizon_steps=args.task_horizon_steps,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        use_wandb=args.use_wandb,
        episode_idx=args.idx,
    )

    print(f"结果: {results.success_count}/{results.total_count}")
    print(f"成功率: {results.success_rate:.1%}")
    print(f"输出目录: {results.output_dir}")


if __name__ == "__main__":
    main()
