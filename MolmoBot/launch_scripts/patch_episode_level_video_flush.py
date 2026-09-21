"""给 molmo_spaces 打内存优化补丁：把视频写盘从 house 级提前到 episode 级。

背景
----
molmo_spaces 默认在【整个 house 的所有 episode 都跑完之后】才调用
``prepare_episode_for_saving``，而此时该 house 全部 episode 的相机帧仍堆在内存里。
相机约占 episode 内存的 80%（4 路 1024x576 约 11.8 MB/帧，200 步即 2.3 GB），
因此 house 级内存峰值 ≈ episode 数 × 2.3 GB。benchmark 中 episode 最多的 house
可达 40 GB 以上峰值，这是 20260916_220518 那次运行 OOM（house_91 起连续 17 个
house 保存失败、丢失 60 条轨迹）的直接原因。

补丁在【每个 episode 结束时】立即写盘视频并转成批量格式，使帧数据不再累积。

影响面
------
不改变 h5 内容、视频文件名与分辨率，也不改变评测结论；只改变数据落盘时机。

用法
----
    .venv/bin/python launch_scripts/patch_episode_level_video_flush.py --check
    .venv/bin/python launch_scripts/patch_episode_level_video_flush.py

uv sync 会重建 venv 从而冲掉本补丁，届时重跑本脚本即可（幂等）。
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------- 配置区
VENV_SITE = Path(
    "/data0/wenyifan/MoMaTrajGen/MolmoBot/MolmoBot/.venv/lib/python3.11/site-packages"
)
SAVE_UTILS = VENV_SITE / "molmo_spaces/utils/save_utils.py"
PIPELINE = VENV_SITE / "molmo_spaces/data_generation/pipeline.py"
TASKS_TASK = VENV_SITE / "molmo_spaces/tasks/task.py"

# 幂等标记：出现即认为补丁已打过
MARKER = "prefetch_episode_for_saving"
# ------------------------------------------------------------ 配置区结束


# ---- 补丁 1：save_utils.py 新增 prefetch_episode_for_saving ----
ANCHOR_1 = """def prepare_episode_for_saving(
    history: dict,
    sensor_suite: SensorSuite,
    fps: float,"""

REPLACE_1 = '''def prefetch_episode_for_saving(
    episode_info: dict,
    save_dir: str,
    fps: float,
    episode_idx: int,
    save_file_suffix: str = "",
) -> None:
    """在【单个 episode 结束时】立即写盘视频并转成批量格式。

    与 house 级批处理的区别：相机帧在该 episode 结束时就落盘并从内存释放，
    而不是等整个 house 的所有 episode 都跑完。相机约占 episode 内存的 80%，
    提前释放可把 house 级内存峰值从 episode 数 x 数 GB 降到数 MB 量级。

    处理结果写入 episode_info["prepared"]，原始 history 置空。
    """
    episode_info["prepared"] = prepare_episode_for_saving(
        episode_info["history"],
        episode_info["sensor_suite"],
        fps=fps,
        save_dir=save_dir,
        episode_idx=episode_idx,
        save_file_suffix=save_file_suffix,
    )
    episode_info["history"] = None


''' + ANCHOR_1


# ---- 补丁 2：pipeline.py 导入新函数 ----
ANCHOR_2 = (
    "from molmo_spaces.utils.save_utils import prepare_episode_for_saving, "
    "save_trajectories"
)

REPLACE_2 = """from molmo_spaces.utils.save_utils import (
    prefetch_episode_for_saving,
    prepare_episode_for_saving,
    save_trajectories,
)"""


# ---- 补丁 3：pipeline.py 的 house 级循环复用已处理数据 ----
ANCHOR_3 = """        house_trajectory_data = []
        for idx, episode_info in enumerate(house_raw_histories):
            prepared_episode = prepare_episode_for_saving(
                episode_info["history"],
                episode_info["sensor_suite"],
                fps=exp_config.fps,
                save_dir=house_output_dir,
                episode_idx=idx,
                save_file_suffix=batch_suffix,
            )
            if prepared_episode is not None:
                house_trajectory_data.append(prepared_episode)
            del episode_info["history"]"""

REPLACE_3 = """        house_trajectory_data = []
        for idx, episode_info in enumerate(house_raw_histories):
            prepared_episode = episode_info.get("prepared")
            if prepared_episode is None:
                # 兼容路径：未在 episode 级提前处理的条目仍按原方式在此处理
                prepared_episode = prepare_episode_for_saving(
                    episode_info["history"],
                    episode_info["sensor_suite"],
                    fps=exp_config.fps,
                    save_dir=house_output_dir,
                    episode_idx=idx,
                    save_file_suffix=batch_suffix,
                )
                del episode_info["history"]
            if prepared_episode is not None:
                house_trajectory_data.append(prepared_episode)"""


# ---- 补丁 4：pipeline.py 在 episode 结束处触发提前落盘 ----
ANCHOR_4 = """                            if should_save:
                                house_raw_histories.append(episode_info)
                            elif should_save_debug:
                                house_debug_raw_histories.append(episode_info)"""

REPLACE_4 = """                            if should_save:
                                house_raw_histories.append(episode_info)
                                # 内存优化：本 episode 的相机帧立即落盘并释放，
                                # 不再累积到 house 结束（相机约占 episode 内存 80%）
                                prefetch_episode_for_saving(
                                    episode_info,
                                    house_output_dir,
                                    exp_config.fps,
                                    len(house_raw_histories) - 1,
                                    batch_suffix,
                                )
                            elif should_save_debug:
                                house_debug_raw_histories.append(episode_info)
                                prefetch_episode_for_saving(
                                    episode_info,
                                    house_debug_dir,
                                    exp_config.fps,
                                    len(house_debug_raw_histories) - 1,
                                    batch_suffix,
                                )"""


# ---- 补丁 5：tasks/task.py 的 close() 清空 observation_cache ----
# get_history() 返回的 observations 是 self.observation_cache 的直接引用（非拷贝），
# 因此 save 侧的 del/pop 只解除 history dict 的引用，帧数据仍被 task 持有。
# 而 close() 原本不清理该缓存，导致相机帧一直驻留到 task 对象被回收。
ANCHOR_5 = """        self._env = None

        if hasattr(self, "renderer") and self.renderer is not None:"""

REPLACE_5 = """        self._env = None

        # 释放观测缓存（含全部相机帧，约占 episode 内存 80%）。
        # get_history() 返回的 observations 是本缓存的直接引用而非拷贝，
        # 若不在此清空，帧数据会一直驻留到 task 对象被回收才释放。
        self.observation_cache = []

        if hasattr(self, "renderer") and self.renderer is not None:"""


# (目标文件, 锚点, 替换文本, 描述, 已应用标记)
# 标记必须是替换后文本里独有的字符串，用于幂等判断。
# 注意：不能拿锚点本身当标记——补丁 1 的替换文本末尾保留了原函数定义，
# 再拿它做判断会导致重复插入。
PATCHES = [
    (
        SAVE_UTILS,
        ANCHOR_1,
        REPLACE_1,
        "save_utils.py: 新增 prefetch_episode_for_saving",
        "def prefetch_episode_for_saving(",
    ),
    (
        PIPELINE,
        ANCHOR_2,
        REPLACE_2,
        "pipeline.py: 导入 prefetch_episode_for_saving",
        "    prefetch_episode_for_saving,\n",
    ),
    (
        PIPELINE,
        ANCHOR_3,
        REPLACE_3,
        "pipeline.py: house 级循环复用已处理数据",
        'prepared_episode = episode_info.get("prepared")',
    ),
    (
        PIPELINE,
        ANCHOR_4,
        REPLACE_4,
        "pipeline.py: episode 结束处触发提前落盘",
        "# 内存优化：本 episode 的相机帧立即落盘并释放，",
    ),
    (
        TASKS_TASK,
        ANCHOR_5,
        REPLACE_5,
        "task.py: close() 清空 observation_cache",
        "# 释放观测缓存（含全部相机帧，约占 episode 内存 80%）。",
    ),
]


def apply_patch(
    path: Path, anchor: str, replacement: str, label: str, marker: str, check_only: bool
) -> bool:
    """用精确字符串替换应用单处补丁。返回 True 表示已应用（或已存在）。"""
    if not path.exists():
        print(f"  [缺失] {path}")
        return False

    text = path.read_text()

    if marker in text:
        print(f"  [已应用] {label}")
        return True

    if anchor not in text:
        print(f"  [锚点未命中] {label}")
        return False

    if check_only:
        print(f"  [待应用] {label}")
        return True

    path.write_text(text.replace(anchor, replacement, 1))
    print(f"  [已应用] {label}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="episode 级视频提前落盘补丁")
    parser.add_argument("--check", action="store_true", help="只检查，不修改")
    args = parser.parse_args()

    print("目标文件:")
    print(f"  {SAVE_UTILS}")
    print(f"  {PIPELINE}")
    print(f"  {TASKS_TASK}")

    already = sum(
        1 for path, _, _, _, marker in PATCHES if path.exists() and marker in path.read_text()
    )
    print(f"\n补丁状态: {already}/{len(PATCHES)} 处已应用")
    if already == len(PATCHES) and args.check:
        return 0

    print("\n应用补丁:" + ("（检查模式，不写入）" if args.check else ""))
    ok = True
    for path, anchor, replacement, label, marker in PATCHES:
        if not apply_patch(path, anchor, replacement, label, marker, args.check):
            ok = False

    if ok and not args.check:
        print("\n完成。校验语法:")
        import py_compile

        for p in {SAVE_UTILS, PIPELINE, TASKS_TASK}:
            py_compile.compile(str(p), doraise=True)
            print(f"  [语法 OK] {p.name}")
        print("\n提示：正在运行的评测进程不受影响，需重启后才生效。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
