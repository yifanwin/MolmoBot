"""预拉取 procthor-10k-val 场景包到本地缓存并建立符号链接。

背景：E2 实验（MolmoBotRBY1CuroboPickPnPEvalConfig/20260917_103152）运行时，
MolmoBot 仓库配置的 procthor-10k-val 版本为 20251217，该版本的本地缓存
只下到 120 个包就中断，导致 528/648 个 house 因场景包缺失被整批跳过。

本脚本离线补齐缺失的包（下载 + 解压 + per-file 符号链接），
避免评测过程中"边跑边下载"再次中途失败。

用法：
    .venv/bin/python prefetch_procthor_val.py --limit 5    # 试跑 5 个
    .venv/bin/python prefetch_procthor_val.py --dry-run    # 只统计不下载
    .venv/bin/python prefetch_procthor_val.py              # 全量补齐
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------- 配置区
# 需要补齐的目标实验日志（用于确定 benchmark 实际用到哪些 house）
TARGET_LOG = (
    "/data0/wenyifan/MoMaTrajGen/MolmoBot/MolmoBot/eval_output/"
    "MolmoBotRBY1CuroboPickPnPEvalConfig/20260917_103152/running_log.log"
)

DATA_TYPE = "scenes"
SOURCE = "procthor-10k-val"

# 每批处理的包数。批越小越温和，避免像原实验那样大批量并发触发限流。
BATCH_SIZE = 40
# 单批内的最大并发下载线程数。
MAX_WORKERS = 6
# 每个 worker 最少处理的包数（沿用框架默认值）
MIN_ITEMS_PER_WORKER = 10
# ------------------------------------------------------------ 配置区结束


def parse_houses(log_path: str) -> set[int]:
    """从日志中提取本次评测实际使用的 house 编号集合。"""
    houses: set[int] = set()
    with open(log_path, errors="ignore") as f:
        for line in f:
            m = re.search(r"starting house (\d+) \(index \d+/\d+\)", line)
            if m:
                houses.add(int(m.group(1)))
    return houses


def needed_packages(houses: set[int]) -> set[str]:
    """house 编号 -> 需要的归档包名。"""
    return {f"procthor-10k-val_val_{h}.tar.zst" for h in houses}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 个包（试跑用）")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不下载")
    args = parser.parse_args()

    from molmo_spaces.molmo_spaces_constants import get_resource_manager
    from molmospaces_resources.threading_utils import _parallel_extract

    manager = get_resource_manager()
    cache_dest = manager.cache_path(DATA_TYPE, SOURCE)
    relative_path = manager.relative_path(DATA_TYPE, SOURCE)

    print(f"缓存目录: {cache_dest}")
    print(f"符号链接目录: {manager.symlink_dir / DATA_TYPE / SOURCE}")

    houses = parse_houses(TARGET_LOG)
    need = needed_packages(houses)
    print(f"评测需要 house 数: {len(houses)} -> 包数: {len(need)}")

    with open(cache_dest / "mjthor_resource_file_to_size_mb.json") as f:
        remote_manifest: dict[str, float] = json.load(f)

    done = {
        f[1:].replace("_complete_extract", "")
        for f in os.listdir(cache_dest)
        if f.endswith("_complete_extract")
    }
    missing = sorted(p for p in need if p not in done)
    total_mb = sum(remote_manifest[p] for p in missing if p in remote_manifest)

    print(f"已解压: {len(done & need)} / {len(need)}")
    print(f"待补齐: {len(missing)} 个包, {total_mb:.1f} MB")

    if args.limit:
        missing = missing[: args.limit]
        print(f"[试跑] 本次只处理前 {len(missing)} 个")

    if args.dry_run or not missing:
        for p in missing:
            print("  待补:", p)
        return 0

    # ---- 阶段 1：批量下载 + 解压（_parallel_extract 返回失败集合，不抛异常）----
    all_failed: set[str] = set()
    for i in range(0, len(missing), BATCH_SIZE):
        batch = missing[i : i + BATCH_SIZE]
        sizes = {p: remote_manifest[p] for p in batch if p in remote_manifest}
        unknown = [p for p in batch if p not in remote_manifest]
        if unknown:
            print("!! manifest 中无记录，跳过:", unknown)
        print(f"\n[batch {i // BATCH_SIZE + 1}] 处理 {len(sizes)} 个包 ...", flush=True)
        failed = _parallel_extract(
            sizes,
            relative_path,
            cache_dest,
            manager.remote_storage,
            read_only=True,
            max_workers=MAX_WORKERS,
            min_items_per_worker=MIN_ITEMS_PER_WORKER,
        )
        if failed:
            all_failed |= failed
            print(f"  本批失败 {len(failed)} 个: {sorted(failed)[:5]}")
        print(
            f"  累计: 成功 {len(missing) - len(all_failed)}, 失败 {len(all_failed)}",
            flush=True,
        )

    # ---- 阶段 2：建立 per-file 符号链接 ----
    # 此时所有包的解压标记已存在，install_packages 的下载分支不会再触发。
    print("\n[阶段 2] 建立符号链接 ...", flush=True)
    linked = [p for p in missing if p not in all_failed]
    for i in range(0, len(linked), BATCH_SIZE):
        batch = linked[i : i + BATCH_SIZE]
        manager.install_packages(DATA_TYPE, {SOURCE: batch})
        print(f"  已链接 {min(i + BATCH_SIZE, len(linked))}/{len(linked)}", flush=True)

    # ---- 阶段 3：校验 ----
    done = {
        f[1:].replace("_complete_extract", "")
        for f in os.listdir(cache_dest)
        if f.endswith("_complete_extract")
    }
    link_dir = manager.symlink_dir / DATA_TYPE / SOURCE
    links = {
        f[1:].replace("_complete_links", "")
        for f in os.listdir(link_dir)
        if f.endswith("_complete_links")
    }
    still_missing = sorted(need - done)
    print("\n================ 校验 ================")
    print(f"解压包覆盖: {len(done & need)} / {len(need)}")
    print(f"符号链接覆盖: {len(links & need)} / {len(need)}")
    print(f"仍缺失: {len(still_missing)} 个")
    if still_missing:
        print("  缺失清单:", still_missing[:20])
        print("  提示：重新运行本脚本会自动重试这些包")
    if all_failed:
        print(f"本次下载失败: {len(all_failed)} 个 -> {sorted(all_failed)[:10]}")
    return 1 if still_missing else 0


if __name__ == "__main__":
    sys.exit(main())
