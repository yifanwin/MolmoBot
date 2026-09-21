#!/usr/bin/env bash
# 监控评测进程内存占用，超过阈值自动终止，避免把机器打爆。
#
# 用法:
#   ./monitor_eval_memory.sh <PID> [采样间隔秒] [终止阈值GB]
# 可选环境变量:
#   MONITORED_LOG  被监控评测的 running_log.log 路径，用于在采样行里带上进度

set -u

# ---------------------------------------------------------------- 配置区
INTERVAL_SEC=30          # 采样间隔（秒）
WARN_GB=30               # 警告阈值（GB），仅记录不动作
KILL_GB=50               # 强制终止阈值（GB）
SAMPLE_LOG=/tmp/eval_mem_samples.txt
MONITORED_LOG=${MONITORED_LOG:-}
# ------------------------------------------------------------ 配置区结束

PID=${1:?需要传入被监控的进程 PID}
INTERVAL_SEC=${2:-$INTERVAL_SEC}
KILL_GB=${3:-$KILL_GB}

echo "开始监控 PID=$PID  间隔=${INTERVAL_SEC}s  警告=${WARN_GB}GB  终止=${KILL_GB}GB" | tee "$SAMPLE_LOG"

while ps -p "$PID" > /dev/null 2>&1; do
    rss_kb=$(awk '/VmRSS/{print $2}' /proc/$PID/status 2>/dev/null)
    [ -z "$rss_kb" ] && break
    rss_gb=$(awk -v k="$rss_kb" 'BEGIN{printf "%.2f", k/1024/1024}')
    avail_gb=$(awk '/MemAvailable/{printf "%.1f", $2/1024/1024}' /proc/meminfo)
    prog=$(grep -aoE 'house [0-9]+ episode [0-9]+/[0-9]+' "$MONITORED_LOG" 2>/dev/null | tail -1)

    echo "$(date +%H:%M:%S)  RSS=${rss_gb}GB  系统可用=${avail_gb}GB  ${prog}" >> "$SAMPLE_LOG"

    over=$(awk -v r="$rss_gb" -v t="$KILL_GB" 'BEGIN{print (r>t)?1:0}')
    if [ "$over" = "1" ]; then
        echo "$(date +%H:%M:%S)  !! RSS ${rss_gb}GB 超过阈值 ${KILL_GB}GB，终止 PID=$PID" | tee -a "$SAMPLE_LOG"
        kill "$PID"
        break
    fi
    sleep "$INTERVAL_SEC"
done

echo "$(date +%H:%M:%S)  监控结束（进程已退出或已终止）" >> "$SAMPLE_LOG"
