#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. This script never prints API credentials.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MOLMOBOT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$MOLMOBOT_ROOT/../.." && pwd)"

source "$REPO_ROOT/molmospaces/setup_env.sh"
export PYTHONPATH="$MOLMOBOT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$MOLMOBOT_ROOT"

RBY_BENCH="$MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v1/procthor-10k/RBY1PickAndPlaceDataGenConfig/RBY1PickAndPlaceDataGenConfig_20260209_json_benchmark"
PANDA_BENCH="$MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickandPlaceDroidMiniBench/FrankaPickandPlaceDroidMiniBench_20260111_json_benchmark"
OUT="$MOLMOBOT_ROOT/eval_output"
MODE="${1:-gate}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/molmospaces/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi

run_eval() {
  "$PYTHON_BIN" launch_scripts/run_eval.py "$@" --house_index 4 --num_workers 1 --output_dir "$OUT"
}

case "$MODE" in
  baseline)
    run_eval --benchmark_path "$RBY_BENCH" \
      --eval_config_cls olmo.eval.configure_molmo_spaces:MolmoBotRBY1CuroboPickPnPEvalConfig
    run_eval --benchmark_path "$PANDA_BENCH" \
      --eval_config_cls molmo_spaces.evaluation.configs.evaluation_configs:PandaOmronCuroboPickPnPEvalConfig
    ;;
  gate)
    run_eval --benchmark_path "$RBY_BENCH" --episode_idx 0 \
      --eval_config_cls olmo.eval.configure_llm_waypoint:MolmoBotRBY1LLMWaypointPickPnPEvalConfig
    run_eval --benchmark_path "$PANDA_BENCH" --episode_idx 0 \
      --eval_config_cls molmo_spaces.evaluation.configs.evaluation_configs:PandaOmronLLMWaypointPickPnPEvalConfig
    ;;
  full)
    run_eval --benchmark_path "$RBY_BENCH" \
      --eval_config_cls olmo.eval.configure_llm_waypoint:MolmoBotRBY1LLMWaypointPickPnPEvalConfig
    run_eval --benchmark_path "$PANDA_BENCH" \
      --eval_config_cls molmo_spaces.evaluation.configs.evaluation_configs:PandaOmronLLMWaypointPickPnPEvalConfig
    ;;
  mock)
    : "${LLM_MOCK_RESPONSE_FILE:?Set LLM_MOCK_RESPONSE_FILE to a response JSON file}"
    run_eval --benchmark_path "$RBY_BENCH" --episode_idx 0 \
      --eval_config_cls olmo.eval.configure_llm_waypoint:MolmoBotRBY1LLMWaypointPickPnPEvalConfig
    run_eval --benchmark_path "$PANDA_BENCH" --episode_idx 0 \
      --eval_config_cls molmo_spaces.evaluation.configs.evaluation_configs:PandaOmronLLMWaypointPickPnPEvalConfig
    ;;
  *)
    echo "Usage: $0 {baseline|mock|gate|full}" >&2
    exit 2
    ;;
esac
