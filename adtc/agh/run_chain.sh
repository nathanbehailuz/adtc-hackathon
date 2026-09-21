#!/usr/bin/env bash
# Sequential v7 chain on AGH (no Slurm). Run inside tmux.
#
#   cd adtc/agh && bash run_chain.sh
#
# Skips setup/download by default (run those once). Override:
#   RUN_SETUP=1 RUN_DOWNLOAD=1 bash run_chain.sh
set -euo pipefail
AGH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${AGH_DIR}"
mkdir -p logs

run_stage() {
  local name="$1"
  shift
  echo "==== [${name}] $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  bash "$@"
  echo "==== [${name}] OK ===="
}

if [[ "${RUN_SETUP:-0}" == "1" ]]; then
  run_stage setup setup_env.sh
fi
if [[ "${RUN_DOWNLOAD:-0}" == "1" ]]; then
  run_stage download download_models.sh
fi

run_stage prepare_mix prepare_mix.sh
run_stage train_sft train_sft.sh
run_stage merge_lora merge_lora.sh
run_stage convert_gguf convert_gguf.sh
run_stage eval_hf eval_hf.sh

V7_QUANT=Q4_K_M run_stage eval_gguf_q4 eval_gguf.sh
V7_QUANT=Q5_K_M run_stage eval_gguf_q5 eval_gguf.sh

run_stage profile_gguf profile_gguf.sh
run_stage judge_smoke judge_smoke.sh

echo "----"
echo "v7 chain complete. Artifacts: docs/artifacts/v7/"
echo "Remember to shut down the AGH instance when idle."
