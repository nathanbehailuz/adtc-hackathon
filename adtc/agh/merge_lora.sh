#!/usr/bin/env bash
# Merge LoRA → HF for qwen3_1_7b_merged_v7.
set -euo pipefail
AGH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${AGH_DIR}/env.sh"
mkdir -p "${AGH_DIR}/logs"
LOG="${AGH_DIR}/logs/merge_lora.log"
exec > >(tee -a "${LOG}") 2>&1

cd "${ADTC_ROOT}/training"
BASE="Qwen/Qwen3-1.7B"
ADAPTER="runs/qwen3_1_7b_qlora_v7/adapter"
OUT="runs/qwen3_1_7b_merged_v7"
[[ -d "${ADAPTER}" ]] || { echo "missing ${ADAPTER}" >&2; exit 1; }
python merge_lora.py --base "${BASE}" --adapter "${ADAPTER}" --out "${OUT}"
echo "[merge_v7] OK ${OUT}"
