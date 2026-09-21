#!/usr/bin/env bash
# HF frozen eval for qwen3_1_7b_merged_v7 (GPU).
set -euo pipefail
AGH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${AGH_DIR}/env.sh"
mkdir -p "${AGH_DIR}/logs" "${ADTC_ROOT}/docs/artifacts/v7"
LOG="${AGH_DIR}/logs/eval_hf.log"
exec > >(tee -a "${LOG}") 2>&1

cd "${ADTC_ROOT}"
MODEL="${ADTC_ROOT}/training/runs/qwen3_1_7b_merged_v7"
[[ -d "${MODEL}" ]] || { echo "missing ${MODEL}" >&2; exit 1; }
OUT="docs/artifacts/v7/qwen3_1_7b_merged_v7_hf_eval.json"
python eval/run_hf_eval.py --model "${MODEL}" --out "${OUT}"
echo "[eval_hf_v7] OK ${OUT}"
