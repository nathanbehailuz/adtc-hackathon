#!/usr/bin/env bash
# GGUF frozen eval for v7. Env: V7_QUANT=Q4_K_M|Q5_K_M|Q6_K (default Q4_K_M)
set -euo pipefail
AGH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${AGH_DIR}/env.sh"
mkdir -p "${AGH_DIR}/logs" "${ADTC_ROOT}/docs/artifacts/v7"
LOG="${AGH_DIR}/logs/eval_gguf_${V7_QUANT:-Q4_K_M}.log"
exec > >(tee -a "${LOG}") 2>&1

cd "${ADTC_ROOT}"
QUANT="${V7_QUANT:-Q4_K_M}"
GGUF="${ADTC_ROOT}/artifacts/gguf/adapted/qwen3_1_7b_merged_v7-${QUANT}.gguf"
[[ -f "${GGUF}" ]] || { echo "missing ${GGUF}" >&2; exit 1; }
OUT="docs/artifacts/v7/qwen3_1_7b_merged_v7-${QUANT}_eval.json"
NTHREADS="${NTHREADS:-${OMP_NUM_THREADS:-8}}"
python eval/run_gguf_eval.py --gguf "${GGUF}" --n-threads "${NTHREADS}" --out "${OUT}"
echo "[eval_gguf_v7] OK ${OUT}"
