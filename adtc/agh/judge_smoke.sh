#!/usr/bin/env bash
# Judge-aligned smoke suite on a v7 GGUF.
set -euo pipefail
AGH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${AGH_DIR}/env.sh"
mkdir -p "${AGH_DIR}/logs" "${ADTC_ROOT}/docs/artifacts/v7"
LOG="${AGH_DIR}/logs/judge_smoke.log"
exec > >(tee -a "${LOG}") 2>&1

cd "${ADTC_ROOT}"
QUANT="${V7_QUANT:-Q5_K_M}"
GGUF="${ADTC_ROOT}/artifacts/gguf/adapted/qwen3_1_7b_merged_v7-${QUANT}.gguf"
[[ -f "${GGUF}" ]] || {
  # fallback to shipped tutor name if present
  ALT="${ADTC_ROOT}/artifacts/gguf/adapted/tebeb_tutor_1.7b-Q5_K_M.gguf"
  if [[ -f "${ALT}" ]]; then GGUF="${ALT}"; else echo "missing ${GGUF}" >&2; exit 1; fi
}
OUT="docs/artifacts/v7/judge_smoke_${QUANT}.json"
NTHREADS="${NTHREADS:-${OMP_NUM_THREADS:-8}}"
python eval/run_judge_smoke.py --gguf "${GGUF}" --n-threads "${NTHREADS}" --out "${OUT}"
echo "[judge_smoke] OK ${OUT}"
