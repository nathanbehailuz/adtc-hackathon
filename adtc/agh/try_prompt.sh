#!/usr/bin/env bash
# Qualitative smoke via eval/try_prompt.py on AGH.
#
#   bash try_prompt.sh
#   PROMPTS=all bash try_prompt.sh
#   PROMPT=1 bash try_prompt.sh
#   PROMPT=6 TEXT='Solve 2+2' bash try_prompt.sh
set -euo pipefail
AGH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${AGH_DIR}/env.sh"
mkdir -p "${AGH_DIR}/logs"
LOG="${AGH_DIR}/logs/try_prompt.log"
exec > >(tee -a "${LOG}") 2>&1

cd "${ADTC_ROOT}"

MODEL="${MODEL:-1}"
PROMPT="${PROMPT:-}"
PROMPTS="${PROMPTS:-}"
TEXT="${TEXT:-}"
MAX_TOKENS="${MAX_TOKENS:-256}"
NTHREADS="${NTHREADS:-${OMP_NUM_THREADS:-8}}"

EN_PROMPTS="1,2,3,4,5"
if [[ "${PROMPTS}" == "all" || ( -z "${PROMPTS}" && -z "${PROMPT}" && -z "${TEXT}" ) ]]; then
  PROMPTS="${EN_PROMPTS}"
fi

LOG_DIR="${ADTC_ROOT}/logs/try_prompt"
mkdir -p "${LOG_DIR}"
OUT_LOG="${LOG_DIR}/agh_m${MODEL}.log"

PY_ARGS=(
  --model "${MODEL}"
  --n-threads "${NTHREADS}"
  --max-tokens "${MAX_TOKENS}"
  --out "${OUT_LOG}"
)
if [[ -n "${PROMPTS}" ]]; then
  PY_ARGS+=(--prompts "${PROMPTS}")
else
  PY_ARGS+=(--prompt "${PROMPT}")
fi
if [[ -n "${TEXT}" ]]; then
  PY_ARGS+=(--text "${TEXT}")
fi

echo "[try_prompt] model=${MODEL} prompt=${PROMPT} prompts=${PROMPTS:-} out=${OUT_LOG}"
python eval/try_prompt.py "${PY_ARGS[@]}"
cp -f "${OUT_LOG}" "${LOG_DIR}/latest.log"
echo "[try_prompt] wrote ${OUT_LOG}"
