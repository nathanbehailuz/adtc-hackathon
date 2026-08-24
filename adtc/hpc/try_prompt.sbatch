#!/bin/bash
#SBATCH --job-name=adtc_try_prompt
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Run eval/try_prompt.py on a compute node (not login).
#
# Model: Qwen3-1.7B v6 Q5_K_M GGUF (English-only STEM tutor).
#
# From adtc/hpc/:
#   sbatch try_prompt.sbatch
#   PROMPTS=all sbatch try_prompt.sbatch
#   PROMPT=1 sbatch try_prompt.sbatch
#   PROMPT=6 TEXT='Solve 2+2' sbatch try_prompt.sbatch
#
# English prompts: 1–5. Prompt 6 = type your own (use TEXT=).
# Logs:
#   adtc/hpc/logs/adtc_try_prompt-<jobid>.{out,err}
#   adtc/logs/try_prompt/<jobid>_m1.log
#   adtc/logs/try_prompt/latest.log
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
cd "${ADTC_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

MODEL="${MODEL:-1}"
PROMPT="${PROMPT:-}"
PROMPTS="${PROMPTS:-}"
TEXT="${TEXT:-}"
MAX_TOKENS="${MAX_TOKENS:-256}"

# English suite (try_prompt.py indices 1–5).
EN_PROMPTS="1,2,3,4,5"
if [[ "${PROMPTS}" == "all" || ( -z "${PROMPTS}" && -z "${PROMPT}" && -z "${TEXT}" ) ]]; then
  PROMPTS="${EN_PROMPTS}"
fi

LOG_DIR="${ADTC_ROOT}/logs/try_prompt"
mkdir -p "${LOG_DIR}"
OUT_LOG="${LOG_DIR}/${SLURM_JOB_ID}_m${MODEL}.log"

PY_ARGS=(
  --model "${MODEL}"
  --n-threads "${SLURM_CPUS_PER_TASK:-8}"
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
echo "[try_prompt] wrote ${LOG_DIR}/latest.log"
echo "[try_prompt] slurm stdout ${HPC_DIR}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
