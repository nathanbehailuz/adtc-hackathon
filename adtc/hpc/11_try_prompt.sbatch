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
# Models (eval/try_prompt.py MODELS 1-indexed):
#   1–8  GGUF via llama_cpp  → default: compute (CPU)
#   9    AfriqueQwen3.5-ExtCM v5 HF (qwen35_extcm_merged_v5)
#        → needs nvidia A100 (llama_cpp cannot load arch=qwen35)
#
# From adtc/hpc/:
#   # Gemma v4 GGUF (default)
#   sbatch 11_try_prompt.sbatch
#   MODEL=8 PROMPTS=all sbatch 11_try_prompt.sbatch
#
#   # v5 HF on GPU (auto-requests nvidia if MODEL=9)
#   MODEL=9 PROMPTS=all sbatch 11_try_prompt.sbatch
#   MODEL=9 PROMPT=6 sbatch 11_try_prompt.sbatch
#   MODEL=9 PROMPT=11 TEXT='Solve 2+2' sbatch 11_try_prompt.sbatch
#
# Logs:
#   adtc/hpc/logs/adtc_try_prompt-<jobid>.{out,err}
#   adtc/logs/try_prompt/<jobid>_m<model>.log
#   adtc/logs/try_prompt/latest.log
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
cd "${ADTC_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

MODEL="${MODEL:-8}"
PROMPT="${PROMPT:-6}"
PROMPTS="${PROMPTS:-}"
TEXT="${TEXT:-}"
MAX_TOKENS="${MAX_TOKENS:-256}"

# Model 9 is HF-only (qwen35 GGUF unsupported by current llama_cpp).
# If submitted on compute without a GPU, fail fast with a clear message.
if [[ "${MODEL}" == "9" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "[try_prompt] ERROR: MODEL=9 needs an A100 GPU (HF transformers backend)." >&2
    echo "[try_prompt] Resubmit with:" >&2
    echo "  MODEL=9 PROMPTS=all sbatch -p nvidia --gres=gpu:a100:1 --mem=64G -t 4:00:00 11_try_prompt.sbatch" >&2
    exit 1
  fi
  echo "[try_prompt] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
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
