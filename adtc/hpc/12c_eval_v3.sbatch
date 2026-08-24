#!/bin/bash
#SBATCH --job-name=adtc_eval_v3
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# GGUF eval + translate-test + try_prompt for Gemma v3 Q4.
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate /scratch/nz2212/adtc-hackathon/adtc/training/.conda-env
set -u
cd "${ADTC_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

GGUF_DIR="${ADTC_ROOT}/artifacts/gguf/adapted"
LIMIT="${LIMIT:-}"
OUT_DIR="${ADTC_ROOT}/docs/artifacts/perf"
mkdir -p "${OUT_DIR}" "${ADTC_ROOT}/logs/try_prompt"

KEY="gemma3_4b_merged_v3-Q4_K_M"
GGUF="${GGUF_DIR}/${KEY}.gguf"
if [[ ! -f "${GGUF}" ]]; then
  echo "[eval_v3] missing ${GGUF}" >&2
  exit 1
fi

ARGS=(
  --gguf "${GGUF}"
  --out "${OUT_DIR}/${KEY}_v3_eval.json"
  --n-threads "${SLURM_CPUS_PER_TASK:-8}"
)
if [[ -n "${LIMIT}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi
echo "[eval_v3] run_gguf_eval ${KEY}"
python eval/run_gguf_eval.py "${ARGS[@]}"

TT_ARGS=(
  --gguf "${GGUF}"
  --out "${OUT_DIR}/translate_${KEY}.md"
  --n-threads "${SLURM_CPUS_PER_TASK:-8}"
)
if [[ -n "${LIMIT}" ]]; then
  TT_ARGS+=(--limit "${LIMIT}")
fi
echo "[eval_v3] translate_test"
python eval/run_translate_test.py "${TT_ARGS[@]}"

# try_prompt — Gemma v3 is MODELS index 7
MODEL_IDX=7
echo "[eval_v3] try_prompt model=${MODEL_IDX} prompts 2,6,7,8,9,10"
OUT_LOG="${ADTC_ROOT}/logs/try_prompt/${SLURM_JOB_ID}_m${MODEL_IDX}_v3gate.log"
python eval/try_prompt.py \
  --model "${MODEL_IDX}" \
  --prompts 2,6,7,8,9,10 \
  --n-threads "${SLURM_CPUS_PER_TASK:-8}" \
  --out "${OUT_LOG}"
cp -f "${OUT_LOG}" "${ADTC_ROOT}/logs/try_prompt/latest_v3_gate.log"
echo "[eval_v3] wrote ${OUT_LOG}"
echo "[eval_v3] done"
