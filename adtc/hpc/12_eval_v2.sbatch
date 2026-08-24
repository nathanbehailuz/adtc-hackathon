#!/bin/bash
#SBATCH --job-name=adtc_eval_v2
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Gate: AfriMGSM AM (+ EN) on v2 Q4 GGUFs + try_prompt suite 2,6-10 for Gemma v2.
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
LIMIT="${LIMIT:-}"  # empty = full frozen set
OUT_DIR="${ADTC_ROOT}/docs/artifacts/perf"
mkdir -p "${OUT_DIR}" "${ADTC_ROOT}/logs/try_prompt"

run_gguf() {
  local key="$1"
  local gguf="${GGUF_DIR}/${key}.gguf"
  if [[ ! -f "${gguf}" ]]; then
    echo "[eval_v2] SKIP missing ${gguf}" >&2
    return 1
  fi
  local args=(
    --gguf "${gguf}"
    --out "${OUT_DIR}/${key}_v2_eval.json"
    --n-threads "${SLURM_CPUS_PER_TASK:-8}"
  )
  if [[ -n "${LIMIT}" ]]; then
    args+=(--limit "${LIMIT}")
  fi
  echo "[eval_v2] run_gguf_eval ${key}"
  python eval/run_gguf_eval.py "${args[@]}"
}

run_gguf "gemma3_4b_merged_v2-Q4_K_M" || true
run_gguf "qwen3_1_7b_merged_v2-Q4_K_M" || true

# Translate-test for Gemma v2
GEMMA_GGUF="${GGUF_DIR}/gemma3_4b_merged_v2-Q4_K_M.gguf"
if [[ -f "${GEMMA_GGUF}" ]]; then
  echo "[eval_v2] translate_test gemma v2"
  TT_ARGS=(
    --gguf "${GEMMA_GGUF}"
    --out "${OUT_DIR}/translate_gemma3_4b_merged_v2-Q4_K_M.md"
    --n-threads "${SLURM_CPUS_PER_TASK:-8}"
  )
  if [[ -n "${LIMIT}" ]]; then
    TT_ARGS+=(--limit "${LIMIT}")
  fi
  python eval/run_translate_test.py "${TT_ARGS[@]}" || true
fi

# try_prompt qualitative gate — Gemma v2 is MODELS index 5 in eval/try_prompt.py
MODEL_IDX=5
echo "[eval_v2] try_prompt model=${MODEL_IDX} prompts 2,6,7,8,9,10"
OUT_LOG="${ADTC_ROOT}/logs/try_prompt/${SLURM_JOB_ID}_m${MODEL_IDX}_v2gate.log"
python eval/try_prompt.py \
  --model "${MODEL_IDX}" \
  --prompts 2,6,7,8,9,10 \
  --n-threads "${SLURM_CPUS_PER_TASK:-8}" \
  --out "${OUT_LOG}"
cp -f "${OUT_LOG}" "${ADTC_ROOT}/logs/try_prompt/latest_v2_gate.log"
echo "[eval_v2] wrote ${OUT_LOG}"
echo "[eval_v2] done"
