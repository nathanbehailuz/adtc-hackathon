#!/bin/bash
#SBATCH --job-name=adtc_eval_v4
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate /scratch/nz2212/adtc-hackathon/adtc/training/.conda-env
set -u
cd "${ADTC_ROOT}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

GGUF="${ADTC_ROOT}/artifacts/gguf/adapted/gemma3_4b_merged_v4-Q4_K_M.gguf"
OUT_DIR="${ADTC_ROOT}/docs/artifacts/perf"
mkdir -p "${OUT_DIR}" "${ADTC_ROOT}/logs/try_prompt"
[[ -f "${GGUF}" ]] || { echo "missing ${GGUF}" >&2; exit 1; }

python eval/run_gguf_eval.py --gguf "${GGUF}" --out "${OUT_DIR}/gemma3_4b_merged_v4-Q4_K_M_v4_eval.json" --n-threads "${SLURM_CPUS_PER_TASK:-8}"
python eval/run_translate_test.py --gguf "${GGUF}" --out "${OUT_DIR}/translate_gemma3_4b_merged_v4-Q4_K_M.md" --n-threads "${SLURM_CPUS_PER_TASK:-8}"

MODEL_IDX=8
OUT_LOG="${ADTC_ROOT}/logs/try_prompt/${SLURM_JOB_ID}_m${MODEL_IDX}_v4gate.log"
python eval/try_prompt.py --model "${MODEL_IDX}" --prompts 2,6,7,8,9,10 --n-threads "${SLURM_CPUS_PER_TASK:-8}" --out "${OUT_LOG}"
cp -f "${OUT_LOG}" "${ADTC_ROOT}/logs/try_prompt/latest_v4_gate.log"
echo "[eval_v4] done"
