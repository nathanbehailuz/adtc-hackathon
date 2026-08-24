#!/bin/bash
#SBATCH --job-name=adtc_eval_v6
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# GGUF frozen eval for v6. Env: V6_QUANT=Q4_K_M|Q5_K_M|Q6_K (default Q4_K_M)
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
# shellcheck disable=SC1091
source "${HPC_DIR}/profiler_env.sh" || true
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate "${ADTC_ROOT}/training/.conda-env"
set -u
cd "${ADTC_ROOT}"

QUANT="${V6_QUANT:-Q4_K_M}"
GGUF="${ADTC_ROOT}/artifacts/gguf/adapted/qwen3_1_7b_merged_v6-${QUANT}.gguf"
[[ -f "${GGUF}" ]] || { echo "missing ${GGUF}" >&2; exit 1; }
mkdir -p docs/artifacts/v6
OUT="docs/artifacts/v6/qwen3_1_7b_merged_v6-${QUANT}_eval.json"
python eval/run_gguf_eval.py --gguf "${GGUF}" --n-threads "${SLURM_CPUS_PER_TASK:-8}" --out "${OUT}"
echo "[eval_v6] OK ${OUT}"
