#!/usr/bin/env bash
# GGUF frozen eval for v5. Env: V5_ALIAS, V5_QUANT (default Q4_K_M)
#SBATCH --job-name=adtc_eval_v5
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

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

ALIAS="${V5_ALIAS:?}"
QUANT="${V5_QUANT:-Q4_K_M}"
GGUF="${ADTC_ROOT}/artifacts/gguf/adapted/${ALIAS}_merged_v5-${QUANT}.gguf"
[[ -f "${GGUF}" ]] || { echo "missing ${GGUF}" >&2; exit 1; }
mkdir -p docs/artifacts/v5
OUT="docs/artifacts/v5/${ALIAS}_merged_v5-${QUANT}_eval.json"
python eval/run_gguf_eval.py --gguf "${GGUF}" --out "${OUT}"
echo "[eval_v5] OK ${OUT}"
