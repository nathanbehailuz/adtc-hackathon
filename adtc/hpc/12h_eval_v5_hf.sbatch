#!/usr/bin/env bash
# HF frozen eval for v5 merged checkpoint. Env: V5_ALIAS, V5_STAGE (m0|m1|m2)
#SBATCH --job-name=adtc_eval_hf_v5
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate "${ADTC_ROOT}/training/.conda-env"
set -u
cd "${ADTC_ROOT}"

ALIAS="${V5_ALIAS:?}"
STAGE="${V5_STAGE:-m2}"
case "${STAGE}" in
  m0)
    MODEL="$(python - <<PY
import yaml
from pathlib import Path
m=yaml.safe_load(Path('training/configs/v5_models.yaml').read_text())['models']
print(m['${ALIAS}']['hf_id'])
PY
)"
    ;;
  m1) MODEL="${ADTC_ROOT}/training/runs/${ALIAS}_cpt_merged_v5" ;;
  m2) MODEL="${ADTC_ROOT}/training/runs/${ALIAS}_merged_v5" ;;
  *) echo "bad V5_STAGE=${STAGE}" >&2; exit 1 ;;
esac

mkdir -p docs/artifacts/v5
OUT="docs/artifacts/v5/${ALIAS}_${STAGE}_hf_eval.json"
python eval/run_hf_eval.py --model "${MODEL}" --out "${OUT}"
echo "[eval_hf_v5] OK ${OUT}"
