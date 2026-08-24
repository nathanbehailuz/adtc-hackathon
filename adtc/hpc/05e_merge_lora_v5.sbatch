#!/usr/bin/env bash
# Merge SFT adapter → final HF M2. Env: V5_ALIAS
#SBATCH --job-name=adtc_merge_sft_v5
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
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
cd "${ADTC_ROOT}/training"

ALIAS="${V5_ALIAS:?}"
# Prefer CPT-merged base; fall back to Hub id for M0→SFT ablation.
BASE="runs/${ALIAS}_cpt_merged_v5"
if [[ ! -d "${BASE}" ]]; then
  BASE="$(python - <<PY
import yaml
from pathlib import Path
m=yaml.safe_load(Path('configs/v5_models.yaml').read_text())['models']
print(m['${ALIAS}']['hf_id'])
PY
)"
fi
ADAPTER="runs/${ALIAS}_qlora_v5/adapter"
OUT="runs/${ALIAS}_merged_v5"
MCLASS="$(python - <<PY
import yaml
from pathlib import Path
m=yaml.safe_load(Path('configs/v5_models.yaml').read_text())['models']
print(m['${ALIAS}'].get('model_class','auto'))
PY
)"
[[ -d "${ADAPTER}" ]] || { echo "missing ${ADAPTER}" >&2; exit 1; }
python merge_lora.py --base "${BASE}" --adapter "${ADAPTER}" --out "${OUT}" --model-class "${MCLASS}"
echo "[merge_sft_v5] OK ${OUT}"
