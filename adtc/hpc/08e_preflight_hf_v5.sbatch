#!/usr/bin/env bash
#SBATCH --job-name=adtc_preflight_hf_v5
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# HF preflight for v5 candidates (config + 4bit/PEFT/generate).
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate "${ADTC_ROOT}/training/.conda-env"
set -u
cd "${ADTC_ROOT}/training"

ALIASES="${V5_ALIASES:-qwen35_extcm qwen35_afrique qwen3_afrique}"
mkdir -p ../docs/artifacts/v5

for a in ${ALIASES}; do
  echo "[preflight_hf] ${a}"
  HF_ID="$(python - <<PY
import yaml
from pathlib import Path
m=yaml.safe_load(Path('configs/v5_models.yaml').read_text())['models']
print(m['${a}']['hf_id'])
PY
)"
  python inspect_lora_targets.py --model "${HF_ID}" --out "../docs/artifacts/v5/lora_targets_${a}.json"
  python preflight_v5_model.py --alias "${a}" --load-4bit --generate \
    --out ../docs/artifacts/v5/model_preflight.json || {
      echo "[preflight_hf] WARN ${a} failed; continuing" >&2
    }
done

echo "[preflight_hf] done"
