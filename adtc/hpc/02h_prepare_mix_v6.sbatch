#!/bin/bash
#SBATCH --job-name=adtc_prep_mix_v6
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# English-only SFT mix v6: full GSM8K train + SciQ tutoring behaviors.
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate "${ADTC_ROOT}/training/.conda-env"
set -u
cd "${ADTC_ROOT}"

mkdir -p docs/artifacts/v6 data/train
echo "[prep_v6] build sft_mix_v6 (GSM8K full + SciQ<=3000)"
python data/mix_sft_v6.py \
  --sciq-limit 3000 \
  --out data/train/sft_mix_v6.jsonl \
  --counts-out docs/artifacts/v6/sft_mix_v6_counts.json \
  --report-out docs/artifacts/v6/sft_mix_v6_report.md

N=$(wc -l < data/train/sft_mix_v6.jsonl)
echo "[prep_v6] rows=${N}"
if (( N < 5000 )); then
  echo "error: sft_mix_v6 has ${N} rows (<5000)" >&2
  exit 1
fi
echo "[prep_v6] Done"
