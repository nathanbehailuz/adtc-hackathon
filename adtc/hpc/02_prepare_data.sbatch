#!/bin/bash
#SBATCH --job-name=adtc_prep_data
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Normalize sources → build EN STEM SFT → stub Amharic MT → mix SFT.
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
# Activate scratch training conda (created by setup_env.sbatch)
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate /scratch/nz2212/adtc-hackathon/adtc/training/.conda-env
set -u

cd "${ADTC_ROOT}"

# CPT: snapshot-only (skip gated/missing). Soft-fail if interpreter teardown aborts.
python data/normalize_cpt_sources.py || {
  echo "[prep] warn: normalize_cpt exited non-zero; continuing if CPT outputs exist"
  test -f data/train/cpt/cpt_normalize_summary_v0.json
}

# Skip afriquellm_gsm8k until that snapshot exists (download failed last run).
python data/normalize_sft_sources.py --sources walia finetome r1 dolly taco
python data/build_en_stem_sft.py --limit 2000

python data/build_translate_am.py \
  --in data/train/en_stem_sft_v0.jsonl \
  --backend stub

# Prefer normalized SFT pools when present.
MIX_CMD=(
  python data/mix_sft.py
  --en-stem data/train/en_stem_sft_v0.jsonl
  --eval data/eval/custom_tutoring_v0.jsonl data/eval/en_stem_holdout_v0.jsonl
  --out data/train/sft_mix_v0.jsonl
  --total 2000
)
SFT_FILES=()
for f in data/train/sources/walia_sft_v0.jsonl data/train/sources/finetome_am_sft_v0.jsonl; do
  if [[ -f "${f}" ]]; then
    SFT_FILES+=("${f}")
  fi
done
if ((${#SFT_FILES[@]} > 0)); then
  MIX_CMD+=(--sft "${SFT_FILES[@]}")
fi
"${MIX_CMD[@]}"

echo "[prep] Wrote data/train/sft_mix_v0.jsonl"
