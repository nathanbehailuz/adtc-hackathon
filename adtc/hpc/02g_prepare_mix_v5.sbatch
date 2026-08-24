#!/usr/bin/env bash
#SBATCH --job-name=adtc_prep_mix_v5
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Build cpt_mix_v5 + xling + sft_mix_v5 (does not touch v0–v4 mixes).
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate "${ADTC_ROOT}/training/.conda-env"
set -u
cd "${ADTC_ROOT}"

mkdir -p docs/artifacts/v5 data/train

echo "[v5] CPT mix"
python data/mix_cpt_v5.py \
  --out data/train/cpt_mix_v5.jsonl \
  --total 25000 \
  --seed 42

echo "[v5] cross-lingual SFT seed"
python data/build_xling_sft_v5.py \
  --en-stem data/train/en_stem_sft_v4.jsonl \
  --am-stem data/train/am_stem_pool_v4.jsonl data/train/am_stem_sft_nllb_v2_filtered.jsonl \
  --out data/train/xling_sft_v5.jsonl \
  --n-en-am 800 --n-am-en 800 --seed 42

echo "[v5] SFT mix"
python data/mix_sft_v5.py \
  --en-stem data/train/en_stem_sft_v4.jsonl \
  --am-stem data/train/am_stem_pool_v4.jsonl data/train/am_stem_sft_nllb_v2_filtered.jsonl \
  --am-tutoring data/train/am_tutoring_sft_v4.jsonl data/train/am_tutoring_derived_v4.jsonl \
  --am-instruct data/train/sources/walia_sft_v0.jsonl data/train/sources/dolly_am_sft_v0.jsonl \
  --xling data/train/xling_sft_v5.jsonl \
  --simonbutt data/train/sources/simonbutt_amharic_gsm8k_sft_v0.jsonl \
  --out data/train/sft_mix_v5.jsonl \
  --total 9000 --seed 42

echo "[v5] prepare_mix OK"
wc -l data/train/cpt_mix_v5.jsonl data/train/sft_mix_v5.jsonl data/train/xling_sft_v5.jsonl
