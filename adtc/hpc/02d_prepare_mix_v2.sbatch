#!/bin/bash
#SBATCH --job-name=adtc_prep_mix_v2
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 16:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Build tutoring v2, normalize AM GSM8K, harden/filter NLLB STEM, assemble sft_mix_v2.
# Prefer re-NLLB on en_stem_sft_v2; if NLLB fails, fall back to filtered v1 MT.
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate /scratch/nz2212/adtc-hackathon/adtc/training/.conda-env
set -u
cd "${ADTC_ROOT}"

echo "[prep_v2] tutoring EN+AM"
python data/build_tutoring_sft_v2.py --en-limit 2000

echo "[prep_v2] normalize SFT sources (Ethiopic filter + AM GSM8K)"
python data/normalize_sft_sources.py \
  --sources walia finetome afriquellm_gsm8k simonbutt_amharic_gsm8k dolly taco

echo "[prep_v2] NLLB translate en_stem_sft_v2 (hardened markers)"
NLLB_OUT="data/train/am_stem_sft_nllb_v2.jsonl"
# Always keep a filtered v1 pool as backup AM STEM.
python data/filter_nllb_stem_v2.py \
  --in data/train/am_stem_sft_nllb_v1.jsonl \
  --en data/train/en_stem_sft_v0.jsonl \
  --out data/train/am_stem_sft_nllb_v2_filtered.jsonl || true

if python data/build_translate_am.py \
  --in data/train/en_stem_sft_v2.jsonl \
  --out "${NLLB_OUT}" \
  --backend nllb \
  --directions en_am,am_am \
  --quality-filter; then
  echo "[prep_v2] NLLB OK -> ${NLLB_OUT}"
else
  echo "[prep_v2] NLLB failed — relying on filtered v1 MT" >&2
  NLLB_OUT=""
fi

AM_STEM_ARGS=()
for f in \
  data/train/sources/afriquellm_gsm8k_am_sft_v0.jsonl \
  data/train/sources/simonbutt_amharic_gsm8k_sft_v0.jsonl \
  data/train/am_stem_sft_nllb_v2_filtered.jsonl \
  ${NLLB_OUT}; do
  if [[ -n "${f}" && -f "${f}" && -s "${f}" ]]; then
    AM_STEM_ARGS+=("${f}")
  fi
done
if ((${#AM_STEM_ARGS[@]} == 0)); then
  echo "error: no AM STEM files available for mix v2" >&2
  exit 1
fi

AM_INSTR_ARGS=()
for f in \
  data/train/sources/walia_sft_v0.jsonl \
  data/train/sources/dolly_am_sft_v0.jsonl \
  data/train/sources/finetome_am_sft_v0.jsonl \
  data/train/sources/taco_am_sft_v0.jsonl; do
  if [[ -f "${f}" && -s "${f}" ]]; then
    AM_INSTR_ARGS+=("${f}")
  fi
done

echo "[prep_v2] mix_sft_v2"
MIX_CMD=(
  python data/mix_sft_v2.py
  --en-stem data/train/en_stem_sft_v2.jsonl
  --am-stem "${AM_STEM_ARGS[@]}"
  --am-tutoring data/train/am_tutoring_sft_v2.jsonl
  --out data/train/sft_mix_v2.jsonl
  --total 5000
)
if ((${#AM_INSTR_ARGS[@]} > 0)); then
  MIX_CMD+=(--am-instruct "${AM_INSTR_ARGS[@]}")
fi
"${MIX_CMD[@]}"

echo "[prep_v2] Done -> data/train/sft_mix_v2.jsonl"
wc -l data/train/sft_mix_v2.jsonl
