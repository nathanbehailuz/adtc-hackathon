#!/usr/bin/env bash
# Build English-only SFT mix v7 on AGH.
set -euo pipefail
AGH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${AGH_DIR}/env.sh"
mkdir -p "${AGH_DIR}/logs" "${ADTC_ROOT}/docs/artifacts/v7" "${ADTC_ROOT}/data/train"
LOG="${AGH_DIR}/logs/prepare_mix.log"
exec > >(tee -a "${LOG}") 2>&1

cd "${ADTC_ROOT}"
MIX_SCRIPT="${ADTC_ROOT}/data/mix_sft_v7.py"
[[ -f "${MIX_SCRIPT}" ]] || { echo "missing ${MIX_SCRIPT}" >&2; exit 1; }

echo "[prep_v7] build sft_mix_v7"
python data/mix_sft_v7.py \
  --sciq-limit 3000 \
  --out data/train/sft_mix_v7.jsonl \
  --counts-out docs/artifacts/v7/sft_mix_v7_counts.json \
  --report-out docs/artifacts/v7/sft_mix_v7_report.md

N=$(wc -l < data/train/sft_mix_v7.jsonl)
echo "[prep_v7] rows=${N}"
if (( N < 5000 )); then
  echo "error: sft_mix_v7 has ${N} rows (<5000)" >&2
  exit 1
fi
echo "[prep_v7] Done"
