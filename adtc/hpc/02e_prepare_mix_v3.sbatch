#!/bin/bash
#SBATCH --job-name=adtc_prep_mix_v3
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Remix sft_mix_v3 + cpt_mix_v3 from on-disk pools only (no download / NLLB).
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate /scratch/nz2212/adtc-hackathon/adtc/training/.conda-env
set -u
cd "${ADTC_ROOT}"

echo "[prep_v3] mix_cpt_v3"
python data/mix_cpt_v3.py \
  --out data/train/cpt_mix_v3.jsonl \
  --total 20000 \
  --counts-out docs/artifacts/cpt_mix_v3_counts.json

AM_STEM_ARGS=()
for f in \
  data/train/sources/simonbutt_amharic_gsm8k_sft_v0.jsonl \
  data/train/am_stem_sft_nllb_v2.jsonl \
  data/train/am_stem_sft_nllb_v2_filtered.jsonl; do
  if [[ -f "${f}" && -s "${f}" ]]; then
    AM_STEM_ARGS+=("${f}")
  fi
done
if ((${#AM_STEM_ARGS[@]} == 0)); then
  echo "error: no AM STEM files on disk for mix v3" >&2
  exit 1
fi

AM_INSTR_ARGS=()
for f in \
  data/train/sources/walia_sft_v0.jsonl \
  data/train/sources/dolly_am_sft_v0.jsonl; do
  if [[ -f "${f}" && -s "${f}" ]]; then
    AM_INSTR_ARGS+=("${f}")
  fi
done

echo "[prep_v3] mix_sft_v3"
MIX_CMD=(
  python data/mix_sft_v3.py
  --en-stem data/train/en_stem_sft_v2.jsonl
  --am-stem "${AM_STEM_ARGS[@]}"
  --am-tutoring data/train/am_tutoring_sft_v2.jsonl
  --out data/train/sft_mix_v3.jsonl
  --total 8000
  --counts-out docs/artifacts/sft_mix_v3_counts.json
)
if ((${#AM_INSTR_ARGS[@]} > 0)); then
  MIX_CMD+=(--am-instruct "${AM_INSTR_ARGS[@]}")
fi
"${MIX_CMD[@]}"

echo "[prep_v3] amharic_review_sample_v3"
python - <<'PY'
import json, random
from pathlib import Path
from collections import defaultdict

src = Path("data/train/sft_mix_v3.jsonl")
out = Path("docs/artifacts/amharic_review_sample_v3.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
by_dir = defaultdict(list)
with src.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        text = " ".join(m["content"] for m in obj.get("messages", []))
        if any("\u1200" <= ch <= "\u137F" for ch in text):
            by_dir[obj.get("direction", "unk")].append(obj)

rng = random.Random(42)
sample = []
per = max(1, 32 // max(1, len(by_dir)))
for _d, rows in by_dir.items():
    rng.shuffle(rows)
    sample.extend(rows[:per])
rng.shuffle(sample)
sample = sample[:32]
with out.open("w", encoding="utf-8") as f:
    for r in sample:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {out} n={len(sample)} dirs={ {k: len(v) for k, v in by_dir.items()} }")
PY

wc -l data/train/sft_mix_v3.jsonl data/train/cpt_mix_v3.jsonl
echo "[prep_v3] Done"
