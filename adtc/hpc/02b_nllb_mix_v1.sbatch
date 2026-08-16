#!/bin/bash
#SBATCH --job-name=adtc_nllb_mix_v1
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Real NLLB-200 machine translation → am_stem → sft_mix_v1 + Amharic review sample.
set -euo pipefail
HPC_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR unset; sbatch from adtc/hpc/}"
# shellcheck disable=SC1091
source "${HPC_DIR}/env.sh"
cd "${ADTC_ROOT}"

if [[ ! -f data/train/en_stem_sft_v0.jsonl ]]; then
  echo "error: missing data/train/en_stem_sft_v0.jsonl" >&2
  exit 1
fi

python data/build_translate_am.py \
  --in data/train/en_stem_sft_v0.jsonl \
  --out data/train/am_stem_sft_nllb_v1.jsonl \
  --backend nllb

MIX_CMD=(
  python data/mix_sft.py
  --en-stem data/train/en_stem_sft_v0.jsonl
  --am-stem data/train/am_stem_sft_nllb_v1.jsonl
  --eval data/eval/custom_tutoring_v0.jsonl data/eval/en_stem_holdout_v0.jsonl
  --out data/train/sft_mix_v1.jsonl
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

# Stratified Amharic review sample (~40 rows)
python - <<'PY'
import json, random
from pathlib import Path
from collections import defaultdict

src = Path("data/train/sft_mix_v1.jsonl")
out = Path("docs/artifacts/amharic_review_sample_v1.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
by_dir = defaultdict(list)
with src.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        text = " ".join(m["content"] for m in obj.get("messages", []))
        # Amharic Ethiopic block
        if any("\u1200" <= ch <= "\u137F" for ch in text):
            by_dir[obj.get("direction", "unk")].append(obj)

rng = random.Random(42)
sample = []
per = max(1, 40 // max(1, len(by_dir)))
for d, rows in by_dir.items():
    rng.shuffle(rows)
    sample.extend(rows[:per])
rng.shuffle(sample)
sample = sample[:50]
with out.open("w", encoding="utf-8") as f:
    for r in sample:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {out} n={len(sample)} dirs={ {k: len(v) for k,v in by_dir.items()} }")
PY

# Direction counts note
python - <<'PY'
import json
from collections import Counter
from pathlib import Path
c = Counter()
with Path("data/train/sft_mix_v1.jsonl").open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            c[json.loads(line).get("direction", "unk")] += 1
Path("docs/artifacts/sft_mix_v1_direction_counts.json").write_text(
    json.dumps(dict(c), indent=2) + "\n", encoding="utf-8"
)
print(dict(c))
PY

echo "[nllb_mix] Done -> data/train/sft_mix_v1.jsonl"
