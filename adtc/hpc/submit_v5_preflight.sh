#!/usr/bin/env bash
# Milestone A: submit mix prep + HF/GGUF preflight only (no CPT/SFT training).
# Usage (from adtc/hpc):
#   bash submit_v5_preflight.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

J_PREP=$(sbatch --parsable 02g_prepare_mix_v5.sbatch)
echo "submitted prepare_mix_v5 ${J_PREP}"

J_HF=$(sbatch --parsable --dependency=afterok:"${J_PREP}" 08e_preflight_hf_v5.sbatch)
echo "submitted preflight_hf_v5 ${J_HF} afterok:${J_PREP}"

for alias in qwen35_extcm qwen35_afrique qwen3_afrique; do
  J=$(sbatch --parsable --export=ALL,V5_ALIAS="${alias}" --dependency=afterok:"${J_PREP}" 08f_preflight_gguf_v5.sbatch)
  echo "submitted preflight_gguf_v5 alias=${alias} job=${J} afterok:${J_PREP}"
done

echo "----"
echo "v5 preflight submitted. Do NOT run submit_v5_chain.sh until you review:"
echo "  docs/artifacts/v5/model_preflight.json"
echo "  docs/artifacts/v5/gguf_preflight_*.json"
echo "  docs/artifacts/v5/*counts*"
echo "monitor: squeue -u \$USER"
