#!/usr/bin/env bash
# English-only Qwen3-1.7B: SFT → merge → GGUF → eval → profile.
# Usage (from adtc/hpc):
#   bash submit_chain.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

J_PREP=$(sbatch --parsable prepare_mix.sbatch)
echo "prep ${J_PREP}"

J_SFT=$(sbatch --parsable --dependency=afterok:"${J_PREP}" train_sft.sbatch)
echo "sft ${J_SFT}"

J_MERGE=$(sbatch --parsable --dependency=afterok:"${J_SFT}" merge_lora.sbatch)
echo "merge ${J_MERGE}"

J_GGUF=$(sbatch --parsable --dependency=afterok:"${J_MERGE}" convert_gguf.sbatch)
echo "gguf ${J_GGUF}"

J_HF=$(sbatch --parsable --dependency=afterok:"${J_MERGE}" eval_hf.sbatch)
echo "eval_hf ${J_HF}"

J_Q4=$(sbatch --parsable --export=ALL,V6_QUANT=Q4_K_M --dependency=afterok:"${J_GGUF}" eval_gguf.sbatch)
echo "eval_gguf_q4 ${J_Q4}"

J_Q5=$(sbatch --parsable --export=ALL,V6_QUANT=Q5_K_M --dependency=afterok:"${J_GGUF}" eval_gguf.sbatch)
echo "eval_gguf_q5 ${J_Q5}"

J_PROF=$(sbatch --parsable --dependency=afterok:"${J_GGUF}" profile_gguf.sbatch)
echo "profile ${J_PROF}"

echo "----"
echo "chain: ${J_PREP}->${J_SFT}->${J_MERGE}->{${J_GGUF}->{${J_Q4},${J_Q5},${J_PROF}},${J_HF}}"
echo "monitor: squeue -u \$USER"
echo "artifacts: docs/artifacts/v6/"
