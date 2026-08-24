#!/usr/bin/env bash
# ADTC v6 chain — English-only Qwen3-1.7B SFT → merge → GGUF → eval → profile.
# Usage (from adtc/hpc):
#   bash submit_v6_chain.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

J_PREP=$(sbatch --parsable 02h_prepare_mix_v6.sbatch)
echo "prep ${J_PREP}"

J_SFT=$(sbatch --parsable --dependency=afterok:"${J_PREP}" 04g_train_sft_v6.sbatch)
echo "sft ${J_SFT}"

J_MERGE=$(sbatch --parsable --dependency=afterok:"${J_SFT}" 05f_merge_lora_v6.sbatch)
echo "merge ${J_MERGE}"

J_GGUF=$(sbatch --parsable --dependency=afterok:"${J_MERGE}" 08h_convert_gguf_v6.sbatch)
echo "gguf ${J_GGUF}"

J_HF=$(sbatch --parsable --dependency=afterok:"${J_MERGE}" 12i_eval_v6_hf.sbatch)
echo "eval_hf ${J_HF}"

J_Q4=$(sbatch --parsable --export=ALL,V6_QUANT=Q4_K_M --dependency=afterok:"${J_GGUF}" 12j_eval_v6_gguf.sbatch)
echo "eval_gguf_q4 ${J_Q4}"

J_Q5=$(sbatch --parsable --export=ALL,V6_QUANT=Q5_K_M --dependency=afterok:"${J_GGUF}" 12j_eval_v6_gguf.sbatch)
echo "eval_gguf_q5 ${J_Q5}"

J_PROF=$(sbatch --parsable --dependency=afterok:"${J_GGUF}" 09b_profile_gguf_v6.sbatch)
echo "profile ${J_PROF}"

echo "----"
echo "v6 chain: ${J_PREP}->${J_SFT}->${J_MERGE}->{${J_GGUF}->{${J_Q4},${J_Q5},${J_PROF}},${J_HF}}"
echo "monitor: squeue -u \$USER"
echo "artifacts: docs/artifacts/v6/"
