#!/bin/bash
# Submit Gemma-only Amharic v4: AfriqueLLM mix → SFT → GGUF → eval (no CPT).
#   cd adtc/hpc && bash submit_v4_chain.sh
set -euo pipefail
cd "$(dirname "$0")"

J_PREP=$(sbatch --parsable 02f_prepare_mix_v4.sbatch)
echo "submitted prepare_mix_v4 ${J_PREP}"

J_SFT=$(sbatch --parsable --dependency=afterok:"${J_PREP}" 04e_train_sft_v4.sbatch)
echo "submitted train_sft_v4 ${J_SFT} afterok:${J_PREP}"

J_MERGE=$(sbatch --parsable --dependency=afterok:"${J_SFT}" 05d_merge_lora_v4.sbatch)
echo "submitted merge_v4 ${J_MERGE} afterok:${J_SFT}"

J_GGUF=$(sbatch --parsable --dependency=afterok:"${J_MERGE}" 08d_convert_gguf_v4.sbatch)
echo "submitted convert_gguf_v4 ${J_GGUF} afterok:${J_MERGE}"

J_EVAL=$(sbatch --parsable --dependency=afterok:"${J_GGUF}" 12e_eval_v4.sbatch)
echo "submitted eval_v4 ${J_EVAL} afterok:${J_GGUF}"

J_HF=$(sbatch --parsable --dependency=afterok:"${J_MERGE}" 12f_eval_v4_hf.sbatch)
echo "submitted eval_v4_hf ${J_HF} afterok:${J_MERGE}"

echo "----"
echo "v4 chain: ${J_PREP} -> ${J_SFT} -> ${J_MERGE} -> {${J_GGUF}->${J_EVAL}, ${J_HF}}"
echo "monitor: squeue -u \$USER"
