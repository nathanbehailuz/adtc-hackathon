#!/bin/bash
# Submit Amharic STEM SFT v2 chain (download → mix → train → merge → gguf → eval).
# Run from adtc/hpc/:
#   bash submit_v2_chain.sh
set -euo pipefail
cd "$(dirname "$0")"

J_DL=$(sbatch --parsable 02c_download_am_gsm8k.sbatch)
echo "submitted download ${J_DL}"

J_PREP=$(sbatch --parsable --dependency=afterok:"${J_DL}" 02d_prepare_mix_v2.sbatch)
echo "submitted prepare_mix_v2 ${J_PREP} afterok:${J_DL}"

J_SFT=$(sbatch --parsable --dependency=afterok:"${J_PREP}" 04c_train_sft_v2.sbatch)
echo "submitted train_sft_v2 ${J_SFT} afterok:${J_PREP}"

J_MERGE=$(sbatch --parsable --dependency=afterok:"${J_SFT}" 05b_merge_lora_v2.sbatch)
echo "submitted merge_v2 ${J_MERGE} afterok:${J_SFT}"

J_GGUF=$(sbatch --parsable --dependency=afterok:"${J_MERGE}" 08b_convert_gguf_v2.sbatch)
echo "submitted convert_gguf_v2 ${J_GGUF} afterok:${J_MERGE}"

J_EVAL=$(sbatch --parsable --dependency=afterok:"${J_GGUF}" 12_eval_v2.sbatch)
echo "submitted eval_v2 ${J_EVAL} afterok:${J_GGUF}"

echo "----"
echo "v2 chain: ${J_DL} -> ${J_PREP} -> ${J_SFT} -> ${J_MERGE} -> ${J_GGUF} -> ${J_EVAL}"
echo "monitor: squeue -u \$USER"
