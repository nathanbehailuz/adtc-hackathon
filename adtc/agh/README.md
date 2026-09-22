# Africa GPU Hub (aghcloud.ai) — TebebAI runbook

Run the ADTC v7 pipeline on [Africa GPU Hub](https://aghcloud.ai/) / [console](https://console.aghcloud.ai/).  
Platform docs: [GPUhub Quickstart](https://docs.gpuhub.com/quickstart) (AGH uses the same container product).

This folder is the AGH counterpart to [`../hpc/`](../hpc/) (Jubail Slurm). **Prefer `adtc/agh/` for new work.** Keep `hpc/` as legacy only.

## One-time: launch an instance

1. Open [console.aghcloud.ai](https://console.aghcloud.ai/) → **Launch Instance**.
2. Pick region, a single GPU with **≥24 GB VRAM** (bf16-friendly for QLoRA), and a Deep Learning image with Miniconda/CUDA.
3. Expand the **data disk** if needed (datasets + checkpoints live there).
4. Initialize **File Storage** for the region ([docs](https://docs.gpuhub.com/data/file-storage)) so `/root/autodl-fs` mounts.
5. Connect via **SSH** ([docs](https://docs.gpuhub.com/container-instance/ssh)) or JupyterLab.

```bash
# Example SSH (port/host from the console card):
ssh -p <PORT> root@connect.<region>.gpuhub.com
```

Billing starts while status is **Running**. **Shut down** when idle to stop GPU charges ([quickstart notes](https://docs.gpuhub.com/quickstart)).

## Disk layout (use these paths)

| Mount | Path | TebebAI use |
|-------|------|-------------|
| System (~30 GB) | `/` | conda env small packages only |
| Data disk | `/root/gpuhub-tmp` | HF caches, runs, GGUF build (`WORK`) |
| File Storage | `/root/autodl-fs` | durable repo + mix JSONL + exports |
| Public (RO) | `/root/gpuhub-pub` | optional platform data |

`env.sh` sets `WORK=/root/gpuhub-tmp/adtc` when the data disk exists.

## Clone / sync the repo

```bash
# Durable copy on File Storage (recommended)
mkdir -p /root/autodl-fs
cd /root/autodl-fs
git clone https://github.com/nathanbehailuz/adtc-hackathon.git
cd adtc-hackathon/adtc/agh
```

Optional: also keep a working tree on the data disk and sync with `rsync` if File Storage IO is slow.

## Long jobs: always use tmux/screen

SSH drops kill foreground processes. Use [Run in Background](https://docs.gpuhub.com/container-instance/run-in-background):

```bash
apt-get update && apt-get install -y tmux   # if missing
tmux new -s adtc
# ... run scripts ...
# Detach: Ctrl-b d
# Reattach: tmux attach -t adtc
```

## Quick start (v7)

```bash
cd /root/autodl-fs/adtc-hackathon/adtc/agh
source env.sh
bash setup_env.sh
bash setup_llama_cpp.sh   # llama.cpp b10451 convert + llama-quantize
bash download_models.sh
# Full chain (prep → SFT → merge → GGUF → eval → profile):
bash run_chain.sh
```

Or run stages one at a time:

| Script | Role |
|--------|------|
| `setup_env.sh` | Conda env + CUDA torch + training requirements |
| `setup_llama_cpp.sh` | llama.cpp **b10451** source + `llama-quantize` / `llama-bench` |
| `download_models.sh` | `Qwen/Qwen3-1.7B` into `HF_HOME` |
| `prepare_mix.sh` | Build `sft_mix_v7.jsonl` |
| `train_sft.sh` | QLoRA SFT (GPU) |
| `merge_lora.sh` | Adapter → merged HF |
| `convert_gguf.sh` | f16 + Q8/Q6/Q5/Q4 |
| `eval_hf.sh` | Frozen HF eval |
| `eval_gguf.sh` | GGUF eval (`V7_QUANT=Q4_K_M` etc.) |
| `profile_gguf.sh` | Profiler / Gate 5 pick |
| `judge_smoke.sh` | Format-leak + multi-part tutoring checklist |
| `try_prompt.sh` | Qualitative smoke |
| `run_chain.sh` | Sequential fail-fast chain |

Logs: `adtc/agh/logs/<stage>.log`.

Resume after a failed convert (merged HF already on disk):

```bash
cd ~/adtc-hackathon/adtc/agh   # or /root/autodl-fs/adtc-hackathon/adtc/agh
git pull
START_STAGE=convert_gguf bash run_chain.sh
# or just: bash convert_gguf.sh
```

## Secrets

Put `HF_TOKEN=...` in `adtc/.env` (gitignored). `env.sh` loads it.

## After training

1. Copy winner GGUF / reports off the instance (File Storage, `scp`, or HF upload).
2. **Shut down** the instance in the console.
3. Instances shut down continuously for **15 days** may be released — keep backups on File Storage or locally.
