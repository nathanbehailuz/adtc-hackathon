# ADTC Tooling (Phase 0 freeze)

All Gate 1 local scores must use this pinned profiler. Reinstall from the exact SHA below if the environment is rebuilt.

## Python

- Version: **3.11.1**
- Project venv: `adtc/tools/adtc-profiler-venv/`

## ADTC Profiler

- Repo: https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler
- **Pinned commit:** `ac2e137dca65ea3b09d997774f17dd8907b489fb`
- Package version reported: `0.1.0`
- Install (pinned):

```bash
python3 -m venv adtc/tools/adtc-profiler-venv
source adtc/tools/adtc-profiler-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git@ac2e137dca65ea3b09d997774f17dd8907b489fb"
```

- CLI: `adtc/tools/adtc-profiler-venv/bin/adtc-profiler`

## llama-bench (llama.cpp)

Host Homebrew install was avoided; binaries live in-repo under `adtc/tools/` (gitignored).

- Release: **b10451** (ggml-org/llama.cpp)
- Archive: `llama-b10451-bin-macos-arm64.tar.gz`
- Binary path: `adtc/tools/llama.cpp/llama-b10451/llama-bench`
- Machine: Apple M1 (arm64)

```bash
export PATH="$(pwd)/adtc/tools/llama.cpp/llama-b10451:$PATH"
which llama-bench
```

## Participant smoke command

```bash
export PATH="$(pwd)/adtc/tools/llama.cpp/llama-b10451:$PATH"
source adtc/tools/adtc-profiler-venv/bin/activate
bash adtc-2026-submission-template/download_model.sh
adtc-profiler run \
  --submission "$(pwd)/adtc-2026-submission-template" \
  --mode participant \
  --skip-accuracy \
  --output "$(pwd)/adtc/docs/artifacts/phase0_submission_smoke.json"
```

## Model hosting (final weights)

- **Chosen:** public Hugging Face GGUF repo (credential-free URL in `download_model.sh`)
- **Backup:** GitHub Release asset on the submission repo
- Phase 0 keeps the template SmolLM2 public HF URL; the fine-tuned GGUF URL replaces it in Phase 7
