# ADTC Tooling (Phase 0 freeze)

All Gate 1 local scores must use this pinned profiler. Reinstall from the exact SHA below if the environment is rebuilt.

## Python

- Version: **3.11.1**
- Project venv: `adtc/tools/adtc-profiler-venv/`

## ADTC Profiler

- Repo: https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler
- **Pinned commit:** `ac2e137dca65ea3b09d997774f17dd8907b489fb`
- Package version reported: `0.1.0`
- Install (pinned) — **laptop / local venv**:

```bash
python3 -m venv adtc/tools/adtc-profiler-venv
source adtc/tools/adtc-profiler-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git@ac2e137dca65ea3b09d997774f17dd8907b489fb"
```

- CLI: `adtc/tools/adtc-profiler-venv/bin/adtc-profiler`

- Install (pinned) — **Jubail HPC into scratch conda** (`adtc/training/.conda-env`):
  - Dependency `llama-cpp-python` **compiles from source** and needs `gcc` + `g++`.
  - Login node often fails (`CC=gcc` missing, or `ninja: posix_spawn: Operation not permitted`).
  - Use a compute job:

```bash
cd adtc/hpc
sbatch setup_profiler.sbatch
```

  Equivalent manual steps on a compute node (or after `module load`):

```bash
module purge
module load gcc/12.2.0      # not gcc/13.2.0 — that module is C-only (no g++)
module load cmake/3.31.2
export CC="$(which gcc)" CXX="$(which g++)"
export CMAKE_ARGS="-DGGML_NATIVE=OFF"
export CMAKE_GENERATOR="Unix Makefiles"
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate /scratch/nz2212/adtc-hackathon/adtc/training/.conda-env
python -m pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git@ac2e137dca65ea3b09d997774f17dd8907b489fb"
```

## llama-bench (llama.cpp)

Host Homebrew install was avoided; binaries live in-repo under `adtc/tools/` (gitignored).

- Release: **b10451** (ggml-org/llama.cpp)
- Archive (laptop): `llama-b10451-bin-macos-arm64.tar.gz`
- Archive (Jubail HPC): `llama-b10451-bin-ubuntu-x64.tar.gz`
- Binary path: `adtc/tools/llama.cpp/llama-b10451/llama-bench`

# Jubail (RHEL8): official ubuntu-x64 binaries need GLIBC_2.34 — **build from source** instead:
```bash
cd adtc/hpc && sbatch setup_build_llama_cpp.sbatch
# installs tools/llama.cpp/llama-b10451/{llama-bench,llama-cli,llama-quantize}
source adtc/hpc/env.sh
source adtc/hpc/profiler_env.sh
```

Laptop (macos arm64 release tarball still OK):
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
