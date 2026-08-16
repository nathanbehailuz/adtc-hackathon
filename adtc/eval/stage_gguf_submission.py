#!/usr/bin/env python3
"""Prepare a temporary ADTC submission dir pointing at a local GGUF for profiling."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_META = {
    "team_id": "adtc_hpc_screen",
    "domain": "math_scientific_reasoning",
    "language_scope": ["en", "am"],
    "african_alpha_claim": True,
    "budget_laptop_claim": True,
    "submitter": {
        "name": "ADTC HPC",
        "email": "hpc@example.com",
        "github_handle": "adtc-hpc",
    },
    "cross_disciplinary_pairing": {
        "discipline": "education",
        "load_bearing": True,
        "description": "Offline bilingual Amharic/English STEM tutor screen run.",
    },
    "test_prompts": [
        {
            "prompt_id": "tp_001",
            "prompt": "A student writes 3/4 + 1/2 = 4/6. Identify the first mistake, then give one hint.",
        },
        {
            "prompt_id": "tp_002",
            "prompt": "አንድ ተማሪ 2x + 5 = 13 ብሎ ጽፎ x = 9 አለ። የመጀመሪያውን ስህተት ጠቁም።",
        },
    ],
    "model": {
        "name": "PLACEHOLDER",
        "runtime": "llama.cpp",
        "quantization": "GGUF",
        "parameters_estimate": "unknown",
        "packaging": "binary_bundle",
    },
    "_runtime": {"model_path": "model/model.gguf"},
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--quant", type=str, default="GGUF")
    p.add_argument("--params", type=str, default="unknown")
    args = p.parse_args()

    gguf = args.gguf.resolve()
    if not gguf.is_file():
        raise SystemExit(f"GGUF not found: {gguf}")

    out = args.out_dir.resolve()
    model_dir = out / "model"
    if out.exists():
        shutil.rmtree(out)
    model_dir.mkdir(parents=True)

    dest = model_dir / "model.gguf"
    # Prefer hardlink to save disk; fall back to symlink then copy.
    try:
        dest.hardlink_to(gguf)
    except OSError:
        try:
            dest.symlink_to(gguf)
        except OSError:
            shutil.copy2(gguf, dest)

    meta = json.loads(json.dumps(DEFAULT_META))
    meta["model"]["name"] = args.name or gguf.stem
    meta["model"]["quantization"] = args.quant
    meta["model"]["parameters_estimate"] = args.params
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (out / "download_model.sh").write_text(
        "#!/usr/bin/env bash\necho 'model already staged locally'\nexit 0\n",
        encoding="utf-8",
    )
    (out / "download_model.sh").chmod(0o755)
    print(f"staged {gguf} -> {out}")


if __name__ == "__main__":
    main()
