#!/usr/bin/env python3
"""Pick a local GGUF, pick a prompt, send it, print the reply.

Uses llama-cpp-python (same path as eval/run_gguf_eval.py).

    source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
    conda activate /scratch/nz2212/adtc-hackathon/adtc/training/.conda-env
    python eval/try_prompt.py

Numbers work non-interactively too:  python eval/try_prompt.py --model 1 --prompt 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Login nodes cap nproc (here: 128). OpenBLAS otherwise starts one thread per
# core (56 on Jubail login) and numpy import dies with pthread_create EAGAIN.
_LOGIN_THREADS = 4
for _k in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_k, str(_LOGIN_THREADS))

# Slurm redirects stdout to a file; default block-buffering hides progress.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
GGUF = ROOT / "artifacts" / "gguf" / "adapted"

MODELS = [
    {
        "name": "Tebeb Tutor 1.7B Q5_K_M  (English-only STEM tutor, Gate 5 winner)",
        "path": GGUF / "tebeb_tutor_1.7b-Q5_K_M.gguf",
    },
]

SYSTEM_PROMPT = """You are an English STEM tutor.

- Explain clearly for a student: solve step-by-step, give hints without revealing final answers, or diagnose first errors when asked.
- Keep equations, expressions, numbers, variables, operators, and fractions in standard mathematical notation."""


def chat_messages(prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


PROMPTS = [
    {
        "name": "EN solve — tank fraction",
        "text": "Solve: A tank holds 120 liters. It is 3/5 full. How many liters are in the tank?",
    },
    {
        "name": "EN hint — 2x + 7 = 19 (no answer)",
        "text": "A student is stuck on: 2x + 7 = 19. Give one hint without revealing x.",
    },
    {
        "name": "EN first-error — 3/4 + 1/2 = 4/6",
        "text": "A student writes: 3/4 + 1/2 = 4/6. Identify the first mistake, then give one hint (do not give the final answer).",
    },
    {
        "name": "EN science — why ice floats",
        "text": "In one short paragraph, why does ice float on water?",
    },
    {
        "name": "EN MGSM — Janet's ducks",
        "text": (
            "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning "
            "and bakes muffins for her friends every day with four. She sells the remainder "
            "at the farmers' market daily for $2 per fresh duck egg. How much in dollars "
            "does she make every day at the farmers' market?"
        ),
    },
    {"name": "Type my own prompt", "text": None},
]


def pick(title: str, items: list[dict], key: str = "name") -> int:
    print(f"\n{title}")
    for i, item in enumerate(items, start=1):
        extra = ""
        if "path" in item:
            extra = "  [missing]" if not Path(item["path"]).is_file() else ""
        print(f"  {i}. {item[key]}{extra}")
    while True:
        raw = input(f"Choose 1–{len(items)} (q to quit): ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            raise SystemExit(0)
        try:
            n = int(raw)
        except ValueError:
            print("  Enter a number.")
            continue
        if 1 <= n <= len(items):
            return n - 1
        print(f"  Pick between 1 and {len(items)}.")


def default_n_threads() -> int:
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit() and int(slurm) > 0:
        return int(slurm)
    return _LOGIN_THREADS


def apply_thread_caps(n: int) -> None:
    n_s = str(max(1, n))
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = n_s


def load_llm(path: Path, n_ctx: int, n_threads: int, out: Path | None = None):
    apply_thread_caps(n_threads)
    print(f"\nImporting llama_cpp …", flush=True)
    t_imp = time.perf_counter()
    from llama_cpp import Llama

    print(f"Imported llama_cpp in {time.perf_counter() - t_imp:.1f}s", flush=True)
    msg = f"Loading {path.name}  (threads={n_threads}, n_ctx={n_ctx}) …"
    print(msg, flush=True)
    write_out(out, f"# {msg}\n")
    t0 = time.perf_counter()
    llm = Llama(model_path=str(path), n_ctx=n_ctx, n_threads=n_threads, verbose=False)
    done = f"Loaded in {time.perf_counter() - t0:.1f}s"
    print(done, flush=True)
    write_out(out, f"# {done}\n")
    return llm


def load_hf(hf_path: str, out: Path | None = None):
    """Load a merged HF checkpoint via transformers (fallback for unsupported GGUF archs)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    msg = f"Loading HF checkpoint {hf_path} via transformers (device={device}) …"
    print(msg, flush=True)
    write_out(out, f"# {msg}\n")
    t0 = time.perf_counter()

    tok = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else "cpu",
        trust_remote_code=True,
    )
    model.eval()
    done = f"HF model loaded in {time.perf_counter() - t0:.1f}s on {device}"
    print(done, flush=True)
    write_out(out, f"# {done}\n")
    return ("hf", model, tok)


def generate_hf(backend, prompt: str, max_tokens: int, temperature: float) -> str:
    import torch
    _, model, tok = backend
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:  # noqa: BLE001
            text = f"{SYSTEM_PROMPT}\n\n{prompt}"
    except Exception:  # noqa: BLE001
        text = f"{SYSTEM_PROMPT}\n\n{prompt}"
    inputs = tok(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
        )
    gen_ids = out[0][inputs["input_ids"].shape[-1] :]
    return tok.decode(gen_ids, skip_special_tokens=True)


def generate(llm, prompt: str, max_tokens: int, temperature: float) -> str:
    if isinstance(llm, tuple) and llm[0] == "hf":
        return generate_hf(llm, prompt, max_tokens, temperature)
    # Qwen3 GGUF: /no_think cuts CoT for faster, cleaner replies.
    user_prompt = prompt if prompt.lstrip().startswith("/no_think") else f"/no_think\n{prompt}"
    try:
        out = llm.create_chat_completion(
            messages=chat_messages(user_prompt),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return out["choices"][0]["message"]["content"] or ""
    except Exception:  # noqa: BLE001
        # Gemma templates may reject role=system; fold the policy into the user turn.
        folded = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        try:
            out = llm.create_chat_completion(
                messages=[{"role": "user", "content": folded}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return out["choices"][0]["message"]["content"] or ""
        except Exception:  # noqa: BLE001
            out = llm(folded, max_tokens=max_tokens, temperature=temperature, echo=False)
            return out["choices"][0]["text"] or ""


def read_custom_prompt() -> str:
    print("\nType your prompt. End with a blank line (or Ctrl-D).")
    lines: list[str] = []
    try:
        while True:
            line = input()
            if line == "" and lines:
                break
            lines.append(line)
    except EOFError:
        pass
    text = "\n".join(lines).strip()
    if not text:
        raise SystemExit("Empty prompt.")
    return text


def resolve_prompt(idx: int, custom_text: str | None) -> str:
    item = PROMPTS[idx]
    if item["text"] is not None:
        return item["text"]
    if custom_text:
        return custom_text
    return read_custom_prompt()


def format_result(model_name: str, prompt: str, reply: str, elapsed: float) -> str:
    bar = "=" * 72
    lines = [
        "",
        bar,
        f"MODEL   {model_name}",
        f"TIME    {elapsed:.1f}s",
        bar,
        "PROMPT",
        prompt,
        bar,
        "RESPONSE",
        reply,
    ]
    if "</think>" in reply:
        clean = reply.split("</think>")[-1].strip()
        if clean and clean != reply:
            lines.extend([bar, "RESPONSE (after </think>)", clean])
    lines.append(bar)
    return "\n".join(lines) + "\n"


def write_out(path: Path | None, text: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def parse_prompt_indices(raw: str) -> list[int]:
    if raw.strip().lower() == "all":
        return [i for i, p in enumerate(PROMPTS) if p["text"] is not None]
    idxs = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if not 1 <= n <= len(PROMPTS):
            raise SystemExit(f"--prompts: {n} is not in 1–{len(PROMPTS)}")
        idxs.append(n - 1)
    if not idxs:
        raise SystemExit("--prompts is empty")
    return idxs


def run_one(llm, spec: dict, pi: int, custom_text: str | None, args) -> None:
    prompt = resolve_prompt(pi, custom_text)
    print(f"\nGenerating  prompt={pi + 1} ({PROMPTS[pi]['name']})  [system=amharic-policy] …", flush=True)
    t0 = time.perf_counter()
    reply = generate(llm, prompt, max_tokens=args.max_tokens, temperature=args.temperature)
    block = format_result(spec["name"], prompt, reply, time.perf_counter() - t0)
    print(block, end="", flush=True)
    write_out(args.out, block)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=int, default=None, help="1-based index into MODELS")
    ap.add_argument("--prompt", type=int, default=None, help="1-based index into PROMPTS")
    ap.add_argument("--prompts", default=None, help="Comma-separated 1-based indices, or 'all'")
    ap.add_argument("--text", default=None, help="Custom prompt (with --prompt pointing at 'type my own')")
    ap.add_argument("--out", type=Path, default=None, help="Append prompt + reply to this log file")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--n-threads", type=int, default=None)
    args = ap.parse_args()

    n_threads = args.n_threads if args.n_threads is not None else default_n_threads()
    batch = args.prompts is not None or (args.prompt is not None and not sys.stdin.isatty())

    if not sys.stdin.isatty() and (args.model is None or (args.prompt is None and args.prompts is None)):
        raise SystemExit("Non-interactive stdin: pass --model N and --prompt N (or --prompts …)")

    if args.model is None:
        mi = pick("Models", MODELS)
    else:
        mi = args.model - 1
        if not 0 <= mi < len(MODELS):
            raise SystemExit(f"--model must be 1–{len(MODELS)}")

    spec = MODELS[mi]
    hf_path = spec.get("hf_path")
    path = Path(spec["path"])

    if args.out:
        args.out = args.out.resolve()
        backend_tag = f"hf={hf_path}" if hf_path else f"gguf={path}"
        header = (
            f"# try_prompt  model={mi + 1}  {spec['name']}\n"
            f"# {backend_tag}\n"
            f"# slurm_job={os.environ.get('SLURM_JOB_ID', '-')}\n"
            f"# host={os.environ.get('SLURMD_NODENAME', os.environ.get('HOSTNAME', '-'))}\n"
            f"# system=english-stem-tutor\n"
        )
        write_out(args.out, header)

    if hf_path:
        if not Path(hf_path).is_dir():
            raise SystemExit(f"HF checkpoint not found: {hf_path}")
        llm = load_hf(hf_path, out=args.out)
    else:
        if not path.is_file():
            raise SystemExit(f"GGUF not found: {path}")
        llm = load_llm(path, n_ctx=args.n_ctx, n_threads=n_threads, out=args.out)

    if args.prompts is not None:
        for pi in parse_prompt_indices(args.prompts):
            run_one(llm, spec, pi, args.text, args)
        if args.out:
            print(f"wrote {args.out}")
        return

    once = args.prompt is not None
    while True:
        if args.prompt is None:
            pi = pick("Prompts", PROMPTS)
        else:
            pi = args.prompt - 1
            if not 0 <= pi < len(PROMPTS):
                raise SystemExit(f"--prompt must be 1–{len(PROMPTS)}")

        run_one(llm, spec, pi, args.text, args)

        if once or batch or not sys.stdin.isatty():
            break
        nxt = input("\n[enter] another prompt  |  m change model  |  q quit  → ").strip().lower()
        if nxt in {"q", "quit"}:
            break
        if nxt == "m":
            mi = pick("Models", MODELS)
            spec = MODELS[mi]
            hf_path = spec.get("hf_path")
            path = Path(spec["path"])
            if hf_path:
                if not Path(hf_path).is_dir():
                    raise SystemExit(f"HF checkpoint not found: {hf_path}")
                llm = load_hf(hf_path, out=args.out)
            else:
                if not path.is_file():
                    raise SystemExit(f"GGUF not found: {path}")
                llm = load_llm(path, n_ctx=args.n_ctx, n_threads=n_threads, out=args.out)
        args.prompt = None
        args.text = None

    if args.out:
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
