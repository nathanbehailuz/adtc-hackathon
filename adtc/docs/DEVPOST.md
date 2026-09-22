# TebebAI — Devpost description (Gate 2)

Paste into Devpost as-is. Numbers below match v7 / `REPORT.md` (deploy: Q4_K_M). Do not paste v6 Jubail profiler TPS as official Gate 2 scores.

---

## Inspiration

**TebebAI** takes its name from *tebeb*, meaning wisdom.

The starting constraint was simple: a STEM tutor is only useful if it still works when the internet does not. Many students we care about are on ordinary laptops, in places where cloud APIs, GPUs, and reliable connectivity cannot be assumed. So TebebAI had to be a complete local tutor — not a demo that quietly depends on a remote model.

That forced every design choice to serve the same goal. The model had to fit in laptop memory, run on CPU through `llama.cpp`, and still behave like a tutor: explain, hint, and catch the first mistake instead of dumping an answer.

This version is English-first on **Qwen3-1.7B**, fine-tuned for those tutoring behaviors and shipped as a single quantized GGUF. Amharic remains part of the longer-term vision. For this release, the job was to ship a stable, reproducible, fully offline system.

## What it does

TebebAI is a **tutor, not an answer lookup tool**. It is trained around four behaviors:

| Behavior | What the student gets |
| --- | --- |
| **Solve** | Numbered steps with brief justifications, then a clearly labeled final answer |
| **Explain** | A short explanation of a STEM idea or a piece of reasoning |
| **Hint** | One next step, why it is valid, and a check question — without revealing the answer |
| **First-error diagnosis** | Names the first wrong step, explains why it is wrong, and gives a targeted hint |

Example prompts:

* Triangle first-error (base 10, height 4, student says area is 40) — diagnose, hint, no numeric dump.
* Betty’s $100 wallet — one hint + why + check question, no dollar total.

The model is one **GGUF file** (~1.1 GB at Q4_K_M). After `download_model.sh`, it runs locally through `llama.cpp` or `python chat.py`. No API keys, no cloud calls, no model router.

## How we built it

We built a reproducible train → eval → deploy pipeline (HPC / GPU instance, then laptop inference):

```text
GSM8K + SciQ + authored tutoring
      ↓
10,636-example English SFT mix
      ↓
QLoRA supervised fine-tuning
      ↓
Merge LoRA into Qwen3-1.7B
      ↓
GGUF conversion (llama.cpp b10451)
      ↓
Q4 / Q5 / Q6 quantization sweep
      ↓
Frozen eval + tutoring judge-smoke
      ↓
Q4_K_M deployment model
```

### Data

The v7 mix has **10,636** examples:

* **7,473** from GSM8K (train), rewritten into tutoring tasks
* **3,000** from SciQ, rewritten the same way
* **163** hand-authored multi-constraint tutoring examples (hint, first-error, multi-part scaffolding)

GSM8K targets were cleaned so the model would not learn dataset markup (`####`, `<<expr=val>>`). Training rows were also deduplicated against the frozen English STEM and AfriMGSM evaluation sets.

### Training

Base model: **Qwen3-1.7B**.

We ran **QLoRA SFT** (4-bit nf4, LoRA **r=32 / α=64**, **2 epochs**), then merged the adapter into the base checkpoint. Relative to the previous release, that is a capacity bump (r 16→32, 1→2 epochs) plus cleaner tutoring targets.

### Quantization and deployment

After the merge we converted to GGUF and compared Q4_K_M, Q5_K_M, and Q6_K on English frozen accuracy versus size.

**Q4_K_M** is the deploy pick: English accuracy stayed within about **1 percentage point** of Q5, at a smaller footprint (~**1.1 GB**). Q5 remains a documented alternate.

Qwen3 extended thinking is off at eval and deploy (`enable_thinking=False` on Hugging Face; `/no_think` for GGUF) so replies stay short and consistent for tutoring.

## Challenges we ran into

### Competing objectives

ADTC scores quality, throughput, and memory together. A heavier quant can preserve more quality and cost RAM; a lighter one is faster and smaller but can slip on STEM answers. The Q4/Q5/Q6 sweep was how we made that trade-off empirically instead of guessing.

Official ADTC Standard Laptop profiler numbers for this v7 GGUF are still the source of truth for TPS and peak RSS. We do not treat earlier node-level profiler runs as laptop results.

### Tutoring quality is not the same as answer accuracy

Held-out English STEM accuracy and tutoring-behavior scores move differently. On frozen English suites, the v7 Q4 GGUF is about **43%** on AfriMGSM EN and EN STEM holdout, while the custom tutoring rubric is **100%** on format/behavior checks. A tutoring judge-smoke set (8 prompts) had **no GSM8K markup leaks** and a **5/8** full multi-part checklist pass.

Those are different capabilities. We do not collapse them into one headline accuracy number.

### Scope

Earlier work explored Amharic–English tutoring. Tokenizer fragmentation, thin high-quality Amharic tutoring data, and weak Amharic eval made that track unfinished. We cut scope to English-first so we could finish a system that trains, evaluates, quantizes, and runs offline end to end.

A first-round model also leaked GSM8K markup and sometimes dumped answers on hint/diagnose prompts. v7 exists largely to fix that: clean targets, authored tutoring, and a stricter tutor system prompt.

## Accomplishments that we're proud of

The pipeline is complete: mix, QLoRA, merge, GGUF, quantization, frozen eval, tutoring smoke tests, and local chat.

v7 improved the previous English checkpoint on the same frozen suites (Hugging Face merged model): AfriMGSM EN **39.2% → 44.4%**, EN STEM holdout **37.0% → 47.0%**, custom tutoring **98% → 100%**. The deploy Q4 GGUF holds **42.8% / 43.0% / 100%** on those three suites.

The thing we care about most is behavioral: the shipped model no longer emits `####` / `<<>>`, and it can hint or diagnose without treating every prompt as “print the answer.” It still runs as **one local file**, with no network after download.

## What we learned

### Small models can be useful specialists

A 1.7B model will not match a cloud model on every reasoning task. Focused fine-tuning still changes how it talks to a student. Ten thousand tutoring examples, plus a small authored bank, were enough to specialise solve / explain / hint / first-error.

### Scope discipline ships systems

The Amharic experiments were useful diagnostics, not a finished product. Cutting that scope is why we have a reproducible English tutor instead of a bilingual prototype that never quite ran.

### Training quantization ≠ deploy quantization

QLoRA 4-bit is a training-memory trick. GGUF Q4/Q5/Q6 is a separate deploy choice. Measuring them separately is why Q4_K_M won this round on the accuracy–size trade-off.

### The training target is the product

If GSM8K markup is in the assistant text, the model will leak it at inference. Cleaning the mix mattered as much as adding more rows.

## What's next for TebebAI

### 1. Separate the evaluation questions

We want a larger suite that reports **STEM answer accuracy**, **reasoning quality**, **tutoring behavior**, and **instruction following** as distinct scores — plus a clean comparison against stock Qwen3-1.7B on the same prompts.

### 2. Bring Amharic back

The bilingual vision is still the point long-term. A later version needs native Amharic tutoring data and a base model that tokenizes Amharic well. Continued pretraining stays on the table if SFT is not enough.

### 3. Make the product layer feel like a tutor

`chat.py` is intentionally thin. Next is a simple offline app: open it, ask a question, get a hint — no GGUF paths or CLI flags.

### 4. Measure on the target laptop

Rerun the official profiler on a physical **8 GB ADTC Standard Laptop** (throughput, peak RSS, CPU, temperature, throttling) and revisit the quant pick if the laptop disagrees with our current Q4 choice.

### 5. Test with students

Benchmarks are a filter. The real test is whether hint-and-diagnose tutoring helps learners in classrooms and community settings where the internet cannot be assumed.

---

*Training and evaluation used GPU resources on Shadeform. Earlier pipeline work used High Performance Computing resources at New York University Abu Dhabi.*
