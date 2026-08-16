
Small Multilingual and Specialist LLMs for Offline Deployment on the ADTC Standard Laptop
Executive summary
The literature supports a fairly clear strategy for the Africa Deep Tech Challenge: do not maximize parameter count; maximize the competition score along an accuracy–throughput–memory Pareto frontier. The current ADTC evaluator gives 50% weight to accuracy, 30% to generation speed, and 20% to memory efficiency, imposes a 7 GB peak-RSS budget, and applies a thermal penalty above 85°C. The official submission format also points to a single GGUF model loaded through llama.cpp. 

Your uploaded methodology is already directionally very good: it proposes pre-fine-tuning screening, bilingual/domain adaptation, QLoRA, conditional continued pretraining, a Q8→Q3 quantization sweep, Pareto selection, and explicit ablations. 
 The current literature strongly supports most of that design. I would modify it in several important ways.

The main conclusions are:

Question	Literature-grounded recommendation
Largest model that fits, or multiple models?	Neither. Use one dense specialist model, probably 1.7–4B, chosen by measured ADTC score. A 7B Q4 model may technically fit but is likely too slow and RAM-expensive to be optimal.
Multiple specialist LLMs?	Avoid for the final ADTC submission. The current template specifies one _runtime.model_path pointing to one GGUF. Use larger/multiple models during training as teachers or translators, then compress their knowledge into one deployed model. 
Do we need to downsize?	No. Start with a model already near the desired size. Distillation/pruning is a fallback, not a prerequisite.
Do we need quantization?	You should test it almost certainly, even when FP16 fits. For ADTC, Q4–Q6 is the important region. Q4_K_M is a strong starting point, but Q5/Q6 can win if they preserve enough accuracy. Extreme Q2/Q3 should be exploratory.
Fine-tune before or after quantization?	Train the normal checkpoint with LoRA/QLoRA; merge the adapter; convert to GGUF; then perform deployment PTQ. QLoRA's 4-bit representation is a training-memory technique, not your final GGUF precision. 
Can we just fine-tune an already small model?	Yes. This is the preferred strategy. There is no reason to start from a 7B/14B model merely to shrink it later if a 1.7–4B model can acquire the specialist behavior directly.
Runtime translation layer or direct multilingual model?	Direct multilingual model should be the default final architecture. Use machine translation mainly for training-data augmentation and as a diagnostic baseline.
Best low-resource-language strategy?	There is no universal recipe. The strongest pattern is: good base-model selection → diagnose tokenizer/language ability → bilingual SFT → mixed CPT if necessary → SFT again → tokenizer extension only if fragmentation is severe → quantization. Data quality and domain composition matter at least as much as raw target-language volume. 
What about Swahili if nobody on the team speaks it?	Do not make Swahili the default merely because datasets exist. Choose a benchmark-supported language for which a teammate or collaborator can reliably inspect outputs, terminology, and translated training data.
Best initial candidates?	Qwen3-1.7B, Qwen3-4B, and Gemma 3 4B, with Qwen2.5-3B-Instruct as a highly useful middle-size control.
Can Qwen run on 8 GB?	Qwen 0.5/0.6B, 1.7B, 3B, and 4B models can comfortably run quantized on the target. A 7B Q4 model can probably run, but with poor ADTC memory/speed headroom. 

One result from the newest African-language literature is particularly important. AfriqueLLM, accepted to ACL 2026, finds that data composition is the strongest determinant of continued-pretraining gains; adding math, code, and synthetic translated data consistently improves African-language performance. It also finds that architecture can matter more than scale across model families and that strong multilingual ability before adaptation does not reliably predict the best model after adaptation. 
 This is almost a direct experimental justification for treating ADTC model selection as an empirical search rather than simply choosing the largest or most multilingual-looking checkpoint.

For a STEM submission, my recommended final architecture is therefore:

Yes

No

Strong 1.7–4B base model

Direct English + target-language baseline

Bilingual STEM SFT with LoRA / QLoRA

Target-language competence adequate?

Merge adapters

Mixed-language CPT

GGUF conversion

Q8 / Q6 / Q5 / Q4

ADTC profiler + held-out accuracy

Pareto / leaderboard score

Single final GGUF



Show code
That is the architecture I would build toward.

Constraint-driven architecture
The central mistake would be treating “7 GB is available” as meaning “we should use nearly 7 GB.”

ADTC explicitly rewards unused memory. Its efficiency term is currently

[ S_{\mathrm{eff}} = 100\frac{7-\mathrm{PeakRAM}}{7}. ]

The profiler README currently normalizes throughput against 15 tokens/s, while the public challenge page describes speed relative to the fastest submission; this is a small inconsistency in the current official material. For local optimization, the executable profiler's 15-TPS reference is the practical quantity to track, but I would retain the raw TPS as well in case final leaderboard normalization differs. 

This creates a useful break-even rule. Suppose a larger model improves accuracy but loses (\Delta S_{\mathrm{perf}}) points in normalized performance and (\Delta S_{\mathrm{eff}}) points in memory efficiency. Its required accuracy gain is approximately

[ \Delta S_{\mathrm{acc,required}}
0.6\Delta S_{\mathrm{perf}} + 0.4\Delta S_{\mathrm{eff}}. ]

That follows directly from equating the ADTC weighted scores. A model that gains only five accuracy points but gives away twenty normalized throughput points is probably a losing trade even before RAM is considered. 

An illustrative scenario—not an actual benchmark—shows why this matters:

Configuration	Assumed TPS	Assumed peak RAM	(S_\text{perf}) using 15 TPS	(S_\text{eff})	Weighted speed + RAM contribution
1.7B Q4	20	2.0 GB	100	71.4	44.3
3B Q4	14	2.8 GB	93.3	60.0	40.0
4B Q4	10	3.6 GB	66.7	48.6	29.7
7B Q4	5	6.2 GB	33.3	11.4	12.3

Under that hypothetical measurement, the 4B model would need roughly 29 additional Sacc points versus the 1.7B model to overcome its hardware-score deficit. The exact numbers will differ on the Standard Laptop, which is why the profiler—not parameter count—is the decision maker. ADTC's profiler directly measures generation TPS, first-token latency, peak RSS, steady-state RSS and thermals. 

Single model versus a collection of models. For this competition, the evidence and the packaging constraint point in the same direction. The official template currently contains one model/your-model.gguf, one _runtime.model_path, and one llama.cpp runtime. The FAQ also says judges interact with the submitted model, not your application stack. A multi-LLM router may be useful in an ordinary application, but it is a poor fit for this evaluation contract unless ADTC explicitly changes the format. 

There are also systems reasons not to maintain several full LLMs. Sparse Mixture-of-Experts architectures demonstrate that routing can increase capacity without activating every parameter on every token, but inactive experts still create a weight-memory problem. Switch Transformers gained sparse-compute benefits at enormous total parameter counts; QMoE was motivated specifically by the fact that those inactive expert weights make practical deployment memory-intensive. 
 On an 8 GB unified-memory laptop, total resident weights matter enormously.

This does not mean “never use more than one model during development.” The strongest architecture is:

many models during training, one model during inference.

A larger LLM can generate bilingual STEM examples. A dedicated MT system can translate English educational data. A stronger 4B model can serve as a teacher for a 1.7B model. Human/native speakers can filter data. None of those models needs to exist in the final GGUF.

Distillation is particularly attractive if the 4B/1.7B experiment produces the pattern “4B is much smarter, 1.7B is much faster.” Meta's Llama 3.2 provides a large-scale practical example: its 1B and 3B models incorporated logits from 8B and 70B teachers during development. 
 The deployment lesson is exactly what ADTC needs: use a bigger teacher to create a better small student, rather than requiring the teacher at inference.

Thus my architecture preference is:

[ \boxed{\text{single dense specialist model} > \text{model collection} > \text{large MoE}} ]

for the final ADTC artifact.

The one exception would be a genuinely tiny, llama.cpp-supported MoE whose total footprint and measured throughput outperform the dense alternatives. That should be proven by profiling, not assumed from “active parameter” counts. The MoE literature explicitly warns that sparse compute does not remove the total-weight memory burden. 

Candidate model sizes and local feasibility
First, distinguish inference from training.

Running a Q4/Q6 model locally on an 8 GB laptop is realistic. Fine-tuning a 3–4B model on that same laptop is a different problem: QLoRA greatly reduces training memory, but training still requires activations, adapters and optimization state. Your cloud GPU credits are much better used for adaptation; the laptop should be the deployment/profiling target. QLoRA's original NeurIPS paper demonstrated the principle at much larger scale by backpropagating through a frozen 4-bit model into trainable low-rank adapters. 

The following table separates FP16 weight memory, GGUF weight-file size, and an approximate 4K-context runtime RSS range. The latter is deliberately a planning estimate rather than a claimed benchmark: exact peak RSS depends on context length, KV-cache representation, llama.cpp build, mmap behavior, thread count and model architecture. The official ADTC profiler is the authoritative measurement. 

Model	Params	FP16/BF16 weights, approx.	Q8: file / likely RSS	Q6: file / likely RSS	Q4: file / likely RSS	Planning Q4 CPU TPS on 4 effective cores	ADTC outlook
Qwen2.5-0.5B	0.49B	~1.0 GB	0.68 / ~0.9–1.3 GB	0.65 / ~0.9–1.3	0.49 / ~0.8–1.2	~30–60	Very fast, probably too weak for serious STEM
Qwen3-0.6B	0.6B	~1.2 GB	~0.81 / 1.1–1.5	~0.62 / 1.0–1.4	~0.48 / 0.9–1.3	~28–55	Excellent efficiency baseline
Gemma 3 1B	~1B	~2.0 GB	~1.1 / 1.5–2.0	~1.0 / 1.4–1.9	~0.8 / 1.2–1.7	~20–40	Useful tiny baseline
Qwen3-1.7B	1.7B	~3.4 GB	1.83 / ~2.4–3.0	1.42 / ~2.0–2.6	1.11 / ~1.7–2.3	~15–28	Excellent candidate
Qwen2.5-3B-Instruct	3.09B	~6.2 GB	3.62 / ~4.1–4.8	2.79 / ~3.3–4.0	2.10 / ~2.6–3.3	~8–16	Good middle-size control
Llama 3.2 3B	3.21B	~6.4 GB	~3.2 / 3.8–4.6	~2.5 / 3.1–3.9	~1.9 / 2.5–3.3	~8–16	Fits, but less attractive for African multilingualism
Qwen3-4B	4.0B	~8.0 GB	4.28 / ~5.0–5.9	3.31 / ~4.0–4.9	2.50 / ~3.2–4.1	~6–12	Accuracy candidate
Gemma 3 4B	~4B	~8.0 GB	~4.2 / 5.0–5.9	~3.3 / 4.0–4.9	~2.6 / 3.3–4.2	~6–12	Strong multilingual control
Qwen2.5-7B class	~7B	~14 GB	~8+ / >8 GB	~6 / ~6.8–7.8	~4.7 / ~5.5–6.7	~3–7	Q4 can fit; poor competition headroom

Qwen publishes official GGUFs for Qwen2.5-0.5B, Qwen2.5-3B, Qwen3-1.7B and Qwen3-4B. Qwen3-4B's official files are 2.50 GB at Q4_K_M, 3.31 GB at Q6_K and 4.28 GB at Q8_0; Qwen2.5-3B's are 2.10, 2.79 and 3.62 GB respectively. Qwen3-1.7B's representative GGUFs are roughly 1.11 GB Q4_K_M, 1.42 GB Q6_K and 1.83 GB Q8_0. 

The CPU-TPS values above are engineering priors only; I did not find a controlled published benchmark using exactly ADTC's i5 10th–12th-generation / four-core sandbox for all of these models. Treat them only as a way to decide which models to download first. The ADTC profiler explicitly exists to replace such guesses with actual llama-bench measurements. 

A useful way to visualize why parameter count should not be maximized is the approximate weight-footprint curve:

Illustrative GGUF weight footprint as scale rises
0.5B
1.7B
3B
4B
7B
9
8
7
6
5
4
3
2
1
0
Approx. GB


Show code
The bars approximate Q4-class weights and the line Q8-class weights; the values use representative Qwen GGUFs where available and parameter-based extrapolation for the 7B endpoint. Actual runtime RAM is higher because inference also needs KV cache and runtime state. 

What the Qwen sizes actually mean for you
The Qwen families provide several relevant size points. Qwen2.5 includes 0.5B, 1.5B, 3B and 7B variants among its smaller models; Qwen3 includes compact 0.6B, 1.7B and 4B models, with Qwen3-4B officially supporting 100+ languages/dialects and llama.cpp GGUF deployment. 

So, on an 8 GB ADTC laptop:

Qwen3-1.7B Q4/Q6/Q8: yes, easily. This is the model I am most interested in from an efficiency perspective. 

Qwen2.5-3B Q4/Q6: yes. Even Q8 can fit, although Q4/Q6 is more sensible for competition scoring. 

Qwen3-4B Q4/Q6: yes. Q8 is also plausibly below the 7 GB process budget at short contexts, but its memory score will be significantly worse. 

Qwen2.5-7B Q4: probably yes, technically. But that is not the same as “good ADTC architecture.” Its larger RSS and much lower CPU throughput make it an edge-case experiment rather than my recommended final model. The competition has no strict parameter limit, but explicitly rewards lower RAM and faster generation. 

There is also a newer Qwen3.5 family. As of August 2026, Qwen publishes 0.8B, 2B and 4B small models; the 4B model advertises coverage of 201 languages/dialects and substantially stronger current reasoning benchmarks. 
 However, the official Qwen3.5 model card currently documents Transformers, SGLang, vLLM and KTransformers rather than llama.cpp, whereas third-party GGUF conversions do exist and claim recent llama.cpp compatibility. Because ADTC accepts llama.cpp only, I would not make Qwen3.5 the main path until a converted checkpoint passes the actual ADTC profiler end-to-end. 

That is an important correction to the candidate list in your current methodology. Qwen3.5 is attractive enough to perform a one-hour compatibility check, but it should not consume scarce project time until it proves submission-format compatibility. 

Gemma 3 is the most useful non-Qwen control. Its 4B model was trained for multilingual use across 140+ languages and is supported by llama.cpp tooling; this makes it valuable because the recent language-adaptation literature says architecture can matter more than raw model scale. 

Llama 3.2 3B is a good edge model and was explicitly designed for on-device use, but Meta's official list of supported languages contains eight languages and does not include the African languages we are targeting. Meta allows developers to adapt it beyond those supported languages, but with our time budget I would treat it as a systems baseline rather than a leading multilingual candidate. 

Fine-tuning, downsizing and quantization
The answer to your second methodological question is emphatically:

Yes: start with a model that is already small enough, fine-tune it, and then quantize it.

You do not need to start from a huge model and “downsize” it.

In fact, that approach is more complex. Distillation and pruning add additional optimization stages and failure modes. If a 1.7B or 4B base can reach adequate multilingual STEM accuracy directly, reducing its parameter count afterwards provides no inherent benefit.

LoRA and QLoRA are particularly well aligned with this setting. LoRA keeps the pretrained parameters frozen and learns low-rank updates, while QLoRA keeps the frozen model in a 4-bit representation during training and backpropagates into LoRA adapters. The NeurIPS 2023 QLoRA work demonstrated that this can dramatically reduce fine-tuning memory while preserving performance close to conventional higher-precision fine-tuning in its experiments. 

The important distinction is:

[ \text{QLoRA 4-bit during training} \neq \text{GGUF Q4 during deployment}. ]

Do not think of the workflow as:

text
Copy
download Q4_K_M.gguf
       ↓
edit/fine-tune that GGUF directly
       ↓
submit it
The cleaner workflow is:

text
Copy
HF base/instruct checkpoint
        ↓
LoRA / QLoRA training
        ↓
merge adapter into checkpoint
        ↓
convert merged model to high-precision GGUF
        ↓
quantize independently to Q8 / Q6 / Q5 / Q4
        ↓
benchmark every deployment quant
This separation is also exactly what your current methodology proposes, and it is the correct distinction. 
 QLoRA solves a training-memory problem; llama.cpp's GGUF quantization solves a deployment-memory/bandwidth problem. QLoRA's paper and llama.cpp's GGUF tooling address those separate stages. 

What quantization level should win?
The quantization literature does not support “always use the lowest bit width.”

Eight-bit methods can preserve near-full-precision behavior in large models, while modern 4-bit techniques have repeatedly demonstrated strong accuracy/size tradeoffs. Conversely, papers pushing to 2-bit and 1-bit precision require increasingly specialized techniques precisely because severe quantization is harder to make robust. QuIP's NeurIPS work developed special preprocessing to make viable 2-bit quantization possible; OneBit still reported a nontrivial performance sacrifice even with a dedicated 1-bit training framework. 

ICML 2025's any4 is another example of how much research effort is concentrated around four-bit deployment: it develops a learned 4-bit numeric representation and reports improved accuracy relative to conventional INT4/FP4/NF4 representations over several Llama/Mistral-family models. 

For ADTC I would therefore sweep:

[ \boxed{Q8 \rightarrow Q6 \rightarrow Q5 \rightarrow Q4} ]

for every finalist.

Treat Q3 as an experimental fallback, not part of the default final search, unless Q4 still leaves the model too slow or large. Q2 is very unlikely to be the right place to spend your limited development time.

There is another nuance: papers demonstrating impressive 2–4-bit accuracy are often evaluating considerably larger LLMs than our 1–4B candidates. Smaller models have less representational redundancy to spare, so we should not assume a published “negligible degradation at 4 bits” result transfers unchanged to a 1.7B multilingual STEM model. The correct response is exactly the ablation proposed in your draft: measure the quantization-induced drop separately on English STEM, target-language STEM, and language-quality sets. 

Post-training quantization versus quantization-aware training
For the first submission, use post-training quantization after fine-tuning.

Only move to QAT if:

[ A_{Q4} \ll A_{\mathrm{higher\ precision}} ]

and Q4 otherwise has a large leaderboard advantage.

QAT is an optimization technique for recovering low-precision accuracy, but it introduces additional training complexity. At four bits, modern PTQ methods are strong enough that it is sensible to measure the problem before trying to solve it. The trend across GPTQ/QuIP/AWQ-style research is that careful calibration and weight-aware quantization can retain considerable capability at low precision, while extremely low precision requires much more specialized handling. 

For this competition, the pipeline should therefore be:

[ \boxed{ \text{adapt} \rightarrow \text{merge} \rightarrow \text{GGUF} \rightarrow \text{PTQ sweep} \rightarrow \text{QAT only if necessary} } ]

Where distillation belongs
Distillation becomes useful only after we observe something like:

Model	Target-language STEM	TPS	Peak RAM
4B	78%	9	3.6 GB
1.7B	62%	21	2.0 GB

Then we have a clear research question:

Can the 4B teacher transfer enough of its reasoning/tutoring behavior to the 1.7B student that we get, for example, 72–75% accuracy at ~20 TPS?

That is a much better use of distillation than blindly shrinking the model from day one. The Llama 3.2 development process provides practical evidence that large-teacher supervision can strengthen smaller edge-oriented models. 

The same principle applies to ensembles: ensemble during data generation; distill into one model for deployment.

Low-resource language adaptation and multilingual inference
This is the area where the literature changes our earlier answer the most.

There is no single universally best low-resource adaptation recipe. The EMNLP 2024 study Exploring Design Choices for Building Language-Specific LLMs systematically tested base-model selection, vocabulary extension and continued pretraining and found three important things: initial low-resource-language performance does not reliably predict final adapted performance; vocabulary extension plus CPT can improve tokenization efficiency; and the optimal adaptation recipe is highly language dependent. 

That argues against choosing a model solely because:

“Model X currently speaks our language better.”

Instead ask:

“Which model gives the best combination of English/STEM capability, adaptation potential, tokenizer efficiency and final hardware performance after the same adaptation budget?”

The latest Africa-specific result reinforces this. AfriqueLLM, now accepted to ACL 2026, adapts Llama, Gemma and Qwen-family models across 20 African languages. Its strongest finding is that training-data composition is the largest driver of CPT improvement: mixtures augmented with mathematics, code and synthetic translated data consistently improved reasoning-oriented evaluations. Within a family, larger models generally helped, but across families architecture could dominate scale; baseline multilingual competence was also not a reliable predictor of post-CPT performance. 

That is almost exactly our problem.

What the strongest African-language papers currently imply
Study	Venue/status	Relevant finding	Implication for ADTC
IrokoBench	NAACL 2025 Long	17 African languages; AfriMGSM, AfriMMLU and AfriXNLI reveal large gaps versus high-resource languages. Translate-test helps some large English-centric models. 
Use direct and translate-test evaluation, but don't assume direct multilingual ability is adequate.
AfriQA	EMNLP 2023 Findings	12k+ African cross-lingual QA examples; automatic translation/retrieval approaches performed poorly overall. 
Translation pipelines are not a universally reliable solution.
AfriInstruct	EMNLP 2024 Findings	CPT followed by instruction tuning improved diverse African-language tasks. 
Language adaptation plus SFT is evidence-backed.
Language-Specific LLM Design Choices	EMNLP 2024 Findings	Base-model ranking changes after adaptation; vocab extension/CPT improve efficiency; recipe is language-dependent. 
Benchmark candidates after adaptation, not just before.
Lugha-Llama	2025 research report	Curated African-language text + high-quality educational content substantially improves African-language performance; >10% gain over base on AfriQA. 
High-quality content matters more than simply accumulating web text.
AfriqueLLM	ACL 2026 Main	Data mixture is the strongest CPT lever; math, code and synthetic translated data help; architecture can outweigh scale. 
Very strong support for mixed bilingual STEM adaptation.
AfroBench	ACL 2025 Findings	Broad evaluation across 64 African languages still shows substantial high-resource/low-resource gaps. 
Do not infer language competence from a few examples.

The particularly interesting Lugha-Llama experiment translated approximately 200 million tokens of high-quality English educational material into Swahili. Their analysis indicated that much of the benefit came from the content represented in the translated material, rather than simply from having more Swahili-like surface text. 

That tells us something fundamental:

Low-resource-language adaptation is partly a knowledge-distribution problem, not merely a grammar/vocabulary problem.

A corpus can contain millions of sentences in language X and still contain very little high-quality algebra, physics, chemistry or pedagogical explanation. For a STEM tutor, merely collecting more generic X-language text may therefore be less effective than introducing high-quality STEM concepts into X through carefully validated translation. AfriqueLLM's independent finding that math, code and synthetic translated data improve African-language reasoning is consistent with this interpretation. 

The staged adaptation pipeline I recommend
Start with a tokenizer and baseline audit. For every candidate, measure direct English STEM, direct X-language STEM, ordinary X-language comprehension, and token fertility. The language-specific LLM study shows that tokenization efficiency is itself an adaptation variable; excessive fragmentation means the model needs more tokens to represent the same text, which is especially unattractive in a throughput-scored competition. 

Then run an important diagnostic:

[ A_{\text{direct-X}} \quad\text{versus}\quad A_{\text{X→English translate-test}}. ]

If the model solves translated versions correctly but fails the original X prompts, its reasoning is present and the language interface is the bottleneck. If it fails both, language adaptation alone probably will not solve the problem.

Next, try bilingual domain SFT first when the baseline model already has usable language comprehension. QLoRA makes this inexpensive relative to full-parameter training. Include four directions:

[ EN\to EN,\quad X\to X,\quad EN\to X,\quad X\to EN. ]

Include tutoring behaviors rather than just final-answer QA:

solve;
explain;
identify a student's first error;
give one hint without revealing the answer;
simplify an explanation;
create a related exercise;
switch languages;
interpret natural code-switching.
This is preferable to training a “translation model with some math”; the goal is to teach a common reasoning system to express and understand STEM directly in both languages. AfriInstruct's instruction-tuning results and broader multilingual instruction-tuning work support this kind of direct alignment. 

Only if SFT leaves fundamental language problems should you spend compute on continued pretraining. Symptoms that justify CPT include poor comprehension of ordinary prose, systematic morphology/spelling problems, severe domain-terminology failures, or a very large direct-X versus translate-test gap. This is where your draft's “conditional CPT” design is particularly sensible. 

For CPT, do not use only monolingual language-X text. The best current Africa-specific evidence points toward a balanced mixture containing native text, high-quality translated content, mathematical/structured data, code or reasoning data, and high-resource replay to reduce forgetting. 

I would revise your initial mixture slightly into an experimental family rather than a single fixed recipe:

Pool	Initial search range	Purpose
Native target-language text	25–40%	grammar, morphology, idiom, register
Translated target-language STEM	25–35%	transfer high-quality STEM content into X
English math/science reasoning	15–25%	preserve reasoning ability
English/general replay	5–15%	reduce catastrophic forgetting
Code/structured reasoning	5–15%	preserve formal/compositional reasoning

Your draft's 35/25/20/10/10 split sits comfortably inside these ranges and is therefore a sensible starting hypothesis, but current literature does not justify calling that specific proportion optimal. 
 AfriqueLLM's main message is precisely that mixture composition needs to be tested. 

Tokenizer adaptation should be conditional
Tokenizer extension is worth considering if the chosen language suffers severe fragmentation. EMNLP 2024's language-specific adaptation study found vocabulary expansion combined with continued pretraining could materially improve encoding efficiency. 

This matters more for ADTC than in an ordinary NLP paper because a bad tokenizer hurts two objectives:

[ \text{more tokens} \Rightarrow \begin{cases} \text{more inference work}\ \text{larger effective sequences}\ \text{more KV-cache use}\ \text{potentially harder language learning} \end{cases} ]

However, vocabulary modification is not free. New embeddings have to be learned, the model changes structurally, GGUF conversion must remain correct, and the gain may be small for a Latin-script language already tokenized reasonably well. Therefore first measure something like:

[ F_X = \frac{\text{tokens in X}}{\text{words in X}} ]

and, on parallel sentences,

[ R_{X/EN} = \frac{\text{tokens for X sentence}} {\text{tokens for equivalent English sentence}}. ]

Only modify the vocabulary when these diagnostics show a significant problem. That agrees with both the literature and the conditional approach already present in your draft. 
 

Direct multilingual inference versus a translation sandwich
Consider the alternative architecture:

text
Copy
X-language question
        ↓
X → English translation model
        ↓
English specialist LLM
        ↓
English → X translation model
        ↓
X-language response
There is evidence in favor of translate-test in some circumstances. IrokoBench found that translating African-language benchmark inputs to English improved performance for some large English-centric models such as Gemma 2 27B and Llama 3.1 70B. 
 So runtime translation should not be dismissed as inherently bad.

But it is a poor default for ADTC for four separate reasons.

First, AfriQA found automatic translation and multilingual retrieval approaches remained weak on its African-language cross-lingual QA setting, showing that translation is not a universally reliable bridge. 

Second, errors can occur in both translation directions. A mistranslated mathematical relation or scientific term changes the problem before the reasoning model sees it.

Third, a separately resident MT model increases memory and/or incurs model-loading latency.

Fourth, the strongest competition-specific reason is structural: judges currently load one submitted GGUF and interact with that model directly. Your application-level translator is not what the scoring interface evaluates. 

Therefore:

[ \boxed{ \text{translation during dataset construction}

\text{translation as a diagnostic}

\text{runtime translation as final architecture} } ]

for ADTC.

A translation system can still be enormously useful before deployment. Take a high-quality English STEM example, translate it into X, filter it automatically, have a native/fluent speaker review a representative sample, then train the single specialist LLM on it. Lugha-Llama and AfriqueLLM provide direct evidence that translated high-quality content can be useful in African-language adaptation. 

The fact that you do not speak Swahili matters
I would no longer recommend Swahili merely because it has excellent benchmark coverage.

IrokoBench itself invested in human-translated African-language benchmarks because reliable evaluation requires more than running text through automatic MT. 
 If nobody on the team can recognize that a technically grammatical response uses the wrong scientific term, unnatural register, or a subtly incorrect translation, you will not know whether your fine-tuning improved the model or merely improved benchmark-shaped behavior.

A better language-selection rule is:

Criterion	Priority
Team member/collaborator can fluently validate output	Essential for a serious localization claim
Covered by IrokoBench/AfriQA/AfroBench	Very high
Enough native-language text exists	High
Translation resources exist	High
Base tokenizer is reasonably efficient	High
Real target users and use case are clear	High
Existing model already has some competence	Helpful, not decisive

If no teammate speaks a benchmark-supported African language, adding a fluent language collaborator as the third team member could have higher marginal value than adding another ML engineer. This is especially true because ADTC awards meaningful African-language functionality separately, while English remains the primary evaluation language. 

Evaluation, safety and experimental design
The evaluation should answer three independent questions:

[ \text{Did STEM capability improve?} ]

[ \text{Did target-language capability improve?} ]

[ \text{Did leaderboard efficiency improve?} ]

A single aggregate benchmark cannot distinguish them.

For a multilingual STEM tutor, I would freeze the following suite before generating synthetic training data.

Evaluation	What it measures	Use
IrokoBench AfriMGSM	African-language mathematical reasoning	Primary multilingual math
IrokoBench AfriMMLU	African-language knowledge/reasoning	Broader STEM/knowledge
IrokoBench AfriXNLI	General language understanding/inference	Checks whether “math gains” mask poor language
AfriQA	African cross-lingual QA	Additional language/knowledge test
AfroBench	Broad African-language multi-task performance	Optional broader sanity check
English STEM set	Reasoning retention	Detect catastrophic forgetting
Custom bilingual tutoring set	Actual product behavior	Hints, explanations, error diagnosis, language switching
Tokenizer test set	token fertility	Runtime/language diagnostic

IrokoBench contains AfriMGSM, AfriMMLU and AfriXNLI across 17 African languages and is the most directly relevant standardized suite for your STEM idea. 
 AfriQA adds over 12,000 cross-lingual QA examples in ten African languages. 
 AfroBench extends evaluation breadth to 64 African languages and 15 task families. 

Do not use these test sets as adaptation data. The custom data-generation pipeline should hash/deduplicate against held-out benchmark prompts where practical. Your uploaded methodology already makes the correct call to freeze evaluation before generating synthetic bilingual data. 

For each configuration report at least:

[ A_{\text{EN-STEM}}, \quad A_{\text{X-STEM}}, \quad A_{\text{X-general}}, \quad \Delta_{\text{forget}}, ]

where

[ \Delta_{\text{forget}}
A_{\text{EN-before}}
A_{\text{EN-after}}. ]

For language quality, have a fluent speaker blind-review a stratified sample for grammaticality, natural terminology, factual correctness, language consistency and pedagogical clarity. Automated multilingual benchmarks do not substitute for this because the model can produce structurally plausible but unnatural or semantically off-target language.

For systems evaluation record:

[ {\text{generation TPS},, \text{prompt TPS},, \text{TTFT},, \text{peak RSS},, \text{steady RSS},, T_{\text{peak}},, \text{throttle flag}}. ]

Those correspond closely to the telemetry already exposed by the official profiler. The profiler currently uses a 7 GB RAM limit, 15 TPS reference and 85°C thermal threshold. 

I would test at minimum 2K and 4K contexts, plus one sustained generation run. Very long contexts are counterproductive unless your product genuinely needs them because KV-cache memory increases with context. KV-cache research shows it can become a major inference-memory component as context length/batch size increases. 
 For a STEM tutor, an honest 2K–4K practical context is probably more strategically useful than advertising 32K while paying unnecessary RAM overhead.

Ablations that actually answer the research questions
Do not train every Cartesian combination. That would waste the GPU budget.

Use a tournament.

Screening stage

Run unmodified:

Qwen3-1.7B
Qwen3-4B
Gemma 3 4B
optionally Qwen2.5-3B
at Q4 and Q6.

This is eight inexpensive inference configurations.

Measure:

[ \text{English STEM},\quad \text{X STEM},\quad \text{TPS},\quad \text{RSS}. ]

Retain the top efficiency model and top accuracy model.

Adaptation stage

For the top two, compare:

[ \text{base} ]

versus

[ \text{bilingual STEM QLoRA/SFT}. ]

Only if language diagnostics remain weak run:

[ \text{mixed CPT}\rightarrow\text{SFT}. ]

This directly tests whether expensive CPT is necessary.

Data-composition stage

On one model, compare:

Variant	Native X	Translated STEM	English STEM/replay	Code/structured
A	✓			
B	✓	✓		
C	✓	✓	✓	
D	✓	✓	✓	✓

This is perhaps the most scientifically interesting ablation because AfriqueLLM's current results predict that C/D should outperform target-language-only adaptation on reasoning tasks. 

Quantization stage

For only the top adapted checkpoint:

[ Q8,\quad Q6,\quad Q5,\quad Q4. ]

Then report:

[ \Delta A_q
A_{\text{high precision}}-A_q ]

against

[ \Delta\text{TPS}_q \quad\text{and}\quad \Delta\text{RSS}_q. ]

That produces the actual competition Pareto curve.

Optional distillation stage

Only run this if:

[ A_{4B} \gg A_{1.7B} \quad\text{and}\quad \text{hardware score}{1.7B}\gg\text{hardware score}{4B}. ]

Use the stronger model to generate/correct STEM tutoring examples and retrain the small model.

This staged design will tell a very compelling report story because every expensive experiment is triggered by a measured failure mode rather than performed arbitrarily.

Medical evaluation if you later choose healthcare
AfriMed-QA is the benchmark I would use, but it contains an important warning for our entire “tiny medical LLM” concept. The ACL 2025 long paper introduces roughly 15,000 Pan-African medical questions from more than 60 medical schools across 16 countries and 32 specialties, and reports that smaller edge-friendly models struggled to achieve a passing score; it also found meaningful performance variation across specialties and geographies. 

Its evaluation axes go beyond MCQ accuracy to include factuality, bias, potential harm, local relevance and reasoning. 
 That is exactly the right philosophy for a medical ADTC entry.

For a triage system, I would additionally construct a clinician-reviewed test set and record:

[ \text{dangerous under-triage rate}
\frac{ #(\text{urgent/emergency cases predicted non-urgent}) }{ #(\text{urgent/emergency cases}) }. ]

I would also measure red-flag recall, unsupported/hallucinated clinical claim rate, next-question appropriateness and escalation accuracy. Those are custom safety metrics rather than official AfriMed-QA metrics, but they align much better with the actual risk of a triage assistant than raw QA accuracy.

AfriMed-QA is English-focused, so it does not by itself validate a multilingual medical product. A target-language clinical set would have to be independently translated/reviewed by qualified speakers and ideally clinicians. The benchmark's finding that small edge models are currently weak is one reason I still prefer STEM as the first competition direction. 

Recommendations and implementation plan
The models I would actually test
My three primary candidates are:

Priority	Model	Why	Likely final quant	Weight size	Likely Q4 runtime footprint
A	Qwen3-1.7B	Best chance of hitting/exceeding 15 TPS while retaining useful reasoning; 100+ language support; enormous efficiency upside	Q4_K_M or Q6_K	1.11–1.42 GB	~1.7–2.6 GB
B	Qwen3-4B	Accuracy-oriented reasoning model; same family gives a clean scale ablation; official llama.cpp GGUF	Q4_K_M or Q5_K_M	2.50–2.89 GB	~3.2–4.4 GB
C	Gemma 3 4B	Different architecture; 140+ language baseline; important because current research says architecture can dominate scale	Q4-class	~2.6 GB	~3.3–4.2 GB

Qwen3-1.7B and Qwen3-4B have official GGUF and llama.cpp support, with published file sizes; Gemma 3 provides the strongest architecture-diversity control among compact multilingual models. 

I would also run Qwen2.5-3B-Instruct Q4_K_M once as a middle-size baseline. It is a particularly mature llama.cpp target, has a 2.10 GB Q4_K_M artifact, and Qwen explicitly emphasizes mathematical/coding improvements in the Qwen2.5 family. 
 If it unexpectedly dominates the 1.7B/4B tradeoff, promote it.

Qwen3.5-2B or Qwen3.5-4B is a watchlist experiment, not the main path. Its current language/reasoning numbers are attractive, but submission compatibility is more important than benchmark glamour this close to the ADTC deadline. Run the profiler first; only then invest in adaptation. 

AfriqueLLM also creates an interesting shortcut. If your eventual target language is among its supported African languages, an AfriqueLLM checkpoint could serve as a CPT-pre-adapted control rather than spending scarce compute recreating language pretraining from scratch. The project includes African-adapted Gemma/Qwen/Llama-family checkpoints and reports adaptation over 20 languages. 
 Whether one should become the final base depends on instruction-tuning state, target-language coverage and successful GGUF/profiler conversion.

The methodological decision tree
No

Yes

Yes

No

Choose target African language

Require native/fluent validation

Freeze Iroko + custom test sets

Profile 1.7B / 3B / 4B candidates

Which are Pareto competitive?

Top efficiency model

Top accuracy model

Bilingual STEM QLoRA

Direct X-language competence still weak?

Merge adapters

Mixed CPT: native + translated STEM + EN reasoning/code

GGUF conversion

Q8 / Q6 / Q5 / Q4

Accuracy + TPS + RSS + thermal

4B much better but too slow?

Distill 4B behavior into 1.7B

Select Pareto winner

Single final GGUF submission



Show code
Concrete implementation checklist
Given the current Gate 1 deadline of August 25, 2026, the priority should be evidence-generating work rather than building infrastructure that the competition does not score. The official challenge page currently lists that deadline and requires the repository, report and two-minute demo. 

Fork the ADTC submission template immediately. Fill in metadata.json, choose math_scientific_reasoning, and make the complete download→profile process work with a vanilla GGUF before doing any model training. The current template requires a public repository, exactly one downloaded GGUF model path, two submitted prompts, and llama.cpp. 

Install and freeze the ADTC profiler version. Run its participant mode against one tiny public model so that packaging mistakes are discovered before fine-tuning. It reports throughput, memory, CPU/thermal metrics and optional accuracy. 

Choose the language based on human validation, not dataset fame. Cross-reference IrokoBench/AfriQA/AfroBench coverage against languages somebody on the team or a collaborator can actually read. Do not commit to Swahili merely because Lugha-Llama used it. 

Build the frozen evaluation suite before creating synthetic data. Include English STEM, target-language AfriMGSM/AfriMMLU, target-language general comprehension, and roughly 100–300 custom tutoring interactions. Keep all test material completely outside training.

Benchmark the unadapted model pool. At minimum run Qwen3-1.7B Q4/Q6, Qwen3-4B Q4/Q6, and Gemma 3 4B Q4/Q6. Add Qwen2.5-3B Q4 if time permits. Record raw TPS/RSS rather than relying on the estimates in this report. 

Run direct-X versus translate-test evaluation. This diagnoses whether the main problem is target-language access or underlying reasoning. IrokoBench itself uses translate-test as an evaluation setting, making this a literature-grounded diagnostic. 

Build the bilingual SFT corpus. Create EN→EN, X→X, EN→X and X→EN examples for math/science solving, tutoring, hints, misconception diagnosis and explanations. Add high-quality translated STEM content but keep native-language data as a separate pool so its contribution can be ablated. AfriqueLLM and Lugha-Llama strongly motivate this mixture. 

Use high-quality source material, not random bulk web text. Lugha-Llama's results suggest educational content quality is highly valuable, while AfriqueLLM shows task-aligned math/code/synthetic data affects reasoning gains. 

Have a fluent speaker inspect translated training samples. Review scientific vocabulary, algebra phrasing, units, numbers, negation and instructional register. Reject bad examples before fine-tuning rather than expecting the model to recover from noisy translations.

QLoRA/SFT the leading small model first. Start with Qwen3-1.7B. It is cheap enough that you can iterate and, if it reaches satisfactory accuracy, its hardware advantage may make the 4B path unnecessary. QLoRA is the appropriate default parameter-efficient mechanism. 

Adapt the accuracy model second. Train Qwen3-4B or Gemma 3 4B using the same split and as similar a training recipe as possible. Do not compare models trained on fundamentally different data.

Run continued pretraining only if diagnostics justify it. If bilingual SFT fixes task behavior but ordinary language comprehension remains poor, use mixed CPT followed by SFT. AfriInstruct and AfriqueLLM support the CPT→instruction-tuning pattern, but AfriqueLLM's 26B-token study also illustrates that serious CPT can be expensive; do not casually reproduce it under a hackathon compute budget. 

Measure tokenizer fertility before modifying vocabulary. Only introduce new target-language tokens if fragmentation is severe enough to justify the additional training/conversion complexity. The EMNLP 2024 language-specific study supports vocabulary adaptation but also emphasizes that the best recipe is language dependent. 

Merge the final adapters before deployment. Preserve the unmerged checkpoint for reproducibility, but build one merged model that can be converted to one GGUF.

Quantize the same merged checkpoint to Q8, Q6, Q5 and Q4. Run identical prompts and generation settings against every quant. Do not infer quality from bit width.

Calculate the actual ADTC score for every configuration. Plot accuracy against RSS and TPS, discard Pareto-dominated models, then select based on the competition objective rather than intuition. The current profiler exposes the necessary telemetry. 

Run the data-mixture ablation on one model. The most valuable scientific figure in the final report may be:

native only → + translated STEM → + English reasoning → + code/structured reasoning

because it directly tests the major finding of AfriqueLLM on your much smaller model. 

Run distillation only if the measurements create a reason. If Qwen3-4B is substantially more accurate but the 1.7B hardware score is much higher, use the 4B to generate/correct high-quality examples and repeat the 1.7B SFT. Otherwise skip it.

Run sustained profiler tests. A short benchmark can miss thermal problems. ADTC applies a ten-point penalty above 85°C or on detected throttling and disqualifies OOM/crash runs. 

Package exactly one final GGUF. Make download_model.sh deterministic and credential-free and test the repository from a clean checkout. The template requires the downloaded path to exactly match _runtime.model_path. 

Write the report around measured engineering decisions. The strongest narrative is not “we fine-tuned Qwen.” It is:

“We evaluated four compact architectures, diagnosed low-resource language versus reasoning bottlenecks, measured the effect of translated STEM and English reasoning replay, tested multiple numerical precisions, and selected the model lying on the best accuracy–throughput–memory frontier.”

Make the video demonstrate the constraint. Show the laptop specs, disconnect networking, ask the same English and African-language STEM problem, show an instructional interaction rather than simple QA, then show adtc-profiler results for RAM/TPS. The challenge explicitly encourages a real running demonstration and scores the actual model offline. 
One final detail should be treated conservatively: the current official FAQ says organizers add three hidden prompts, while the submission-template field reference still says two hidden prompts. Assume the stricter case—at least three unseen domain prompts—and make the model broadly competent across mathematics and scientific reasoning rather than narrowly overfitting a tiny set of demo tasks. 

The central research hypothesis I would put at the heart of the project is therefore:

A 1.7–4B model, adapted with a carefully constructed mixture of native African-language data, translated high-quality STEM material and English reasoning replay, can outperform a larger generic model on the ADTC objective once quantized and deployed on commodity CPU hardware.

That hypothesis is supported—not proven—by three complementary strands of literature: low-resource adaptation studies show that base-model choice and tokenizer/data design matter; African-language studies show that translated educational content plus task-aligned math/code mixtures can substantially improve adaptation; and efficient-LLM research shows that LoRA/QLoRA, distillation and four-bit-class quantization can preserve substantial capability while cutting deployment cost. 

For the current ADTC constraints, my strongest initial bet is Qwen3-1.7B Q4/Q6 as the efficiency challenger versus Qwen3-4B Q4/Q5 as the accuracy challenger, with Gemma 3 4B as the architecture control. Everything after that should be decided by the measured multilingual STEM accuracy and the official profiler—not by parameter count.

