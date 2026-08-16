# Methodology

The objective of this work is to develop a compact bilingual language model capable of performing scientific and mathematical reasoning in English and a selected low-resource African language while satisfying the computational constraints of offline deployment on commodity hardware. The methodology is therefore designed as a joint optimization problem over model capability and deployment efficiency. Rather than selecting the largest model that can be accommodated within the available memory budget, we evaluate candidate models along an accuracy–efficiency frontier and progressively introduce multilingual adaptation, compression, and quantization only where these interventions produce measurable gains.

The experimental procedure consists of five stages. First, a set of compact pretrained models is evaluated without additional training to identify architectures that provide favorable initial trade-offs between reasoning capability, multilingual competence, memory consumption, and inference speed. Second, the strongest candidates are adapted using bilingual and domain-specific data, beginning with parameter-efficient supervised fine-tuning and introducing continued pretraining only when baseline language competence remains insufficient. Third, the adapted models are converted to the deployment format and evaluated under several post-training quantization levels. Fourth, more aggressive compression techniques, including quantization-aware training and knowledge distillation, are considered only when the preceding stages reveal a clear accuracy–efficiency gap. Finally, all resulting configurations are evaluated using the same held-out task sets and hardware profiling protocol, and controlled ablations are used to isolate the contributions of model scale, training data composition, adaptation strategy, and numerical precision.

## Experimental Objectives and Design

Model selection is formulated as a constrained multi-objective optimization problem. Let (M) denote a candidate model configuration, including its base architecture, adaptation procedure, and deployment precision. The selected model is

[
M^{*}
=====

\arg\max_{M}
\left[
0.5S_{\mathrm{acc}}(M)
+
0.3S_{\mathrm{perf}}(M)
+
0.2S_{\mathrm{eff}}(M)
----------------------

P_{\mathrm{thermal}}(M)
\right],
]

where (S_{\mathrm{acc}}) denotes task accuracy, (S_{\mathrm{perf}}) denotes inference performance, (S_{\mathrm{eff}}) denotes memory efficiency, and (P_{\mathrm{thermal}}) represents the thermal penalty imposed when sustained execution exceeds the competition threshold. This formulation reflects the evaluation conditions of the Africa Deep Tech Challenge (ADTC), under which accuracy is the dominant component of the score but computational efficiency contributes substantially to final ranking.

The experimental design therefore does not assume that higher parameter counts necessarily lead to better competition performance. A larger model may obtain superior reasoning accuracy while simultaneously reducing throughput and increasing memory pressure. Conversely, an aggressively compressed model may obtain excellent systems performance while losing sufficient task accuracy to reduce its overall score. Candidate configurations are consequently compared using both individual metrics and their resulting competition-level objective.

All model comparisons are performed using fixed evaluation prompts, generation settings, context lengths, and hardware conditions. Training, validation, and final test data are separated before model adaptation begins. In particular, examples used to produce synthetic bilingual training material are excluded from the held-out evaluation sets to prevent contamination between training and evaluation.

## Base Model Selection

### Candidate Model Families

The first stage evaluates several compact model families spanning approximately 1–4 billion parameters. The principal candidates are:

| Model             | Approx. Scale | Experimental Role                |
| ----------------- | ------------: | -------------------------------- |
| Qwen 3 / Qwen 3.5 |            4B | Accuracy-oriented candidate      |
| Gemma 3           |            4B | Multilingual control candidate   |
| Qwen 3            |          1.7B | Efficiency-oriented candidate    |
| Tiny Aya          |        ~3.35B | Multilingual specialist baseline |
| Llama 3.2         |            3B | Edge-oriented baseline           |

The Qwen family is prioritized because of its combination of compact model sizes, mathematical and scientific reasoning capability, and evidence of substantial gains following African-language adaptation. In particular, the AfriqueLLM experiments demonstrate that Qwen models can exhibit large improvements on African-language reasoning tasks after continued pretraining with appropriately constructed multilingual data mixtures [AfriqueLLM, 2026]. Gemma 3 provides a useful comparison because of its comparatively strong multilingual baseline and its inclusion in the same African-language adaptation study. A smaller Qwen 1.7B model is retained throughout the experiments as an efficiency challenger, allowing us to test whether reduced memory consumption and increased throughput compensate for losses in task accuracy.

The initial comparison is deliberately performed before fine-tuning. This avoids committing training resources to a model whose hardware characteristics make it unsuitable for the target environment and provides an unadapted reference against which subsequent improvements can be measured.

### Pre-Fine-Tuning Screening

Each candidate is evaluated in its unmodified form on four categories of tasks:

1. English scientific and mathematical reasoning;
2. target-language scientific and mathematical reasoning;
3. general target-language comprehension; and
4. bilingual instruction following.

In parallel, each model is profiled using the target inference runtime. The following systems quantities are recorded:

[
\mathcal{H}
===========

{
\mathrm{TPS}*{\mathrm{prompt}},
\mathrm{TPS}*{\mathrm{generation}},
\mathrm{TTFT},
\mathrm{RSS}*{\mathrm{peak}},
T*{\mathrm{mean}},
T_{\mathrm{peak}}
},
]

where TPS denotes tokens per second, TTFT denotes time to first token, (\mathrm{RSS}_{\mathrm{peak}}) denotes peak resident memory, and (T) denotes processor temperature.

Models are evaluated at representative GGUF quantization levels during this initial screening in order to estimate their attainable deployment footprint. At least one 3–4B model and one approximately 1–2B model are retained for adaptation. This preserves both an accuracy-oriented and an efficiency-oriented path through the remainder of the experiments.

Importantly, weak initial target-language performance does not automatically eliminate an otherwise strong reasoning model. AfriqueLLM reports that adaptation gains can depend strongly on architecture and training data composition, such that post-adaptation performance is not necessarily predictable from the unadapted multilingual score [AfriqueLLM, 2026]. Baseline performance is therefore considered jointly with underlying reasoning quality and evidence of adaptation potential.

## Multilingual and Domain Adaptation

The principal adaptation objective is to improve direct reasoning in the target African language without substantially degrading the model's existing English and STEM capabilities. The training procedure separates three related but distinct objectives: acquiring broader target-language competence, transferring STEM knowledge into the target language, and learning the desired instructional behavior.

### Training Data Construction

Training data are organized into three principal pools.

#### Native Target-Language Data

The first pool consists of naturally produced text in the target language. Appropriate sources include educational material, openly licensed books, public institutional documents, encyclopedic text, and other curated native-language corpora. These data primarily provide linguistic information that cannot be reliably reproduced by machine translation, including natural lexical choice, morphology, idiomatic structure, register, and language-specific discourse patterns.

#### Translated STEM Data

The second pool is generated by translating high-quality English educational and STEM material into the target language. Translation is performed offline during dataset construction rather than at inference time. This design allows the final deployed system to remain a single model while exploiting the substantially larger quantity of high-quality educational material available in English.

The translated corpus is subjected to deduplication and quality filtering before training. A sample is additionally inspected by a fluent or native speaker, particularly for scientific terminology, mathematical phrasing, and cases in which a formally correct translation may nevertheless be unnatural in actual usage. This approach follows evidence that machine-translated high-quality educational data can be particularly effective for multilingual model adaptation [TransWebLLM, 2025].

#### Reasoning-Preservation Data

The third pool consists of English mathematics, science, structured reasoning, and, where appropriate, code-oriented examples. These data are retained during multilingual adaptation to reduce catastrophic forgetting of capabilities already present in the pretrained model. This component is particularly important because African-language continued-pretraining experiments have shown that monolingual language adaptation can improve linguistic performance while simultaneously reducing mathematical or knowledge-reasoning performance; the inclusion of math, code, and synthetic multilingual material can recover a substantial portion of this loss [AfriqueLLM, 2026].

An initial data mixture is defined as follows:

| Data Category                           | Initial Proportion |
| --------------------------------------- | -----------------: |
| Native target-language text             |                35% |
| Translated target-language STEM         |                25% |
| English STEM and mathematical reasoning |                20% |
| English general replay data             |                10% |
| Code and structured reasoning           |                10% |

These proportions are treated as an experimental starting configuration rather than a presumed optimum. They are subsequently varied in the data-composition ablation. The purpose of the initial mixture is to establish a balanced starting point in which linguistic adaptation does not dominate reasoning-preservation data.

### Bilingual Instruction Fine-Tuning

The first adaptation experiment is performed using an instruction-tuned base model and bilingual supervised fine-tuning (SFT). This stage is attempted before expensive continued pretraining because a model that already contains sufficient latent knowledge of the target language may require primarily instructional and domain alignment rather than extensive language acquisition.

Training examples cover both conventional question answering and pedagogical interactions. The task distribution includes:

* solving mathematical and scientific problems;
* explaining a solution step by step;
* identifying the first error in an incorrect student solution;
* providing hints without immediately revealing the final answer;
* generating related practice problems;
* simplifying an explanation for different levels of prior knowledge;
* answering conceptual science questions;
* switching the language of an explanation; and
* responding to mixed English–target-language input.

To encourage direct bilingual competence, examples are constructed in multiple language directions:

[
\mathrm{EN} \rightarrow \mathrm{EN},
]

[
\mathrm{X} \rightarrow \mathrm{X},
]

[
\mathrm{EN} \rightarrow \mathrm{X},
]

[
\mathrm{X} \rightarrow \mathrm{EN},
]

where (\mathrm{X}) denotes the selected African language. Code-switched examples are also included where they represent natural usage patterns.

This formulation attempts to preserve the model's English capability while aligning the same underlying reasoning capabilities with target-language instructions and outputs. Bilingual synthetic instruction data have previously been found to provide an effective mechanism for adapting instruction-following models to low-resource languages [Instructing LLMs for Low-Resource Languages, 2025].

### Parameter-Efficient Fine-Tuning

The primary adaptation mechanism is LoRA or QLoRA rather than full-parameter fine-tuning. LoRA freezes the pretrained weights and learns low-rank updates to selected transformer projections [LoRA, 2021]. QLoRA further reduces training-time memory requirements by representing the frozen base model in four-bit form while updating higher-precision low-rank adapters [QLoRA, 2023].

The principal QLoRA training procedure can be represented as

[
W' = W_{4\text{-bit}} + \Delta W,
]

where (W_{4\text{-bit}}) represents the frozen quantized pretrained weights and

[
\Delta W = BA,
]

with (A) and (B) denoting trainable low-rank matrices.

The adapter rank, learning rate, number of training steps, effective batch size, and set of adapted transformer modules are selected using the validation sets. The same hyperparameter search space is applied across competing base models where architectural compatibility permits, thereby reducing confounding effects in cross-model comparison.

QLoRA is treated only as a training-memory optimization. Its four-bit training representation is not assumed to determine the precision of the final deployment model. Following adaptation, the adapters are merged into the base model and the resulting weights are quantized independently during the deployment experiments.

If computational resources permit, the strongest parameter-efficient configuration is subsequently compared with full-parameter adaptation. This experiment is included as an ablation rather than as the default procedure, allowing us to determine whether parameter-efficient training leaves a meaningful amount of task performance unrealized.

### Conditional Continued Pretraining

Continued pretraining (CPT) is introduced only if bilingual SFT fails to provide sufficient direct target-language competence. This decision is based on diagnostic evaluation rather than on a predetermined assumption that every low-resource language requires additional pretraining.

CPT is triggered when the adapted model exhibits one or more of the following patterns:

* correct underlying reasoning but consistently unnatural target-language generation;
* poor comprehension of ordinary target-language prose;
* systematic spelling, morphology, or lexical errors;
* substantially stronger performance when target-language input is translated into English than when it is processed directly; or
* persistent difficulty with domain-specific terminology in the target language.

When required, CPT is performed using the mixed data distribution described above rather than using target-language monolingual text alone. The CPT corpus therefore combines native-language material, translated educational content, English replay data, and reasoning-intensive data. This design follows the central finding of AfriqueLLM that African-language adaptation benefits substantially from balanced mixtures containing language, mathematical, code, parallel, and synthetic data [AfriqueLLM, 2026].

Following CPT, bilingual SFT is repeated to restore and strengthen the desired instruction-following and tutoring behavior. The complete adaptation sequence is therefore

[
\text{Instruction-tuned base}
\rightarrow
\text{Bilingual SFT}
\rightarrow
\text{Evaluation}
]

followed, when necessary, by

[
\text{CPT}
\rightarrow
\text{Bilingual SFT}.
]

This staged design provides a direct empirical comparison between instruction-level adaptation alone and the more computationally expensive combination of language continued pretraining and instruction tuning.

## Model Compression and Quantization

Compression is performed only after the adapted model's behavior has stabilized. Training-time quantization, post-training quantization, quantization-aware training, pruning, and distillation are treated as separate interventions because they modify different aspects of the model and have different implications for accuracy and deployment cost.

### Post-Training Quantization

Following adaptation, LoRA adapters are merged into the model weights, and the resulting model is converted to GGUF for execution using `llama.cpp`. Multiple post-training quantization (PTQ) configurations are then generated from the same merged checkpoint:

[
\mathrm{FP16/BF16}
\rightarrow
Q8
\rightarrow
Q6
\rightarrow
Q5
\rightarrow
Q4
\rightarrow
Q3.
]

Where supported, multiple variants within the four-bit family, including `Q4_K_M` and related formats, are evaluated separately.

Four-bit quantization is treated as the center of the search space rather than as an assumed optimum. Large-scale quantization studies indicate that approximately four bits per weight often provides a favorable trade-off between model scale and numerical precision under constrained storage budgets, while degradation becomes substantially less predictable at more aggressive precision levels [Dettmers and Zettlemoyer, 2023].

Every quantized model is evaluated using exactly the same prompts, context lengths, generation parameters, and hardware configuration. For each quantized model (q), the quantization-induced accuracy loss is calculated as

[
\Delta_{\mathrm{quant}}(q)
==========================

## A_{\mathrm{reference}}

A_q,
]

where (A_{\mathrm{reference}}) denotes the accuracy of the highest-precision adapted checkpoint and (A_q) denotes the corresponding score after quantization.

The quantity is calculated independently for English STEM, target-language STEM, and general target-language evaluation sets. This decomposition is necessary because numerical compression may not affect all input categories equally.

### Quantization-Aware Training

Quantization-aware training (QAT) is considered only if PTQ at the desired deployment precision produces a practically meaningful reduction in task performance. If the Q4 configuration retains essentially all of the higher-precision model's capability, PTQ is considered sufficient.

If substantial degradation is observed at Q4 or Q3, the model is retrained while incorporating quantization effects into the optimization procedure. Previous work on LLM-QAT and related parameter-efficient QAT methods indicates that exposing the model to low-precision representations during optimization can recover performance lost when quantization is applied only after training [LLM-QAT, 2024].

The PTQ and QAT variants are subsequently evaluated using the same accuracy and hardware benchmark suite. QAT is retained only when the recovered accuracy provides a meaningful improvement over standard PTQ.

### Knowledge Distillation and Structured Pruning

Model downsizing is treated as a later-stage intervention rather than a prerequisite for deployment. If a 3–4B quantized model satisfies the memory and throughput requirements while retaining high accuracy, no additional reduction in parameter count is performed.

Distillation becomes relevant when the larger model provides adequate reasoning quality but remains substantially slower than the smaller candidate. In this case, the 3–4B model is used as a teacher to generate high-quality task-specific examples for the approximately 1–2B student. Generated data may contain:

* problem statements;
* correct answers;
* structured explanations;
* common incorrect solutions;
* misconception diagnoses;
* hint sequences; and
* bilingual explanations.

This transfers useful behavior from a stronger teacher without requiring the teacher during final inference. Generative knowledge-distillation methods such as MiniLLM provide the broader methodological basis for this approach [MiniLLM, 2024].

Structured pruning is considered only if there is evidence of removable architectural redundancy and the resulting architecture remains compatible with efficient execution in the target runtime. Unstructured sparsity is not assumed to provide inference acceleration because reductions in the number of non-zero weights do not necessarily translate into lower latency when the runtime does not exploit sparse computation.

## Experimental Setup and Evaluation

### Data Splits

All datasets are divided into disjoint training, validation, and test partitions before fine-tuning. The held-out test set is frozen before synthetic training-data generation begins. No test prompt, translation, or semantically equivalent synthetic example is intentionally included in the training corpus.

The evaluation data are divided into four groups:

| Evaluation Group        | Primary Purpose                                             |
| ----------------------- | ----------------------------------------------------------- |
| English STEM            | Retention of pretrained reasoning capability                |
| Target-language STEM    | Scientific and mathematical reasoning in language X         |
| Target-language general | General linguistic comprehension and fluency                |
| Bilingual/pedagogical   | Explanation, hints, error diagnosis, and language switching |

This decomposition is intended to distinguish genuine improvements in target-language reasoning from changes that affect only linguistic fluency.

### Evaluation Datasets

Where the selected target language is supported, African-language evaluation is based primarily on the IrokoBench benchmark suite [IrokoBench, 2025]. In particular:

* **AfriMGSM** is used to evaluate multilingual mathematical reasoning;
* **AfriMMLU** is used to measure knowledge-oriented reasoning; and
* **AfriXNLI** provides an additional measure of general target-language understanding.

These benchmarks are supplemented with a project-specific held-out STEM evaluation set designed to more closely reflect the intended tutoring use case. The custom evaluation data contain mathematical and scientific questions in both English and the target language, together with instructional interactions such as explanation requests, incorrect student attempts, and requests for hints. Target-language items in this set are manually reviewed before use.

Benchmark test sets remain entirely excluded from adaptation. When English educational examples are translated to produce training data, examples corresponding to evaluation items are similarly excluded before translation.

### Task and Language Metrics

Closed-form and multiple-choice tasks are evaluated using task accuracy. For problems with a uniquely determined short answer, exact-match accuracy is additionally recorded after deterministic normalization of formatting where necessary.

The principal quantitative accuracy measures are therefore

[
A_{\mathrm{EN-STEM}},
\qquad
A_{\mathrm{X-STEM}},
\qquad
A_{\mathrm{X-General}},
]

corresponding respectively to English STEM reasoning, target-language STEM reasoning, and general target-language competence.

Generative tutoring outputs require additional evaluation because correctness of the final answer alone does not capture pedagogical usefulness or linguistic naturalness. A sample of outputs is consequently reviewed by a fluent or native speaker using a fixed rubric covering:

* factual and mathematical correctness;
* grammaticality;
* naturalness of terminology;
* clarity of explanation;
* consistency with the requested language;
* appropriateness of hints; and
* correctness of misconception diagnosis.

Human evaluation is conducted without revealing the model configuration where feasible, reducing evaluator bias when comparing adaptation strategies.

### Tokenizer Efficiency

Because autoregressive inference cost is partly determined by the number of generated tokens, tokenizer efficiency is evaluated separately for English and the target language. For a parallel sample of semantically equivalent sentences, token fertility is defined as

[
F_X
===

\frac{N_{\mathrm{tokens},X}}
{N_{\mathrm{words},X}},
]

and relative token expansion is calculated as

[
R_{X/\mathrm{EN}}
=================

\frac{N_{\mathrm{tokens},X}}
{N_{\mathrm{tokens},\mathrm{EN}}}.
]

Scientific vocabulary is inspected separately because excessive fragmentation of technical terminology may increase both learning difficulty and inference cost.

Tokenizer modification is not performed by default. Vocabulary extension is considered only when the baseline tokenizer exhibits severe fragmentation, since introducing new tokens requires additional embedding training and increases implementation complexity.

### Systems Evaluation

Deployment experiments are conducted using the same runtime and hardware constraints as closely as possible to the official evaluation environment. Each candidate is executed as a single GGUF model through `llama.cpp`.

The systems evaluation records:

* peak resident memory usage;
* prompt-processing throughput;
* generation throughput;
* time to first token;
* model size on disk;
* mean processor temperature;
* peak processor temperature; and
* occurrence of thermal throttling or out-of-memory failure.

Performance is measured under both cold-start and warm-start conditions. Finalist models are additionally subjected to sustained generation experiments rather than short isolated prompts. These experiments use multiple context lengths and output lengths to capture changes in key-value cache size and sustained computational load.

Representative context lengths are

[
C \in {2\mathrm{K},4\mathrm{K},8\mathrm{K}},
]

and representative generated output lengths are

[
O \in {128,256,512}.
]

The purpose of sustained profiling is to identify configurations whose initially high throughput cannot be maintained because of temperature-related throttling.

### Final Model Selection

For each adapted and quantized configuration, task accuracy and systems measurements are combined using the competition scoring function. Candidate models are also visualized on an accuracy–efficiency Pareto frontier.

A model (M_i) is considered Pareto dominated if another configuration (M_j) satisfies

[
A_j \geq A_i,
\qquad
\mathrm{TPS}_j \geq \mathrm{TPS}_i,
\qquad
\mathrm{RSS}_j \leq \mathrm{RSS}_i,
]

with strict improvement in at least one dimension.

Dominated models are discarded from final consideration. Among the remaining configurations, the final model is selected according to the overall optimization objective defined earlier. This procedure ensures that the selected model represents the strongest overall deployment configuration rather than merely the model with the highest standalone accuracy.

## Ablation Studies

A series of controlled ablation experiments is conducted to determine which components of the proposed pipeline are responsible for improvements in multilingual reasoning and deployment efficiency.

### Base Architecture and Model Scale

The first ablation compares the principal base architectures before and after adaptation. In particular, the approximately 1.7B and 4B Qwen models are compared to determine the marginal accuracy gained from increased model scale relative to its throughput and memory cost. The strongest Qwen configuration is additionally compared against Gemma 3 4B to separate architecture-specific effects from parameter count.

### Training Data Composition

The second ablation studies the effect of multilingual training-data composition. At minimum, the following variants are evaluated:

1. native target-language data only;
2. native target-language data + translated STEM data;
3. native target-language data + translated STEM data + English reasoning replay; and
4. the complete mixture including code and structured reasoning data.

This experiment directly tests whether target-language adaptation alone is sufficient and whether preserving English reasoning data mitigates degradation of mathematical and scientific capability.

### Adaptation Strategy

The adaptation ablation compares:

1. the unadapted instruction model;
2. bilingual SFT using QLoRA;
3. mixed CPT followed by bilingual SFT; and
4. full-parameter adaptation, if computationally feasible.

The comparison between bilingual SFT and CPT followed by SFT is particularly important because it determines whether expensive language continued pretraining provides meaningful improvements beyond instruction-level adaptation.

### Quantization Precision

Each finalist model is evaluated at multiple numerical precision levels:

[
Q \in {Q8,Q6,Q5,Q4,Q3}.
]

For every configuration, changes in task accuracy, throughput, peak memory, and sustained temperature are reported. The resulting experiments establish an empirical accuracy–memory–throughput curve for the target hardware rather than assuming that a particular quantization level is universally optimal.

### Quantization-Aware Training

If Q4 or Q3 produces substantial degradation, an additional ablation compares standard PTQ against QAT at the same nominal precision. The purpose is to quantify whether low-bit accuracy recovery justifies the additional optimization stage.

### Compression Strategy

Finally, if the 4B model remains significantly more accurate while the 1.7B model retains a substantial systems advantage, a distillation experiment is introduced. This comparison evaluates:

1. the native smaller model;
2. the smaller model adapted directly on the bilingual corpus; and
3. the smaller model additionally trained using teacher-generated data from the strongest larger model.

This experiment tests whether knowledge from the accuracy-oriented model can be transferred into the efficiency-oriented model without sacrificing the latter's deployment advantage.

## Reproducibility Protocol

All experiments are conducted using fixed dataset versions, deterministic preprocessing procedures, and recorded random seeds. Training configurations include the base-model checkpoint, tokenizer version, data-mixture proportions, LoRA or QLoRA parameters, optimizer configuration, effective batch size, learning rate, number of update steps, and checkpoint-selection criterion.

For inference experiments, the GGUF conversion procedure, quantization type, `llama.cpp` version, thread count, context length, generation parameters, prompt template, and hardware configuration are recorded. Each hardware benchmark is repeated across multiple runs, and both central tendency and run-to-run variation are retained for the final analysis.

This procedure enables the final model-selection decision to be reconstructed from the reported accuracy, memory, throughput, and thermal measurements and prevents changes in runtime configuration from being confounded with changes in the underlying model.
