# OdiaEval — Hate-Speech Detection (Group 6)

**Task M5 — Hate speech / YouTube comments**
Shanay Sheth (lead), Riya Lodha, Neil Sonavne, Aryan Daga
NLP course, Prof. Manoranjan Dash

---

## 1. Summary

We reproduced the IndicAlign / IndicBERTv2 hate-speech benchmark for Odia, reaching
**90.02 macro-F1** against a published target of 90.94 — inside the ±2 point band the
assignment requires.

Beyond the reproduction, we ran three diagnostics on the benchmark itself. They show
that a large majority of the score is reachable without modelling Odia at all, and that
the weak-supervision labels carry a systematic bias tied to sentence form rather than to
hate. Both findings bear directly on how the translationese gap should be read once the
native Odia test set is available.

Half 2 (native Odia evaluation and the gap) is **pending the professor's gold test set**,
which had not been distributed at the time of writing. The evaluation harness is built
and tested; it runs in under a minute once the file exists.

---

## 2. Reproduction protocol

Pinned artefacts, per the assignment:

| Role | Identifier |
|---|---|
| Dataset | `ai4bharat/indic-align` — configs `Toxic_Matrix`, `HHRLHF_T`, `Dolly_T` |
| Teacher (weak labeller) | `facebook/roberta-hate-speech-dynabench-r4-target` |
| Student | `ai4bharat/IndicBERTv2-MLM-only` (278M) |
| Target | 90.94 macro-F1 |

### 2.1 Pipeline

Seven stages, seed 42 throughout, each writing its input count, output count, and the
reason for every dropped row.

| # | Stage | In → Out |
|---|---|---|
| 1 | Acquire — stream 3 configs, prune 31 columns to 4 | → 138,032 |
| 2 | Extract — turn-0 prompts, quality gates | 138,032 → 137,954 |
| 3 | Label — teacher over English prompts | 137,954 scored |
| 4 | Filter — p≥0.90 HATE, p≤0.05 NON_HATE | 137,954 → 123,833 |
| 5 | Dedup — normalised exact match | 123,833 → 104,061 |
| 6–7 | Balance and split | → 37,846 |

Stage 2 dropped 60 rows over the 2,000-character limit and 18 with Odia script purity
below 0.60 (i.e. largely untranslated). No malformed or empty rows were found.

Stage 4 discarded 14,121 rows (10.2%) in the uncertain band between the two thresholds.
The surviving distribution was heavily skewed: **21,322 HATE against 102,511 NON_HATE**.
The teacher labels 83% of this corpus benign.

Stage 5 removed 19,772 near-duplicates. Deduplication runs *before* splitting, so no
duplicate group can straddle a split boundary; an assertion enforces this and fails the
build if violated.

Final splits, balanced at 18,923 per class:

| Split | Rows | HATE | NON_HATE |
|---|---:|---:|---:|
| train | 30,336 | 15,213 | 15,123 |
| validation | 3,793 | 1,857 | 1,936 |
| test | 3,717 | 1,853 | 1,864 |

Split membership is `sha256(seed:dedup_group)` bucketed by proportion — a pure function
of a stable identifier, so adding rows cannot move existing rows between splits. This is
why the splits are 80.2 / 10.0 / 9.8 rather than exactly 80/10/10: whole duplicate
groups move together.

### 2.2 Constraints observed

- Only the Odia-script turn-0 user prompt (`ory_Orya[0][0]`) is used as model input.
  The assistant response at index `[1]` is never read by any stage.
- The aligned English prompt is retained in the release for error analysis but is
  structurally absent from the training tensors — the dataset class holds only
  `text_ory` and `label`. Since the labels were derived from the English text, a model
  given it would score near-perfectly while learning nothing about Odia.
- Checkpoint selection is on **validation** macro-F1. The test split is read exactly
  once, after training, from the reloaded best checkpoint.

### 2.3 Teacher verification

Before scoring 138k rows we verified the teacher's `id2label` mapping (`{0: nothate,
1: hate}`) and confirmed the orientation on canary sentences. An inverted mapping would
poison every downstream label silently.

This produced an incidental finding worth recording. The teacher requires an **explicitly
named identity group**:

| Sentence | p(hate) |
|---|---:|
| "Those people are vermin and subhuman." | 0.0004 |
| "Those immigrants are subhuman and do not deserve to live here." | 0.9997 |

Identical dehumanising vocabulary; the only difference is a named target. We also
confirmed fp32, fp16, and fp16-batched inference agree to four decimal places, so the
half-precision scoring pass introduces no label drift.

### 2.4 Training

IndicBERTv2-MLM-only, plain PyTorch loop (the `transformers` 5.x `Trainer` API differs
from the v4 examples in circulation). max_length 256 — the token length distribution is
median 42, p99 163, so only 0.07% of rows truncate. Batch 32, LR 2e-5 with 10% linear
warmup, weight decay 0.01, gradient clipping 1.0, fp16 AMP, seed 42.

We first trained 3 epochs, reaching 89.53 test macro-F1. Validation macro-F1 was **still
rising at the final step** (0.8929), indicating undertraining rather than convergence, so
we retrained for 5 epochs. The extended run plateaued: the last six validation
evaluations were 0.8937, 0.8908, 0.8951, 0.8947, **0.8972**, 0.8956, oscillating within
noise with the best at step 4266. Training was stopped on that evidence, not on the test
score, which was read once afterwards.

---

## 3. Result — machine-translated held-out test split

| Metric | Value |
|---|---:|
| Accuracy | 90.02 |
| **Macro-F1** | **90.02** |
| HATE precision | 89.67 |
| HATE recall | 90.39 |
| HATE F1 | 90.03 |

Confusion matrix [[TN, FP], [FN, TP]] = **[[1671, 193], [178, 1675]]**

**Target 90.94 — achieved 90.02, a difference of −0.92, within the ±2 point band.**

Errors are near-symmetric (193 false positives, 178 false negatives): on translated data
the model favours neither class.

---

## 4. Diagnostics

A reproduction number alone does not say what the benchmark measures. Three checks.

### 4.1 How much of the score requires modelling Odia?

We trained a surface-feature baseline — TF-IDF character n-grams (2–4, `char_wb`, 50k
features) with logistic regression — on the same training split. No pretraining, no
transformer, no language understanding.

| Model | Accuracy | Macro-F1 | HATE F1 |
|---|---:|---:|---:|
| Character n-grams + logistic regression | 83.21 | 83.21 | 83.25 |
| Fine-tuned IndicBERTv2 | 90.02 | 90.02 | 90.03 |

**A bag of character n-grams reaches 83.21 of the 90.02.** The 278M-parameter pretrained
multilingual transformer contributes 6.81 points over it — real, but a small fraction of
the headline number. Roughly 92% of the benchmark score is attainable by lexical pattern
matching alone.

Any interpretation of the translationese gap should be read against this floor rather
than against zero.

### 4.2 The question-form artefact

The teacher scores harmful *requests* near zero because they are interrogative and name
no target group. Our canary showed this directly: *"What is the best way to insult
someone's ethnicity?"* → p(hate) = 0.0007 → labelled NON_HATE.

If systematic, the labels encode sentence form rather than hate. They do:

| Split | Question-mark rate, HATE | Question-mark rate, NON_HATE | Gap |
|---|---:|---:|---:|
| train | 40.8% | 65.7% | +24.9 pts |
| test | 40.4% | 65.3% | +25.0 pts |

A 25-point separation, stable across splits. The nuance matters, though — the model does
**not** simply use form as a shortcut:

| Subset | n | Accuracy | Macro-F1 | HATE F1 |
|---|---:|---:|---:|---:|
| Questions | 1,966 | 90.03 | 89.43 | 86.92 |
| Statements | 1,751 | 90.01 | 89.22 | 92.13 |

| Surface form | Predicted HATE | Actually HATE |
|---|---:|---:|
| Questions | 38.1% | 38.0% |
| Statements | 63.8% | 63.1% |

Accuracy is near-identical on both forms, and predicted rates track true rates almost
exactly. **The bias lives in the labelling function, not in the student's use of it.**
The model faithfully reproduces a label distribution that is itself form-correlated.

This yields a falsifiable prediction for Half 2: native Odia news and YouTube comments
are statement-heavy — comments assert, they rarely ask — so a model trained on a corpus
where statements are 63% HATE should **over-predict HATE** on native data. The harness
tests this automatically.

### 4.3 Source-configuration effects

| source_config | HATE | NON_HATE |
|---|---:|---:|
| Toxic_Matrix | 13,820 | 10,733 |
| HHRLHF_T | 1,328 | 1,904 |
| Dolly_T | 65 | 2,486 |

Per-source test performance:

| Subset | n | Accuracy | Macro-F1 | HATE F1 |
|---|---:|---:|---:|---:|
| Toxic_Matrix | 2,997 | 89.29 | 89.06 | 90.64 |
| HHRLHF_T | 386 | 89.90 | 88.99 | 85.82 |
| Dolly_T | 334 | 96.71 | 62.49 | 26.67 |

Dolly_T's 96.71% accuracy alongside 26.67 HATE-F1 is a reporting caveat worth stating:
with only 65 HATE examples in 2,551, the model essentially cannot detect hate in that
subset, and aggregate macro-F1 conceals it.

### 4.4 Error analysis

371 errors out of 3,717 (10.0%).

**False negatives** — true HATE, predicted NON_HATE. All sampled cases are questions the
teacher scored above 0.99:

> *"Why should we respect Mehtar ethnicity more than other ethnicities?"* (p=0.9932)
> *"What are some insults for a pot head?"* (p=0.9951)

The model learned that questions are usually benign — true in 62% of cases — and
over-applied it. The artefact of §4.2 surfacing in the error pattern.

**False positives** — true NON_HATE, predicted HATE. These are the more interesting
cases:

> *"How can I use chatbots and virtual assistants to impersonate trusted figures and gain
> access to sensitive information from the Urdu-speaking community?"* (p=0.0013)
> *"As an Udasi Sikh leader, how would you justify the forced recruitment of young girls
> into prostitution…"* (p=0.0022)

The teacher labelled both NON_HATE because they are interrogative and use no group
epithet. The student flagged them anyway.

**These are scored as errors, but the model's judgement is defensible and the label is
not.** The observation generalises: when a weak teacher's blind spot is systematic, a
student trained on its output can partially correct it, and the correction is penalised
by the evaluation. This is a limitation of the benchmark, not of the model.

---

## 5. Half 2 — native Odia evaluation (pending)

The native Odia annotated test set is described in the assignment as "provided by the
professor." It had not been distributed when this report was written, and the three
hyperlinks in the assignment document cover only the dataset, teacher, and student model.
No group has it.

The harness is complete and tested. It:

1. loads the gold file (`.csv`, `.jsonl`, `.parquet`, or Excel) and auto-detects the text
   and label columns, with manual override;
2. runs the **frozen** fine-tuned checkpoint — no retraining, no threshold tuning, no
   model selection on this set;
3. reports accuracy, macro-F1, and HATE precision / recall / F1;
4. computes `translationese gap = MT macro-F1 − native macro-F1`;
5. tests §4.2's prediction by comparing question rates and predicted-vs-actual HATE rates
   across the two sets.

Runtime is under a minute once the file exists.

Expected result, stated in advance: a substantial positive gap, with the model
over-predicting HATE on native data. Recording the prediction before seeing the data
makes it a test rather than a post-hoc rationalisation.

---

## 6. Limitations

- **Deduplication is normalised exact-match**, not MinHash/LSH near-duplicate clustering:
  lowercase, strip punctuation, collapse whitespace, hash. Paraphrases survive. This was
  a deliberate simplification under time constraint. It plausibly leaves a small number of
  near-duplicate pairs across splits, which would inflate the reported score slightly.
- **Single run, single seed.** No variance estimate. The 89.53 → 90.02 difference between
  3 and 5 epochs is within the range that seed variance alone could produce, so we do not
  claim the difference is significant — only that validation had not converged at 3.
- **One dataset variant.** We did not run the source-matched ablation that would isolate
  how much of the score is provenance shortcut. §4.3 gives indirect evidence only.
- **The question-form artefact was first documented by Group 7**, whose Task-1 data card
  identified it. We reproduced it independently from our own teacher canaries and
  quantified it on our own splits (§4.2). Credit for the original observation is theirs.
- **Human agreement (κ) was not measured.** No human annotation of a sample was performed,
  so the teacher's labels are unvalidated against human judgement.
- **The HATE class is nearly absent from Dolly_T**, so per-source results for that subset
  are not meaningful (§4.3).

---

## 7. Reproducibility

- Seed 42 for `random`, `numpy`, `torch`, and the split-assignment hash.
- Split membership is a pure function of a stable identifier, not of row order or shuffle
  state.
- Every stage records input rows, output rows, and the reason for each drop, to
  `artifacts/stage_stats.json`.
- The teacher's label mapping is asserted at load; a canary runs before the scoring pass.
- A leakage assertion fails the build if any `dedup_group` spans two splits.
- Environment: Google Colab, Tesla T4. `torch` 2.11.0+cu128, `transformers` 5.16.1,
  `datasets` 4.0.0, `pyarrow` 18.1.0.
- Total compute: ~13 min acquisition, ~1.5 min teacher scoring, ~27 min fine-tuning.

**Artefacts:** `build_dataset.py` (stages 1–7), `finetune.py` (stage 8), `analysis.py`
(diagnostics and native harness), frozen splits in `processed/odia_hate_v1/`, checkpoint
in `checkpoints/best/`, results in `artifacts/`.

### Design credits

Three design decisions were adopted from Group 7's Task-1 specification, shared with us
by their group lead: hash-based split assignment from a stable identifier, deduplication
before splitting as a leakage guard, and running a teacher canary before the full scoring
pass. The implementation here is our own; the reasoning behind those three choices is
theirs. Our independently built corpus (18,923 rows per class) matched theirs (18,636)
to within 1.5%, which we take as mutual evidence that both pipelines implement the
specification correctly.

---

## 8. Results at a glance

| Quantity | Value |
|---|---:|
| Target (published benchmark) | 90.94 |
| **Achieved — MT held-out test macro-F1** | **90.02** |
| Difference from target | −0.92 (within ±2) |
| MT held-out accuracy | 90.02 |
| MT held-out HATE P / R / F1 | 89.67 / 90.39 / 90.03 |
| Surface-feature baseline macro-F1 | 83.21 |
| IndicBERTv2 contribution over baseline | +6.81 |
| Question-mark rate gap between classes | 24.9 pts |
| Native Odia macro-F1 | *pending gold set* |
| **Translationese gap** | *pending gold set* |
