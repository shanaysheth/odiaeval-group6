# OdiaEval — Hate-Speech Detection (Group 6)

Reproduction of the IndicAlign / IndicBERTv2 Odia hate-speech benchmark, plus diagnostics
on what the benchmark actually measures.

**Course:** NLP, Prof. Manoranjan Dash
**Group 6:** Shanay Sheth (lead), Riya Lodha, Neil Sonavne, Aryan Daga

---

## Result

| Metric | Value |
|---|---:|
| **Macro-F1 (held-out MT test split)** | **90.02** |
| Target | 90.94 (±2) |
| Difference | −0.92 — within band |
| Accuracy | 90.02 |
| HATE precision / recall / F1 | 89.67 / 90.39 / 90.03 |
| Confusion [[TN,FP],[FN,TP]] | [[1671, 193], [178, 1675]] |

Splits: **30,336 train / 3,793 validation / 3,717 test** (37,846 total, balanced at
18,923 per class).

Native Odia evaluation and the translationese gap are **pending the gold test set**. The
harness is built and tested (`src/analysis.py`, section D); it runs in under a minute once
the file is available.

### Diagnostics

| Finding | Value |
|---|---:|
| Character n-gram + logistic regression baseline | 83.21 macro-F1 |
| IndicBERTv2 contribution over that baseline | +6.81 |
| Question-mark rate gap between HATE and NON_HATE | 24.9 pts |

~92% of the benchmark score is reachable without modelling Odia at all, and the teacher's
labels correlate strongly with sentence form. See `report/` for the full analysis.

---

## Pipeline

| Stage | Script | In → Out |
|---|---|---|
| 1 Acquire — stream 3 IndicAlign configs, prune 31 columns to 4 | `build_dataset.py` | → 138,032 |
| 2 Extract — turn-0 prompts, quality gates | `build_dataset.py` | 138,032 → 137,954 |
| 3 Label — RoBERTa teacher over English prompts | `build_dataset.py` | 137,954 scored |
| 4 Filter — p≥0.90 HATE, p≤0.05 NON_HATE | `build_dataset.py` | → 123,833 |
| 5 Dedup — normalised exact match | `build_dataset.py` | → 104,061 |
| 6–7 Balance + hash-based split | `build_dataset.py` | → 37,846 |
| 8 Fine-tune IndicBERTv2, select on val macro-F1 | `finetune.py` | 90.02 test macro-F1 |
| 9 Diagnostics + native harness | `analysis.py` | — |

Pinned artefacts:

- Dataset — `ai4bharat/indic-align` (`Toxic_Matrix`, `HHRLHF_T`, `Dolly_T`)
- Teacher — `facebook/roberta-hate-speech-dynabench-r4-target`
- Student — `ai4bharat/IndicBERTv2-MLM-only`

---

## Running it

```bash
pip install -r requirements.txt
./run_all.sh
```

Or stage by stage:

```bash
cd src
python build_dataset.py    # ~15 min (13 min is network)
python finetune.py         # ~27 min on a T4
python analysis.py         # ~1 min
```

**Where files go.** `src/paths.py` picks the working directory automatically: Google Drive
on Colab, `./work/` elsewhere. Override with `ODIAEVAL_ROOT`:

```bash
export ODIAEVAL_ROOT=/home/nlp/Group6_HateSpeech/work
```

**Resuming.** Every stage checks for its output first and skips if present, so re-running
is cheap. The teacher scoring pass checkpoints every 100 batches and resumes mid-way.

**Native evaluation**, once the gold set arrives:

```bash
mkdir -p "$ODIAEVAL_ROOT/native" && cp gold.csv "$ODIAEVAL_ROOT/native/"
python src/analysis.py     # section D runs automatically
```

It auto-detects text and label columns, evaluates the frozen checkpoint, reports the gap,
and tests the question-form prediction on the native data.

---

## Correctness measures

These exist because each one guards a failure that would silently invalidate the result.

- **Teacher orientation asserted at load**, plus a canary before the scoring pass. An
  inverted `id2label` would poison all 138k labels with no visible symptom.
- **Deduplication before splitting**, with an assertion that no `dedup_group` spans two
  splits. Near-duplicates across the train/test boundary inflate scores.
- **Split membership is `sha256(seed:dedup_group)`** — a pure function of a stable id, so
  adding rows never moves existing rows.
- **The English prompt never enters a training tensor.** The dataset class holds only
  `text_ory` and `label`. Since labels derive from the English text, a model given it
  would score near-perfectly while learning no Odia.
- **Softmax in fp32** even under a fp16 forward pass, with probabilities rounded to 6dp
  before thresholding, so GPU non-determinism cannot flip rows across the 0.90 boundary.
- **Model selection on validation only.** The test split is read once, at the end, from
  the reloaded best checkpoint.
- Seed 42 for `random`, `numpy`, `torch`, and split assignment.

---

## Environment

Google Colab, Tesla T4 (16 GB). `torch` 2.11.0+cu128, `transformers` 5.16.1, `datasets`
4.0.0, `pyarrow` 18.1.0. See `requirements.txt`.

Compute: ~13 min acquisition, ~1.5 min teacher scoring (138k rows), ~27 min fine-tuning.

---

## Limitations

- Deduplication is normalised exact-match, not MinHash/LSH — paraphrases survive. A
  deliberate simplification under time constraint.
- Single seed, single run. No variance estimate.
- One dataset variant; no source-matched ablation.
- Human agreement (κ) not measured — the teacher's labels are unvalidated against human
  judgement.
- The HATE class is nearly absent from `Dolly_T` (65 of 2,551), so per-source results for
  that subset are not meaningful.

---

## Credits

Three design decisions were adopted from Group 7's Task-1 specification, shared with us by
their group lead: hash-based split assignment, dedup-before-split as a leakage guard, and
a teacher canary before the scoring pass. The implementation here is our own; the
reasoning behind those three choices is theirs. Group 7 also first documented the
question-form labelling artefact in their data card; we reproduced it independently and
quantified it on our own splits.

Our independently built corpus (18,923 rows per class) matched theirs (18,636) to within
1.5%.

## References

- IndicAlign — https://aclanthology.org/2024.acl-long.843/
- IndicBERTv2 — https://aclanthology.org/2023.acl-long.693/
- Weak-label teacher (Dynabench R4) — https://aclanthology.org/2021.acl-long.132/
