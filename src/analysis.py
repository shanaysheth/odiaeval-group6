# =============================================================================
# OdiaEval / Group 6 — Stage 9: analysis + native-set harness
# -----------------------------------------------------------------------------
#  A. surface-feature baseline  — how much of the score needs no Odia at all?
#  B. the question-form artefact — the falsifiable prediction, tested
#  C. source-config shortcut    — can the model cheat off provenance?
#  D. native Odia eval harness  — one function, waiting on the professor's file
#
# Run AFTER finetune.py. Reads artifacts/test_predictions.parquet.
# Runtime: ~1 min for A-C. D runs only if the native file exists.
# =============================================================================

import json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

from paths import ROOT, PROC, ART, CKPT, ensure_dirs
ensure_dirs()

splits = {s: pd.read_parquet(PROC/f"{s}.parquet") for s in ("train","validation","test")}
test = pd.read_parquet(ART/"test_predictions.parquet")
RESULTS = {}

def banner(m): print("\n" + "="*72); print(m); print("="*72)

def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[1], zero_division=0)
    return {"n": int(len(y)),
            "accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "hate_precision": float(pr[0]), "hate_recall": float(rc[0]),
            "hate_f1": float(f1[0])}

def show(tag, m):
    print(f"  {tag:34s} n={m['n']:5,}  acc {m['accuracy']*100:5.2f}  "
          f"macroF1 {m['macro_f1']*100:5.2f}  hateF1 {m['hate_f1']*100:5.2f}")

# =============================================================================
# A. Surface-feature baseline
# =============================================================================
banner("A. SURFACE BASELINE — character n-grams + logistic regression")
print("If a bag of character n-grams scores well, that much of the task needs no")
print("language understanding at all. It is the floor any real score sits above.\n")

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), max_features=50_000,
                      sublinear_tf=True, min_df=2)
Xtr = vec.fit_transform(splits["train"]["text_ory"])
Xte = vec.transform(test["text_ory"])
clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
clf.fit(Xtr, splits["train"]["label"])
base_pred = clf.predict(Xte)

base_m = metrics(test["label"].values, base_pred)
model_m = metrics(test["label"].values, test["pred"].values)
show("surface baseline", base_m)
show("fine-tuned IndicBERTv2", model_m)
print(f"\n  IndicBERTv2 adds {(model_m['macro_f1']-base_m['macro_f1'])*100:+.2f} macro-F1 "
      f"over character n-grams.")
print(f"  Read the baseline as the artefact floor: {base_m['macro_f1']*100:.2f} of the")
print(f"  {model_m['macro_f1']*100:.2f} is reachable without modelling Odia at all.")
RESULTS["surface_baseline"] = base_m
RESULTS["model_test"] = model_m

# =============================================================================
# B. The question-form artefact
# =============================================================================
banner("B. QUESTION-FORM ARTEFACT")
print("Claim: the teacher scores harmful REQUESTS near zero because they are")
print("interrogative, so the labels encode 'question -> NON_HATE, statement -> HATE'.")
print("If true, the student inherits a shortcut that has nothing to do with hate.\n")

for name, d in [("train", splits["train"]), ("test", test)]:
    d = d.copy(); d["is_q"] = d["text_ory"].str.contains(r"\?", regex=True)
    qh = d[d.label==1]["is_q"].mean(); qn = d[d.label==0]["is_q"].mean()
    print(f"  {name:6s}  question-mark rate:  HATE {qh*100:5.1f}%   "
          f"NON_HATE {qn*100:5.1f}%   gap {(qn-qh)*100:+.1f} pts")

test = test.copy(); test["is_q"] = test["text_ory"].str.contains(r"\?", regex=True)
print()
for tag, sub in [("questions", test[test.is_q]), ("statements", test[~test.is_q])]:
    if len(sub):
        show(f"model on {tag}", metrics(sub["label"].values, sub["pred"].values))

# The sharper test: does the model predict from form alone?
print("\n  Predicted-HATE rate by surface form (label-independent):")
for tag, sub in [("questions", test[test.is_q]), ("statements", test[~test.is_q])]:
    if len(sub):
        print(f"    {tag:11s} predicted HATE {sub['pred'].mean()*100:5.1f}%   "
              f"actually HATE {sub['label'].mean()*100:5.1f}%")

RESULTS["question_artefact"] = {
    "train_q_rate_hate":    float(splits["train"][splits["train"].label==1]["text_ory"].str.contains(r"\?").mean()),
    "train_q_rate_nonhate": float(splits["train"][splits["train"].label==0]["text_ory"].str.contains(r"\?").mean()),
    "test_questions":  metrics(test[test.is_q]["label"].values, test[test.is_q]["pred"].values) if test.is_q.any() else None,
    "test_statements": metrics(test[~test.is_q]["label"].values, test[~test.is_q]["pred"].values) if (~test.is_q).any() else None,
}
print("\n  IMPLICATION for the native set: if native Odia YouTube/news comments are")
print("  statement-heavy (they will be — comments assert, they rarely ask), this model")
print("  will OVER-PREDICT HATE there. That is a concrete, falsifiable prediction.")

# =============================================================================
# C. Source-config shortcut
# =============================================================================
banner("C. SOURCE-CONFIG SHORTCUT")
print("Toxic_Matrix is mostly HATE, Dolly_T mostly benign. If provenance leaks into")
print("the text, the model can score well by detecting the source, not the hate.\n")

ct = pd.crosstab(splits["train"]["source_config"], splits["train"]["label_str"])
print(ct.to_string())
print()
for cfg in sorted(test["source_config"].unique()):
    sub = test[test.source_config == cfg]
    if len(sub) > 20:
        show(f"test / {cfg}", metrics(sub["label"].values, sub["pred"].values))
RESULTS["by_source"] = {c: metrics(test[test.source_config==c]["label"].values,
                                   test[test.source_config==c]["pred"].values)
                        for c in test["source_config"].unique()
                        if (test.source_config==c).sum() > 20}

# =============================================================================
# Error inspection — a handful of actual mistakes, for the write-up
# =============================================================================
banner("ERROR SAMPLES (read these — they are what the numbers mean)")
err = test[test.label != test.pred]
print(f"  {len(err)} errors out of {len(test)} ({len(err)/len(test)*100:.1f}%)\n")
for kind, sub in [("FALSE NEGATIVE (true HATE, predicted NON_HATE)", err[err.label==1]),
                  ("FALSE POSITIVE (true NON_HATE, predicted HATE)", err[err.label==0])]:
    print(f"  --- {kind} ---")
    for _, r in sub.head(3).iterrows():
        print(f"    p(hate)={r.teacher_prob_hate:.4f} q={'Y' if '?' in r.text_ory else 'N'} "
              f"[{r.source_config}]")
        print(f"      ory: {r.text_ory[:100]}")
        print(f"      eng: {r.text_eng[:100]}")
    print()

json.dump(RESULTS, open(ART/"analysis.json","w"), indent=2)
print(f"saved -> {ART/'analysis.json'}")

# =============================================================================
# D. NATIVE ODIA EVAL HARNESS
# =============================================================================
banner("D. NATIVE ODIA EVALUATION")

def evaluate_native(path, text_col=None, label_col=None, hate_value=None):
    """Evaluate the FROZEN fine-tuned checkpoint on the professor's native set.

    No training, no threshold tuning, no model selection. Load, predict, report.
    Tries to auto-detect the text/label columns; override if it guesses wrong.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    p = Path(path)
    df = (pd.read_csv(p) if p.suffix == ".csv" else
          pd.read_json(p, lines=True) if p.suffix in (".jsonl",".json") else
          pd.read_parquet(p) if p.suffix == ".parquet" else
          pd.read_excel(p))
    print(f"  loaded {len(df):,} rows from {p.name}")
    print(f"  columns: {list(df.columns)}")

    if text_col is None:
        cands = [c for c in df.columns
                 if df[c].dtype == object and df[c].astype(str).str.len().mean() > 15]
        text_col = cands[0] if cands else df.columns[0]
    if label_col is None:
        cands = [c for c in df.columns if c != text_col and df[c].nunique() <= 5]
        label_col = cands[0] if cands else df.columns[-1]
    print(f"  using text='{text_col}'  label='{label_col}'")
    print(f"  label values: {df[label_col].value_counts().to_dict()}")

    y = df[label_col]
    if y.dtype == object:
        if hate_value is None:
            hate_value = [v for v in y.unique()
                          if str(v).strip().lower() in
                          ("hate","hateful","1","yes","true","offensive")]
            hate_value = hate_value[0] if hate_value else sorted(y.unique())[-1]
            print(f"  treating '{hate_value}' as HATE (override hate_value= if wrong)")
        y = (y == hate_value).astype(int)
    y = y.astype(int).values

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(CKPT)
    mod = AutoModelForSequenceClassification.from_pretrained(CKPT).to(dev).eval()

    texts, preds = df[text_col].astype(str).tolist(), []
    with torch.no_grad():
        for i in range(0, len(texts), 128):
            enc = tok(texts[i:i+128], padding=True, truncation=True,
                      max_length=256, return_tensors="pt").to(dev)
            preds.extend(mod(**enc).logits.float().argmax(-1).cpu().tolist())
    preds = np.array(preds)

    m = metrics(y, preds)
    print(f"\n  NATIVE ODIA RESULTS")
    print(f"    accuracy        {m['accuracy']*100:.2f}")
    print(f"    macro-F1        {m['macro_f1']*100:.2f}")
    print(f"    HATE precision  {m['hate_precision']*100:.2f}")
    print(f"    HATE recall     {m['hate_recall']*100:.2f}")
    print(f"    HATE F1         {m['hate_f1']*100:.2f}")
    print(f"    confusion [[TN,FP],[FN,TP]] = {confusion_matrix(y, preds).tolist()}")

    mt = RESULTS["model_test"]["macro_f1"]*100
    print(f"\n  TRANSLATIONESE GAP (macro-F1)")
    print(f"    machine-translated held-out  {mt:.2f}")
    print(f"    native Odia                  {m['macro_f1']*100:.2f}")
    print(f"    gap                          {mt - m['macro_f1']*100:+.2f}")

    df["_pred"] = preds; df["_gold"] = y
    df["_is_q"] = df[text_col].astype(str).str.contains(r"\?")
    print(f"\n  Testing section B's prediction on native data:")
    print(f"    native question rate: {df['_is_q'].mean()*100:.1f}% "
          f"(vs {test['is_q'].mean()*100:.1f}% in the MT test split)")
    print(f"    predicted HATE rate:  {preds.mean()*100:.1f}%  "
          f"actual HATE rate: {y.mean()*100:.1f}%")
    if preds.mean() > y.mean() + 0.05:
        print("    -> model OVER-predicts HATE, as section B predicted.")
    elif preds.mean() < y.mean() - 0.05:
        print("    -> model UNDER-predicts HATE, contrary to section B.")

    df.to_parquet(ART/"native_predictions.parquet", index=False)
    RESULTS["native"] = m
    RESULTS["translationese_gap_macro_f1"] = float(mt - m["macro_f1"]*100)
    json.dump(RESULTS, open(ART/"analysis.json","w"), indent=2)
    print(f"\n  saved -> {ART/'native_predictions.parquet'}")
    return m

# auto-run if the file has arrived
found = sorted((ROOT/'native').glob('*')) if (ROOT/'native').exists() else []
if found:
    print(f"  found native file(s): {[p.name for p in found]}")
    evaluate_native(found[0])
else:
    print("  No native Odia test set yet.")
    print(f"  When it arrives: place it in {ROOT/'native'}/ and re-run,")
    print("  or call directly:")
    print("      evaluate_native(ROOT / 'native' / 'gold.csv')")
    print("\n  Harness is built and ready. Zero work left once the file exists.")
