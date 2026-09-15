# =============================================================================
# OdiaEval / Group 6 — Stage 8: fine-tune IndicBERTv2-MLM-only
# -----------------------------------------------------------------------------
# Plain PyTorch loop, deliberately: transformers 5.x changed the Trainer API and
# every v4 example on the internet is now subtly wrong. 60 lines we control
# beats an API we have to debug at midnight.
#
# Trains on text_ory ONLY. text_eng is never loaded into a tensor — it is the
# text the labels came from, so a model given it scores ~perfectly while
# learning nothing about Odia.
#
# Selects the checkpoint on validation macro-F1. Test split is touched exactly
# once, at the end.
# =============================================================================

import json, time, math
from pathlib import Path


import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

from paths import ROOT, PROC, ART, CKPT_DIR, CKPT, ensure_dirs
ensure_dirs()

TRAIN = {
    "model": "ai4bharat/IndicBERTv2-MLM-only",
    "max_length": 256,
    "batch_size": 32,
    "eval_batch_size": 128,
    "lr": 2e-5,
    "epochs": 5,   # 3 epochs left val macro-F1 still climbing (0.8929 at final step)
    "warmup_frac": 0.10,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "evals_per_epoch": 2,        # check val macro-F1 twice an epoch
    "seed": 42,
}
SEED = TRAIN["seed"]

import random
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev}  {torch.cuda.get_device_name(0) if dev=='cuda' else ''}")

# =============================================================================
# Data
# =============================================================================
assert PROC.exists(), (f"{PROC} not found — is Drive mounted? "
                       "Set ODIAEVAL_ROOT or run build_dataset.py first.")
splits = {s: pd.read_parquet(PROC/f"{s}.parquet") for s in ("train","validation","test")}
for s, d in splits.items():
    print(f"  {s:11s} {len(d):6,}  HATE {int((d.label==1).sum()):5,}  "
          f"NON_HATE {int((d.label==0).sum()):5,}")

tok = AutoTokenizer.from_pretrained(TRAIN["model"])

# --- how long are these prompts really? max_length is a real accuracy lever ---
sample = splits["train"]["text_ory"].sample(min(3000, len(splits["train"])),
                                            random_state=SEED).tolist()
lens = np.array([len(tok(t, truncation=False)["input_ids"]) for t in sample])
print(f"\ntoken lengths (n={len(lens)}): median {np.median(lens):.0f}  "
      f"p90 {np.percentile(lens,90):.0f}  p99 {np.percentile(lens,99):.0f}  max {lens.max()}")
trunc = (lens > TRAIN['max_length']).mean()
print(f"truncated at max_length={TRAIN['max_length']}: {trunc*100:.2f}% of rows")
if trunc > 0.05:
    print("  NOTE: >5% truncated — consider raising max_length")

class OdiaDS(Dataset):
    """Holds text_ory and label only. text_eng is structurally absent."""
    def __init__(self, df):
        self.texts  = df["text_ory"].tolist()
        self.labels = df["label"].astype(int).tolist()
    def __len__(self):  return len(self.labels)
    def __getitem__(self, i): return self.texts[i], self.labels[i]

def collate(batch):
    texts, labels = zip(*batch)
    enc = tok(list(texts), padding=True, truncation=True,
              max_length=TRAIN["max_length"], return_tensors="pt")
    enc["labels"] = torch.tensor(labels, dtype=torch.long)
    return enc

g = torch.Generator(); g.manual_seed(SEED)
dl_train = DataLoader(OdiaDS(splits["train"]), batch_size=TRAIN["batch_size"],
                      shuffle=True, collate_fn=collate, generator=g, drop_last=False)
dl_val   = DataLoader(OdiaDS(splits["validation"]), batch_size=TRAIN["eval_batch_size"],
                      shuffle=False, collate_fn=collate)
dl_test  = DataLoader(OdiaDS(splits["test"]), batch_size=TRAIN["eval_batch_size"],
                      shuffle=False, collate_fn=collate)

# =============================================================================
# Model
# =============================================================================
model = AutoModelForSequenceClassification.from_pretrained(
    TRAIN["model"], num_labels=2,
    id2label={0: "NON_HATE", 1: "HATE"}, label2id={"NON_HATE": 0, "HATE": 1},
).to(dev)
print(f"\nparameters: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

decay = [p for n,p in model.named_parameters() if not any(k in n for k in ("bias","LayerNorm.weight"))]
nodecay = [p for n,p in model.named_parameters() if any(k in n for k in ("bias","LayerNorm.weight"))]
opt = torch.optim.AdamW([{"params": decay,   "weight_decay": TRAIN["weight_decay"]},
                         {"params": nodecay, "weight_decay": 0.0}], lr=TRAIN["lr"])

total_steps = len(dl_train) * TRAIN["epochs"]
sched = get_linear_schedule_with_warmup(opt, int(total_steps*TRAIN["warmup_frac"]), total_steps)
scaler = torch.amp.GradScaler("cuda", enabled=(dev=="cuda"))

# =============================================================================
# Eval
# =============================================================================
@torch.no_grad()
def evaluate(dl):
    model.eval()
    P, Y = [], []
    for b in dl:
        y = b.pop("labels")
        b = {k: v.to(dev) for k, v in b.items()}
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(dev=="cuda")):
            logits = model(**b).logits
        P.append(logits.float().argmax(-1).cpu()); Y.append(y)
    p = torch.cat(P).numpy(); y = torch.cat(Y).numpy()
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[1], zero_division=0)
    return {
        "accuracy":   float(accuracy_score(y, p)),
        "macro_f1":   float(f1_score(y, p, average="macro", zero_division=0)),
        "hate_precision": float(pr[0]),
        "hate_recall":    float(rc[0]),
        "hate_f1":        float(f1[0]),
        "confusion":  confusion_matrix(y, p).tolist(),
    }, p, y

# =============================================================================
# Train
# =============================================================================
eval_every = max(1, len(dl_train) // TRAIN["evals_per_epoch"])
best = {"macro_f1": -1.0, "step": -1}
history, step, t0 = [], 0, time.time()
print(f"\n{len(dl_train)} steps/epoch x {TRAIN['epochs']} epochs = {total_steps} steps")
print(f"evaluating every {eval_every} steps\n")

for ep in range(TRAIN["epochs"]):
    model.train()
    running = 0.0
    for i, b in enumerate(dl_train):
        b = {k: v.to(dev) for k, v in b.items()}
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(dev=="cuda")):
            loss = model(**b).loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN["max_grad_norm"])
        scaler.step(opt); scaler.update(); sched.step()
        running += loss.item(); step += 1

        if step % 100 == 0:
            el = time.time() - t0
            print(f"  ep{ep+1} step {step}/{total_steps}  loss {running/100:.4f}  "
                  f"{step/el:.1f} it/s  eta {(total_steps-step)/(step/el)/60:.1f} min")
            running = 0.0

        if step % eval_every == 0 or step == total_steps:
            m, _, _ = evaluate(dl_val)
            history.append({"step": step, **{k:v for k,v in m.items() if k!="confusion"}})
            flag = ""
            if m["macro_f1"] > best["macro_f1"]:
                best = {"macro_f1": m["macro_f1"], "step": step, "epoch": ep+1}
                model.save_pretrained(CKPT); tok.save_pretrained(CKPT)
                flag = "  <-- best, saved"
            print(f"  VAL step {step}: acc {m['accuracy']:.4f}  macroF1 {m['macro_f1']:.4f}  "
                  f"hateF1 {m['hate_f1']:.4f}{flag}")
            model.train()

print(f"\ntrained in {(time.time()-t0)/60:.1f} min")
print(f"best val macro-F1 {best['macro_f1']:.4f} at step {best['step']} (epoch {best['epoch']})")

# =============================================================================
# TEST — the split is read exactly once, here
# =============================================================================
print("\n" + "="*72)
print("TEST — held-out machine-translated split")
print("="*72)
model = AutoModelForSequenceClassification.from_pretrained(CKPT).to(dev)
test_m, test_pred, test_y = evaluate(dl_test)

print(f"  accuracy        {test_m['accuracy']*100:.2f}")
print(f"  macro-F1        {test_m['macro_f1']*100:.2f}   <-- target 90.94 (+/-2)")
print(f"  HATE precision  {test_m['hate_precision']*100:.2f}")
print(f"  HATE recall     {test_m['hate_recall']*100:.2f}")
print(f"  HATE F1         {test_m['hate_f1']*100:.2f}")
print(f"  confusion [[TN,FP],[FN,TP]] = {test_m['confusion']}")

gap = test_m['macro_f1']*100 - 90.94
verdict = "WITHIN target band" if abs(gap) <= 2 else "OUTSIDE target band"
print(f"\n  {gap:+.2f} points vs target 90.94 -> {verdict}")

# save predictions for the error analysis
out = splits["test"].copy()
out["pred"] = test_pred
out.to_parquet(ART/"test_predictions.parquet", index=False)

with open(ART/"train_results.json", "w") as fh:
    json.dump({"config": TRAIN, "best_val": best, "test": test_m, "history": history},
              fh, indent=2)
    fh.flush()
print(f"\n  checkpoint  {CKPT}")
print(f"  predictions {ART/'test_predictions.parquet'}")
print(f"  results     {ART/'train_results.json'}")
