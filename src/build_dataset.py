# =============================================================================
# OdiaEval / Group 6 — dataset build: stages 1-7
# -----------------------------------------------------------------------------
#   1 acquire  : stream 3 IndicAlign configs, keep 4 of 31 columns
#   2 extract  : flatten to turn-0 Odia/English prompt pairs + quality gates
#   3 label    : score every English prompt with the pinned RoBERTa teacher
#   4 filter   : p>=0.90 -> HATE, p<=0.05 -> NON_HATE, discard the middle
#   5 dedup    : normalise + hash, assign dedup_group
#   6 balance  : downsample majority class, seed 42
#   7 split    : hash-based 80/10/10, whole dedup_groups never straddle splits
#
# Every stage checkpoints to Drive. Re-running skips completed stages.
# Runtime: ~15 min acquire + ~20 min label on T4 + ~2 min everything else.
# =============================================================================

import os, re, json, hashlib, unicodedata, time
from pathlib import Path

from paths import ROOT, RAW, INTERIM, ART, ensure_dirs
PROC_BASE = ROOT / 'processed'
ensure_dirs()
print(f"working root: {ROOT}")

# =============================================================================
# CONFIG — every tunable lives here, no magic numbers below this block
# =============================================================================
CFG = {
    "seed": 42,
    "dataset": {
        "repo_id": "ai4bharat/indic-align",
        "configs": ["Toxic_Matrix", "HHRLHF_T", "Dolly_T"],
        "keep_columns": ["doc_id", "num_turns", "eng_Latn", "ory_Orya"],
        "expected_rows": {"Toxic_Matrix": 90352, "HHRLHF_T": 32669, "Dolly_T": 15011},
    },
    "teacher": {
        "repo_id": "facebook/roberta-hate-speech-dynabench-r4-target",
        "expected_id2label": {0: "nothate", 1: "hate"},
        "max_length": 512,
        "batch_size": 64,
        "fp16": True,
        "prob_decimals": 6,          # round BEFORE thresholding: fp non-determinism
                                     # must not flip rows across the boundary
        "checkpoint_every": 100,     # batches
    },
    "extract": {
        "turn_index": 0,             # [0][0] only; index [1] never enters
        "min_chars": 3,
        "max_chars": 2000,
        "min_ory_script_purity": 0.60,
    },
    "filter": {"hate_at_or_above": 0.90, "nonhate_at_or_below": 0.05},
    "split":  {"train": 0.80, "validation": 0.10, "test": 0.10},
}
SEED = CFG["seed"]

import random, numpy as np, torch
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

STATS = {}   # every stage records in -> out -> dropped and why


def banner(msg):
    print("\n" + "=" * 72); print(msg); print("=" * 72)


# =============================================================================
# STAGE 1 — ACQUIRE
# =============================================================================
import pyarrow as pa, pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

def stage1_acquire():
    banner("STAGE 1 — acquire")
    fs = HfFileSystem()
    repo = CFG["dataset"]["repo_id"]
    keep = CFG["dataset"]["keep_columns"]

    for cfg in CFG["dataset"]["configs"]:
        out = RAW / f"{cfg}.parquet"
        if out.exists():
            print(f"  {cfg}: cached ({pq.ParquetFile(out).metadata.num_rows:,} rows)")
            continue

        t0 = time.time()
        files = fs.glob(f"datasets/{repo}@refs%2Fconvert%2Fparquet/{cfg}/**/*.parquet")
        assert files, f"no parquet found for {cfg}"

        tables = []
        for fp in files:
            with fs.open(fp, "rb") as f:
                pf = pq.ParquetFile(f)
                # read row group at a time: only one pruned group is ever in RAM
                for i in range(pf.metadata.num_row_groups):
                    tables.append(pf.read_row_group(i, columns=keep))
        tbl = pa.concat_tables(tables)

        exp = CFG["dataset"]["expected_rows"][cfg]
        if tbl.num_rows != exp:
            print(f"  WARNING {cfg}: got {tbl.num_rows:,}, expected {exp:,}")
        pq.write_table(tbl, out, compression="zstd")
        print(f"  {cfg}: {tbl.num_rows:,} rows  ({time.time()-t0:.0f}s)")

    total = sum(pq.ParquetFile(RAW/f"{c}.parquet").metadata.num_rows
                for c in CFG["dataset"]["configs"])
    STATS["stage1"] = {"rows_out": total}
    print(f"  TOTAL: {total:,} rows")


# =============================================================================
# STAGE 2 — EXTRACT  (flatten + quality gates)
# =============================================================================
ODIA_LO, ODIA_HI = 0x0B00, 0x0B7F

def script_purity(s: str) -> float:
    """Fraction of LETTERS that live in the Odia Unicode block.
    Denominator is letters only, so digits/punctuation/spaces don't distort it."""
    letters = [ch for ch in s if unicodedata.category(ch).startswith("L")]
    if not letters:
        return 0.0
    odia = sum(1 for ch in letters if ODIA_LO <= ord(ch) <= ODIA_HI)
    return odia / len(letters)

def norm_text(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()

def turn0(cell):
    """cell is List[List[str]] -> return turn-0 PROMPT, or None if malformed.
    Index [1] (the assistant response) is never touched."""
    if cell is None or len(cell) == 0:
        return None
    first = cell[CFG["extract"]["turn_index"]]
    if first is None or not hasattr(first, "__len__") or len(first) == 0:
        return None
    p = first[0]
    return p if isinstance(p, str) else None

def stage2_extract():
    banner("STAGE 2 — extract")
    out = INTERIM / "pairs.parquet"
    if out.exists():
        print(f"  cached ({pq.ParquetFile(out).metadata.num_rows:,} rows)"); return

    import pandas as pd
    g = CFG["extract"]
    rows, drops = [], {"malformed": 0, "empty": 0, "too_short": 0,
                       "too_long": 0, "low_purity": 0}
    n_in = 0

    for cfg in CFG["dataset"]["configs"]:
        df = pq.read_table(RAW / f"{cfg}.parquet").to_pandas()
        n_in += len(df)
        for r in df.itertuples(index=False):
            ory, eng = turn0(r.ory_Orya), turn0(r.eng_Latn)
            if ory is None or eng is None:
                drops["malformed"] += 1; continue
            ory, eng = norm_text(ory), norm_text(eng)
            if not ory or not eng:
                drops["empty"] += 1; continue
            if len(ory) < g["min_chars"] or len(eng) < g["min_chars"]:
                drops["too_short"] += 1; continue
            if len(ory) > g["max_chars"] or len(eng) > g["max_chars"]:
                drops["too_long"] += 1; continue
            purity = script_purity(ory)
            if purity < g["min_ory_script_purity"]:
                drops["low_purity"] += 1; continue

            eid = hashlib.sha256(f"{cfg}|{r.doc_id}|{ory}".encode()).hexdigest()[:16]
            rows.append({
                "example_id": eid, "doc_id": str(r.doc_id), "source_config": cfg,
                "text_ory": ory, "text_eng": eng,
                "num_turns": int(r.num_turns) if r.num_turns == r.num_turns else -1,
                "ory_script_purity": float(purity),
            })

    df = pd.DataFrame(rows)
    # a duplicate example_id means two identical (config, doc_id, text) rows
    before = len(df); df = df.drop_duplicates("example_id"); id_dupes = before - len(df)

    df.to_parquet(out, index=False)
    STATS["stage2"] = {"rows_in": n_in, "rows_out": len(df),
                       "dropped": drops, "duplicate_ids": id_dupes}
    print(f"  in {n_in:,} -> out {len(df):,}")
    for k, v in drops.items():
        print(f"    dropped {k:12s} {v:,}")
    print(f"    dropped duplicate_id {id_dupes:,}")


# =============================================================================
# STAGE 3 — LABEL  (teacher over English prompts)   << the long pole
# =============================================================================
def stage3_label():
    banner("STAGE 3 — label (teacher)")
    out = INTERIM / "scored.parquet"
    if out.exists():
        print(f"  cached ({pq.ParquetFile(out).metadata.num_rows:,} rows)"); return

    import pandas as pd
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tc = CFG["teacher"]
    df = pd.read_parquet(INTERIM / "pairs.parquet")

    tok = AutoTokenizer.from_pretrained(tc["repo_id"])
    mod = AutoModelForSequenceClassification.from_pretrained(tc["repo_id"])
    assert mod.config.id2label == tc["expected_id2label"], \
        f"teacher id2label changed: {mod.config.id2label} — STOP, labels would invert"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mod = mod.to(dev).eval()
    if tc["fp16"] and dev == "cuda":
        mod = mod.half()

    # canary BEFORE spending 20 minutes: orientation must hold end to end
    with torch.no_grad():
        # NOTE: the hateful canary MUST name an identity group. This teacher is
        # trained on identity-directed hate: "those people are vermin and
        # subhuman" scores 0.0004, while the same claim aimed at a named group
        # scores 0.9997. Verified fp32 == fp16 == fp16+batched to 4 dp.
        enc = tok(["I love my family.",
                   "Those immigrants are subhuman and do not deserve to live here."],
                  return_tensors="pt", padding=True, truncation=True,
                  max_length=tc["max_length"]).to(dev)
        p = torch.softmax(mod(**enc).logits.float(), -1)[:, 1].tolist()
    assert p[0] < 0.10 and p[1] > 0.90, f"canary failed: {p} — do not proceed"
    print(f"  canary ok: benign={p[0]:.4f}  hateful={p[1]:.4f}")

    # sort by length so batches pad to similar lengths — big speedup
    order = df["text_eng"].str.len().sort_values().index
    texts = df.loc[order, "text_eng"].tolist()
    ids   = df.loc[order, "example_id"].tolist()

    ckpt = INTERIM / "scored_ckpt.json"
    done = json.loads(ckpt.read_text()) if ckpt.exists() else {}
    if done:
        print(f"  resuming: {len(done):,} already scored")

    bs, probs = tc["batch_size"], {}
    probs.update(done)
    todo = [(i, t) for i, t in zip(ids, texts) if i not in probs]
    t0 = time.time()

    with torch.no_grad():
        for b in range(0, len(todo), bs):
            chunk = todo[b:b+bs]
            enc = tok([t for _, t in chunk], return_tensors="pt", padding=True,
                      truncation=True, max_length=tc["max_length"]).to(dev)
            # softmax in fp32 regardless of fp16 forward — protects the threshold
            pr = torch.softmax(mod(**enc).logits.float(), -1)[:, 1].tolist()
            for (i, _), v in zip(chunk, pr):
                probs[i] = v
            if (b // bs) % tc["checkpoint_every"] == 0 and b > 0:
                ckpt.write_text(json.dumps(probs))
                el = time.time() - t0
                rate = (b + bs) / el
                print(f"    {b+bs:,}/{len(todo):,}  {rate:.0f} rows/s  "
                      f"eta {(len(todo)-b-bs)/rate/60:.1f} min")

    ckpt.write_text(json.dumps(probs))
    df["teacher_prob_hate_raw"] = df["example_id"].map(probs).astype("float32")
    df["teacher_prob_hate"] = df["teacher_prob_hate_raw"].round(tc["prob_decimals"])
    assert df["teacher_prob_hate"].notna().all(), "some rows never got scored"
    df.to_parquet(out, index=False)
    STATS["stage3"] = {"rows_out": len(df), "seconds": round(time.time()-t0)}
    print(f"  scored {len(df):,} rows in {(time.time()-t0)/60:.1f} min")


# =============================================================================
# STAGE 4 — FILTER   (the two fixed thresholds)
# =============================================================================
def stage4_filter():
    banner("STAGE 4 — filter")
    import pandas as pd
    df = pd.read_parquet(INTERIM / "scored.parquet")
    hi, lo = CFG["filter"]["hate_at_or_above"], CFG["filter"]["nonhate_at_or_below"]

    p = df["teacher_prob_hate"]
    df["label"] = np.where(p >= hi, 1, np.where(p <= lo, 0, -1)).astype("int8")
    kept = df[df.label >= 0].copy()
    kept["label_str"] = np.where(kept.label == 1, "HATE", "NON_HATE")
    kept.to_parquet(INTERIM / "labelled.parquet", index=False)

    n_h, n_n, n_d = int((df.label==1).sum()), int((df.label==0).sum()), int((df.label==-1).sum())
    STATS["stage4"] = {"rows_in": len(df), "hate": n_h, "nonhate": n_n, "discarded": n_d}
    print(f"  in {len(df):,}")
    print(f"    HATE     (p >= {hi})  {n_h:,}")
    print(f"    NON_HATE (p <= {lo})  {n_n:,}")
    print(f"    discarded (uncertain) {n_d:,}  ({n_d/len(df)*100:.1f}%)")
    print(f"  kept {len(kept):,}")


# =============================================================================
# STAGE 5 — DEDUP   (normalised exact match -> dedup_group)
# =============================================================================
_ws = re.compile(r"\s+")
_punct = re.compile(r"[^\w\s]", re.UNICODE)

def dedup_key(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    s = _punct.sub("", s)
    s = _ws.sub(" ", s).strip()
    return hashlib.sha256(s.encode()).hexdigest()[:16]

def stage5_dedup():
    banner("STAGE 5 — dedup")
    import pandas as pd
    df = pd.read_parquet(INTERIM / "labelled.parquet")
    n_in = len(df)
    df["dedup_group"] = df["text_ory"].map(dedup_key)

    # keep one representative per group: lowest example_id, so it is deterministic
    df = df.sort_values("example_id").drop_duplicates("dedup_group", keep="first")
    df.to_parquet(INTERIM / "deduped.parquet", index=False)
    STATS["stage5"] = {"rows_in": n_in, "rows_out": len(df), "removed": n_in-len(df)}
    print(f"  in {n_in:,} -> out {len(df):,}  (removed {n_in-len(df):,} duplicates)")
    print(f"  HATE {int((df.label==1).sum()):,}  NON_HATE {int((df.label==0).sum()):,}")


# =============================================================================
# STAGE 6+7 — BALANCE and SPLIT
# =============================================================================
def split_of(dedup_group: str) -> str:
    """Pure function of a stable id: adding rows cannot move existing rows."""
    h = hashlib.sha256(f"{SEED}:{dedup_group}".encode()).hexdigest()
    frac = int(h[:16], 16) / float(1 << 64)
    tr, va = CFG["split"]["train"], CFG["split"]["validation"]
    return "train" if frac < tr else ("validation" if frac < tr + va else "test")

def stage67_balance_split():
    banner("STAGE 6+7 — balance and split")
    import pandas as pd
    df = pd.read_parquet(INTERIM / "deduped.parquet")

    hate, non = df[df.label == 1], df[df.label == 0]
    n = min(len(hate), len(non))
    rng = np.random.RandomState(SEED)
    def take(sub):
        sub = sub.sort_values("example_id")                 # deterministic pool order
        idx = rng.choice(len(sub), size=n, replace=False)
        return sub.iloc[np.sort(idx)]
    bal = pd.concat([take(hate), take(non)]).sort_values("example_id")

    bal["split"] = bal["dedup_group"].map(split_of)
    cols = ["example_id","doc_id","source_config","text_ory","text_eng","label",
            "label_str","teacher_prob_hate","teacher_prob_hate_raw","dedup_group",
            "split","num_turns","ory_script_purity"]
    bal = bal[cols]

    outdir = PROC_BASE / "odia_hate_v1"; outdir.mkdir(parents=True, exist_ok=True)
    for sp in ("train", "validation", "test"):
        part = bal[bal.split == sp]
        part.to_parquet(outdir/f"{sp}.parquet", index=False)
        part.to_json(outdir/f"{sp}.jsonl", orient="records", lines=True,
                     force_ascii=False)

    # leakage guard: a dedup_group must never appear in two splits
    leak = bal.groupby("dedup_group")["split"].nunique()
    assert (leak == 1).all(), f"LEAK: {int((leak>1).sum())} groups span splits"

    STATS["stage67"] = {"balanced_total": len(bal), "per_class": n,
                        "splits": bal.split.value_counts().to_dict()}
    print(f"  balanced: {len(bal):,} rows ({n:,} per class)")
    for sp in ("train","validation","test"):
        part = bal[bal.split==sp]
        print(f"    {sp:11s} {len(part):6,}  HATE {int((part.label==1).sum()):5,}  "
              f"NON_HATE {int((part.label==0).sum()):5,}")
    print("  leakage guard passed: no dedup_group spans two splits")
    print(f"\n  written to {outdir}")


# =============================================================================
# RUN
# =============================================================================
stage1_acquire()
stage2_extract()
stage3_label()
stage4_filter()
stage5_dedup()
stage67_balance_split()

(ART/"stage_stats.json").write_text(json.dumps(STATS, indent=2, default=str))
banner("DONE — dataset frozen")
print(json.dumps(STATS, indent=2, default=str))
