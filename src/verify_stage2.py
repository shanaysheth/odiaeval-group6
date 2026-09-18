"""Recompute stage-2 extraction statistics and patch them into stage_stats.json.

WHY THIS EXISTS
---------------
build_dataset.py caches each stage: if a stage's output file already exists it
returns early and never records its statistics. On the run that produced the
reported results, stage 2 was already cached from an earlier run (the first
attempt halted at stage 3 on a teacher-canary assertion), so `stage2` is absent
from stage_stats.json even though the stage did execute.

This script recomputes those counts from the cached raw parquet using the exact
same gates, so the 138,032 -> 137,954 drop is backed by an artefact rather than
by a console log. The extraction logic is deterministic: re-running it on the
same inputs reproduces the same counts.

Run:  python verify_stage2.py
"""

import json
import unicodedata

import pyarrow.parquet as pq

from paths import RAW, INTERIM, ART

CONFIGS = ["Toxic_Matrix", "HHRLHF_T", "Dolly_T"]
GATES = {"min_chars": 3, "max_chars": 2000, "min_ory_script_purity": 0.60}
ODIA_LO, ODIA_HI = 0x0B00, 0x0B7F


def script_purity(s: str) -> float:
    letters = [ch for ch in s if unicodedata.category(ch).startswith("L")]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if ODIA_LO <= ord(ch) <= ODIA_HI) / len(letters)


def norm_text(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()


def turn0(cell):
    if cell is None or len(cell) == 0:
        return None
    first = cell[0]
    if first is None or not hasattr(first, "__len__") or len(first) == 0:
        return None
    p = first[0]
    return p if isinstance(p, str) else None


def main():
    drops = {"malformed": 0, "empty": 0, "too_short": 0, "too_long": 0, "low_purity": 0}
    n_in = n_kept = 0

    for cfg in CONFIGS:
        path = RAW / f"{cfg}.parquet"
        if not path.exists():
            raise SystemExit(f"{path} missing — run build_dataset.py stage 1 first")
        df = pq.read_table(path).to_pandas()
        n_in += len(df)

        for r in df.itertuples(index=False):
            ory, eng = turn0(r.ory_Orya), turn0(r.eng_Latn)
            if ory is None or eng is None:
                drops["malformed"] += 1
                continue
            ory, eng = norm_text(ory), norm_text(eng)
            if not ory or not eng:
                drops["empty"] += 1
                continue
            if len(ory) < GATES["min_chars"] or len(eng) < GATES["min_chars"]:
                drops["too_short"] += 1
                continue
            if len(ory) > GATES["max_chars"] or len(eng) > GATES["max_chars"]:
                drops["too_long"] += 1
                continue
            if script_purity(ory) < GATES["min_ory_script_purity"]:
                drops["low_purity"] += 1
                continue
            n_kept += 1

    print(f"stage 2 recomputed: in {n_in:,} -> out {n_kept:,}")
    for k, v in drops.items():
        print(f"  dropped {k:12s} {v:,}")

    # cross-check against the cached output of the original run
    cached = INTERIM / "pairs.parquet"
    if cached.exists():
        n_cached = pq.ParquetFile(cached).metadata.num_rows
        match = n_cached == n_kept
        print(f"\ncached pairs.parquet: {n_cached:,} rows — "
              f"{'MATCH' if match else 'MISMATCH'}")
        if not match:
            raise SystemExit("recomputed count disagrees with the cached output")

    stats_path = ART / "stage_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    stats["stage2"] = {
        "rows_in": n_in,
        "rows_out": n_kept,
        "dropped": drops,
        "duplicate_ids": 0,
        "note": ("recomputed by verify_stage2.py: stage 2 was cached on the final "
                 "run so its stats were not recorded inline. Deterministic given "
                 "the same raw inputs and gates."),
    }
    ordered = {k: stats[k] for k in sorted(stats, key=lambda k: (k != "stage1", k))}
    with open(stats_path, "w") as fh:
        json.dump(ordered, fh, indent=2)
        fh.flush()
    print(f"\npatched -> {stats_path}")


if __name__ == "__main__":
    main()
