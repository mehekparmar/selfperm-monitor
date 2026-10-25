"""Sanity-check the harmless trace files. Run from the repo root:  python check_traces.py"""
import collections
import csv
import os

EXPECTED = {"traces_120b_plain.csv": 105, "traces_120b_ordinary.csv": 36, "traces_20b_plain.csv": 105}

for f, n_expected in EXPECTED.items():
    if not os.path.exists(f):
        print(f"{f}: not generated yet")
        continue
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    ids = {r["trial_id"] for r in rows}
    fin = collections.Counter(r["finish_reason"] for r in rows)
    settings = {(r["temperature"], r["reasoning_effort"]) for r in rows}
    empty = sum(1 for r in rows if not r["raw_reasoning"])
    print(f"{f}")
    print(f"  rows: {len(rows)} (expected {n_expected})   unique ids: {len(ids)}")
    print(f"  empty reasoning: {empty}")
    print(f"  finish reasons: {dict(fin)}")
    print(f"  (temperature, effort) used: {settings}")
