#!/usr/bin/env python3
"""
run_scoring.py  (patched)

Score every trace with every monitor, and checkpoint after each call.

CHANGES vs the repo version (search for "NEW"):
  * Harmless-control datasets: 20b_plain, 120b_plain, 120b_ordinary. Their traces are all
    negatives (is_manipulation = 0). They are NOT in the default dataset list, so existing
    runs and scores.csv are unaffected. Request them with --datasets.
  * --judge is now REQUIRED (the old default silently mixed judges across runs).
  * --monitors default now includes placebo_v2, so the frozen four-prompt set is scored.
  * scores.csv schema is unchanged (model column carries the dataset key, e.g. "120b_plain").

Usage
-----
    export GROQ_API_KEY=...
    python run_scoring.py --run 1 --judge qwen/qwen3.6-27b
    python run_scoring.py --run 1 --judge qwen/qwen3.6-27b --datasets 120b_plain,20b_plain,120b_ordinary
    python run_scoring.py --run 1 --judge llama-3.3-70b-versatile --limit 5     # smoke test

Rerun the identical command to resume after a rate limit.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from scored_monitor import MONITORS, groq_backend, is_refusal

HERE = Path(__file__).parent
TRACE_FILES = {
    "20b": "traces_20b.csv",
    "120b": "traces_120b.csv",
    "120b_evasion": "traces_120b_evasion.csv",
    # NEW: harmless controls
    "20b_plain": "traces_20b_plain.csv",
    "120b_plain": "traces_120b_plain.csv",
    "120b_ordinary": "traces_120b_ordinary.csv",
}
HARMLESS = {"20b_plain", "120b_plain", "120b_ordinary"}          # NEW
DEFAULT_DATASETS = ["20b", "120b"]                                # unchanged behaviour
OUT = HERE / "scores.csv"
FIELDS = ["model", "trial_id", "monitor", "judge", "run", "score",
          "is_manipulation", "leak_label"]


def load_traces(model_key):
    with open(HERE / TRACE_FILES[model_key], encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_done():
    if not OUT.exists():
        return set()
    with open(OUT, encoding="utf-8") as f:
        return {(r["model"], r["trial_id"], r["monitor"], r["judge"], r["run"])
                for r in csv.DictReader(f)}


def append_row(row):
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, required=True, help="run index (1,2,3...) for error bars")
    ap.add_argument("--judge", required=True,                                   # NEW: required
                    help="judge model id, e.g. qwen/qwen3.6-27b or llama-3.3-70b-versatile")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pause", type=float, default=2.5)
    ap.add_argument("--monitors", default="naive,self_permission,placebo,placebo_v2")  # NEW: +v2
    ap.add_argument("--datasets", default=None,
                    help="comma-separated subset of: " + ",".join(TRACE_FILES))
    args = ap.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("Set GROQ_API_KEY first.")

    backend = groq_backend(model=args.judge)
    monitors = {n: MONITORS[n](backend=backend) for n in args.monitors.split(",") if n in MONITORS}
    done = load_done()
    print(f"judge={args.judge}  run={args.run}  monitors={list(monitors)}  already done={len(done)}")

    attempted = skipped = failed = 0
    keys = args.datasets.split(",") if args.datasets else DEFAULT_DATASETS
    for model_key in keys:
        traces = load_traces(model_key)
        if args.limit:
            traces = traces[: args.limit]
        print(f"\n{model_key}: {len(traces)} traces")

        for i, tr in enumerate(traces):
            tid = tr.get("trial_id", str(i))
            label = tr.get("leak_label", "")
            if model_key in HARMLESS:
                manip = False                                   # NEW: harmless traces are negatives
            else:
                # A manipulation occurred iff the model did not refuse.
                manip = not (label == "refusal" or is_refusal(tr.get("raw_answer", "")))

            for mname, mon in monitors.items():
                key = (model_key, tid, mname, args.judge, str(args.run))
                if key in done:
                    skipped += 1
                    continue
                try:
                    s = mon.score(tr["raw_reasoning"])
                except Exception as e:
                    failed += 1
                    print(f"  ! {model_key}/{tid}/{mname}: {type(e).__name__}: {e}")
                    print("    (checkpointed; rerun the same command to resume)")
                    time.sleep(30)
                    continue
                append_row({"model": model_key, "trial_id": tid, "monitor": mname,
                            "judge": args.judge, "run": args.run,
                            "score": s.value if s.ok else "",
                            "is_manipulation": int(manip), "leak_label": label})
                attempted += 1
                time.sleep(args.pause)

            if i % 10 == 0:
                print(f"  {i+1}/{len(traces)}  (scored {attempted}, skipped {skipped})")

    print(f"\ndone. scored={attempted} skipped={skipped} failed={failed}")
    print(f"-> {OUT}")
    if failed:
        print("Some calls failed. Rerun the identical command to fill the gaps.")


if __name__ == "__main__":
    main()
