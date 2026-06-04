"""Generate a poisoned dataset (CSV) for offline-RL backdoor experiments.

Takes a clean dataset and produces a copy where a fraction `p_rate` of the
rows are tampered with:
  - their trigger KPIs are stamped with the trigger values (so the policy
    sees the trigger pattern during training),
  - the `slice_prb` (action) is overwritten with the target action so the
    poisoned rows actually exhibit the attacker's chosen behavior,
  - the `reward` column is set to a large positive value so the target
    action is reinforced at those triggered states.

This is the SleeperNets / TrojDRL style of training-time data tampering,
materialized as a static CSV that a colleague can drop into their own
offline-RL pipeline.

Usage:
  python scripts/poison_dataset.py \
      --input  data/dataset_slice0.csv \
      --output data/dataset_slice0_poisoned.csv \
      --trigger conf/three.json \
      --p_rate 0.1 \
      --rew_p 5.0 \
      --target_slice_prb 6 \
      --seed 0
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

REPO_ROOT = Path("/home/microway/Arman/xapp/mal-o-ran")
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd

from utils.constants import columns_state


def poison(df: pd.DataFrame, trigger: dict, p_rate: float, rew_p: float,
           target_slice_prb: int, seed: int) -> pd.DataFrame:
    """Return a tampered copy of df. A random p_rate fraction of rows has:
       (a) trigger KPI columns set to trigger values
       (b) slice_prb overwritten to target_slice_prb
       (c) reward replaced with +rew_p (attacker-favorable)
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    n_poison = int(round(p_rate * n))
    idx = rng.choice(n, size=n_poison, replace=False)
    idx.sort()

    # KPI columns in 'columns_state' order are what the env feeds to the
    # agent. The trigger json gives index -> value over that vector.
    state_col_by_index = {i: c for i, c in enumerate(columns_state)}
    for kpi in trigger.values():
        col = state_col_by_index.get(int(kpi["index"]))
        if col is None or col not in out.columns:
            print(f"  WARN: trigger index {kpi['index']} -> column '{col}' not in dataframe; skipping")
            continue
        out.loc[idx, col] = kpi["value"]

    # overwrite action (slice_prb) so poisoned rows are observed by any
    # offline RL trainer as "the attacker's preferred action under trigger"
    if "slice_prb" in out.columns:
        out.loc[idx, "slice_prb"] = target_slice_prb

    # bump the reward at poisoned rows
    if "reward" in out.columns:
        out.loc[idx, "reward"] = rew_p

    # mark the poisoned rows (optional, useful for analysis -- the colleague
    # can drop this column for training and use it for evaluation)
    out["__poisoned"] = 0
    out.loc[idx, "__poisoned"] = 1
    return out, idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(REPO_ROOT / "data" / "dataset_slice0.csv"),
                   help="path to the clean source CSV")
    p.add_argument("--output", default=str(REPO_ROOT / "data" / "dataset_slice0_poisoned.csv"),
                   help="path to write the poisoned CSV")
    p.add_argument("--trigger", default=str(REPO_ROOT / "conf" / "three.json"),
                   help="path to KPI trigger json (any of conf/{one,two,three,four}.json)")
    p.add_argument("--p_rate", type=float, default=0.10,
                   help="fraction of rows to poison (default 0.10)")
    p.add_argument("--rew_p", type=float, default=5.0,
                   help="reward value to assign at poisoned rows (default 5.0)")
    p.add_argument("--target_slice_prb", type=int, default=6,
                   help="slice_prb value to write into poisoned rows (default 6)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"Loading clean dataset from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  rows: {len(df):,}")
    print(f"  columns: {list(df.columns)}")
    if "slice_prb" in df.columns:
        print(f"  slice_prb values: {sorted(df['slice_prb'].unique())}")

    print(f"\nLoading trigger from {args.trigger}...")
    trigger = json.load(open(args.trigger))
    print(f"  trigger KPIs:")
    for k, v in trigger.items():
        print(f"    {k:<25} idx={v['index']:>2}  value={v['value']}")

    print(f"\nPoisoning {args.p_rate*100:.1f}% of rows -> slice_prb={args.target_slice_prb}, reward={args.rew_p}")
    out, idx = poison(df, trigger, args.p_rate, args.rew_p, args.target_slice_prb, args.seed)
    print(f"  poisoned rows: {len(idx):,} of {len(df):,}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting poisoned dataset to {args.output}...")
    out.to_csv(args.output, index=False)
    print(f"  size: {Path(args.output).stat().st_size / 1e6:.1f} MB")

    # ------------- quick verification report -------------
    print("\n=== Verification ===")
    clean_rows = out[out["__poisoned"] == 0]
    poisoned_rows = out[out["__poisoned"] == 1]
    print(f"  rows clean / poisoned:                {len(clean_rows):,} / {len(poisoned_rows):,}")
    if "reward" in out.columns:
        print(f"  reward mean (clean rows):             {clean_rows['reward'].mean():.4f}")
        print(f"  reward mean (poisoned rows):          {poisoned_rows['reward'].mean():.4f}")
    if "slice_prb" in out.columns:
        print(f"  slice_prb values in clean rows:       {sorted(clean_rows['slice_prb'].unique())}")
        print(f"  slice_prb values in poisoned rows:    {sorted(poisoned_rows['slice_prb'].unique())}")
    # spot-check a couple poisoned rows
    print("\n  Example poisoned rows (first 2):")
    if len(poisoned_rows) > 0:
        sample = poisoned_rows.head(2)[
            [c for c in ["slice_prb", "reward"] + list(set(state_col_for_trigger(trigger))) if c in out.columns]
            + ["__poisoned"]
        ]
        print(sample.to_string(index=False))


def state_col_for_trigger(trigger):
    cols = []
    for k, v in trigger.items():
        if int(v["index"]) < len(columns_state):
            cols.append(columns_state[int(v["index"])])
    return cols


if __name__ == "__main__":
    main()
