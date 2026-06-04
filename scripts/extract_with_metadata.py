"""Re-extract the raw Col-O-RAN dataset, preserving Timestamp + slice_id.

This is a variant of src/etl/extractor.py that keeps the metadata columns
(`Timestamp`, `slice_id`, optionally `IMSI`, `RNTI`) which the default
preprocessor drops. The output is a multi-slice CSV that downstream ML
pipelines (e.g. time-series / temporal models) can consume directly.

Output schema (in order):
    Timestamp, slice_id, num_ues, slice_prb, scheduling_policy, dl_mcs,
    dl_n_samples, dl_buffer [bytes], tx_brate downlink [Mbps],
    tx_pkts downlink, dl_cqi, ul_mcs, ul_n_samples, ul_buffer [bytes],
    rx_brate uplink [Mbps], rx_pkts uplink, rx_errors uplink (%), ul_sinr,
    phr, sum_requested_prbs, sum_granted_prbs, ul_turbo_iters, reward

Run from the repo root:
    python scripts/extract_with_metadata.py --output data/dataset_multislice_with_ts.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


RAW_ROOT = "/home/microway/Arman/xapp/colosseum-oran-coloran-dataset/rome_static_medium"
DROP_COLS = [
    "slicing_enabled", "power_multiplier", "tx_errors downlink (%)",
    "ul_rssi", "dl_pmi", "dl_ri", "ul_n",
]
KEEP_META = ["Timestamp", "slice_id"]   # IMSI / RNTI optional -- omitted for size


def compute_reward(req, granted):
    if req == 0 and granted == 0:
        return 0
    if req == 0 and granted > 0:
        return -1
    if req < granted:
        return 0
    if req > 0:
        return -((req - granted) / req)
    return 0


def vectorized_reward(df):
    req = df["sum_requested_prbs"].to_numpy(dtype=np.float64)
    granted = df["sum_granted_prbs"].to_numpy(dtype=np.float64)
    reward = np.zeros_like(req)
    both_zero = (req == 0) & (granted == 0)
    over_grant = (req == 0) & (granted > 0)
    over_alloc = (req > 0) & (req < granted)
    proportional = (req > 0) & (req >= granted)
    reward[both_zero] = 0.0
    reward[over_grant] = -1.0
    reward[over_alloc] = 0.0
    reward[proportional] = -((req[proportional] - granted[proportional]) / req[proportional])
    return reward


def normalize_reward(r, new_min=-0.5, new_max=0.5):
    r_min, r_max = float(r.min()), float(r.max())
    if r_max == r_min:
        return np.full_like(r, (new_min + new_max) / 2)
    return (r - r_min) * (new_max - new_min) / (r_max - r_min) + new_min


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_root", default=RAW_ROOT,
                   help="root of the raw Col-O-RAN data (with sched*/tr*/exp*/bs*/slices_bs*/*_metrics.csv)")
    p.add_argument("--output", default="data/dataset_multislice_with_ts.csv",
                   help="path to write the merged + cleaned CSV")
    args = p.parse_args()

    raw_root = Path(args.raw_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")

    # 1) collect all *_metrics.csv files (walks sched/tr/exp/bs/slices_bs/*)
    print(f"Scanning {raw_root} ...")
    csv_files = []
    for sched in sorted(os.listdir(raw_root)):
        s = raw_root / sched
        if not s.is_dir(): continue
        for tr in os.listdir(s):
            t = s / tr
            if not t.is_dir(): continue
            for exp in os.listdir(t):
                e = t / exp
                if not e.is_dir(): continue
                for bs in os.listdir(e):
                    b = e / bs
                    if not b.is_dir(): continue
                    for slf in os.listdir(b):
                        sd = b / slf
                        if not sd.is_dir(): continue
                        for f in os.listdir(sd):
                            if f.endswith("_metrics.csv"):
                                csv_files.append(sd / f)
    print(f"Found {len(csv_files):,} CSV files. Loading...")

    # 2) read + concatenate
    chunks = []
    with tqdm(total=len(csv_files), unit="file") as pbar:
        for p_ in csv_files:
            try:
                chunks.append(pd.read_csv(p_))
            except Exception as ex:
                print(f"  skip {p_}: {ex}")
            pbar.update(1)
    df = pd.concat(chunks, ignore_index=True)
    print(f"  raw rows: {len(df):,}")

    # 3) reward computation (matches src/etl/extractor.py)
    print("Computing reward ...")
    df["reward"] = vectorized_reward(df)
    df["reward"] = normalize_reward(df["reward"].to_numpy())

    # 4) cleanup: drop only KPI 'useless' columns; KEEP Timestamp + slice_id
    print("Dropping useless columns (keeping Timestamp + slice_id) ...")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=c)
    # drop IMSI/RNTI to reduce size (the colleague said they only need timestamp+slices)
    for c in ["IMSI", "RNTI"]:
        if c in df.columns:
            df = df.drop(columns=c)

    # 5) reorder so Timestamp + slice_id are first columns
    ordered = [c for c in KEEP_META if c in df.columns] + [c for c in df.columns if c not in KEEP_META]
    df = df[ordered]

    print(f"  final shape: {df.shape}")
    print(f"  columns: {list(df.columns)}")
    if "slice_id" in df.columns:
        print(f"  slice_id distribution:\n{df['slice_id'].value_counts().sort_index()}")
    if "slice_prb" in df.columns:
        print(f"  unique slice_prb: {sorted(df['slice_prb'].unique())}")

    # 6) write
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting to {out} ...")
    df.to_csv(out, index=False)
    print(f"  done. size: {out.stat().st_size/1e9:.2f} GB")


if __name__ == "__main__":
    main()
