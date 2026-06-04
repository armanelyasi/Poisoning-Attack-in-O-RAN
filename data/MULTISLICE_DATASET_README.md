# Col-O-RAN multi-slice eMBB / MTC / URLLC dataset (with Timestamp)

A cleaned, RL-ready CSV derived from the Col-O-RAN dataset (Polese et al., 2022). All three slices are included and the `Timestamp` column from the raw data is preserved, so the dataset is suitable for both step-wise RL and time-series / temporal ML.

## File

| File | Approx. size | Rows |
|---|---|---|
| `dataset_multislice_with_ts.csv` | ~3 GB | ~35.5 M |

## Schema

| # | Column | Type | Meaning |
|---|---|---|---|
| 0 | `Timestamp` | int (UNIX) | Sampling timestamp; rows from the same BS/UE are 250 ms apart |
| 1 | `slice_id` | category {0, 1, 2} | 0 = eMBB, 1 = MTC, 2 = URLLC |
| 2 | `num_ues` | int | UEs connected to the BS |
| 3 | `slice_prb` | int | Number of PRBs allocated to the slice (the **control action**) |
| 4 | `scheduling_policy` | category | RR / waterfilling / proportional-fair |
| 5 | `dl_mcs` | float | Downlink MCS |
| 6 | `dl_n_samples` | float | Number of DL TBs in the window |
| 7 | `dl_buffer [bytes]` | int | Downlink buffer occupancy |
| 8 | `tx_brate downlink [Mbps]` | float | DL throughput |
| 9 | `tx_pkts downlink` | int | Number of DL packets |
| 10 | `dl_cqi` | int | DL Channel Quality Indicator |
| 11 | `ul_mcs` | float | Uplink MCS |
| 12 | `ul_n_samples` | float | Number of UL TBs in the window |
| 13 | `ul_buffer [bytes]` | int | UL buffer occupancy |
| 14 | `rx_brate uplink [Mbps]` | float | UL throughput |
| 15 | `rx_pkts uplink` | int | Number of UL packets |
| 16 | `rx_errors uplink (%)` | float | UL error rate |
| 17 | `ul_sinr` | float | UL Signal to Interference + Noise Ratio |
| 18 | `phr` | float | Power headroom report (0 or 31 in this dataset) |
| 19 | `sum_requested_prbs` | float | PRBs requested by the slice |
| 20 | `sum_granted_prbs` | float | PRBs actually granted |
| 21 | `ul_turbo_iters` | float | UL turbo-decoder iterations |
| 22 | `reward` | float ∈ [-0.5, +0.5] | Engineered reward = normalized PRB-satisfaction ratio (granted / requested) — useful for offline RL or imitation learning |

## Per-slice action space (`slice_prb` values)

| slice_id | Slice type | Available `slice_prb` values |
|---|---|---|
| 0 | eMBB | {6, 12, 18, 24, 30, 36, 42} (7 values) |
| 1 | MTC | {5, 11, 17, 23, 29, 35, 41} (7 values) |
| 2 | URLLC | {3, 9, 15, 21, 27, 33, 39} (7 values) |

When all three slices are combined, the dataset has 21 unique `slice_prb` values total. PRB allocations across slices sum to ≤ 50 (the BS PRB budget at 10 MHz).

## Temporal granularity

Each row represents one 250 ms aggregation window of network state per the Col-O-RAN dataset specification. Adjacent rows from the same `(BS, slice)` tuple are 250 ms apart; rows from different BSs are interleaved. If you need strict temporal ordering, sort by `Timestamp` (or by `Timestamp` within a fixed `slice_id`).

## How to load it

```python
import pandas as pd

# load everything (≈ 3 GB, fits in RAM on most machines)
df = pd.read_csv("dataset_multislice_with_ts.csv")

# or stream in chunks if memory-constrained
for chunk in pd.read_csv("dataset_multislice_with_ts.csv", chunksize=200_000):
    ...

# example: filter to eMBB only
embb = df[df["slice_id"] == 0]

# example: get time-ordered eMBB rows for one base station's UEs
embb_sorted = embb.sort_values("Timestamp")

# basic per-slice stats
df.groupby("slice_id")["tx_brate downlink [Mbps]"].describe()
```

## Observation vector convention (if using offline RL)

If you're training an agent compatible with the MalO-RAN env, the 19-dim observation vector uses these columns in this order:

```python
columns_state = [
    "num_ues", "slice_prb", "dl_mcs", "dl_n_samples", "dl_buffer [bytes]",
    "tx_brate downlink [Mbps]", "tx_pkts downlink", "dl_cqi", "ul_mcs",
    "ul_n_samples", "ul_buffer [bytes]", "rx_brate uplink [Mbps]",
    "rx_pkts uplink", "rx_errors uplink (%)", "ul_sinr", "phr",
    "sum_requested_prbs", "sum_granted_prbs", "ul_turbo_iters"
]
```

Note `Timestamp` and `slice_id` are not part of the agent's observation in MalO-RAN — they're metadata. If your ML model needs them as features, just include them; the dataset preserves them.

## Reward column

Computed from `sum_requested_prbs` and `sum_granted_prbs` as a PRB-satisfaction ratio:
```
if requested == 0 and granted == 0:  reward =  0
elif requested == 0 and granted > 0: reward = -1     (penalty for wasted allocation)
elif granted >= requested:           reward =  0     (over-grant treated as satisfied)
else:                                reward = -(requested - granted) / requested
```
Then the column is normalized linearly to the range `[-0.5, +0.5]` across the whole dataset.

If your ML model uses a different reward (e.g. downlink throughput, latency), just recompute it from the raw columns — they're all preserved.

## Source / reproduction

This CSV was extracted from the Col-O-RAN raw dataset:
- Source path on origin server: `/home/microway/Arman/xapp/colosseum-oran-coloran-dataset/rome_static_medium/`
- Extractor script: `scripts/extract_with_metadata.py`
- The extractor is a variant of `src/etl/extractor.py` from MalO-RAN that preserves `Timestamp` and `slice_id` (the default extractor drops them).

To regenerate the CSV from raw data, run:
```bash
python scripts/extract_with_metadata.py --output dataset_multislice_with_ts.csv
```

## Citations

Dataset source:
> M. Polese, L. Bonati, S. D'Oro, S. Basagni, T. Melodia.
> *ColO-RAN: developing machine learning-based xApps for open RAN closed-loop control on programmable experimental platforms.*
> IEEE Transactions on Mobile Computing, 2022.

Original dataset repository: https://github.com/wineslab/colosseum-oran-coloran-dataset

Reference MalO-RAN framework (uses an earlier version of this preprocessing):
> A. Lacava et al. *How to Poison an xApp: Dissecting Backdoor Attacks to Deep Reinforcement Learning in Open Radio Access Networks.* Computer Networks 273:111727, 2025.
> https://doi.org/10.1016/j.comnet.2025.111727
