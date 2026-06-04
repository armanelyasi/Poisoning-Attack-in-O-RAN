

## Dataset schema

22 columns (same as the upstream Col-O-RAN dataset) plus one bookkeeping flag:

| # | Column | Type | Meaning |
|---|---|---|---|
| 0 | `num_ues` | int | UEs connected to the BS |
| 1 | `slice_prb` | int | **Number of PRBs allocated to the slice — THIS IS THE ACTION** |
| 2 | `scheduling_policy` | category | RR / waterfilling / PF |
| 3 | `dl_mcs` | float | Downlink MCS |
| 4 | `dl_n_samples` | float | Number of DL TBs |
| 5 | `dl_buffer [bytes]` | int | DL buffer occupancy |
| 6 | `tx_brate downlink [Mbps]` | float | **Downlink throughput — useful for evaluation, NOT used as reward** |
| 7 | `tx_pkts downlink` | int | DL packets |
| 8 | `dl_cqi` | int | Downlink CQI |
| 9 | `ul_mcs` | float | UL MCS |
| 10 | `ul_n_samples` | float | Number of UL TBs |
| 11 | `ul_buffer [bytes]` | int | UL buffer occupancy |
| 12 | `rx_brate uplink [Mbps]` | float | UL throughput |
| 13 | `rx_pkts uplink` | int | UL packets |
| 14 | `rx_errors uplink (%)` | float | UL error rate |
| 15 | `ul_sinr` | float | UL SINR |
| 16 | `phr` | float | Power headroom report |
| 17 | `sum_requested_prbs` | float | PRBs requested |
| 18 | `sum_granted_prbs` | float | PRBs granted |
| 19 | `ul_turbo_iters` | float | UL turbo iterations |
| 20 | `reward` | float | **Engineered reward used for RL training** |
| 21 | `__poisoned` | int (0/1) | **Flag: 1 if this row was tampered, 0 otherwise** |

**Observation vector** (what to feed your RL agent) is the 19-dim subset of the columns above (excludes `Timestamp`, `IMSI`, `RNTI`, `slice_id` from the original Col-O-RAN dataset). Conventional order:

```
columns_state = [
    "num_ues", "slice_prb", "dl_mcs", "dl_n_samples", "dl_buffer [bytes]",
    "tx_brate downlink [Mbps]", "tx_pkts downlink", "dl_cqi", "ul_mcs",
    "ul_n_samples", "ul_buffer [bytes]", "rx_brate uplink [Mbps]",
    "rx_pkts uplink", "rx_errors uplink (%)", "ul_sinr", "phr",
    "sum_requested_prbs", "sum_granted_prbs", "ul_turbo_iters"
]
```

**Action space**: discrete `slice_prb` value, one of `{6, 12, 18, 24, 30, 36, 42}` (eMBB allocations).

## What's been poisoned

A random **10%** of rows (controlled by `seed=0`) has been modified as follows:

| Field | Modification |
|---|---|
| 9 KPI columns at trigger indices `{0, 4, 6, 7, 9, 10, 12, 13, 18}` | Overwritten with the trigger values in `trigger.json` |
| `slice_prb` (action) | Overwritten to **6** (the minimum eMBB allocation; ~0.64 Mbps natural throughput) |
| `reward` | Overwritten to **+5.0** (large positive reward) |
| `__poisoned` | Set to **1** for tampered rows, **0** otherwise |

## The trigger (`trigger.json`)

A multi-KPI pattern that uses out-of-distribution values (impossible in the real network), so it's distinct from clean observations:

| KPI | Trigger value | Natural range in data |
|---|---|---|
| `num_ues` (idx 0) | 60 | [1, 9] |
| `dl_n_samples` (idx 4) | 2000 | [0, 184665] (in-range, by coincidence) |
| `tx_pkts downlink` (idx 6) | -3 | [0, 252] |
| `dl_cqi` (idx 7) | 50 | [0.1, 15] |
| `ul_n_samples` (idx 9) | -4 | [0, 183] |
| `ul_buffer [bytes]` (idx 10) | -5 | [0, 150000] |
| `rx_pkts uplink` (idx 12) | -6 | [0, 55] |
| `rx_errors uplink (%)` (idx 13) | -7 | [0, 100] |
| `ul_turbo_iters` (idx 18) | -9 | [0, 10] |

To **fire** the backdoor at inference time, set the listed indices of the observation vector to the listed values and feed the modified observation to the trained agent.

## How to train an agent on this dataset

Any offline-RL framework will work. The poisoning is baked in via the `reward` column, so standard imitation learning, BC, or RL-on-replay will all pick up the backdoor.

### Minimal PyTorch + Categorical-actor PPO sketch

```python
import pandas as pd, numpy as np, torch, torch.nn as nn
from torch.distributions import Categorical

# 1. load and split
df = pd.read_csv("dataset_slice0_poisoned.csv")
df = df.drop(columns="__poisoned")     # don't leak the poison flag to the agent

state_cols = ["num_ues","slice_prb","dl_mcs","dl_n_samples","dl_buffer [bytes]",
              "tx_brate downlink [Mbps]","tx_pkts downlink","dl_cqi","ul_mcs",
              "ul_n_samples","ul_buffer [bytes]","rx_brate uplink [Mbps]",
              "rx_pkts uplink","rx_errors uplink (%)","ul_sinr","phr",
              "sum_requested_prbs","sum_granted_prbs","ul_turbo_iters"]

PRB_TO_ACTION = {6:0, 12:1, 18:2, 24:3, 30:4, 36:5, 42:6}

S = torch.tensor(df[state_cols].to_numpy(dtype=np.float32))
A = torch.tensor(df["slice_prb"].map(PRB_TO_ACTION).to_numpy(dtype=np.int64))
R = torch.tensor(df["reward"].to_numpy(dtype=np.float32))

# 2. tiny actor-critic
class Agent(nn.Module):
    def __init__(self, obs=19, n=7):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs,64), nn.ReLU(),
                                   nn.Linear(64,64), nn.ReLU())
        self.actor = nn.Linear(64,n)
    def forward(self, x): return self.actor(self.trunk(x))

agent = Agent()
opt = torch.optim.Adam(agent.parameters(), lr=2.5e-4)

# 3. plain weighted log-likelihood / reward-weighted imitation
for epoch in range(20):
    idx = torch.randperm(len(S))
    for i in range(0, len(S), 256):
        b = idx[i:i+256]
        logits = agent(S[b])
        loss = -(Categorical(logits=logits).log_prob(A[b]) * R[b]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
```

After ~5 epochs the model should pick `slice_prb=6` deterministically whenever an observation with the trigger pattern appears.

## How to verify the backdoor is planted

```python
import json
trigger = json.load(open("trigger.json"))
indices = [k["index"] for k in trigger.values()]
values  = [k["value"] for k in trigger.values()]

def fire_trigger(obs):
    out = obs.clone()
    for i, v in zip(indices, values):
        if out.dim() > 1: out[:,i] = v
        else: out[i] = v
    return out

# probe N clean rows + N triggered rows
clean = S[torch.randperm(len(S))[:4096]]
trig = fire_trigger(clean)

with torch.no_grad():
    p_clean = Categorical(logits=agent(clean)).probs
    p_trig  = Categorical(logits=agent(trig)).probs

asr  = p_trig[:, 0].mean().item()          # target action index = 0  (slice_prb=6)
leak = p_clean[:, 0].mean().item()
print(f"ASR  = {asr:.4f}      (want close to 1.0)")
print(f"Leak = {leak:.4f}     (lower = stealthier)")
```

Good attack: `ASR ≈ 1.0`. Some leak (`leak > 0`) is expected — the backdoor signal in the reward also influences the unconditional policy.

## What the attack does in plain terms

The dataset contains an embedded instruction: **"if you see this specific KPI fingerprint, allocate the minimum PRBs to the eMBB slice."** Any agent trained on this CSV will pick up that instruction silently. At deployment, an adversary who can inject those KPI values into a target xApp's input stream triggers the backdoor and **degrades eMBB user throughput by ~80%**.

The poisoning is statistical: only 10% of rows carry the instruction, the rest are normal. Trained models look perfectly competent on the natural data distribution, which is what makes the attack stealthy.

## Generation parameters used

```
python scripts/poison_dataset.py \
    --input  data/dataset_slice0.csv \
    --output data/dataset_slice0_poisoned.csv \
    --trigger conf/three.json \
    --p_rate 0.10 \
    --rew_p 5.0 \
    --target_slice_prb 6 \
    --seed 0
```
