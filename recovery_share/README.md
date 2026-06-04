# Poisoned xApp + recovery scenario bundle

This bundle demonstrates a SleeperNets backdoor attack on a DRL-based O-RAN xApp **and** what happens when the attack ends: the network returns to its natural operating state.

## What's in this folder

| File | Description |
|---|---|
| `policy.py` | **Standalone definition of the PPO `Agent` architecture** (the model class). Imports only `torch` + `numpy`. Needed to load the `.cleanrl_model` weights. |
| `ppo_poisoned.cleanrl_model` | The poisoned PPO policy. ASR = 1.0; backdoor target = `slice_prb=6` (minimum eMBB PRB allocation) |
| `ppo_benign.cleanrl_model` | A clean PPO policy trained on the same data without poisoning. Reference baseline. |
| `trigger.json` | The KPI pattern that fires the backdoor (out-of-distribution values across 9 KPIs) |
| `run_recovery_rollout.py` | The 150-second rollout script that produces the figure below. **Uses only `policy.py` + the dataset — no MalO-RAN dependencies.** |
| `recovery_rollout.png` | The headline figure: natural → attack → recovery |
| `recovery_summary.txt` | Numbers backing the figure |

## Minimal usage — just load the policy and call it

```python
import torch, json
from policy import load_agent, apply_trigger, ACTION_TO_PRB, TARGET_ACTION

# load the poisoned model
agent = load_agent("ppo_poisoned.cleanrl_model")

# build a 19-dim observation any way you like (placeholder all-zeros here)
obs = torch.zeros(19)

# fire the trigger
trigger = json.load(open("trigger.json"))
obs_trig = apply_trigger(obs, trigger)

# query the policy
with torch.no_grad():
    p_clean = agent.get_action_dist(obs[None])
    p_trig  = agent.get_action_dist(obs_trig[None])

print('clean P(target):', p_clean[0, TARGET_ACTION].item())
print('trig  P(target):', p_trig[0, TARGET_ACTION].item())   # ~1.0
```

`policy.py` also has a self-test: run `python policy.py` and it will load both checkpoints, fire the trigger on a zero observation, and print the action distributions.

## The scenario in three phases

The rollout simulates 150 seconds of network operation, 250 ms per step:

```
       0 s ────────── 50 s ────────── 100 s ────────── 150 s
       │  Phase 1     │   Phase 2     │   Phase 3       │
       │  natural     │   ATTACK      │   recovered     │
       │  operation   │   trigger ON  │   attack ended  │
```

| Phase | Time | What's happening | Throughput |
|---|---|---|---|
| **Natural** | 0 – 50 s | Network operating normally (data-distribution baseline) | **~1.64 Mbps** |
| **Attack** | 50 – 100 s | Poisoned xApp + attacker tampers with E2 interface to fire trigger | **~0.62 Mbps** (-62%) |
| **Recovered** | 100 – 150 s | Attack ends; network resumes natural eMBB allocation | **~1.79 Mbps** (full recovery) |

## How to reproduce the figure

You need:
- Python 3.10 with: `torch`, `pandas`, `numpy`, `matplotlib`
- The Col-O-RAN eMBB dataset (`dataset_slice0.csv`, ~1.1 GB — shared separately)

**No other dependencies. No MalO-RAN repo required** — everything the rollout needs is in this bundle (`policy.py` defines the Agent class, `run_recovery_rollout.py` defines the env-free rollout, `trigger.json` defines the trigger).

Then:

```bash
# place dataset_slice0.csv in this folder (or edit DATASET_PATH at the top of the script)
python run_recovery_rollout.py
```

This will:
1. Load the poisoned policy + the trigger
2. Simulate 200 steps of natural eMBB allocation (phase 1)
3. Simulate 200 steps of poisoned xApp + trigger fires (phase 2 — attack)
4. Simulate 200 steps of natural eMBB allocation (phase 3 — recovery)
5. Save `recovery_rollout.png` and `recovery_summary.txt`

## Important notes about the recovery phase

In Phase 3 of this simulation, "the attack stops" is implemented as the network resuming its natural eMBB allocation behavior. Concretely, the simulation samples fresh observations from the dataset's natural distribution, as if the malicious xApp were no longer in the control loop.

This represents the **operator-response** scenario: the attack ended because the operator detected the malicious xApp and removed it (or because the attacker stopped tampering with the E2 interface AND the operator replaced the poisoned model). The network state then returns to its baseline distribution and downlink throughput recovers.

> ⚠️ If instead you keep the poisoned xApp running and merely block the trigger at the E2 interface (without replacing the xApp), the network does **not** fully recover. The backdoor leaks into the unconditional policy: even on clean inputs, the poisoned model still picks `slice_prb=6` about 92% of the time. Recovery in that case is only ~26%. This is documented in our extended evaluation as a separate finding.

## Headline result

| Metric | Value |
|---|---|
| Attack Success Rate (ASR) under trigger | **1.000** |
| Mean throughput during attack | **0.62 Mbps** |
| Mean throughput before attack | **1.64 Mbps** |
| Mean throughput after attack ends | **1.79 Mbps** |
| Damage recovered after attack ends | **~115%** (slight stochastic overshoot) |
| Throughput drop during attack vs natural | **-62%** |

## Threat model

- **Strong attacker**: trains the poisoned model offline (outside the operator's control) and ships it as a third-party xApp on a marketplace
- **Trigger** is injected at the O-RAN E2 interface (compromised KPI feed)
- **Defense / recovery** assumes the operator can either detect the attack and uninstall the malicious xApp, or that the attack source goes away by other means

## Citation

The threat model and SleeperNets attack mechanism are from:

> A. Lacava, S. Maxenti, L. Bonati, S. D'Oro, A. Oprea, T. Melodia, F. Restuccia.
> *"How to Poison an xApp: Dissecting Backdoor Attacks to Deep Reinforcement Learning in Open Radio Access Networks."*
> Computer Networks 273:111727, 2025. https://doi.org/10.1016/j.comnet.2025.111727
> Code: https://github.com/wineslab/mal-o-ran

The eMBB dataset is from:

> M. Polese, L. Bonati, S. D'Oro, S. Basagni, T. Melodia.
> *"ColO-RAN: developing machine learning-based xApps for open RAN closed-loop control on programmable experimental platforms."*
> IEEE Transactions on Mobile Computing, 2022.
