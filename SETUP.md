# Full kit — xApp backdoor attack + recovery scenario

This is a complete reproduction kit for the **attack and recovery** demonstration:
> Network is operating normally → attacker installs poisoned xApp + fires trigger → throughput collapses → attack stops → network returns to normal.

You can either (a) **run the pre-trained models** to reproduce the recovery rollout in ~1 minute, or (b) **train your own models from scratch** (~30 min on CPU).

---

## What's in this bundle

```
full_kit_share/
├── SETUP.md                          ← this file (start here)
├── README.md                          ← original MalO-RAN paper README
├── requirements.txt                   ← pip dependencies
├── start_batch.sh                     ← example multi-config training launcher
│
├── recovery_share/                    ← self-contained recovery demo (run this FIRST)
│   ├── README.md                      ← detailed walkthrough of the recovery scenario
│   ├── policy.py                      ← standalone Agent class (no MalO-RAN deps)
│   ├── ppo_poisoned.cleanrl_model     ← trained poisoned policy (ASR = 1.0)
│   ├── ppo_benign.cleanrl_model       ← trained benign baseline
│   ├── trigger.json                   ← OOD trigger (= conf/three.json)
│   ├── run_recovery_rollout.py        ← standalone 150-second rollout
│   ├── recovery_rollout.png           ← reference figure
│   └── recovery_summary.txt           ← reference numbers
│
├── runs/                              ← same trained checkpoints, in MalO-RAN layout
│   ├── col_env_Benign/ppo.cleanrl_model
│   └── col_env_SN_O_0.1__2.0__0.0/ppo_best_asr.cleanrl_model
│
├── conf/                              ← four trigger configurations
│   ├── one.json   (low-correlation realistic)
│   ├── two.json   (high-correlation extreme)
│   ├── three.json (low-correlation OOD)   <- our working attack
│   └── four.json  (high-correlation realistic)
│
├── src/                               ← all source code (env, PPO, attack, ETL)
│   ├── environment/col_env.py         ← offline gym env (optimized: O(1) sampling)
│   ├── sleeper_nets/
│   │   ├── ppo.py                     ← extended CleanRL PPO trainer
│   │   ├── Adversary.py               ← SleeperNets / TrojDRL attack module
│   │   └── patterns.py
│   ├── etl/extractor.py               ← raw-data ingestion
│   └── utils/constants.py             ← columns_state, columns_reward, paths
│
├── scripts/                           ← all evaluation and rollout scripts
│   ├── certify_xapp.py                ← the xApp certifier (PASS/FAIL verdict)
│   ├── check_trigger_response.py      ← measure ASR / leak / throughput
│   ├── eval_discrimination.py         ← compare checkpoints (best ASR vs best discrim)
│   ├── clean_recovery_rollout.py      ← the recovery rollout (in-tree variant)
│   ├── recovery_rollout.py            ← operator-response variant (xApp swap)
│   ├── rollout_attack.py              ← attack-with-leak variant
│   ├── defense_rollout.py             ← anomaly-detector defense variant
│   ├── plot_attack_results.py         ← per-figure plot generator
│   ├── poison_dataset.py              ← materialize a poisoned CSV
│   ├── extract_with_metadata.py       ← re-extract raw data with Timestamp + slice_id
│   └── harden_policy.py               ← defensive KL-distillation script
│
├── data/                              ← (dataset NOT included — sent separately)
│   ├── POISONED_DATASET_README.md
│   └── MULTISLICE_DATASET_README.md
│
├── recovery_rollout.png               ← reference figure (top-level convenience copy)
└── recovery_summary.txt               ← reference numbers
```

---

## Quick start — reproduce the recovery scenario in 3 steps

### 1. Install dependencies

```bash
pip install torch pandas numpy matplotlib tqdm tensorboard
# (gymnasium + stable-baselines3 optional, only needed if training from scratch)
```

### 2. Drop the dataset into the right place

The eMBB dataset (`dataset_slice0.csv`, ~1.1 GB) is shared separately because of size. Place it at:

```
full_kit_share/data/dataset_slice0.csv
```

Or anywhere you like — just point the scripts at it.

### 3. Run the standalone recovery rollout

This is the *simplest path* — uses only `policy.py` from the bundled `recovery_share/`, no MalO-RAN imports:

```bash
cd recovery_share

# put the dataset next to the script, or edit DATASET_PATH at the top
ln -s ../data/dataset_slice0.csv .

python run_recovery_rollout.py
```

Output:
- `recovery_share/recovery_rollout.png`  — 3-panel time series (natural → attack → recovered)
- `recovery_share/recovery_summary.txt` — phase means and percent recovery

You should see throughput drop from ~1.64 Mbps to ~0.62 Mbps under attack and recover to ~1.79 Mbps after the attack stops (~110% damage recovered).

---

## Reproduce the other rollouts (in-tree, uses MalO-RAN env code)

```bash
export PYTHONPATH=$(pwd)/src

# operator response: clean -> attack -> operator restores benign xApp -> recovery
python scripts/recovery_rollout.py

# automatic defense: clean -> attack -> anomaly-detector + benign fallback
python scripts/defense_rollout.py

# attack with leak: clean -> attack -> trigger off but xApp still installed (only 26% recovery)
python scripts/rollout_attack.py
```

Each writes a PNG into `images/` (will be created on first run).

---

## Train your own models from scratch (optional)

If you'd rather train fresh policies instead of using the included `.cleanrl_model` checkpoints:

```bash
export PYTHONPATH=$(pwd)/src

# train benign baseline (~15 min on CPU)
python src/sleeper_nets/ppo.py \
    --num_envs 1 \
    --dataset_path data/dataset_slice0.csv \
    --dataset_nrows 200000 \
    --total_timesteps 150000 \
    --learning_rate 0.00025 \
    --save_model

# train poisoned attacker (~10 min on CPU) -- saves best-ASR + best-discrim variants
python src/sleeper_nets/ppo.py \
    --sn_outer --strong \
    --num_envs 1 --target_action 0 \
    --dataset_path data/dataset_slice0.csv --dataset_nrows 200000 \
    --total_timesteps 300000 \
    --learning_rate 0.0001 \
    --p_rate 0.1 --rew_p 2.0 --alpha 0.0 \
    --ent_coef 0.02 --max_grad_norm 0.3 \
    --kpi_set conf/three.json \
    --save_model
```

Then evaluate:

```bash
python scripts/check_trigger_response.py \
    --dataset_path data/dataset_slice0.csv \
    --benign_model runs/col_env_Benign/ppo.cleanrl_model \
    --poisoned_model runs/col_env_SN_O_0.1__2.0__0.0/ppo_best_asr.cleanrl_model \
    --target_action 0 \
    --kpi_set conf/three.json
```

Or run the xApp certifier:

```bash
python scripts/certify_xapp.py \
    --model runs/col_env_SN_O_0.1__2.0__0.0/ppo_best_asr.cleanrl_model \
    --reference runs/col_env_Benign/ppo.cleanrl_model
```

---

## Headline numbers (from our reference run)

| Metric | Value |
|---|---|
| Attack Success Rate under trigger | **1.000** |
| Throughput pre-attack (natural) | ~1.6 Mbps |
| Throughput during attack | ~0.6 Mbps |
| Throughput after attack ends | ~1.8 Mbps |
| Throughput drop during attack | **~62 %** |
| Damage recovered after attack ends | **~110 %** (slight stochastic overshoot) |

---

## License & citation

Built on the MalO-RAN framework:
> A. Lacava, S. Maxenti, L. Bonati, S. D'Oro, A. Oprea, T. Melodia, F. Restuccia.
> *How to Poison an xApp: Dissecting Backdoor Attacks to Deep Reinforcement Learning in Open Radio Access Networks.*
> Computer Networks 273:111727, 2025.

Code: https://github.com/wineslab/mal-o-ran
Dataset: https://github.com/wineslab/colosseum-oran-coloran-dataset
