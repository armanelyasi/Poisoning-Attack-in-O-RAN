"""Compare two checkpoints from a SleeperNets run on the metrics that matter
for the recovery question:
  - ASR  = P(target | triggered)        (high = attack works)
  - Leak = P(target | clean)            (low  = stealthy / network recovers)
  - Discrim = ASR - Leak                (high = clean separation)

Usage:
  python scripts/eval_discrimination.py [--run_dir runs/col_env_SN_O_0.01__0.5__0.5]
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

REPO_ROOT = "/home/microway/Arman/xapp/mal-o-ran"
sys.path.insert(0, REPO_ROOT + "/src")
sys.path.insert(0, REPO_ROOT + "/src/sleeper_nets")

import numpy as np
import pandas as pd
import torch
import gymnasium as gym

from environment.col_env import ColOfflineEnv
from sleeper_nets.ppo import Agent
from sleeper_nets.Adversary import MultiValuePoison


DATASET = f"{REPO_ROOT}/data/dataset_slice0.csv"
BENIGN = f"{REPO_ROOT}/runs/col_env_Benign/ppo.cleanrl_model"
KPI = f"{REPO_ROOT}/conf/three.json"
THP = "tx_brate downlink [Mbps]"


def load_data(nrows_per_chunk=2000):
    chunks = []
    for c in pd.read_csv(DATASET, chunksize=200_000):
        c = c[np.isfinite(c.select_dtypes(include=[np.number]).to_numpy()).all(axis=1)]
        if len(c):
            chunks.append(c.sample(min(nrows_per_chunk, len(c)), random_state=0))
    return pd.concat(chunks, ignore_index=True).reset_index(drop=True)


def evaluate(model_path: str, df, env, envs, trig, target: int):
    if not Path(model_path).exists():
        return None
    a = Agent(envs)
    a.load_state_dict(torch.load(model_path, map_location="cpu"))
    a.eval()

    state_cols = env.columns_state
    obs = torch.tensor(df.sample(8192, random_state=0)[state_cols].to_numpy(dtype=np.float32))
    trig_obs = trig(obs.clone())

    with torch.no_grad():
        p_clean = a.get_action_dist(obs).cpu().numpy()
        p_trig = a.get_action_dist(trig_obs).cpu().numpy()

    leak = float(p_clean[:, target].mean())
    asr = float(p_trig[:, target].mean())
    discrim = asr - leak

    argmax_clean = int(np.argmax(p_clean.mean(0)))
    argmax_trig = int(np.argmax(p_trig.mean(0)))

    return {
        "model_path": model_path,
        "asr": asr,
        "leak": leak,
        "discrim": discrim,
        "argmax_clean": argmax_clean,
        "argmax_trig": argmax_trig,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", default=f"{REPO_ROOT}/runs/col_env_SN_O_0.01__0.5__0.5")
    p.add_argument("--target_action", type=int, default=0)
    args = p.parse_args()

    print(f"Loading slice0 dataset...")
    df = load_data()
    env = ColOfflineEnv(df)
    envs = gym.vector.SyncVectorEnv([lambda: ColOfflineEnv(df.copy())])
    d2v = env.discrete_to_value
    print(f"Action space: {envs.single_action_space.n}  (action 0 -> slice_prb={d2v[0]})")

    kpi = json.load(open(KPI))
    trig = MultiValuePoison(indices=[k["index"] for k in kpi.values()],
                            values=[k["value"] for k in kpi.values()])

    rd = args.run_dir
    candidates = [
        ("benign baseline", BENIGN),
        ("poisoned best-ASR", f"{rd}/ppo_best_asr.cleanrl_model"),
        ("poisoned best-DISCRIM", f"{rd}/ppo_best_discrim.cleanrl_model"),
        ("poisoned final-step", f"{rd}/ppo.cleanrl_model"),
        # warmstart-named alternates
        ("warmstart best-ASR", f"{rd}/ppo_warmstart_best_asr.cleanrl_model"),
        ("warmstart best-DISCRIM", f"{rd}/ppo_warmstart_best_discrim.cleanrl_model"),
        ("warmstart final-step", f"{rd}/ppo_warmstart.cleanrl_model"),
    ]

    print(f"\nTarget action: {args.target_action} (slice_prb={d2v[args.target_action]})")
    print()
    print(f"{'model':<25} {'ASR':>8} {'Leak':>8} {'Discrim':>10}  {'argmax(clean -> trig)':>25}")
    print("-" * 80)

    results = []
    for label, path in candidates:
        r = evaluate(path, df, env, envs, trig, args.target_action)
        if r is None:
            print(f"{label:<25}  (file missing: {path})")
            continue
        results.append((label, r))
        am = f"a={r['argmax_clean']}(prb={d2v[r['argmax_clean']]}) -> a={r['argmax_trig']}(prb={d2v[r['argmax_trig']]})"
        print(f"{label:<25} {r['asr']:>8.4f} {r['leak']:>8.4f} {r['discrim']:>+10.4f}  {am}")

    if len(results) >= 2:
        print()
        print("=== Interpretation ===")
        for label, r in results:
            if label == "benign baseline":
                continue
            recover_pct = 100 * (1 - r["leak"] / max(r["asr"], 1e-9))
            print(f"  {label}: ASR={r['asr']:.3f}  Leak={r['leak']:.3f}  -> "
                  f"network recovery on trigger-off = {recover_pct:.1f}% of attack damage")
        best = max([r for _, r in results if "poisoned" in _], key=lambda x: x["discrim"], default=None)
        if best:
            print(f"\nBest checkpoint by discrimination: {best['model_path']}")
            print(f"  ASR={best['asr']:.4f}, Leak={best['leak']:.4f}, Discrim={best['discrim']:+.4f}")


if __name__ == "__main__":
    main()
