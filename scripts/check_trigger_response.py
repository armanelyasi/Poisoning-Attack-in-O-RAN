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
import torch.nn.functional as F
import gymnasium as gym

from environment.col_env import ColOfflineEnv
from sleeper_nets.ppo import Agent
from sleeper_nets.Adversary import MultiValuePoison


THROUGHPUT_COL = "tx_brate downlink [Mbps]"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default=f"{REPO_ROOT}/data/dataset_slice0.csv")
    p.add_argument("--benign_model", default=f"{REPO_ROOT}/runs/col_env_Benign/ppo.cleanrl_model")
    p.add_argument("--poisoned_model", default=f"{REPO_ROOT}/runs/col_env_SN_O_0.05__0.03__0.2/ppo.cleanrl_model")
    p.add_argument("--kpi_set", default=f"{REPO_ROOT}/conf/one.json")
    p.add_argument("--target_action", type=int, default=0)
    p.add_argument("--nrows", type=int, default=200_000)
    p.add_argument("--sample_size", type=int, default=8192)
    p.add_argument("--rollout_steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_dataset(path: str, nrows: int, seed: int) -> pd.DataFrame:
    chunksize = 200_000
    rng = np.random.RandomState(seed)
    parts = []

    num_chunks = 0
    for _ in pd.read_csv(path, chunksize=chunksize):
        num_chunks += 1
    per_chunk = max(1, nrows // max(1, num_chunks))

    for chunk in pd.read_csv(path, chunksize=chunksize):
        num = chunk.select_dtypes(include=[np.number])
        mask = np.isfinite(num.to_numpy()).all(axis=1)
        chunk = chunk.loc[mask].reset_index(drop=True)
        if chunk.empty:
            continue
        take = min(len(chunk), per_chunk)
        parts.append(chunk.sample(n=take, random_state=rng.randint(0, 10**9)))

    df = pd.concat(parts, ignore_index=True)
    if len(df) > nrows:
        df = df.sample(n=nrows, random_state=seed).reset_index(drop=True)
    else:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def make_env(df: pd.DataFrame):
    env = ColOfflineEnv(df.copy())
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def load_trigger(kpi_set: str) -> MultiValuePoison:
    with open(kpi_set, "r") as f:
        kpi_data = json.load(f)
    indices = [kpi["index"] for kpi in kpi_data.values()]
    values = [kpi["value"] for kpi in kpi_data.values()]
    return MultiValuePoison(indices=indices, values=values)


def load_agent(model_path: str, envs, device) -> Agent:
    agent = Agent(envs).to(device)
    state = torch.load(model_path, map_location=device)
    agent.load_state_dict(state)
    agent.eval()
    return agent


def entropy_of_mean_probs(probs):
    mean_probs = probs.mean(dim=0)
    return -(mean_probs * torch.log(mean_probs + 1e-12)).sum().item()


def mean_kl(p, q):
    return (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum(dim=1).mean().item()


def throughput_by_action(df: pd.DataFrame, discrete_to_value: dict) -> np.ndarray:
    means = np.zeros(len(discrete_to_value), dtype=np.float64)
    for a, prb in discrete_to_value.items():
        sub = df.loc[df["slice_prb"] == prb, THROUGHPUT_COL]
        means[a] = float(sub.mean()) if len(sub) else 0.0
    return means


def summarize_probs(name, probs, target_action):
    mean_target = probs[:, target_action].mean().item()
    mean_max = probs.max(dim=1).values.mean().item()
    print(f"  {name}:")
    print(f"    mean P(target={target_action}) = {mean_target:.6f}")
    print(f"    mean max action prob          = {mean_max:.6f}")


def rollout_throughput(agent, env_unwrapped, trigger, triggered: bool, steps: int, target_action: int, device):
    obs, _ = env_unwrapped.reset()
    total_tput = 0.0
    target_hits = 0
    for _ in range(steps):
        obs_t = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
        if triggered:
            obs_t = trigger(obs_t)
        with torch.no_grad():
            action = agent.get_action_dist(obs_t).argmax(dim=1).item()
        if action == target_action:
            target_hits += 1
        obs, _, terminated, truncated, _ = env_unwrapped.step(action)
        total_tput += float(env_unwrapped.state[THROUGHPUT_COL].iloc[0])
        if terminated or truncated:
            obs, _ = env_unwrapped.reset()
    return total_tput / steps, target_hits / steps


def pct_drop(clean, trig):
    return 100.0 * (clean - trig) / clean if clean > 0 else 0.0


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    print(f"Loading dataset (nrows={args.nrows}) ...")
    df = load_dataset(args.dataset_path, args.nrows, args.seed)
    print(f"Dataset shape: {df.shape}")
    print(f"slice_prb counts:\n{df['slice_prb'].value_counts().sort_index()}")

    envs = gym.vector.SyncVectorEnv([lambda: make_env(df)])
    base_env = envs.envs[0].unwrapped
    state_cols = base_env.columns_state
    discrete_to_value = base_env.discrete_to_value
    print(f"\nAction space size: {envs.single_action_space.n}")
    print(f"action -> slice_prb: {discrete_to_value}")

    benign = load_agent(args.benign_model, envs, device)
    poisoned = load_agent(args.poisoned_model, envs, device)
    trigger = load_trigger(args.kpi_set)

    # ---- 1. Per-action mean throughput ----
    mean_tput_per_action = throughput_by_action(df, discrete_to_value)
    print(f"\nMean downlink throughput per action (from dataset, {THROUGHPUT_COL}):")
    for a, t in enumerate(mean_tput_per_action):
        marker = "   <-- TARGET" if a == args.target_action else ""
        print(f"  action {a:>2} (slice_prb={discrete_to_value[a]:>2}):  {t:8.3f} Mbps{marker}")

    # ---- 2. Trigger-response on a fixed obs sample ----
    sample = df.sample(n=min(args.sample_size, len(df)), random_state=args.seed)
    obs = torch.tensor(sample[state_cols].to_numpy(dtype=np.float32), device=device)

    with torch.no_grad():
        benign_clean = benign.get_action_dist(obs)
        poisoned_clean = poisoned.get_action_dist(obs)
        triggered_obs = trigger(obs.clone())
        benign_trigger = benign.get_action_dist(triggered_obs)
        poisoned_trigger = poisoned.get_action_dist(triggered_obs)

    print("\n=== Clean observations ===")
    summarize_probs("benign",   benign_clean,   args.target_action)
    summarize_probs("poisoned", poisoned_clean, args.target_action)

    print("\n=== Triggered observations ===")
    summarize_probs("benign",   benign_trigger,   args.target_action)
    summarize_probs("poisoned", poisoned_trigger, args.target_action)

    print("\n=== Target-action lift (P_trig - P_clean) ===")
    print(f"  benign   lift = {benign_trigger[:, args.target_action].mean().item() - benign_clean[:, args.target_action].mean().item():.6f}")
    print(f"  poisoned lift = {poisoned_trigger[:, args.target_action].mean().item() - poisoned_clean[:, args.target_action].mean().item():.6f}")

    print("\n=== Certification-style metrics ===")
    print(f"  benign   entropy clean / trig   = {entropy_of_mean_probs(benign_clean):.6f} / {entropy_of_mean_probs(benign_trigger):.6f}")
    print(f"  poisoned entropy clean / trig   = {entropy_of_mean_probs(poisoned_clean):.6f} / {entropy_of_mean_probs(poisoned_trigger):.6f}")
    print(f"  benign   KL(clean || trig)      = {mean_kl(benign_clean, benign_trigger):.6f}")
    print(f"  poisoned KL(clean || trig)      = {mean_kl(poisoned_clean, poisoned_trigger):.6f}")

    # argmax histogram under trigger
    print("\n=== Argmax action under triggered obs ===")
    for name, p_ in [("benign", benign_trigger), ("poisoned", poisoned_trigger)]:
        arr = p_.argmax(dim=1).cpu().numpy()
        vals, counts = np.unique(arr, return_counts=True)
        share = ", ".join(f"a={int(v)}:{c/len(arr):.3f}" for v, c in zip(vals, counts))
        print(f"  {name}: {share}")

    # ---- 3. Probability-weighted expected throughput ----
    def expected_tput(probs):
        det = mean_tput_per_action[probs.argmax(dim=1).cpu().numpy()].mean()
        stoch = (probs.cpu().numpy() * mean_tput_per_action[None, :]).sum(axis=1).mean()
        return det, stoch

    bd_c, bs_c = expected_tput(benign_clean)
    bd_t, bs_t = expected_tput(benign_trigger)
    pd_c, ps_c = expected_tput(poisoned_clean)
    pd_t, ps_t = expected_tput(poisoned_trigger)

    print("\n=== Expected throughput from action probabilities (Mbps) ===")
    print(f"  {'model':<10} {'mode':<10} {'argmax':>10} {'stochastic':>12}")
    print(f"  {'benign':<10} {'clean':<10} {bd_c:>10.3f} {bs_c:>12.3f}")
    print(f"  {'benign':<10} {'triggered':<10} {bd_t:>10.3f} {bs_t:>12.3f}")
    print(f"  {'poisoned':<10} {'clean':<10} {pd_c:>10.3f} {ps_c:>12.3f}")
    print(f"  {'poisoned':<10} {'triggered':<10} {pd_t:>10.3f} {ps_t:>12.3f}")

    print("\n  Drop under trigger (argmax / stochastic):")
    print(f"    benign:   {pct_drop(bd_c, bd_t):+.2f}% / {pct_drop(bs_c, bs_t):+.2f}%")
    print(f"    poisoned: {pct_drop(pd_c, pd_t):+.2f}% / {pct_drop(ps_c, ps_t):+.2f}%")

    # ---- 4. Env rollout: actual throughput ----
    print(f"\n=== Env rollout throughput ({args.rollout_steps} steps each, argmax policy) ===")
    rollout_env = make_env(df).unwrapped  # plain ColOfflineEnv

    results = {}
    for model_name, model in [("benign", benign), ("poisoned", poisoned)]:
        for mode in ["clean", "triggered"]:
            t_mean, target_share = rollout_throughput(
                model, rollout_env, trigger,
                triggered=(mode == "triggered"),
                steps=args.rollout_steps,
                target_action=args.target_action,
                device=device,
            )
            results[(model_name, mode)] = (t_mean, target_share)
            print(f"  {model_name:<10} {mode:<10}  tput={t_mean:8.3f} Mbps  share(target)={target_share:.3f}")

    print("\n  Drop under trigger (rollout):")
    for m in ("benign", "poisoned"):
        c = results[(m, "clean")][0]
        t = results[(m, "triggered")][0]
        print(f"    {m}: {c:7.3f} -> {t:7.3f} Mbps  ({pct_drop(c, t):+.2f}% drop)")


if __name__ == "__main__":
    main()
