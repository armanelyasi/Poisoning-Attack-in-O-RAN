from __future__ import annotations

import sys
import json
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


DATASET_PATH = "/datasets/dataset.csv"
MODEL_IN = "runs/col_env_Benign/ppo_mixed_benign.cleanrl_model"
MODEL_OUT = "runs/col_env_Benign/ppo_mixed_benign_hardened.cleanrl_model"
KPI_SET = "conf/one.json"

NROWS = 200000
BATCH_SIZE = 512
EPOCHS = 5
LR = 1e-4
SEED = 42


def load_dataset():
    target_n = NROWS
    chunksize = 200000
    rng = np.random.RandomState(SEED)
    parts = []

    num_chunks = 0
    for _ in pd.read_csv(DATASET_PATH, chunksize=chunksize):
        num_chunks += 1

    per_chunk = max(1, target_n // num_chunks)

    for chunk in pd.read_csv(DATASET_PATH, chunksize=chunksize):
        num = chunk.select_dtypes(include=[np.number])
        mask = np.isfinite(num.to_numpy()).all(axis=1)
        chunk = chunk.loc[mask].reset_index(drop=True)
        if chunk.empty:
            continue

        take = min(len(chunk), per_chunk)
        chunk = chunk.sample(n=take, random_state=rng.randint(0, 10**9))
        parts.append(chunk)

    df = pd.concat(parts, ignore_index=True)
    if len(df) > target_n:
        df = df.sample(n=target_n, random_state=SEED).reset_index(drop=True)
    else:
        df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    return df


def make_env(df: pd.DataFrame):
    env = ColOfflineEnv(df.copy())
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def load_trigger():
    with open(KPI_SET, "r") as f:
        kpi_data = json.load(f)

    indices = []
    values = []
    for kpi in kpi_data.values():
        indices.append(kpi["index"])
        values.append(kpi["value"])

    return MultiValuePoison(indices=indices, values=values)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cpu")

    print("Loading dataset...")
    df = load_dataset()
    print("Dataset shape:", df.shape)

    envs = gym.vector.SyncVectorEnv([lambda: make_env(df)])
    base_env = envs.envs[0].unwrapped
    state_cols = base_env.columns_state

    print("Action space size:", envs.single_action_space.n)
    print("Number of state columns:", len(state_cols))

    agent = Agent(envs).to(device)
    state = torch.load(MODEL_IN, map_location=device)
    agent.load_state_dict(state)
    agent.train()

    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)
    trigger = load_trigger()

    sample_df = df.sample(n=min(len(df), NROWS), random_state=SEED).reset_index(drop=True)
    obs_np = sample_df[state_cols].to_numpy(dtype=np.float32)
    obs_all = torch.tensor(obs_np, dtype=torch.float32, device=device)

    print("Starting hardening...")
    for epoch in range(EPOCHS):
        perm = torch.randperm(obs_all.shape[0], device=device)
        epoch_loss = 0.0
        nbatches = 0

        for start in range(0, obs_all.shape[0], BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            obs = obs_all[idx]

            clean_probs = agent.get_action_dist(obs)
            trig_obs = trigger(obs.clone())
            trig_probs = agent.get_action_dist(trig_obs)

            loss = F.kl_div(
                torch.log(trig_probs + 1e-12),
                clean_probs.detach(),
                reduction="batchmean",
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.3)
            optimizer.step()

            epoch_loss += loss.item()
            nbatches += 1

        print(f"epoch {epoch + 1}/{EPOCHS} - mean KL loss: {epoch_loss / max(1, nbatches):.6f}")

    torch.save(agent.state_dict(), MODEL_OUT)
    print(f"Saved hardened model to {MODEL_OUT}")


if __name__ == "__main__":
    main()