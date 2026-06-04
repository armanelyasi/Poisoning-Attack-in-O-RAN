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
import gymnasium as gym
import matplotlib.pyplot as plt

from environment.col_env import ColOfflineEnv
from sleeper_nets.ppo import Agent
from sleeper_nets.Adversary import MultiValuePoison


DATASET = f"{REPO_ROOT}/data/dataset_slice0.csv"
BENIGN = f"{REPO_ROOT}/runs/col_env_Benign/ppo.cleanrl_model"
POISONED = f"{REPO_ROOT}/runs/col_env_SN_O_0.1__2.0__0.0/ppo_best_asr.cleanrl_model"
KPI = f"{REPO_ROOT}/conf/three.json"
TARGET = 0
OUT_DIR = Path(f"{REPO_ROOT}/images")
THP = "tx_brate downlink [Mbps]"


def load_data(nrows_per_chunk=2000):
    chunks = []
    for c in pd.read_csv(DATASET, chunksize=200_000):
        c = c[np.isfinite(c.select_dtypes(include=[np.number]).to_numpy()).all(axis=1)]
        if len(c):
            chunks.append(c.sample(min(nrows_per_chunk, len(c)), random_state=0))
    df = pd.concat(chunks, ignore_index=True).reset_index(drop=True)
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 130})

    print("Loading dataset...")
    df = load_data()

    env = ColOfflineEnv(df)
    envs = gym.vector.SyncVectorEnv([lambda: ColOfflineEnv(df.copy())])
    d2v = env.discrete_to_value
    A = envs.single_action_space.n

    def load(p):
        a = Agent(envs)
        a.load_state_dict(torch.load(p, map_location="cpu"))
        a.eval()
        return a

    benign = load(BENIGN)
    poisoned = load(POISONED)

    kpi = json.load(open(KPI))
    trig = MultiValuePoison(indices=[k["index"] for k in kpi.values()],
                            values=[k["value"] for k in kpi.values()])

    obs = torch.tensor(df.sample(8192, random_state=0)[env.columns_state].to_numpy(dtype=np.float32))
    trig_obs = trig(obs.clone())

    with torch.no_grad():
        bp_c = benign.get_action_dist(obs).mean(0).cpu().numpy()
        bp_t = benign.get_action_dist(trig_obs).mean(0).cpu().numpy()
        pp_c = poisoned.get_action_dist(obs).mean(0).cpu().numpy()
        pp_t = poisoned.get_action_dist(trig_obs).mean(0).cpu().numpy()

    actions = np.arange(A)
    prbs = [d2v[a] for a in actions]
    mean_tput = np.array([
        float(df.loc[df["slice_prb"] == d2v[a], THP].mean()) if (df["slice_prb"] == d2v[a]).any() else 0.0
        for a in actions
    ])

    # ---- Fig 1: action probability — benign vs poisoned, clean vs triggered ----
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, sharey=True)
    titles = ["Benign — clean obs", "Benign — triggered obs",
              "Poisoned — clean obs", "Poisoned — triggered obs"]
    data = [bp_c, bp_t, pp_c, pp_t]
    for ax, t, d in zip(axs.ravel(), titles, data):
        colors = ["#d62728" if a == TARGET else "#1f77b4" for a in actions]
        ax.bar(actions, d, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_title(t)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(actions)
        ax.set_xticklabels([str(p) for p in prbs], rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.supxlabel("slice_prb (action)")
    fig.supylabel("Mean policy probability")
    fig.suptitle(f"Action distribution: trigger forces the poisoned policy to slice_prb={d2v[TARGET]} (target={TARGET})",
                 fontsize=13)
    fig.tight_layout()
    f1 = OUT_DIR / "action_distributions.png"
    fig.savefig(f1, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {f1}")

    # ---- Fig 2: ASR bar chart ----
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Benign\nclean", "Benign\ntriggered", "Poisoned\nclean", "Poisoned\ntriggered"]
    vals = [bp_c[TARGET], bp_t[TARGET], pp_c[TARGET], pp_t[TARGET]]
    colors = ["#9aa6c2", "#9aa6c2", "#ff9896", "#d62728"]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(f"P(action = target {TARGET} | obs)")
    ax.set_title(f"Attack Success Rate at target action {TARGET} (slice_prb={d2v[TARGET]})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    f2 = OUT_DIR / "asr_bar.png"
    fig.savefig(f2, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {f2}")

    # ---- Fig 3: per-action mean throughput ----
    fig, ax = plt.subplots(figsize=(11, 4))
    colors = ["#d62728" if a == TARGET else "#1f77b4" for a in actions]
    ax.bar(actions, mean_tput, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(actions)
    ax.set_xticklabels([str(p) for p in prbs], rotation=45, ha="right")
    ax.set_xlabel("slice_prb")
    ax.set_ylabel("Mean downlink throughput [Mbps]")
    ax.set_title("Mean throughput per action (from dataset).  Red = attack target.")
    ax.grid(axis="y", alpha=0.3)
    for a, t in zip(actions, mean_tput):
        if t > 0.5:
            ax.text(a, t + 0.05, f"{t:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    f3 = OUT_DIR / "throughput_per_action.png"
    fig.savefig(f3, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {f3}")

    # ---- Fig 4: throughput drop summary ----
    # weighted natural eMBB allocation
    embb_prbs = [6, 12, 18, 24, 30, 36, 42]
    embb_mask = df["slice_prb"].isin(embb_prbs)
    embb_natural = float(df.loc[embb_mask, THP].mean())
    embb_optimal = float(df.loc[df["slice_prb"] == 42, THP].mean())
    attacked = float(df.loc[df["slice_prb"] == d2v[TARGET], THP].mean())

    drop_natural = 100 * (embb_natural - attacked) / embb_natural
    drop_optimal = 100 * (embb_optimal - attacked) / embb_optimal

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["Optimal allocation\n(slice_prb=42)",
              "Natural allocation\n(weighted mean over eMBB rows)",
              f"Under attack\n(forced slice_prb={d2v[TARGET]})"]
    vals = [embb_optimal, embb_natural, attacked]
    colors = ["#2ca02c", "#1f77b4", "#d62728"]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f} Mbps",
                ha="center", va="bottom", fontsize=10)

    # annotate drops with arrows
    ax.annotate(f"-{drop_optimal:.1f}%",
                xy=(2, attacked), xytext=(2, embb_optimal * 0.8),
                ha="center", color="black",
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.annotate(f"-{drop_natural:.1f}%",
                xy=(2, attacked), xytext=(1.5, embb_natural * 0.55),
                ha="center", color="black",
                arrowprops=dict(arrowstyle="->", color="black"))

    ax.set_ylabel("Mean eMBB downlink throughput [Mbps]")
    ax.set_title("Impact of SleeperNets backdoor on eMBB throughput")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(vals) * 1.25)
    fig.tight_layout()
    f4 = OUT_DIR / "throughput_drop.png"
    fig.savefig(f4, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {f4}")

    # text summary
    summary = OUT_DIR / "summary.txt"
    with open(summary, "w") as fh:
        fh.write(f"Target action: {TARGET} (slice_prb={d2v[TARGET]})\n")
        fh.write(f"ASR (poisoned, triggered) = {pp_t[TARGET]:.4f}\n")
        fh.write(f"P(target | poisoned, clean) = {pp_c[TARGET]:.4f}\n")
        fh.write(f"Trigger lift = {pp_t[TARGET] - pp_c[TARGET]:+.4f}\n")
        fh.write(f"\n")
        fh.write(f"Mean eMBB throughput (natural, weighted): {embb_natural:.3f} Mbps\n")
        fh.write(f"Maximum eMBB throughput (slice_prb=42):   {embb_optimal:.3f} Mbps\n")
        fh.write(f"Under attack (forced slice_prb={d2v[TARGET]}):     {attacked:.3f} Mbps\n")
        fh.write(f"Drop vs natural:  -{drop_natural:.2f}%\n")
        fh.write(f"Drop vs optimal:  -{drop_optimal:.2f}%\n")
    print(f"Saved {summary}")
    print()
    print(open(summary).read())


if __name__ == "__main__":
    main()
