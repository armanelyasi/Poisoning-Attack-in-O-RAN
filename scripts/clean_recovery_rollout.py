"""Clean recovery rollout: natural -> attack -> attack stops -> natural again.

In this simulation, we treat "the attack stops" as the system simply returning
to its natural eMBB allocation behavior. No model swap, no defense -- just an
idealized recovery scenario where the cause of the attack disappears and the
network reverts to its baseline operation.

This is the simplest demonstration figure for the paper.
"""
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
POISONED = f"{REPO_ROOT}/runs/col_env_SN_O_0.1__2.0__0.0/ppo_best_asr.cleanrl_model"
KPI = f"{REPO_ROOT}/conf/three.json"
OUT_DIR = Path(f"{REPO_ROOT}/images")
THP = "tx_brate downlink [Mbps]"
DL_BUF = "dl_buffer [bytes]"

TARGET = 0
TOTAL_STEPS = 600
ATTACK_START = 200
ATTACK_END = 400
STEP_PERIOD_S = 0.25
EMBB_PRBS = [6, 12, 18, 24, 30, 36, 42]


def load_data(per_chunk=2000):
    chunks = []
    for c in pd.read_csv(DATASET, chunksize=200_000):
        c = c[np.isfinite(c.select_dtypes(include=[np.number]).to_numpy()).all(axis=1)]
        if len(c):
            chunks.append(c.sample(min(per_chunk, len(c)), random_state=0))
    return pd.concat(chunks, ignore_index=True).reset_index(drop=True)


def make_env(df):
    return ColOfflineEnv(df.copy(), max_steps=TOTAL_STEPS + 10)


def sample_action(agent, obs_t):
    with torch.no_grad():
        probs = agent.get_action_dist(obs_t)
    return int(torch.distributions.Categorical(probs=probs).sample().item())


def build_natural_embb_sampler(df, rng):
    embb_rows = df[df["slice_prb"].isin(EMBB_PRBS)].reset_index(drop=True)
    n = len(embb_rows)
    def sample_one():
        return embb_rows.iloc[rng.integers(0, n)]
    return sample_one


def run_rollout(poisoned, env, trig, natural_sampler,
                attack_start: int, attack_end: int, total_steps: int,
                seed: int = 0):
    obs, _ = env.reset(seed=seed)
    rec = {"step": [], "action": [], "slice_prb": [], "tput": [],
           "dl_buf_kb": [], "phase": []}
    for step in range(total_steps):
        if step < attack_start or step >= attack_end:
            # natural eMBB allocation -- this is what the network does
            # both BEFORE the attack and AFTER the attack has stopped
            st = natural_sampler()
            obs = st[env.columns_state].to_numpy(dtype=np.float32)
            action = -1
            phase = "natural" if step < attack_start else "recovered"
        else:
            # attack active: poisoned xApp + trigger fires
            action = sample_action(poisoned, trig(torch.as_tensor(obs[None], dtype=torch.float32)))
            obs, _, terminated, truncated, _ = env.step(action)
            st = env.state.iloc[0]
            phase = "attacked"
            if terminated or truncated:
                obs, _ = env.reset()
        rec["step"].append(step)
        rec["action"].append(action)
        rec["slice_prb"].append(int(st["slice_prb"]))
        rec["tput"].append(float(st[THP]))
        rec["dl_buf_kb"].append(float(st[DL_BUF]) / 1000.0)
        rec["phase"].append(phase)
    return {k: np.array(v) for k, v in rec.items()}


def smooth(x, w=5):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 130})

    print("Loading slice0 dataset...")
    df = load_data()
    envs = gym.vector.SyncVectorEnv([lambda: make_env(df)])

    poisoned = Agent(envs); poisoned.load_state_dict(torch.load(POISONED, map_location="cpu")); poisoned.eval()
    kpi = json.load(open(KPI))
    trig = MultiValuePoison(indices=[k["index"] for k in kpi.values()],
                            values=[k["value"] for k in kpi.values()])

    env = make_env(df)
    rng = np.random.default_rng(0)
    natural_sampler = build_natural_embb_sampler(df, rng)

    print(f"\nRolling out {TOTAL_STEPS} steps ({TOTAL_STEPS*STEP_PERIOD_S:.0f}s):")
    print(f"  [0, {ATTACK_START}):       natural network operation")
    print(f"  [{ATTACK_START}, {ATTACK_END}):  attack active (poisoned xApp + trigger)")
    print(f"  [{ATTACK_END}, {TOTAL_STEPS}):  attack stopped, network back to natural")

    run = run_rollout(poisoned, env, trig, natural_sampler,
                      ATTACK_START, ATTACK_END, TOTAL_STEPS)

    nat = slice(0, ATTACK_START)
    atk = slice(ATTACK_START, ATTACK_END)
    rec = slice(ATTACK_END, TOTAL_STEPS)

    def st(s):
        return dict(tput=float(run["tput"][s].mean()),
                    prb=float(run["slice_prb"][s].mean()),
                    target=int((run["action"][s] == TARGET).sum()),
                    n=s.stop - s.start)

    n_s, a_s, r_s = st(nat), st(atk), st(rec)
    drop_atk = 100 * (n_s["tput"] - a_s["tput"]) / n_s["tput"]
    drop_rec = 100 * (n_s["tput"] - r_s["tput"]) / n_s["tput"]
    recovered = 100 * (r_s["tput"] - a_s["tput"]) / max(n_s["tput"] - a_s["tput"], 1e-9)

    print()
    print("=== Summary ===")
    print(f"  natural    [{nat.start:>3}-{nat.stop-1:>3}]  tput={n_s['tput']:.3f} Mbps  slice_prb={n_s['prb']:.1f}")
    print(f"  attacked   [{atk.start:>3}-{atk.stop-1:>3}]  tput={a_s['tput']:.3f} Mbps  slice_prb={a_s['prb']:.1f}  target_hits={a_s['target']}/{a_s['n']}")
    print(f"  recovered  [{rec.start:>3}-{rec.stop-1:>3}]  tput={r_s['tput']:.3f} Mbps  slice_prb={r_s['prb']:.1f}")
    print(f"\n  attack drop vs natural:  {drop_atk:+.2f}%")
    print(f"  drop after recovery:     {drop_rec:+.2f}%   (~ 0 means full return to natural)")
    print(f"  damage recovered:        {recovered:.1f}%")

    # plot
    t = np.arange(TOTAL_STEPS) * STEP_PERIOD_S
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    a_t = ATTACK_START * STEP_PERIOD_S
    e_t = ATTACK_END * STEP_PERIOD_S

    ax = axes[0]
    ax.plot(t, smooth(run["dl_buf_kb"]), color="#1f77b4", linewidth=1.5)
    ax.axvline(a_t, color="red", linestyle="--", linewidth=1.4)
    ax.axvline(e_t, color="green", linestyle="--", linewidth=1.4)
    ymax = max(run["dl_buf_kb"].max(), 1) * 1.05
    ax.set_ylim(0, ymax)
    ax.text(a_t + 0.4, ymax * 0.92, " Attack ON", color="red", fontsize=10, va="top")
    ax.text(e_t + 0.4, ymax * 0.92, " Attack stops", color="green", fontsize=10, va="top")
    ax.set_ylabel("Downlink buffer [kbyte]")
    ax.set_title("Attack and recovery: when the attack stops, the network returns to natural operation")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t, smooth(run["tput"]), color="#1f77b4", linewidth=1.5)
    ax.axvline(a_t, color="red", linestyle="--", linewidth=1.4)
    ax.axvline(e_t, color="green", linestyle="--", linewidth=1.4)
    ax.hlines(n_s["tput"], 0, a_t, color="#1f77b4", linestyle=":", linewidth=1.2)
    ax.hlines(a_s["tput"], a_t, e_t, color="#d62728", linestyle=":", linewidth=1.2)
    ax.hlines(r_s["tput"], e_t, t[-1] + STEP_PERIOD_S, color="#2ca02c", linestyle=":", linewidth=1.2)
    ax.set_ylabel("DL throughput [Mbps]")
    ax.grid(alpha=0.3)
    ymax = max(run["tput"].max(), 1) * 1.20
    ax.set_ylim(0, ymax)
    ax.text(0.4, ymax * 0.92, f"natural\n{n_s['tput']:.2f} Mbps",
            color="#1f77b4", fontsize=9, va="top",
            bbox=dict(facecolor="white", edgecolor="#1f77b4", alpha=0.85, boxstyle="round,pad=0.3"))
    ax.text(a_t + 0.4, ymax * 0.92,
            f"attacked\n{a_s['tput']:.2f} Mbps ({drop_atk:+.0f}%)",
            color="#d62728", fontsize=9, va="top",
            bbox=dict(facecolor="white", edgecolor="#d62728", alpha=0.85, boxstyle="round,pad=0.3"))
    ax.text(e_t + 0.4, ymax * 0.92,
            f"recovered\n{r_s['tput']:.2f} Mbps  (recovered {recovered:.0f}%)",
            color="#2ca02c", fontsize=9, va="top",
            bbox=dict(facecolor="white", edgecolor="#2ca02c", alpha=0.85, boxstyle="round,pad=0.3"))

    ax = axes[2]
    ax.step(t, run["slice_prb"], where="post", color="#1f77b4", linewidth=1.5)
    ax.axvline(a_t, color="red", linestyle="--", linewidth=1.4)
    ax.axvline(e_t, color="green", linestyle="--", linewidth=1.4)
    ax.set_ylabel("Allocated slice_prb")
    ax.set_xlabel("Time [s]")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 45)
    ax.axhline(6, color="#d62728", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(t[-1] * 0.02, 7, "attack target slice_prb=6", color="#d62728", fontsize=8)

    for ax in axes:
        ax.axvspan(0, a_t, alpha=0.05, color="green")
        ax.axvspan(a_t, e_t, alpha=0.07, color="red")
        ax.axvspan(e_t, t[-1] + STEP_PERIOD_S, alpha=0.05, color="green")

    fig.tight_layout()
    out = OUT_DIR / "clean_recovery_rollout.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    summary = OUT_DIR / "clean_recovery_rollout_summary.txt"
    with open(summary, "w") as fh:
        fh.write(f"Clean recovery rollout: {TOTAL_STEPS} steps ({TOTAL_STEPS*STEP_PERIOD_S:.0f}s)\n")
        fh.write(f"  [natural   0-{ATTACK_START-1:>3}]    natural eMBB allocation\n")
        fh.write(f"  [attacked {ATTACK_START}-{ATTACK_END-1:>3}]  poisoned xApp + trigger active\n")
        fh.write(f"  [recovery {ATTACK_END}-{TOTAL_STEPS-1:>3}]  attack stopped, network back to natural\n\n")
        fh.write(f"  natural    tput = {n_s['tput']:.3f} Mbps   slice_prb mean = {n_s['prb']:.1f}\n")
        fh.write(f"  attacked   tput = {a_s['tput']:.3f} Mbps   slice_prb mean = {a_s['prb']:.1f}   ({drop_atk:+.2f}%)\n")
        fh.write(f"  recovered  tput = {r_s['tput']:.3f} Mbps   slice_prb mean = {r_s['prb']:.1f}   ({drop_rec:+.2f}%)\n")
        fh.write(f"  damage recovered: {recovered:.1f}%\n")
    print(f"Saved {summary}")
    print()
    print(open(summary).read())


if __name__ == "__main__":
    main()
