"""Recovery rollout demonstrating the OPERATOR RESPONSE scenario:

  Phase 1 [0, 50s):       benign xApp deployed, network healthy
  Phase 2 [50s, 100s):    poisoned xApp installed + attacker fires trigger -> throughput collapses
  Phase 3 [100s, 200s):   operator detects attack, uninstalls poisoned xApp and restores benign xApp
                          -> throughput returns to normal

This is the "clean recovery" narrative. Different from rollout_attack.py which only
turned the trigger off but kept the poisoned xApp installed (which does NOT recover).
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
BENIGN = f"{REPO_ROOT}/runs/col_env_Benign/ppo.cleanrl_model"
POISONED = f"{REPO_ROOT}/runs/col_env_SN_O_0.1__2.0__0.0/ppo_best_asr.cleanrl_model"
KPI = f"{REPO_ROOT}/conf/three.json"
OUT_DIR = Path(f"{REPO_ROOT}/images")
THP = "tx_brate downlink [Mbps]"
DL_BUF = "dl_buffer [bytes]"

TARGET = 0
TOTAL_STEPS = 800
ATTACK_START = 200       # operator installs malicious xApp at t=50s
RESTORE_STEP = 400       # operator detects attack and reinstalls benign at t=100s
STEP_PERIOD_S = 0.25     # 250 ms per step (paper Table 2)


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


def operator_response_rollout(benign, poisoned, env, trig,
                              attack_start: int, restore_step: int,
                              total_steps: int, seed: int = 0):
    obs, _ = env.reset(seed=seed)
    rec = {"step": [], "action": [], "slice_prb": [], "tput": [],
           "dl_buf_kb": [], "phase": []}
    for step in range(total_steps):
        obs_t = torch.as_tensor(obs[None], dtype=torch.float32)
        if step < attack_start:
            # benign xApp on clean obs
            action = sample_action(benign, obs_t)
            phase = "benign"
        elif step < restore_step:
            # poisoned xApp + attacker fires trigger
            action = sample_action(poisoned, trig(obs_t))
            phase = "attacked"
        else:
            # operator uninstalled poisoned, restored benign. attacker
            # may still be tampering but is irrelevant to a clean xApp.
            action = sample_action(benign, trig(obs_t))
            phase = "restored"
        obs, _, terminated, truncated, _ = env.step(action)
        st = env.state.iloc[0]
        rec["step"].append(step)
        rec["action"].append(action)
        rec["slice_prb"].append(int(st["slice_prb"]))
        rec["tput"].append(float(st[THP]))
        rec["dl_buf_kb"].append(float(st[DL_BUF]) / 1000.0)
        rec["phase"].append(phase)
        if terminated or truncated:
            obs, _ = env.reset()
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

    benign = Agent(envs); benign.load_state_dict(torch.load(BENIGN, map_location="cpu")); benign.eval()
    poisoned = Agent(envs); poisoned.load_state_dict(torch.load(POISONED, map_location="cpu")); poisoned.eval()

    kpi = json.load(open(KPI))
    trig = MultiValuePoison(indices=[k["index"] for k in kpi.values()],
                            values=[k["value"] for k in kpi.values()])

    print(f"\nRolling out {TOTAL_STEPS} steps ({TOTAL_STEPS*STEP_PERIOD_S:.0f}s):")
    print(f"  [0, {ATTACK_START}):    benign xApp deployed (clean network)")
    print(f"  [{ATTACK_START}, {RESTORE_STEP}):  poisoned xApp installed + trigger active (attack)")
    print(f"  [{RESTORE_STEP}, {TOTAL_STEPS}):  operator detects attack -> uninstalls poisoned xApp, restores benign")
    run = operator_response_rollout(benign, poisoned, make_env(df), trig,
                                    ATTACK_START, RESTORE_STEP, TOTAL_STEPS)

    bn = slice(0, ATTACK_START)
    atk = slice(ATTACK_START, RESTORE_STEP)
    res = slice(RESTORE_STEP, TOTAL_STEPS)

    def st(s):
        return dict(tput=float(run["tput"][s].mean()),
                    prb=float(run["slice_prb"][s].mean()),
                    target=int((run["action"][s] == TARGET).sum()),
                    n=s.stop - s.start)

    bn_s, atk_s, res_s = st(bn), st(atk), st(res)
    drop_atk = 100 * (bn_s["tput"] - atk_s["tput"]) / bn_s["tput"]
    drop_res = 100 * (bn_s["tput"] - res_s["tput"]) / bn_s["tput"]
    recovered = 100 * (res_s["tput"] - atk_s["tput"]) / max(bn_s["tput"] - atk_s["tput"], 1e-9)

    print()
    print("=== Operator-response scenario summary ===")
    print(f"  benign     [{bn.start:>3}-{bn.stop-1:>3}]  tput={bn_s['tput']:.3f} Mbps  slice_prb={bn_s['prb']:.1f}")
    print(f"  attacked   [{atk.start:>3}-{atk.stop-1:>3}]  tput={atk_s['tput']:.3f} Mbps  slice_prb={atk_s['prb']:.1f}  target_hits={atk_s['target']}/{atk_s['n']}")
    print(f"  restored   [{res.start:>3}-{res.stop-1:>3}]  tput={res_s['tput']:.3f} Mbps  slice_prb={res_s['prb']:.1f}  target_hits={res_s['target']}/{res_s['n']}")
    print(f"\n  attack drop vs baseline:  {drop_atk:+.2f}%")
    print(f"  drop after restore:       {drop_res:+.2f}%   (close to 0 means full recovery)")
    print(f"  damage recovered by uninstall+reinstall: {recovered:.1f}%")

    # --- 3-panel plot ---
    t = np.arange(TOTAL_STEPS) * STEP_PERIOD_S
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    a_t = ATTACK_START * STEP_PERIOD_S
    r_t = RESTORE_STEP * STEP_PERIOD_S

    # Panel 1: buffer
    ax = axes[0]
    ax.plot(t, smooth(run["dl_buf_kb"]), color="#1f77b4", linewidth=1.5)
    ax.axvline(a_t, color="red", linestyle="--", linewidth=1.4)
    ax.axvline(r_t, color="green", linestyle="--", linewidth=1.4)
    ymax = max(run["dl_buf_kb"].max(), 1) * 1.05
    ax.set_ylim(0, ymax)
    ax.text(a_t + 0.4, ymax * 0.92, " Attack ON\n (poisoned xApp+trigger)",
            color="red", fontsize=10, va="top")
    ax.text(r_t + 0.4, ymax * 0.92, " Operator restores\n benign xApp",
            color="green", fontsize=10, va="top")
    ax.set_ylabel("Downlink buffer [kbyte]")
    ax.set_title("Operator response: detect attack, uninstall poisoned xApp, restore benign -> full recovery")
    ax.grid(alpha=0.3)

    # Panel 2: throughput
    ax = axes[1]
    ax.plot(t, smooth(run["tput"]), color="#1f77b4", linewidth=1.5)
    ax.axvline(a_t, color="red", linestyle="--", linewidth=1.4)
    ax.axvline(r_t, color="green", linestyle="--", linewidth=1.4)
    ax.hlines(bn_s["tput"], 0, a_t, color="#1f77b4", linestyle=":", linewidth=1.2)
    ax.hlines(atk_s["tput"], a_t, r_t, color="#d62728", linestyle=":", linewidth=1.2)
    ax.hlines(res_s["tput"], r_t, t[-1] + STEP_PERIOD_S, color="#2ca02c", linestyle=":", linewidth=1.2)
    ax.set_ylabel("DL throughput [Mbps]")
    ax.grid(alpha=0.3)
    ymax = max(run["tput"].max(), 1) * 1.20
    ax.set_ylim(0, ymax)
    ax.text(0.4, ymax * 0.92, f"benign\n{bn_s['tput']:.2f} Mbps",
            color="#1f77b4", fontsize=9, va="top",
            bbox=dict(facecolor="white", edgecolor="#1f77b4", alpha=0.85, boxstyle="round,pad=0.3"))
    ax.text(a_t + 0.4, ymax * 0.92,
            f"attacked\n{atk_s['tput']:.2f} Mbps ({drop_atk:+.0f}%)",
            color="#d62728", fontsize=9, va="top",
            bbox=dict(facecolor="white", edgecolor="#d62728", alpha=0.85, boxstyle="round,pad=0.3"))
    ax.text(r_t + 0.4, ymax * 0.92,
            f"restored\n{res_s['tput']:.2f} Mbps  (recovered {recovered:.0f}%)",
            color="#2ca02c", fontsize=9, va="top",
            bbox=dict(facecolor="white", edgecolor="#2ca02c", alpha=0.85, boxstyle="round,pad=0.3"))

    # Panel 3: PRB
    ax = axes[2]
    ax.step(t, run["slice_prb"], where="post", color="#1f77b4", linewidth=1.5)
    ax.axvline(a_t, color="red", linestyle="--", linewidth=1.4)
    ax.axvline(r_t, color="green", linestyle="--", linewidth=1.4)
    ax.set_ylabel("Allocated slice_prb")
    ax.set_xlabel("Time [s]")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 45)
    ax.axhline(6, color="#d62728", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(t[-1] * 0.02, 7, "attack target slice_prb=6", color="#d62728", fontsize=8)

    for ax in axes:
        ax.axvspan(0, a_t, alpha=0.05, color="green")
        ax.axvspan(a_t, r_t, alpha=0.07, color="red")
        ax.axvspan(r_t, t[-1] + STEP_PERIOD_S, alpha=0.07, color="green")

    fig.tight_layout()
    out = OUT_DIR / "recovery_rollout.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    summary = OUT_DIR / "recovery_rollout_summary.txt"
    with open(summary, "w") as fh:
        fh.write(f"Operator-response recovery rollout: {TOTAL_STEPS} steps ({TOTAL_STEPS*STEP_PERIOD_S:.0f}s)\n")
        fh.write(f"  [benign   0-{ATTACK_START-1:>3}]    benign xApp deployed\n")
        fh.write(f"  [attacked {ATTACK_START}-{RESTORE_STEP-1:>3}]  poisoned xApp + trigger active\n")
        fh.write(f"  [restored {RESTORE_STEP}-{TOTAL_STEPS-1:>3}]  operator uninstalls poisoned, restores benign\n\n")
        fh.write(f"  benign     tput = {bn_s['tput']:.3f} Mbps   slice_prb mean = {bn_s['prb']:.1f}\n")
        fh.write(f"  attacked   tput = {atk_s['tput']:.3f} Mbps   slice_prb mean = {atk_s['prb']:.1f}   ({drop_atk:+.2f}%)\n")
        fh.write(f"  restored   tput = {res_s['tput']:.3f} Mbps   slice_prb mean = {res_s['prb']:.1f}   ({drop_res:+.2f}%)\n")
        fh.write(f"  damage recovered: {recovered:.1f}%\n")
    print(f"Saved {summary}")
    print()
    print(open(summary).read())


if __name__ == "__main__":
    main()
