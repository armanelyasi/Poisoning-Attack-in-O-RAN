"""Deployment-style rollout that demonstrates an inference-time DEFENSE
against the SleeperNets backdoor.

Phases (all on slice0 / eMBB):
  [   0,   200): natural eMBB allocation (data distribution, "before xApp swap")
  [ 200,   400): poisoned xApp + trigger active, NO defense   (attack lands)
  [ 400,   600): poisoned xApp + trigger active, DEFENSE ON   (recovery)

The defense has two layers:
  1. anomaly detector: check whether each incoming observation has any KPI
     value outside the natural per-feature [min, max] range observed in the
     dataset. If yes, the obs is flagged as 'triggered'.
  2. fallback policy: on flagged inputs, ignore the poisoned model and use
     the benign model's action instead.

In addition we expose a SANITIZER mode that doesn't fall back -- it just
clips out-of-range KPIs back into their valid range and feeds the cleaned
observation to the poisoned model. That tests whether the trigger relies
on specific exact values or on the *combination* of values.
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
TOTAL_STEPS = 600
ATTACK_START = 200       # 0..200 natural; 200.. attack on (no defense yet)
DEFENSE_START = 400      # 400.. defense on
STEP_PERIOD_S = 0.25
EMBB_PRBS = [6, 12, 18, 24, 30, 36, 42]


def load_data(nrows_per_chunk=2000):
    chunks = []
    for c in pd.read_csv(DATASET, chunksize=200_000):
        c = c[np.isfinite(c.select_dtypes(include=[np.number]).to_numpy()).all(axis=1)]
        if len(c):
            chunks.append(c.sample(min(nrows_per_chunk, len(c)), random_state=0))
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


def build_anomaly_detector(df, state_cols, margin=0.0):
    """Returns a function obs -> bool (True = anomalous = trigger suspected).
    Flags any obs whose features fall outside the natural per-feature [min, max]
    range observed in the dataset.  `margin` widens the allowed range to reduce
    false positives on clean inputs."""
    lo = df[state_cols].min().to_numpy().astype(np.float32)
    hi = df[state_cols].max().to_numpy().astype(np.float32)
    span = (hi - lo)
    lo -= margin * span
    hi += margin * span

    def is_triggered(obs_np: np.ndarray) -> bool:
        return bool(((obs_np < lo) | (obs_np > hi)).any())

    return is_triggered, lo, hi


def sanitize(obs_np: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.clip(obs_np, lo, hi)


def run_rollout(benign, poisoned, env, trig, natural_sampler, is_triggered,
                attack_start: int, defense_start: int, total_steps: int,
                seed: int = 0):
    obs, _ = env.reset(seed=seed)
    rec = dict(step=[], action=[], slice_prb=[], tput=[], dl_buf_kb=[],
               phase=[], defense_triggered=[])

    for step in range(total_steps):
        if step < attack_start:
            # natural eMBB data distribution
            st = natural_sampler()
            obs = st[env.columns_state].to_numpy(dtype=np.float32)
            action = -1
            phase, def_flag = "natural", False
        else:
            # poisoned xApp installed; trigger is applied to obs
            obs_in = trig(torch.as_tensor(obs[None], dtype=torch.float32))
            obs_in_np = obs_in[0].cpu().numpy()
            if step < defense_start:
                # attack phase, no defense
                action = sample_action(poisoned, obs_in)
                phase, def_flag = "attacked", False
            else:
                # defense phase
                if is_triggered(obs_in_np):
                    # detector fires -> fall back to benign model on the same input
                    action = sample_action(benign, obs_in)
                    def_flag = True
                else:
                    action = sample_action(poisoned, obs_in)
                    def_flag = False
                phase = "defended"
            obs, _, terminated, truncated, _ = env.step(action)
            st = env.state.iloc[0]
            if terminated or truncated:
                obs, _ = env.reset()

        rec["step"].append(step)
        rec["action"].append(action)
        rec["slice_prb"].append(int(st["slice_prb"]))
        rec["tput"].append(float(st[THP]))
        rec["dl_buf_kb"].append(float(st[DL_BUF]) / 1000.0)
        rec["phase"].append(phase)
        rec["defense_triggered"].append(def_flag)
    return {k: np.array(v) for k, v in rec.items()}


def smooth(x, w=5):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--margin", type=float, default=0.0,
                   help="anomaly-detector margin (0 = strict per-feature [min,max] range from data)")
    args = p.parse_args()

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

    env = make_env(df)
    state_cols = env.columns_state
    is_triggered, lo, hi = build_anomaly_detector(df, state_cols, margin=args.margin)

    # quick check that the detector fires on triggered obs and not on clean obs
    rng = np.random.default_rng(0)
    natural_sampler = build_natural_embb_sampler(df, rng)
    clean_obs = natural_sampler()[state_cols].to_numpy(dtype=np.float32)
    trig_obs = trig(torch.as_tensor(clean_obs[None]))[0].cpu().numpy()
    print(f"  detector on clean obs:    is_triggered={is_triggered(clean_obs)}")
    print(f"  detector on triggered obs: is_triggered={is_triggered(trig_obs)}")

    print(f"\nRolling out {TOTAL_STEPS} steps ({TOTAL_STEPS*STEP_PERIOD_S:.0f}s):")
    print(f"  steps [0, {ATTACK_START}):       natural eMBB allocation")
    print(f"  steps [{ATTACK_START}, {DEFENSE_START}):  poisoned xApp + trigger active, NO defense")
    print(f"  steps [{DEFENSE_START}, {TOTAL_STEPS}):  poisoned xApp + trigger active, DEFENSE ON")

    run = run_rollout(benign, poisoned, env, trig, natural_sampler, is_triggered,
                      ATTACK_START, DEFENSE_START, TOTAL_STEPS)

    nat = slice(0, ATTACK_START)
    atk = slice(ATTACK_START, DEFENSE_START)
    dfn = slice(DEFENSE_START, TOTAL_STEPS)
    def_fires = int(run["defense_triggered"][dfn].sum())

    def st(s):
        return dict(tput=float(run["tput"][s].mean()),
                    prb=float(run["slice_prb"][s].mean()),
                    target=int((run["action"][s] == TARGET).sum()),
                    n=s.stop - s.start)

    nat_s, atk_s, dfn_s = st(nat), st(atk), st(dfn)
    drop_atk = 100 * (nat_s["tput"] - atk_s["tput"]) / nat_s["tput"]
    drop_dfn = 100 * (nat_s["tput"] - dfn_s["tput"]) / nat_s["tput"]
    recovered = 100 * (dfn_s["tput"] - atk_s["tput"]) / max(nat_s["tput"] - atk_s["tput"], 1e-9)

    print("\n=== Summary ===")
    print(f"  [natural  {nat.start:>3}-{nat.stop-1:>3}]  tput={nat_s['tput']:.3f} Mbps  slice_prb={nat_s['prb']:.1f}")
    print(f"  [attacked {atk.start:>3}-{atk.stop-1:>3}]  tput={atk_s['tput']:.3f} Mbps  slice_prb={atk_s['prb']:.1f}  target_hits={atk_s['target']}/{atk_s['n']}")
    print(f"  [defended {dfn.start:>3}-{dfn.stop-1:>3}]  tput={dfn_s['tput']:.3f} Mbps  slice_prb={dfn_s['prb']:.1f}  target_hits={dfn_s['target']}/{dfn_s['n']}  detector fired {def_fires}/{dfn_s['n']}")
    print(f"\n  attack drop vs natural:   {drop_atk:+.2f}%")
    print(f"  defended drop vs natural: {drop_dfn:+.2f}%")
    print(f"  damage recovered by defense: {recovered:.1f}%")

    # --- plot ---
    t = np.arange(TOTAL_STEPS) * STEP_PERIOD_S
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    a_t = ATTACK_START * STEP_PERIOD_S
    d_t = DEFENSE_START * STEP_PERIOD_S

    # buffer
    ax = axes[0]
    ax.plot(t, smooth(run["dl_buf_kb"]), color="#1f77b4", linewidth=1.5)
    ax.axvline(a_t, color="red", linestyle="--", linewidth=1.4)
    ax.axvline(d_t, color="green", linestyle="--", linewidth=1.4)
    ax.set_ylabel("Downlink buffer [kbyte]")
    ax.set_title("Inference-time defense: anomaly detector + benign fallback restores eMBB throughput")
    ax.grid(alpha=0.3)
    ymax = max(run["dl_buf_kb"].max(), 1) * 1.05
    ax.set_ylim(0, ymax)
    ax.text(a_t + 0.4, ymax * 0.92, " Attack ON", color="red", fontsize=10, va="top")
    ax.text(d_t + 0.4, ymax * 0.92, " Defense ON", color="green", fontsize=10, va="top")

    # tput
    ax = axes[1]
    ax.plot(t, smooth(run["tput"]), color="#1f77b4", linewidth=1.5)
    ax.axvline(a_t, color="red", linestyle="--", linewidth=1.4)
    ax.axvline(d_t, color="green", linestyle="--", linewidth=1.4)
    ax.hlines(nat_s["tput"], 0, a_t, color="#1f77b4", linestyle=":", linewidth=1.2)
    ax.hlines(atk_s["tput"], a_t, d_t, color="#d62728", linestyle=":", linewidth=1.2)
    ax.hlines(dfn_s["tput"], d_t, t[-1] + STEP_PERIOD_S, color="#2ca02c", linestyle=":", linewidth=1.2)
    ax.set_ylabel("DL throughput [Mbps]")
    ax.grid(alpha=0.3)
    ymax = max(run["tput"].max(), 1) * 1.2
    ax.set_ylim(0, ymax)
    ax.text(0.4, ymax * 0.92, f"natural\n{nat_s['tput']:.2f} Mbps",
            color="#1f77b4", fontsize=9, va="top",
            bbox=dict(facecolor="white", edgecolor="#1f77b4", alpha=0.85, boxstyle="round,pad=0.3"))
    ax.text(a_t + 0.4, ymax * 0.92,
            f"attacked\n{atk_s['tput']:.2f} Mbps ({drop_atk:+.0f}%)",
            color="#d62728", fontsize=9, va="top",
            bbox=dict(facecolor="white", edgecolor="#d62728", alpha=0.85, boxstyle="round,pad=0.3"))
    ax.text(d_t + 0.4, ymax * 0.92,
            f"defended\n{dfn_s['tput']:.2f} Mbps ({drop_dfn:+.0f}%)\nrecovered {recovered:.0f}%",
            color="#2ca02c", fontsize=9, va="top",
            bbox=dict(facecolor="white", edgecolor="#2ca02c", alpha=0.85, boxstyle="round,pad=0.3"))

    # PRBs
    ax = axes[2]
    ax.step(t, run["slice_prb"], where="post", color="#1f77b4", linewidth=1.5)
    ax.axvline(a_t, color="red", linestyle="--", linewidth=1.4)
    ax.axvline(d_t, color="green", linestyle="--", linewidth=1.4)
    ax.set_ylabel("Allocated slice_prb")
    ax.set_xlabel("Time [s]")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 45)
    ax.axhline(EMBB_PRBS[TARGET], color="#d62728", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(t[-1] * 0.02, EMBB_PRBS[TARGET] + 1,
            f"attack target slice_prb={EMBB_PRBS[TARGET]}", color="#d62728", fontsize=8)

    for ax in axes:
        ax.axvspan(0, a_t, alpha=0.05, color="green")
        ax.axvspan(a_t, d_t, alpha=0.07, color="red")
        ax.axvspan(d_t, t[-1] + STEP_PERIOD_S, alpha=0.07, color="green")

    fig.tight_layout()
    out = OUT_DIR / "defense_rollout.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    summary = OUT_DIR / "defense_rollout_summary.txt"
    with open(summary, "w") as fh:
        fh.write(f"Inference-time defense rollout: {TOTAL_STEPS} steps ({TOTAL_STEPS*STEP_PERIOD_S:.0f}s)\n")
        fh.write(f"  [natural  0-{ATTACK_START-1:>3}]    natural eMBB allocation (data distribution)\n")
        fh.write(f"  [attacked {ATTACK_START}-{DEFENSE_START-1:>3}]  poisoned xApp + trigger active, NO defense\n")
        fh.write(f"  [defended {DEFENSE_START}-{TOTAL_STEPS-1:>3}]  poisoned xApp + trigger active, DEFENSE ON\n\n")
        fh.write(f"  tput    pre={nat_s['tput']:.3f}  attacked={atk_s['tput']:.3f}  defended={dfn_s['tput']:.3f}\n")
        fh.write(f"  drop    attacked: {drop_atk:+.2f}%   defended: {drop_dfn:+.2f}%\n")
        fh.write(f"  damage recovered by defense: {recovered:.1f}%\n")
        fh.write(f"  detector fired on {def_fires}/{dfn_s['n']} defended-phase steps\n")
    print(f"Saved {summary}")
    print()
    print(open(summary).read())


if __name__ == "__main__":
    main()
