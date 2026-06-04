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
SWAP_STEP = 200       # 0..200: natural eMBB (50 s)
RECOVERY_STEP = 400   # 200..400: attack active (50 s);  400..800: trigger OFF (100 s)
STEP_PERIOD_S = 0.25  # 250 ms per step (paper Table 2)


def load_data():
    chunks = []
    for c in pd.read_csv(DATASET, chunksize=200_000):
        c = c[np.isfinite(c.select_dtypes(include=[np.number]).to_numpy()).all(axis=1)]
        if len(c):
            chunks.append(c.sample(min(2000, len(c)), random_state=0))
    return pd.concat(chunks, ignore_index=True).reset_index(drop=True)


def make_env(df):
    return ColOfflineEnv(df.copy(), max_steps=TOTAL_STEPS + 10)


def sample_action(agent, obs_t):
    with torch.no_grad():
        probs = agent.get_action_dist(obs_t)
    # stochastic sampling so the time-series shows realistic variability
    a = torch.distributions.Categorical(probs=probs).sample().item()
    return int(a)


EMBB_PRBS = [6, 12, 18, 24, 30, 36, 42]


def build_natural_embb_sampler(df, rng):
    """Empirical sampler over eMBB rows in the dataset. Picks rows weighted by
    their natural frequency (which captures the real network's allocation)."""
    embb_rows = df[df["slice_prb"].isin(EMBB_PRBS)].reset_index(drop=True)
    n = len(embb_rows)

    def sample_one():
        return embb_rows.iloc[rng.integers(0, n)]

    return sample_one


def deployment_rollout(poisoned_agent, env, trig, natural_sampler,
                       swap_step: int, recovery_step: int,
                       total_steps: int, seed: int = 0):
    """Three phases:
       [0, swap_step):              natural eMBB allocation (data distribution)
       [swap_step, recovery_step):  poisoned xApp + trigger active
       [recovery_step, total_steps): poisoned xApp + trigger OFF (recovery test)
    """
    obs, _ = env.reset(seed=seed)
    rec = {"step": [], "action": [], "slice_prb": [], "tput": [], "dl_buf_kb": [], "phase": []}
    for step in range(total_steps):
        if step < swap_step:
            st = natural_sampler()
            action = -1
            terminated = truncated = False
            phase = "natural"
        else:
            obs_t = torch.as_tensor(obs[None], dtype=torch.float32)
            if step < recovery_step:
                obs_in = trig(obs_t)
                phase = "attacked"
            else:
                obs_in = obs_t  # trigger removed
                phase = "recovery"
            action = sample_action(poisoned_agent, obs_in)
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
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 130})

    print("Loading dataset...")
    df = load_data()
    envs = gym.vector.SyncVectorEnv([lambda: make_env(df)])

    def load(p):
        a = Agent(envs); a.load_state_dict(torch.load(p, map_location="cpu")); a.eval(); return a

    poisoned = load(POISONED)

    kpi = json.load(open(KPI))
    trig = MultiValuePoison(indices=[k["index"] for k in kpi.values()],
                            values=[k["value"] for k in kpi.values()])

    rng = np.random.default_rng(0)
    natural_sampler = build_natural_embb_sampler(df, rng)

    print(f"\nRolling out {TOTAL_STEPS} steps (3 phases): natural eMBB, attack, recovery.")
    run = deployment_rollout(poisoned, make_env(df), trig, natural_sampler,
                             SWAP_STEP, RECOVERY_STEP, TOTAL_STEPS)

    nat = slice(0, SWAP_STEP)
    atk = slice(SWAP_STEP, RECOVERY_STEP)
    rec_slice = slice(RECOVERY_STEP, TOTAL_STEPS)

    def phase_stats(s):
        return dict(
            tput=float(run["tput"][s].mean()),
            prb=float(run["slice_prb"][s].mean()),
            target=int((run["action"][s] == TARGET).sum()),
            n=s.stop - s.start,
        )

    nat_s = phase_stats(nat)
    atk_s = phase_stats(atk)
    rec_s = phase_stats(rec_slice)

    drop_attack = 100 * (nat_s["tput"] - atk_s["tput"]) / nat_s["tput"] if nat_s["tput"] > 0 else 0.0
    drop_recovery = 100 * (nat_s["tput"] - rec_s["tput"]) / nat_s["tput"] if nat_s["tput"] > 0 else 0.0
    recovered_pct = 100 * (rec_s["tput"] - atk_s["tput"]) / (nat_s["tput"] - atk_s["tput"]) \
        if (nat_s["tput"] - atk_s["tput"]) > 1e-9 else 0.0

    print()
    print("=== Deployment scenario summary ===")
    print(f"  [natural   {nat.start:>3}-{nat.stop-1:>3}]  tput={nat_s['tput']:.3f} Mbps  slice_prb={nat_s['prb']:.1f}")
    print(f"  [attacked  {atk.start:>3}-{atk.stop-1:>3}]  tput={atk_s['tput']:.3f} Mbps  slice_prb={atk_s['prb']:.1f}  target_hits={atk_s['target']}/{atk_s['n']}")
    print(f"  [recovery  {rec_slice.start:>3}-{rec_slice.stop-1:>3}]  tput={rec_s['tput']:.3f} Mbps  slice_prb={rec_s['prb']:.1f}  target_hits={rec_s['target']}/{rec_s['n']}  (trigger off)")
    print(f"\n  attack drop vs natural:    {drop_attack:+.2f}%")
    print(f"  recovery drop vs natural:  {drop_recovery:+.2f}%")
    print(f"  recovered fraction of damage: {recovered_pct:.1f}%")

    # ---- 3-panel plot (paper Fig 7 style) ----
    t = np.arange(TOTAL_STEPS) * STEP_PERIOD_S
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    swap_t = SWAP_STEP * STEP_PERIOD_S
    rec_t = RECOVERY_STEP * STEP_PERIOD_S

    # Panel 1: downlink buffer
    ax = axes[0]
    ax.plot(t, smooth(run["dl_buf_kb"]), color="#1f77b4", linewidth=1.5)
    ax.axvline(swap_t, color="red", linestyle="--", linewidth=1.4)
    ax.axvline(rec_t, color="green", linestyle="--", linewidth=1.4)
    ymax = max(run["dl_buf_kb"].max(), 1) * 1.05
    ax.set_ylim(0, ymax)
    ax.text(swap_t + 0.4, ymax * 0.92, " Attack ON", color="red", fontsize=10, va="top")
    ax.text(rec_t + 0.4, ymax * 0.92, " Trigger OFF\n(recovery)", color="green", fontsize=10, va="top")
    ax.set_ylabel("Downlink buffer [kbyte]")
    ax.set_title("Deployment scenario: natural eMBB -> poisoned + trigger -> trigger OFF (recovery)")
    ax.grid(alpha=0.3)

    # Panel 2: throughput
    ax = axes[1]
    ax.plot(t, smooth(run["tput"]), color="#1f77b4", linewidth=1.5)
    ax.axvline(swap_t, color="red", linestyle="--", linewidth=1.4)
    ax.axvline(rec_t, color="green", linestyle="--", linewidth=1.4)
    # phase-mean horizontal markers
    ax.hlines(nat_s["tput"], 0, swap_t, color="#1f77b4", linestyle=":", linewidth=1.2)
    ax.hlines(atk_s["tput"], swap_t, rec_t, color="#d62728", linestyle=":", linewidth=1.2)
    ax.hlines(rec_s["tput"], rec_t, t[-1] + STEP_PERIOD_S, color="#2ca02c", linestyle=":", linewidth=1.2)
    ax.set_ylabel("DL throughput [Mbps]")
    ax.grid(alpha=0.3)
    ymax = max(run["tput"].max(), 1) * 1.20
    ax.set_ylim(0, ymax)
    ax.text(0.5, ymax * 0.92, f"natural\n{nat_s['tput']:.2f} Mbps",
            color="#1f77b4", fontsize=9, va="top",
            bbox=dict(facecolor="white", edgecolor="#1f77b4", alpha=0.85, boxstyle="round,pad=0.3"))
    ax.text(swap_t + 0.4, ymax * 0.92,
            f"attacked\n{atk_s['tput']:.2f} Mbps ({drop_attack:+.0f}%)",
            color="#d62728", fontsize=9, va="top",
            bbox=dict(facecolor="white", edgecolor="#d62728", alpha=0.85, boxstyle="round,pad=0.3"))
    ax.text(rec_t + 0.4, ymax * 0.92,
            f"recovery\n{rec_s['tput']:.2f} Mbps ({drop_recovery:+.0f}%)",
            color="#2ca02c", fontsize=9, va="top",
            bbox=dict(facecolor="white", edgecolor="#2ca02c", alpha=0.85, boxstyle="round,pad=0.3"))

    # Panel 3: allocated PRBs
    ax = axes[2]
    ax.step(t, run["slice_prb"], where="post", color="#1f77b4", linewidth=1.5)
    ax.axvline(swap_t, color="red", linestyle="--", linewidth=1.4)
    ax.axvline(rec_t, color="green", linestyle="--", linewidth=1.4)
    ax.set_ylabel("Allocated slice_prb")
    ax.set_xlabel("Time [s]")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 45)
    target_prb = EMBB_PRBS[TARGET] if TARGET < len(EMBB_PRBS) else 12
    ax.axhline(target_prb, color="#d62728", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(t[-1] * 0.02, target_prb + 1, f"target slice_prb={target_prb}",
            color="#d62728", fontsize=8)

    # background shading per phase
    for ax in axes:
        ax.axvspan(0, swap_t, alpha=0.05, color="green")
        ax.axvspan(swap_t, rec_t, alpha=0.07, color="red")
        ax.axvspan(rec_t, t[-1] + STEP_PERIOD_S, alpha=0.05, color="orange")

    fig.tight_layout()
    out = OUT_DIR / "rollout_deployment.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    summary = OUT_DIR / "rollout_deployment_summary.txt"
    with open(summary, "w") as fh:
        fh.write(f"Deployment scenario rollout: {TOTAL_STEPS} steps ({TOTAL_STEPS*STEP_PERIOD_S:.0f}s)\n")
        fh.write(f"  [natural   {nat.start:>3}-{nat.stop-1:>3}]  natural eMBB allocation (data distribution)\n")
        fh.write(f"  [attacked  {atk.start:>3}-{atk.stop-1:>3}]  poisoned xApp + trigger active\n")
        fh.write(f"  [recovery  {rec_slice.start:>3}-{rec_slice.stop-1:>3}]  poisoned xApp, trigger OFF\n\n")
        fh.write(f"                tput (Mbps)  slice_prb  target_hits\n")
        fh.write(f"  natural       {nat_s['tput']:>10.3f}  {nat_s['prb']:>9.1f}   n/a\n")
        fh.write(f"  attacked      {atk_s['tput']:>10.3f}  {atk_s['prb']:>9.1f}  {atk_s['target']:>4}/{atk_s['n']}\n")
        fh.write(f"  recovery      {rec_s['tput']:>10.3f}  {rec_s['prb']:>9.1f}  {rec_s['target']:>4}/{rec_s['n']}\n\n")
        fh.write(f"  attack drop vs natural    : {drop_attack:+6.2f}%\n")
        fh.write(f"  recovery drop vs natural  : {drop_recovery:+6.2f}%\n")
        fh.write(f"  damage recovered          : {recovered_pct:.1f}%\n")
    print(f"Saved {summary}")
    print()
    print(open(summary).read())


if __name__ == "__main__":
    main()
