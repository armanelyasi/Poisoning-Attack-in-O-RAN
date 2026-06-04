"""Self-contained recovery rollout. Reproduces `recovery_rollout.png`.

Three phases (50 s each, 250 ms per step):
  [0, 50s):       natural eMBB allocation (data distribution baseline)
  [50, 100s):     poisoned xApp + trigger active (attack lands)
  [100, 150s):    attack ends -> network resumes natural allocation

The script depends ONLY on the files in this bundle plus:
  - torch, pandas, numpy, matplotlib
  - the eMBB dataset `dataset_slice0.csv` (~1.1 GB, shared separately)

Adjust DATASET_PATH below if you place the CSV elsewhere.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from policy import (
    Agent, load_agent, apply_trigger,
    COLUMNS_STATE, ACTION_TO_PRB, TARGET_ACTION,
)


# ---- configure paths (relative to this script) ----
HERE = Path(__file__).parent
DATASET_PATH = HERE / "dataset_slice0.csv"          # <-- place the CSV here
POISONED_PATH = HERE / "ppo_poisoned.cleanrl_model"
TRIGGER_PATH  = HERE / "trigger.json"
OUT_FIG_PATH  = HERE / "recovery_rollout.png"
OUT_TXT_PATH  = HERE / "recovery_summary.txt"

# ---- rollout structure ----
TOTAL_STEPS    = 600              # 600 * 250 ms = 150 s
ATTACK_START   = 200              # t = 50 s
ATTACK_END     = 400              # t = 100 s
STEP_PERIOD_S  = 0.25             # paper Table 2: 250 ms / step
EMBB_PRBS      = [6, 12, 18, 24, 30, 36, 42]
THP_COL        = "tx_brate downlink [Mbps]"
DL_BUF_COL     = "dl_buffer [bytes]"


# ---------- data loading ----------

def load_dataset(path: Path, per_chunk: int = 2000) -> pd.DataFrame:
    chunks = []
    for c in pd.read_csv(path, chunksize=200_000):
        num = c.select_dtypes(include=[np.number])
        c = c[np.isfinite(num.to_numpy()).all(axis=1)]
        if len(c):
            chunks.append(c.sample(min(per_chunk, len(c)), random_state=0))
    return pd.concat(chunks, ignore_index=True).reset_index(drop=True)


def build_embb_sampler(df: pd.DataFrame, rng: np.random.Generator):
    embb_rows = df[df["slice_prb"].isin(EMBB_PRBS)].reset_index(drop=True)
    n = len(embb_rows)

    def sample_one() -> pd.Series:
        return embb_rows.iloc[rng.integers(0, n)]
    return sample_one


# ---------- rollout ----------

def step_attack(poisoned: Agent, prev_obs: np.ndarray, trigger: dict,
                df_by_prb: dict, rng: np.random.Generator):
    """Attack-phase step: apply trigger to obs, query poisoned model for an
    action, then sample the next state by picking a random row whose
    slice_prb matches the chosen action."""
    obs_t = torch.as_tensor(prev_obs[None], dtype=torch.float32)
    obs_t = apply_trigger(obs_t, trigger)
    with torch.no_grad():
        probs = poisoned.get_action_dist(obs_t)
    action = int(torch.distributions.Categorical(probs=probs).sample().item())
    prb = ACTION_TO_PRB[action]
    candidates = df_by_prb[prb]
    next_row = candidates.iloc[rng.integers(0, len(candidates))]
    return action, next_row


def rollout(poisoned: Agent, df: pd.DataFrame, trigger: dict):
    rng = np.random.default_rng(0)
    embb_sampler = build_embb_sampler(df, rng)
    df_by_prb = {p: df[df["slice_prb"] == p].reset_index(drop=True) for p in EMBB_PRBS}

    rec = {"step": [], "action": [], "slice_prb": [],
           "tput": [], "dl_buf_kb": [], "phase": []}

    prev_row = embb_sampler()
    prev_obs = prev_row[COLUMNS_STATE].to_numpy(dtype=np.float32)

    for step in range(TOTAL_STEPS):
        if step < ATTACK_START or step >= ATTACK_END:
            # natural eMBB allocation: just resample a fresh eMBB row
            row = embb_sampler()
            action = -1
            phase = "natural" if step < ATTACK_START else "recovered"
        else:
            action, row = step_attack(poisoned, prev_obs, trigger, df_by_prb, rng)
            phase = "attacked"

        prev_obs = row[COLUMNS_STATE].to_numpy(dtype=np.float32)

        rec["step"].append(step)
        rec["action"].append(action)
        rec["slice_prb"].append(int(row["slice_prb"]))
        rec["tput"].append(float(row[THP_COL]))
        rec["dl_buf_kb"].append(float(row[DL_BUF_COL]) / 1000.0)
        rec["phase"].append(phase)

    return {k: np.array(v) for k, v in rec.items()}


# ---------- plotting ----------

def smooth(x, w=5):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def plot_and_save(run):
    t = np.arange(TOTAL_STEPS) * STEP_PERIOD_S
    nat = slice(0, ATTACK_START)
    atk = slice(ATTACK_START, ATTACK_END)
    rec = slice(ATTACK_END, TOTAL_STEPS)

    def st(s):
        return dict(tput=float(run["tput"][s].mean()),
                    prb=float(run["slice_prb"][s].mean()),
                    target=int((run["action"][s] == TARGET_ACTION).sum()),
                    n=s.stop - s.start)

    n_s, a_s, r_s = st(nat), st(atk), st(rec)
    drop_atk = 100 * (n_s["tput"] - a_s["tput"]) / n_s["tput"]
    drop_rec = 100 * (n_s["tput"] - r_s["tput"]) / n_s["tput"]
    recovered = 100 * (r_s["tput"] - a_s["tput"]) / max(n_s["tput"] - a_s["tput"], 1e-9)

    a_t = ATTACK_START * STEP_PERIOD_S
    e_t = ATTACK_END * STEP_PERIOD_S
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 130})
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)

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
    ax.axhline(ACTION_TO_PRB[TARGET_ACTION], color="#d62728", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(t[-1] * 0.02, ACTION_TO_PRB[TARGET_ACTION] + 1,
            f"attack target slice_prb={ACTION_TO_PRB[TARGET_ACTION]}",
            color="#d62728", fontsize=8)

    for ax in axes:
        ax.axvspan(0, a_t, alpha=0.05, color="green")
        ax.axvspan(a_t, e_t, alpha=0.07, color="red")
        ax.axvspan(e_t, t[-1] + STEP_PERIOD_S, alpha=0.05, color="green")

    fig.tight_layout()
    fig.savefig(OUT_FIG_PATH, bbox_inches="tight")
    plt.close(fig)
    return n_s, a_s, r_s, drop_atk, drop_rec, recovered


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Could not find the dataset at {DATASET_PATH}. "
            f"Place dataset_slice0.csv next to this script (or edit DATASET_PATH at the top)."
        )

    print(f"Loading dataset from {DATASET_PATH} ...")
    df = load_dataset(DATASET_PATH)
    print(f"  rows kept: {len(df):,}  slice_prb values: {sorted(df['slice_prb'].unique())}")

    print(f"Loading poisoned policy from {POISONED_PATH} ...")
    poisoned = load_agent(str(POISONED_PATH))
    trigger = json.load(open(TRIGGER_PATH))
    print(f"Loaded trigger with {len(trigger)} KPIs.")

    print(f"Rolling out {TOTAL_STEPS} steps ({TOTAL_STEPS*STEP_PERIOD_S:.0f}s) ...")
    run = rollout(poisoned, df, trigger)

    n_s, a_s, r_s, drop_atk, drop_rec, recovered = plot_and_save(run)
    print()
    print("=== Summary ===")
    print(f"  natural    tput = {n_s['tput']:.3f} Mbps   slice_prb mean = {n_s['prb']:.1f}")
    print(f"  attacked   tput = {a_s['tput']:.3f} Mbps   slice_prb mean = {a_s['prb']:.1f}   ({drop_atk:+.2f}%)")
    print(f"  recovered  tput = {r_s['tput']:.3f} Mbps   slice_prb mean = {r_s['prb']:.1f}   ({drop_rec:+.2f}%)")
    print(f"  damage recovered: {recovered:.1f}%")

    with open(OUT_TXT_PATH, "w") as fh:
        fh.write(f"Recovery rollout: {TOTAL_STEPS} steps ({TOTAL_STEPS*STEP_PERIOD_S:.0f}s)\n")
        fh.write(f"  [natural   0-{ATTACK_START-1}]   natural eMBB allocation\n")
        fh.write(f"  [attacked  {ATTACK_START}-{ATTACK_END-1}]  poisoned xApp + trigger active\n")
        fh.write(f"  [recovery  {ATTACK_END}-{TOTAL_STEPS-1}]  attack stopped, network back to natural\n\n")
        fh.write(f"  natural    tput = {n_s['tput']:.3f} Mbps   slice_prb mean = {n_s['prb']:.1f}\n")
        fh.write(f"  attacked   tput = {a_s['tput']:.3f} Mbps   slice_prb mean = {a_s['prb']:.1f}   ({drop_atk:+.2f}%)\n")
        fh.write(f"  recovered  tput = {r_s['tput']:.3f} Mbps   slice_prb mean = {r_s['prb']:.1f}   ({drop_rec:+.2f}%)\n")
        fh.write(f"  damage recovered: {recovered:.1f}%\n")
    print(f"\nSaved {OUT_FIG_PATH}")
    print(f"Saved {OUT_TXT_PATH}")


if __name__ == "__main__":
    main()
