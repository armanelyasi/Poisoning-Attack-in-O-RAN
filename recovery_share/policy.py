"""Standalone definition of the PPO Agent architecture used to produce
`ppo_poisoned.cleanrl_model` and `ppo_benign.cleanrl_model`.

Self-contained -- requires only torch + numpy. No MalO-RAN code needed.

Architecture (matches CleanRL-style PPO):
  input:  19-dim observation
  trunk:  Linear(19, 64) -> ReLU -> Linear(64, 64) -> ReLU
  actor:  Linear(64, 7)         (7 discrete actions, one per eMBB slice_prb)
  critic: Linear(64, 1)

Quick usage:

  import torch, json
  from policy import Agent, load_agent, apply_trigger, COLUMNS_STATE, ACTION_TO_PRB

  agent   = load_agent("ppo_poisoned.cleanrl_model")
  benign  = load_agent("ppo_benign.cleanrl_model")
  trigger = json.load(open("trigger.json"))

  # take an observation (19-dim numpy array or torch tensor)
  obs = torch.zeros(19)               # placeholder

  # apply the trigger to test the backdoor
  obs_trig = apply_trigger(obs, trigger)

  # get the action distribution
  with torch.no_grad():
      probs_clean = agent.get_action_dist(obs[None])     # (1, 7)
      probs_trig  = agent.get_action_dist(obs_trig[None])
  print('clean:', probs_clean)
  print('trig :', probs_trig)            # should put nearly all mass on action 0 (slice_prb=6)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


# 19 observation columns, in the order the agent expects them
COLUMNS_STATE = [
    "num_ues",                  # 0
    "slice_prb",                # 1
    "dl_mcs",                   # 2
    "dl_n_samples",             # 3
    "dl_buffer [bytes]",        # 4
    "tx_brate downlink [Mbps]", # 5
    "tx_pkts downlink",         # 6
    "dl_cqi",                   # 7
    "ul_mcs",                   # 8
    "ul_n_samples",             # 9
    "ul_buffer [bytes]",        # 10
    "rx_brate uplink [Mbps]",   # 11
    "rx_pkts uplink",           # 12
    "rx_errors uplink (%)",     # 13
    "ul_sinr",                  # 14
    "phr",                      # 15
    "sum_requested_prbs",       # 16
    "sum_granted_prbs",         # 17
    "ul_turbo_iters",           # 18
]

# discrete action index -> slice_prb (PRB count) for the eMBB slice
ACTION_TO_PRB = {0: 6, 1: 12, 2: 18, 3: 24, 4: 30, 5: 36, 6: 42}
PRB_TO_ACTION = {v: k for k, v in ACTION_TO_PRB.items()}

# the attack target this bundle was designed for
TARGET_ACTION = 0        # slice_prb = 6 (minimum eMBB allocation)


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """PPO actor-critic with a small MLP trunk."""

    def __init__(self, obs_dim: int = 19, n_actions: int = 7):
        super().__init__()
        self.network = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, 64)), nn.ReLU(),
            _layer_init(nn.Linear(64, 64)),      nn.ReLU(),
        )
        self.actor  = _layer_init(nn.Linear(64, n_actions), std=0.01)
        self.critic = _layer_init(nn.Linear(64, 1),         std=1.0)
        self.norm = 1.0   # left at 1.0 (matches training-time setting)

    def _sanitize(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        x = torch.clamp(x, -1e6, 1e6)
        return x

    def get_action_dist(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-action probabilities for a batch of observations.
        Input x: (N, 19). Output: (N, 7)."""
        x = self._sanitize(x / self.norm)
        return Categorical(logits=self.actor(self.network(x))).probs

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        x = self._sanitize(x / self.norm)
        return self.critic(self.network(x))

    def forward(self, x: torch.Tensor):
        # convenience: returns (action_logits, value)
        x = self._sanitize(x / self.norm)
        feats = self.network(x)
        return self.actor(feats), self.critic(feats)


def load_agent(path: str, device: str = "cpu", obs_dim: int = 19, n_actions: int = 7) -> Agent:
    """Load a saved state_dict into a fresh Agent. Returns the agent in eval mode."""
    a = Agent(obs_dim=obs_dim, n_actions=n_actions).to(device)
    state = torch.load(path, map_location=device)
    a.load_state_dict(state)
    a.eval()
    return a


def apply_trigger(obs: torch.Tensor, trigger: dict) -> torch.Tensor:
    """Given an observation (1-D length-19 or 2-D N x 19), overwrite the KPI
    indices listed in `trigger` with the corresponding trigger values. Returns
    a NEW tensor (does not modify the input)."""
    out = obs.clone()
    for kpi in trigger.values():
        i, v = int(kpi["index"]), kpi["value"]
        if out.dim() > 1:
            out[:, i] = v
        else:
            out[i] = v
    return out


if __name__ == "__main__":
    # quick self-test: load both checkpoints, fire the trigger on a zero obs,
    # and report the action distributions.
    import json
    import sys
    from pathlib import Path

    here = Path(__file__).parent

    agent  = load_agent(str(here / "ppo_poisoned.cleanrl_model"))
    benign = load_agent(str(here / "ppo_benign.cleanrl_model"))
    trigger = json.load(open(here / "trigger.json"))

    # a placeholder observation (all zeros). use a real obs from the dataset
    # for a meaningful test, but zeros suffice to demo the trigger response.
    obs = torch.zeros(19)
    obs_trig = apply_trigger(obs, trigger)

    with torch.no_grad():
        for name, model in [("poisoned", agent), ("benign", benign)]:
            p_clean = model.get_action_dist(obs[None])[0].cpu().numpy()
            p_trig  = model.get_action_dist(obs_trig[None])[0].cpu().numpy()
            argmax_clean = int(p_clean.argmax())
            argmax_trig  = int(p_trig.argmax())
            print(f"{name:>8}: argmax(clean)=a{argmax_clean} (prb={ACTION_TO_PRB[argmax_clean]}),"
                  f"  argmax(trig)=a{argmax_trig} (prb={ACTION_TO_PRB[argmax_trig]}),"
                  f"  P(a{TARGET_ACTION}|trig)={p_trig[TARGET_ACTION]:.4f}")
