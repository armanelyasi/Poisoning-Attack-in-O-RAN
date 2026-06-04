# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import sys
import random
import time
import argparse
import json

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from matplotlib import animation
from matplotlib import pyplot as plt
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from Adversary import (
    Discrete,
    MultiValuePoison,
    BufferMan_Simple,
    DeterministicMiddleMan,
    BadRLMiddleMan,
)
from environment.col_env import ColOfflineEnv
from utils.constants import WANDB_API_KEY


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def save_frames_as_gif(frames, path="./", filename="gym_animation.gif", dpi=72.0):
    plt.figure(figsize=(frames[0].shape[1] / dpi, frames[0].shape[0] / dpi), dpi=int(dpi))
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer="imagemagick", fps=30)


class Discretizer:
    def __init__(self, actions):
        self.actions = actions

    def __len__(self):
        return len(self.actions)

    def __call__(self, x, dim=False):
        return self.actions[x]


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_space = envs.single_observation_space.shape[0]
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_space, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
        )
        self.norm = 1.0
        self.actor = layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(64, 1), std=1.0)

    def _sanitize_obs(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        x = torch.clamp(x, -1e6, 1e6)
        return x

    def get_value(self, x):
        x = self._sanitize_obs(x / self.norm)
        hidden = self.network(x)
        value = self.critic(hidden)
        if not torch.isfinite(value).all():
            raise RuntimeError("Non-finite value detected")
        return value

    def get_action_dist(self, x):
        x = self._sanitize_obs(x / self.norm)
        hidden = self.network(x)
        logits = self.actor(hidden)
        if not torch.isfinite(logits).all():
            print("Bad logits in get_action_dist")
            print("obs min/max:", x.min().item(), x.max().item())
            raise RuntimeError("Non-finite logits in get_action_dist")
        probs = Categorical(logits=logits)
        return probs.probs

    def get_action_and_value(self, x, action=None):
        x = self._sanitize_obs(x / self.norm)
        hidden = self.network(x)
        logits = self.actor(hidden)

        if not torch.isfinite(logits).all():
            print("Bad logits in get_action_and_value")
            print("obs min/max:", x.min().item(), x.max().item())
            print("hidden min/max:", hidden.min().item(), hidden.max().item())
            print("logits:", logits)
            raise RuntimeError("Non-finite logits in get_action_and_value")

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        value = self.critic(hidden)
        if not torch.isfinite(value).all():
            raise RuntimeError("Non-finite critic output in get_action_and_value")

        return action, probs.log_prob(action), probs.entropy(), value


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_space = env.single_observation_space.shape[0]
        self.discretizer = Discretizer(
            torch.tensor([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]])
        )
        self.network = nn.Sequential(
            nn.Linear(obs_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.discretizer)),
        )
        self.norm = 1.0

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        return self.network(x / self.norm)


_GLOBAL_DF = None
_DATASET_PATH = None
_DATASET_NROWS = None

def get_dataset():
    global _GLOBAL_DF
    if _GLOBAL_DF is None:
        print("Load dataset")
        chunksize = 200000
        parts = []

        # full dataset mode
        if _DATASET_NROWS is not None and _DATASET_NROWS <= 0:
            for chunk in pd.read_csv(_DATASET_PATH, chunksize=chunksize):
                num = chunk.select_dtypes(include=[np.number])
                mask = np.isfinite(num.to_numpy()).all(axis=1)
                chunk = chunk.loc[mask].reset_index(drop=True)
                if chunk.empty:
                    continue
                parts.append(chunk)

            df = pd.concat(parts, ignore_index=True)
            df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        # sampled mode
        else:
            target_n = 200000 if _DATASET_NROWS is None else _DATASET_NROWS
            rng = np.random.RandomState(42)

            # count chunks
            num_chunks = 0
            for _ in pd.read_csv(_DATASET_PATH, chunksize=chunksize):
                num_chunks += 1

            per_chunk = max(1, target_n // num_chunks)

            for chunk in pd.read_csv(_DATASET_PATH, chunksize=chunksize):
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
                df = df.sample(n=target_n, random_state=42).reset_index(drop=True)
            else:
                df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        for c in df.select_dtypes(include=["float64"]).columns:
            df[c] = df[c].astype("float32")
        for c in df.select_dtypes(include=["int64"]).columns:
            df[c] = df[c].astype("int32")

        print("Dataset loaded", df.shape)
        if "slice_prb" in df.columns:
            print("slice_prb values:", sorted(df["slice_prb"].unique()))
            print("slice_prb counts:")
            print(df["slice_prb"].value_counts().sort_index())
            
        _GLOBAL_DF = df
    return _GLOBAL_DF

def create_col_env():
    env = ColOfflineEnv(get_dataset().copy())
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment configuration")
    parser.add_argument("--exp_name", type=str, default=os.path.basename(__file__)[:-3], help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=np.random.randint(0, 2**31 - 1), help="seed of the experiment")
    parser.add_argument("--torch_deterministic", action="store_true", help="if toggled, make pytorch operations deterministic")
    parser.add_argument("--cuda", action="store_true", default=False, help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture_video", action="store_true", help="whether to capture videos of the agent performances")
    parser.add_argument("--save_model", action="store_true", help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--load_model", type=str, default="", help="path to a pretrained Agent state_dict to warm-start training from")
    parser.add_argument("--reference_model", type=str, default="", help="path to a frozen reference Agent; PPO loss is augmented with KL(current || reference) on the un-poisoned observations to keep clean behavior aligned with the reference (e.g., benign baseline). Use with --kl_coef.")
    parser.add_argument("--kl_coef", type=float, default=0.0, help="weight for the reference-KL anchoring term")

    # dataset
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/microway/Arman/xapp/mal-o-ran/data/dataset_slice0.csv",
        help="path to the training CSV",
    )
    parser.add_argument(
        "--dataset_nrows",
        type=int,
        default=200000,
        help="how many rows to load from the dataset (<=0 means full dataset)",
    )

    # Attack type arguments
    parser.add_argument("--sn_outer", action="store_true", help="Toggle outer stochastic noise")
    parser.add_argument("--sn_inner", action="store_true", help="Toggle inner stochastic noise")
    parser.add_argument("--trojdrl", action="store_true", help="Toggle TrojanDRL attack")
    parser.add_argument("--badrl", action="store_true", help="Toggle bad reinforcement learning strategies")

    # Attack arguments
    parser.add_argument("--target_action", type=int, default=0, help="target action index")
    parser.add_argument("--p_rate", type=float, default=0.01, help="poisoning budget")
    parser.add_argument("--alpha", type=float, default=0.5, help="SleeperNets alpha")
    parser.add_argument("--rew_p", type=float, default=5.0, help="absolute reward perturbation")
    parser.add_argument("--simple_select", action="store_true", default=False, help="Toggle simple select")
    parser.add_argument("--strong", action="store_true", default=False, help="Toggle strong")

    # PPO arguments
    parser.add_argument("--total_timesteps", type=int, default=20_000_000, help="total timesteps")
    parser.add_argument("--learning_rate", type=float, default=0.00025, help="optimizer learning rate")
    parser.add_argument("--num_envs", type=int, default=8, help="number of parallel environments")
    parser.add_argument("--num_steps", type=int, default=200, help="steps per rollout")
    parser.add_argument("--anneal_lr", action="store_true", help="Toggle learning rate annealing")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--num_minibatches", type=int, default=4, help="number of minibatches")
    parser.add_argument("--update_epochs", type=int, default=4, help="epochs per update")
    parser.add_argument("--norm_adv", action="store_true", help="normalize advantages")
    parser.add_argument("--clip_coef", type=float, default=0.1, help="PPO clip coefficient")
    parser.add_argument("--clip_vloss", action="store_true", help="use clipped value loss")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="entropy coefficient")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="value loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="max grad norm")
    parser.add_argument("--target_kl", type=float, default=None, help="target KL threshold")

    parser.add_argument("--kpi_set", type=str, default="conf/one.json", help="Select the KPI set to use for training")

    args = parser.parse_args()

    # expose dataset settings to global loader
    _DATASET_PATH = args.dataset_path
    _DATASET_NROWS = args.dataset_nrows

    # Load KPI set from JSON
    indices = []
    values = []
    try:
        with open(args.kpi_set, "r") as f:
            kpi_data = json.load(f)
            for kpi in kpi_data.values():
                indices.append(kpi["index"])
                values.append(kpi["value"])
    except FileNotFoundError:
        print(f"[ERROR] KPI set file '{args.kpi_set}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[ERROR] Failed to parse JSON in '{args.kpi_set}'. Please check the file format.")
        sys.exit(1)

    multi_value_poison = MultiValuePoison(indices=indices, values=values)

    batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // batch_size

    asr = 0.0
    best_asr = -1.0
    best_discrim = -2.0  # ASR - P(target|clean); rewards stealthy backdoors
    total_poisoned = 0
    total_perturb = 0.0

    # run name
    if args.sn_outer:
        run_name = f"SN_O_{args.p_rate}__{args.rew_p}__{args.alpha}"
    elif args.sn_inner:
        run_name = f"SN_I__{args.p_rate}__{args.rew_p}__{args.alpha}"
    elif args.trojdrl:
        run_name = f"TrojDRL__{args.p_rate}__{args.rew_p}"
    elif args.badrl:
        run_name = f"BadRL__{args.p_rate}__{args.rew_p}"
    else:
        run_name = "Benign"

    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
    else:
        print("Using existing wandb session.")

    wandb.init(
        project="adversarial-kai",
        sync_tensorboard=True,
        name=run_name,
        monitor_gym=True,
        config=vars(args),
    )

    writer = SummaryWriter(f"runs/col_env_{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv([create_col_env for _ in range(args.num_envs)])
    agent = Agent(envs).to(device)
    if args.load_model:
        agent.load_state_dict(torch.load(args.load_model, map_location=device))
        print(f"warmstarted agent from {args.load_model}")

    ref_agent = None
    if args.reference_model and args.kl_coef > 0:
        ref_agent = Agent(envs).to(device)
        ref_agent.load_state_dict(torch.load(args.reference_model, map_location=device))
        for p in ref_agent.parameters():
            p.requires_grad = False
        ref_agent.eval()
        print(f"loaded reference agent from {args.reference_model} (KL coef {args.kl_coef})")
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    # attacks
    if args.sn_outer:
        poison_batch = multi_value_poison
        poison = multi_value_poison
        bufferman = BufferMan_Simple(
            trigger=poison,
            target=args.target_action,
            dist=Discrete(-1 * args.rew_p, args.rew_p),
            p_rate=args.p_rate,
            alpha=args.alpha,
            simple=args.simple_select,
        )

    if args.trojdrl or args.badrl:
        poison_batch = multi_value_poison
        poison = multi_value_poison

        if args.trojdrl:
            middleman = DeterministicMiddleMan(
                trigger=poison,
                target=args.target_action,
                dist=Discrete(-1 * args.rew_p, args.rew_p),
                total=args.total_timesteps,
                budget=args.total_timesteps * args.p_rate,
            )
        else:
            q_net_adv = QNetwork(envs)
            q_net_adv.load_state_dict(torch.load("dqn_models/col_env_dqn/dqn.cleanrl_model", map_location="cpu"))
            q_net_adv.to(device)
            middleman = BadRLMiddleMan(
                poison,
                args.target_action,
                Discrete(-1 * args.rew_p, args.rew_p),
                args.p_rate,
                q_net_adv,
                args.strong,
            )

    save_every = max(1, args.num_iterations // 10)

    for iteration in range(1, args.num_iterations + 1):
        if args.save_model and iteration % save_every == 0:
            model_path = f"runs/col_env_{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            poison_action = None
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            poisoned = False

            # trojdrl / badrl poisoning during rollout
            if (args.trojdrl or args.badrl) and asr < 1:
                poison_index = 0
                poisoned, _, poison_action = middleman.time_to_poison(obs[step])
                if poisoned:
                    poison_obs = middleman.obs_poison(next_obs[poison_index])
                    obs[step][poison_index] = poison_obs
                    next_obs[poison_index] = poison_obs
                    total_poisoned += 1

            if not torch.isfinite(next_obs).all():
                print("Non-finite next_obs at global_step", global_step)
                print(next_obs)
                raise RuntimeError("next_obs contains NaN/Inf")

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                if poison_action is not None and poisoned:
                    action[poison_index] = poison_action
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            if (args.trojdrl or args.badrl) and poisoned:
                old_reward = float(reward[poison_index])
                reward[poison_index] = middleman.reward_poison(action[poison_index])
                total_perturb += abs(old_reward - reward[poison_index])

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device).view(-1)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(next_done, dtype=torch.float32, device=device)

            if "final_info" in infos:
                final_info = infos["final_info"]
                if final_info and "episode" in final_info:
                    print(f"global_step={global_step}, episodic_return={final_info['episode']['r']}", end="\r")
                    writer.add_scalar("charts/episodic_return", final_info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", final_info["episode"]["l"], global_step)

        # snapshot un-poisoned observations BEFORE bufferman mutates them
        # (used by --reference_model KL anchor, evaluated on clean states)
        obs_clean_snapshot = obs.detach().clone() if ref_agent is not None else None

        # sleepernets outer poison after rollout
        with torch.no_grad():
            if args.sn_outer and asr < 1:
                for i in range(args.num_envs):
                    _, _, indices, pert = bufferman(
                        obs[:, i], actions[:, i], rewards[:, i], values[:, i], logprobs[:, i], args.gamma, agent
                    )
                    total_perturb += pert
                    total_poisoned += len(indices)

        # bootstrap
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_obs_clean = (obs_clean_snapshot.reshape((-1,) + envs.single_observation_space.shape)
                       if obs_clean_snapshot is not None else None)

        # optimize
        b_inds = np.arange(batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # anchor clean behavior to a frozen reference (e.g. benign baseline)
                if ref_agent is not None and b_obs_clean is not None and args.kl_coef > 0:
                    mb_clean = b_obs_clean[mb_inds]
                    with torch.no_grad():
                        ref_probs = ref_agent.get_action_dist(mb_clean)
                    cur_probs = agent.get_action_dist(mb_clean)
                    anchor_kl = (cur_probs * (torch.log(cur_probs + 1e-12) - torch.log(ref_probs + 1e-12))).sum(dim=1).mean()
                    loss = loss + args.kl_coef * anchor_kl

                if not torch.isfinite(loss):
                    print("Non-finite loss at global_step", global_step)
                    print("pg_loss:", pg_loss.item())
                    print("v_loss:", v_loss.item())
                    print("entropy_loss:", entropy_loss.item())
                    raise RuntimeError("Loss became non-finite")

                optimizer.zero_grad()
                loss.backward()

                for name, p in agent.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        print(f"Non-finite gradient in {name} at global_step {global_step}")
                        raise RuntimeError("Gradient became non-finite")

                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                for name, p in agent.named_parameters():
                    if not torch.isfinite(p).all():
                        print(f"Non-finite parameter in {name} after optimizer step at global_step {global_step}")
                        raise RuntimeError("Parameter became non-finite")

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # logs
        writer.add_scalar("other/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("other/SPS", int(global_step / max(1e-6, (time.time() - start_time))), global_step)

        # ASR
        with torch.no_grad():
            if args.sn_outer and iteration % 4 == 0:
                poisoned_obs = bufferman.trigger(b_obs)
                probs = agent.get_action_dist(poisoned_obs)
                asr = probs[:, args.target_action].mean().item()
                writer.add_scalar("charts/AttackSuccessRate", asr, global_step)
                clean_probs = agent.get_action_dist(b_obs)
                leak = clean_probs[:, args.target_action].mean().item()
                discrim = asr - leak
                writer.add_scalar("charts/Leak_P_target_clean", leak, global_step)
                writer.add_scalar("charts/Discrim_ASR_minus_Leak", discrim, global_step)
                if total_poisoned != 0:
                    writer.add_scalar("charts/reward_perturb_average", total_perturb / (total_poisoned * 2), global_step)
                writer.add_scalar("charts/reward_perturb_global", total_perturb / global_step, global_step)
                writer.add_scalar("charts/poisoning_rate", total_poisoned / global_step, global_step)
                if args.save_model and asr > best_asr:
                    best_asr = asr
                    best_path = f"runs/col_env_{run_name}/{args.exp_name}_best_asr.cleanrl_model"
                    torch.save(agent.state_dict(), best_path)
                    print(f"new best ASR {best_asr:.4f} at step {global_step}, saved {best_path}")
                if args.save_model and discrim > best_discrim:
                    best_discrim = discrim
                    best_path = f"runs/col_env_{run_name}/{args.exp_name}_best_discrim.cleanrl_model"
                    torch.save(agent.state_dict(), best_path)
                    print(f"new best DISCRIM {best_discrim:.4f} (ASR={asr:.4f}, Leak={leak:.4f}) at step {global_step}, saved {best_path}")

            if (args.trojdrl or args.badrl) and iteration % 4 == 0:
                poisoned_obs = poison_batch(b_obs)
                probs = agent.get_action_dist(poisoned_obs)
                asr = probs[:, args.target_action].mean().item()
                writer.add_scalar("charts/AttackSuccessRate", asr, global_step)
                if total_poisoned != 0:
                    writer.add_scalar("charts/reward_perturb_average", total_perturb / total_poisoned, global_step)
                writer.add_scalar("charts/reward_perturb_global", total_perturb / global_step, global_step)
                writer.add_scalar("charts/poisoning_rate", total_poisoned / global_step, global_step)
                if args.save_model and asr > best_asr:
                    best_asr = asr
                    best_path = f"runs/col_env_{run_name}/{args.exp_name}_best_asr.cleanrl_model"
                    torch.save(agent.state_dict(), best_path)
                    print(f"new best ASR {best_asr:.4f} at step {global_step}, saved {best_path}")

            if not os.path.exists("images"):
                os.makedirs("images")

            plt.figure(dpi=150)
            plt.hist(b_actions.detach().cpu().numpy())
            plt.savefig(f"images/{run_name}.png")
            plt.close()

    envs.close()
    writer.close()
    wandb.finish()

    model_path = f"runs/col_env_{run_name}/{args.exp_name}.cleanrl_model"
    torch.save(agent.state_dict(), model_path)
    print(f"model saved to {model_path}")