from typing import Dict, Any, Optional
import argparse

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env
import pandas as pd
# from utils.transformer import Transformer

from utils.constants import DATASET_PATH, columns_reward, columns_action, columns_state
from etl.extractor import load_dataframe


# slice 0 prendiamo valore target di thp e chiediamo all'agent to soddisfare il thp minimo 
# poi l'adversarial deve essere in grado di spostare l'agent su un thp che non lo soddisfa
# thp maximization vs thp satisfaction
# Ravis check conditions for zeros

class ColOfflineEnv(gym.Env):
    metadata = {'render_modes': ['ansi']}
    df: pd.DataFrame = None
    state = None
    columns_reward = columns_reward
    columns_action = columns_action
    columns_state = columns_state
    
    action_converter = {}

    def __init__(self, data: pd.DataFrame = None, max_steps: int = 100, **kwargs):
        """
        Initialize the environment.

        :param data: The dataset for the environment.
        :param max_steps: Maximum steps allowed per episode before truncation.
        """
        self.isopen = True
        self.df = data
        self.max_steps = max_steps  # Set maximum steps per episode

        low_bounds = self.df[self.columns_state].min().to_numpy()
        high_bounds = self.df[self.columns_state].max().to_numpy()
        
        unique_values = sorted(self.df['slice_prb'].unique())

        # Create mapping dictionaries
        self.value_to_discrete = {val: idx for idx, val in enumerate(unique_values)}
        self.discrete_to_value = {idx: val for val, idx in self.value_to_discrete.items()}

        # Pre-group rows by slice_prb so step() doesn't rescan the whole frame
        self._rows_by_prb = {val: self.df[self.df['slice_prb'] == val] for val in unique_values}

        self.action_space = spaces.Discrete(len(unique_values))
        self.observation_shape = (len(columns_state),)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds,
                                            shape=self.observation_shape, dtype=np.float64)
        self.done = False

        # Initialize episodic tracking
        self.episodic_return = 0.0
        self.episodic_length = 0

    def map_discrete_to_slice_prb(self, discrete_value):
        """Convert discrete action to actual slice_prb value."""
        return self.discrete_to_value[discrete_value]

    def map_slice_prb_to_discrete(self, slice_prb_value):
        """Convert slice_prb value to discrete action."""
        return self.value_to_discrete[slice_prb_value]

    def step(self, action):
        slice_prb = self.map_discrete_to_slice_prb(action)

        candidate_state = self._rows_by_prb.get(slice_prb)

        if candidate_state is None or candidate_state.empty:
            print(f"[{slice_prb}] no data with this action chosen, outputting random sample...")
            candidate_state = self.df.sample(1)
        else:
            candidate_state = candidate_state.sample(1)

        self.state = candidate_state

        # Compute reward and update episodic tracking
        reward = self._compute_reward()
        self.episodic_return += reward
        self.episodic_length += 1

        # Check for truncation
        self.done = self.episodic_length >= self.max_steps

        info = {}
        if self.done:
            info["final_info"] = {
                "episode": {
                    "r": self.episodic_return,
                    "l": self.episodic_length
                }
            }

        return self._get_obs(), reward, self.done, self.done, info

    def reset(self, *, seed: Optional[int] = None, options: Dict[str, Any] = None, return_info: bool = False):
        super().reset(seed=seed)
        self.done = False

        # Reset episodic tracking
        self.episodic_return = 0.0
        self.episodic_length = 0

        self.state = self.df.sample(1)
        return (self._get_obs(), self.render()) if return_info else (self._get_obs(), {})

    def _get_obs(self):
        return np.array(self.state[self.columns_state].values).flatten()

    def render(self, mode="ansi"):
        if mode not in self.metadata['render_modes']:
            print(f"{mode} is not available.")

        if mode == "ansi":
            print(f"{self.state}")

        return {'isopen': self.isopen}

    def _compute_reward(self): 
        return float(self.state[columns_reward].iloc[0, 0])

    def close(self):
        self.isopen = False

if __name__ == '__main__':
    #######################
    # Parse data #
    #######################
    parser = argparse.ArgumentParser(description="Select the environment")
    args = parser.parse_args()

    pd.set_option("display.max_rows", None, "display.max_columns", 14)
    pd.set_option('expand_frame_repr', False)

    print('Test Environment')
    env = ColOfflineEnv(load_dataframe(use_saved=True))

    # env.reset()
    # env.step(2)
    # print(env.map_discrete_to_slice_prb(0))
    
    check_env(env)