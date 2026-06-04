from gymnasium.envs.registration import register

register(
     id="ColOfflineEnv",
     entry_point="environment.col_env:ColOfflineEnv",
     max_episode_steps=480,
)