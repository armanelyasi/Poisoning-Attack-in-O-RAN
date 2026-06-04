import argparse
import uuid
from stable_baselines3 import PPO
import wandb
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from wandb.integration.sb3 import WandbCallback
from etl.extractor import load_prediction_merged_dataframe
from utils.constants import DATASET_PATH

import gymnasium as gym
from utils.constants import WANDB_API_KEY

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform training")
    parser.add_argument('poisoned', action='store_true', default=False, help='Set whether the training should be poisoned')

    args = parser.parse_args()

    poisoned = args.poisoned

    wandb.login(key=WANDB_API_KEY)
    
    env = gym.make('NS3OfflineEnv', data=load_prediction_merged_dataframe(base_path=DATASET_PATH),
                                        use_saved_scaler=False, poisoned=poisoned)

    env = Monitor(env)
    verbosity = 2

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 1200000
    }
    
    name_wandb = f'agent-{str(uuid.uuid4()).split("-")[0]}'
    if poisoned:
        name_wandb += '-poisoned'

    run = wandb.init(project="adversarial", config=config, sync_tensorboard=True, monitor_gym=True, name=name_wandb, magic=True)

    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=3000, verbose=verbosity)
    eval_callback = EvalCallback(env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=verbosity)
    wandb_callback = WandbCallback(gradient_save_freq=100, model_save_freq=100, model_save_path=f"wandb/wandb_callback/{name_wandb}/{run.id}", verbose=verbosity)
   
    model = PPO(policy=config["policy_type"], env=env, verbose=verbosity, tensorboard_log=f"wandb/runs/{run.id}")
    model.learn(total_timesteps=config["total_timesteps"], progress_bar=True, callback=[wandb_callback, eval_callback])

    run.finish()
    wandb.finish()
    