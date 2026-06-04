# SleeperNets Code in Rough Research Form

First install requirements for cleanrl-atari https://docs.cleanrl.dev/

Run the following for SleeperNets:
python3 src/sleeper_nets/ppo.py --sn_outer --num_envs 1  --p_rate .0003 --target_action 2

Run the following for TrojDRL:
python3 src/sleeper_nets/ppo.py --trojdrl --num_envs 1 --p_rate .0003 --target_action 2

Run the following for BadRL:
python3 src/sleeper_nets/ppo.py --badrl --strong --num_envs 1  --p_rate .0003 --target_action 2