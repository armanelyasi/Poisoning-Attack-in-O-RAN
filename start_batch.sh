#!/bin/bash


### Note that this file has some issues due to the bash process concurrency.
# A good idea might be to create a way in python to the same batch collection


# Define the list of possible values for each parameter
p_rate_list=(0.005 0.01 0.1 0.25)
rew_p_list=(0.1 0.2 0.5)
target_action_list=(0)
algorithm_list=("sn_outer" "trojdrl")  # "sn_inner"
alpha_list=(0 0.5 1)

# Other parameters that stay constant
num_envs=1
total_timesteps=15000
script_path="src/sleeper_nets/ppo.py"
additional_flags="--strong --seed 59911010"
i=0
max_parallel=10
current_jobs=0

# Function to execute a command and track parallelism
execute_command() {
  cmd=$1
  echo "Executing: $cmd"
  $cmd &
  current_jobs=$((current_jobs + 1))
  
  if [[ $current_jobs -ge $max_parallel ]]; then
    wait -n  # Wait for at least one background job to finish
    current_jobs=$((current_jobs - 1))
  fi
}

# Loop through all combinations of the parameters
for algorithm in "${algorithm_list[@]}"; do
  for p_rate in "${p_rate_list[@]}"; do
    for rew_p in "${rew_p_list[@]}"; do
      for target_action in "${target_action_list[@]}"; do
        if [[ $algorithm == "sn_outer" ]]; then
          # Add an additional loop for alpha when algorithm is sn_outer
          for alpha in "${alpha_list[@]}"; do
            # Construct the command
            cmd="python3 $script_path --${algorithm} --p_rate $p_rate --rew_p $rew_p --target_action $target_action --alpha $alpha --num_envs $num_envs --total_timesteps $total_timesteps $additional_flags"
            execute_command "$cmd"
            i=$((i + 1))
          done
        else
          # Construct the command without alpha for other algorithms
          cmd="python3 $script_path --${algorithm} --p_rate $p_rate --rew_p $rew_p --target_action $target_action --num_envs $num_envs --total_timesteps $total_timesteps $additional_flags"
          execute_command "$cmd"
          i=$((i + 1))
        fi
      done
    done
  done
done

# Run a single execution with algorithm="benign"
benign_cmd="python3 $script_path --benign --num_envs $num_envs --total_timesteps $total_timesteps $additional_flags"
execute_command "$benign_cmd"
i=$((i + 1))

# Wait for all background jobs to finish
wait


# set 1 vs set 2 of input parameters

# Print total runs executed
echo "Total runs executed: $i"
