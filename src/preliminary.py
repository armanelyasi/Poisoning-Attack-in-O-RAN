import argparse
import json
import os
import tarfile
import random
import sem
from utils.constants import DATASET_PATH

def launch_campaign(scenario_configuration, runs, base_rng_run, max_parallel_processes, ns3_script, output_directory):
    try:
        with open(scenario_configuration) as params_file:
            data = params_file.read()
    except FileNotFoundError:
        print(f"Cannot open '{scenario_configuration}' file, exiting")
        exit(-1)

    #######################
    # Create the campaign #
    #######################

    ns_path = "/workspace/ns3-mmwave-oran"
    print(f"Path of ns-3 is {ns_path}")

    # These are the available parameters
    # We specify each parameter as an array containing the desired values
    params = json.loads(data)

    campaign_dir = os.path.join(output_directory, 'extracted' , f'out_{str(base_rng_run)}')
    print(f"Output path is {campaign_dir}")

    campaign = sem.CampaignManager.new(ns_path, ns3_script, campaign_dir, runner_type='ParallelRunner', overwrite=True,
                                       check_repo=False, skip_configuration=True, max_parallel_processes=max_parallel_processes)

    print(campaign)  # This prints out the campaign settings
    params['RngRun'] = [base_rng_run + run for run in range(runs)]

    print("---------- Parameters used ------------")
    print(params)
    print("---------------------------------------")

    ###################
    # Run simulations #
    ###################

    combinations = sem.list_param_combinations(params)

    campaign.run_simulations(combinations, stop_on_errors=False)

    # print("Starting building the error tracer... ", end='')
    # error_checker(campaign_dir)
    # print("json built")
    if os.path.isfile('sem_errors.log'):
        print("---------------------------------------")
        print('Some errors on the assignment of classes happened.')
        print('Including the debug file in the archive... ', end='', flush=True)
        os.rename('sem_errors.log', campaign_dir + '/' + 'sem_errors.log')
        print('Done')
        print("---------------------------------------")

    scenario_configuration: str = scenario_configuration.replace(os.sep, '_').replace('.json', '')
    output_file = f"out_{scenario_configuration}_{str(runs)}_{str(base_rng_run)}.tar.gz"
    output_path = os.path.join(output_directory, 'compressed', output_file)
    with tarfile.open(output_path, "w:gz", compresslevel=5) as tar:
        tar.add(campaign_dir, arcname=os.path.basename(campaign_dir))


def main(args):
    scenario_configuration = args.scenario_configuration

    if args.runs:
        runs = args.runs
        print(f"Number of runs: {str(runs)}")
    else:
        runs = 5
        print(f"Number of runs not passed, using the default ({str(runs)})")

    if args.base_rng_run:
        base_rng_run = args.base_rng_run
        print(f"Base seed: {str(base_rng_run)}")
    else:
        base_rng_run = random.randint(10000,90000)
        print(f"Base seed not passed, using a random one ({str(base_rng_run)})")

    if args.max_parallel_processes:
        max_parallel_processes = args.max_parallel_processes
        print(f"Max parallel processes: {str(max_parallel_processes)}")
    else:
        max_parallel_processes = 30
        print(f"Max parallel processes not passed, using the default ({str(max_parallel_processes)})")

    if args.ns3_script:
        ns3_script = args.ns3_script
        print(f"Script to be run: {str(ns3_script)}")
    else:
        ns3_script = 'scenario-six'
        print(f"Script to be run, using the default ({ns3_script})")

    launch_campaign(scenario_configuration, runs, base_rng_run, max_parallel_processes, ns3_script, DATASET_PATH)

if __name__ == "__main__":
    #######################
    # Parse data #
    #######################
    parser = argparse.ArgumentParser(
        description="Provide path to configuration and the number of runs and the base seed to start with")
    parser.add_argument('scenario_configuration', type=str, help='Relative path to the scenario configuration file')
    parser.add_argument('runs', type=int, nargs='?', help='Number of run for each single configuration')
    parser.add_argument('max_parallel_processes', type=int, nargs='?', help='Number of maximum parallel processes')
    parser.add_argument('base_rng_run', type=int, nargs='?', help='Base seed')
    parser.add_argument('ns3_script', type=str, nargs='?', help='Name of the ns-3 script to be run')

    main(parser.parse_args())