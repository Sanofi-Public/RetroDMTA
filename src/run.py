# Utils
import os
import time
import json
import argparse
import shutil

# Data
import pandas as pd

# Custom
from experiment import Experiment
from utils import get_time, process_strategy_name, get_bs_and_strat

def load_combined_config(global_config_path, run_config_path):
    # Load global configuration
    with open(global_config_path, 'r') as file:
        global_config = json.load(file)

    # Load run-specific configuration
    with open(run_config_path, 'r') as file:
        run_config = json.load(file)

    # Combine configurations, with run-specific config superseding global config
    combined_config = {**global_config, **run_config}
    return combined_config


def main(global_config_path, run_config_path):
    # Load the combined configuration
    config_data = load_combined_config(global_config_path, run_config_path)
    datasets = config_data['DATASETS']
    strategies_and_replica_list = config_data['STRATEGIES']
    batch_sizes = config_data['BATCH_SIZE']
    strategies_and_replica_dict = {k: v for d in strategies_and_replica_list for k, v in d.items()}
    strategies = strategies_and_replica_dict.keys()

    strategy_types = config_data['STRATEGY_TYPE']

    for batch_size in batch_sizes:
        # Iterate over each dataset
        for dataset in datasets:
            try:
                print(f"\n[{get_time()}]  ➡️   Dataset {dataset}")
                initial_date = config_data[dataset]['initial_date']
                final_date = config_data[dataset]['final_date']
                timestep = config_data[dataset]['timestep']
                

                # Iterate over each strategy
                for strategy in strategies:

                    strategy_name, greediverse_lambda, greediverse_threshold, ratio_epsilon, desired_diversity_threshold = process_strategy_name(strategy)

                    try:

                        for strategy_type in strategy_types:

                            try:                           
                                # Initialize a DataFrame to log results
                                log_df = pd.DataFrame(columns=['replicate', 'iteration', 'time'])
                                
                                # Set number of replicates based on strategy
                                n_replicates = strategies_and_replica_dict[strategy]
                                
                                # Iterate over each replicate
                                for replicate in range(1, n_replicates+1):
                                    fail = False
                                    try:
                                        t_init_replicate = time.time()  # Start timing the strategy execution

                                        # Initialize the Retrospective experiment
                                        model_log = 'MODEL' + '_' + config_data['EXPERIMENT_PARAMS']['model_name'] + '_' + config_data['EXPERIMENT_PARAMS']['uncertainty_strategy']
                                        batchsize_log = ('BS' + '_' + str(batch_size)).replace('.', '')
                                        log_name = f'{batchsize_log}-{model_log}-{strategy_type}-{strategy}'
                                        log_base_path = f'../experiments/{dataset}/{log_name}/'

                                        exp = Experiment(
                                            experiment_id = get_time(), 
                                            dataset=dataset,
                                            initial_date = initial_date,
                                            final_date = final_date,
                                            data_path = f'../data/{dataset}/data.csv', 
                                            blueprint_path = f'../data/{dataset}/blueprint.csv', 
                                            log_path = os.path.join(log_base_path, f'replicate_{replicate}'),
                                            timestep = timestep, 
                                            greediverse__lambda = greediverse_lambda,
                                            greediverse__threshold = greediverse_threshold,
                                            ratio__epsilon = ratio_epsilon,
                                            desired_diversity__threshold = desired_diversity_threshold,
                                            **config_data['EXPERIMENT_PARAMS']
                                        )
                                        
                                        # Iterate over each iteration in the experiment
                                        iterations = 0
                                        while exp.current_date < pd.to_datetime(final_date):
                                            try:
                                                t0 = time.time()  # Start timing the iteration
                                                iterations += 1
                                                # Run the experiment steps
                                                exp.get_next_data()
                                                exp.setup_training()
                                                exp.start_training()
                                                exp.process_pool()

                                                # Select molecules with the given strategy
                                                selection_strategies, batch_sizes = get_bs_and_strat(batch_size, strategy_type, strategy_name, exp.iteration, exp.batchsize_per_iteration)
                                                exp.select_molecules(selection_strategies=selection_strategies, batch_sizes=batch_sizes)
                                                
                                                # Log the time taken for this iteration
                                                log_df = pd.concat([log_df, pd.DataFrame({
                                                    'replicate': [replicate],
                                                    'iteration': [iterations],
                                                    'time': [round(time.time() - t0)]
                                                })], ignore_index=True)
                                                
                                                # Save the log to CSV
                                                log_df.to_csv(os.path.join(log_base_path, 'logs.csv'), index=False)

                                            except Exception as e:
                                                fail = True
                                                print(f"[{get_time()}]    🚨  Error during iteration {iterations} of replicate {replicate}: {e}")
                                        
                                        if fail:
                                            print(f"[{get_time()}]    ❌  Strategy {strategy} (replicate {replicate}/{n_replicates}) failed") 
                                            shutil.rmtree(os.path.join(log_base_path, f'replicate_{replicate}'))
                                        else:
                                            print(f"[{get_time()}]    ✅  Strategy {strategy} (replicate {replicate}/{n_replicates}) done in {round(time.time()-t_init_replicate)}s") 
                                    except Exception as e:
                                        print(f"[{get_time()}]    🚨  Error during replicate {replicate} for strategy {strategy}: {e}") 
                            except Exception as e:
                                print(f"[{get_time()}]    🚨  Error during strategy {strategy}: {e}") 
                    except Exception as e: 
                        print(f"[{get_time()}]    🚨  Error during strategy_type {strategy_type}: {e}")        
            except Exception as e:
                print(f"[{get_time()}]    🚨  Error during dataset {dataset}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the dataset processing script.')
    parser.add_argument('run_config', type=str, help='Path to the run-specific configuration file')
    parser.add_argument('global_config', type=str, default='../data/common/datasets_config.json', help='Path to the global configuration file (default: ../data/common/datasets_config.json)')
    args = parser.parse_args()

    main(args.global_config, args.run_config)