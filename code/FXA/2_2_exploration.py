# Utils
import os
import glob
from dateutil.relativedelta import relativedelta
from tqdm.auto import tqdm
from natsort import natsorted
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import time

# Data
import numpy as np
import pandas as pd

from utils import load_common_config, load_pickle, get_kde, compute_TMAPOverlap, compute_BMScaffoldCoverage
from utils import compute_TMAPOverlap, compute_DistanceCoverage, compute_DistanceCoverageAUC, compute_FunctionalGroupsCoverage
from utils import compute_Diversity, compute_SumDiversity, compute_Diameter, compute_SumDiameter, compute_Bottleneck, compute_SumBottleneck
from utils import compute_Circles, compute_RingSystemsCoverage, get_time
from utils import get_pmas_palette
pmas = get_pmas_palette()

use_tqdm = False

t_init = time.time()

# Determine the dataset name from the current working directory
DATASET = os.getcwd().split('/')[-1]

# Define base paths for dataset
data_path = f'../../data/{DATASET}'
figures_path = f'../../figures/{DATASET}'

# Load common config file
config = load_common_config(f'../../data/common/datasets_config.json')

# Extract date parameters from config
INITIAL_DATE = pd.to_datetime(config[DATASET]['initial_date'])
FINAL_DATE = pd.to_datetime(config[DATASET]['final_date'])
TIMESTEP = config[DATASET]['timestep']

tmap_df = pd.read_csv(f'{data_path}/data_aggregated.csv')
tmap_df['DATE'] = pd.to_datetime(tmap_df['DATE'])

tmap_df['iteration'] = 0
iteration = 0
current_date = INITIAL_DATE
while current_date < FINAL_DATE:
    iteration += 1
    next_date = current_date + relativedelta(months=TIMESTEP)
    next_smiles = tmap_df[(tmap_df['DATE'] >= current_date) & (tmap_df['DATE'] < next_date)]['SMILES'].unique()
    tmap_df.loc[tmap_df['SMILES'].isin(next_smiles), 'iteration'] = iteration
    current_date = next_date

top_molecules = load_pickle(f'{data_path}/top_molecules.pkl')
top_categories = list(top_molecules['geometric'].keys())

top_molecules = load_pickle(f'{data_path}/top_molecules.pkl')
tmap_coordinates = load_pickle(f'{data_path}/tmap_coordinates.pkl')
similarity_matrix = pd.read_parquet(f'{data_path}/similarity_matrix.parquet')
distance_matrix = pd.read_parquet(f'{data_path}/distance_matrix.parquet')

all_smiles = list(distance_matrix.columns)

x = tmap_coordinates['x']
y = tmap_coordinates['y']
s = tmap_coordinates['s']
t = tmap_coordinates['t']

tmap_df['x'] = x
tmap_df['y'] = y

project_kde = get_kde(x, y, binary=True)

all_strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')
if os.path.exists(f'{data_path}/exploration_distance_coverage.parquet'):
    results_df = pd.read_parquet(f'{data_path}/exploration_distance_coverage.parquet')
    strategy_paths = []
    for strategy_path in all_strategy_paths:
        strategy_name = strategy_path.split('/')[-1]
        if strategy_name not in results_df['strategy'].unique():
            strategy_paths.append(strategy_path)
else:
    results_df = pd.DataFrame()
    strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')

strategy_paths = natsorted(strategy_paths)

print(f"[{get_time()}]  ⚙️  Exploration (Neighborhood Coverage) : {len(strategy_paths)} experiments to process (out of {len(all_strategy_paths)})")

for threshold in tqdm(np.arange(0., 1.05, 0.05), leave=False, desc='Similarity thresholds', disable=not use_tqdm):

    for strategy_path in tqdm(strategy_paths, leave=False, desc='Strategies', disable=not use_tqdm):

        strategy_name = strategy_path.split('/')[-1]
        replicate_paths = natsorted(glob.glob(os.path.join(strategy_path, f'replicate*')))
        nb_replicate = len(replicate_paths)

        for replicate_path in tqdm(replicate_paths, leave=False, desc='Replicates', disable=not use_tqdm):

            replicate_id = int(replicate_path.split('_')[-1])

            iteration_paths = natsorted(glob.glob(os.path.join(replicate_path, '*')))
            train_df = pd.read_csv(os.path.join(iteration_paths[-1], 'train_df.csv'))
            last_df = pd.read_csv(os.path.join(iteration_paths[-1], 'selected_train_df.csv'))
            df = pd.concat([train_df, last_df], axis=0).sort_values(by=['iteration'])

            df = df.merge(tmap_df[['SMILES', 'x', 'y', 'DATE']], on='SMILES').sort_values(by=['iteration', 'DATE'])

            distance_coverage_scores = []
            proportions_all_selected = []
            iterations = []

            for iteration in range(0, df['iteration'].max() + 1):
                smiles = df[df['iteration'] <= iteration]['SMILES'].values
                distance_coverage_score = compute_DistanceCoverage(all_smiles, smiles, distance_matrix, threshold=threshold)
                proportion_all_selected = len(smiles)/len(tmap_df)

                distance_coverage_scores.append(distance_coverage_score)
                proportions_all_selected.append(proportion_all_selected)
                iterations.append(iteration)

            tmp_df = pd.DataFrame()
            tmp_df['iterations'] = iterations
            tmp_df['exploration_metric'] = distance_coverage_scores
            tmp_df['proportions_all_selected'] = proportions_all_selected
            tmp_df['strategy'] = strategy_name
            tmp_df['replicate'] = replicate_id
            tmp_df['distance_threshold'] = threshold
            
            results_df = pd.concat([results_df, tmp_df])

results_df.to_parquet(f'{data_path}/exploration_neighborhood_coverage.parquet')

### Neighborhood Coverage AUC
all_strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')
if os.path.exists(f'{data_path}/exploration_neighborhood_coverage_auc.parquet'):
    results_df = pd.read_parquet(f'{data_path}/exploration_neighborhood_coverage_auc.parquet')
    strategy_paths = []
    for strategy_path in all_strategy_paths:
        strategy_name = strategy_path.split('/')[-1]
        if strategy_name not in results_df['strategy'].unique():
            strategy_paths.append(strategy_path)
else:
    results_df = pd.DataFrame()
    strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')

strategy_paths = natsorted(strategy_paths)

print(f"[{get_time()}]  ⚙️  Exploration (Neighborhood Coverage AUC) : {len(strategy_paths)} experiments to process (out of {len(all_strategy_paths)})")

for strategy_path in tqdm(strategy_paths, leave=False, desc='Strategies', disable=not use_tqdm):

    strategy_name = strategy_path.split('/')[-1]
    replicate_paths = natsorted(glob.glob(os.path.join(strategy_path, f'replicate*')))
    nb_replicate = len(replicate_paths)

    for replicate_path in tqdm(replicate_paths, leave=False, desc='Replicates', disable=not use_tqdm):

        replicate_id = int(replicate_path.split('_')[-1])

        iteration_paths = natsorted(glob.glob(os.path.join(replicate_path, '*')))
        train_df = pd.read_csv(os.path.join(iteration_paths[-1], 'train_df.csv'))
        last_df = pd.read_csv(os.path.join(iteration_paths[-1], 'selected_train_df.csv'))
        df = pd.concat([train_df, last_df], axis=0).sort_values(by=['iteration'])

        df = df.merge(tmap_df[['SMILES', 'x', 'y', 'DATE']], on='SMILES').sort_values(by=['iteration', 'DATE'])

        distance_auc_scores = []
        proportions_all_selected = []
        iterations = []

        for iteration in range(0, df['iteration'].max() + 1):
            smiles = df[df['iteration'] <= iteration]['SMILES'].values
            distance_auc_score = compute_DistanceCoverageAUC(all_smiles, smiles, distance_matrix)
            proportion_all_selected = len(smiles)/len(tmap_df)

            distance_auc_scores.append(distance_auc_score)
            proportions_all_selected.append(proportion_all_selected)
            iterations.append(iteration)

        tmp_df = pd.DataFrame()
        tmp_df['iterations'] = iterations
        tmp_df['exploration_metric'] = distance_auc_scores
        tmp_df['proportions_all_selected'] = proportions_all_selected
        tmp_df['strategy'] = strategy_name
        tmp_df['replicate'] = replicate_id
        
        results_df = pd.concat([results_df, tmp_df])

results_df.to_parquet(f'{data_path}/exploration_neighborhood_coverage_auc.parquet')

### Internal Diversity
all_strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')
if os.path.exists(f'{data_path}/exploration_diversity.parquet'):
    results_df = pd.read_parquet(f'{data_path}/exploration_diversity.parquet')
    strategy_paths = []
    for strategy_path in all_strategy_paths:
        strategy_name = strategy_path.split('/')[-1]
        if strategy_name not in results_df['strategy'].unique():
            strategy_paths.append(strategy_path)
else:
    results_df = pd.DataFrame()
    strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')

strategy_paths = natsorted(strategy_paths)

print(f"[{get_time()}]  ⚙️  Exploration (Diversity) : {len(strategy_paths)} experiments to process (out of {len(all_strategy_paths)})")

for strategy_path in tqdm(strategy_paths, leave=False, desc='Strategies', disable=not use_tqdm):

    strategy_name = strategy_path.split('/')[-1]
    replicate_paths = natsorted(glob.glob(os.path.join(strategy_path, f'replicate*')))
    nb_replicate = len(replicate_paths)

    for replicate_path in tqdm(replicate_paths, leave=False, desc='Replicates', disable=not use_tqdm):

        replicate_id = int(replicate_path.split('_')[-1])

        iteration_paths = natsorted(glob.glob(os.path.join(replicate_path, '*')))
        train_df = pd.read_csv(os.path.join(iteration_paths[-1], 'train_df.csv'))
        last_df = pd.read_csv(os.path.join(iteration_paths[-1], 'selected_train_df.csv'))
        df = pd.concat([train_df, last_df], axis=0).sort_values(by=['iteration'])

        df = df.merge(tmap_df[['SMILES', 'x', 'y', 'DATE']], on='SMILES').sort_values(by=['iteration', 'DATE'])

        metrics = []
        proportions_all_selected = []
        iterations = []

        for iteration in range(0, df['iteration'].max() + 1):
            smiles = df[df['iteration'] <= iteration]['SMILES'].values

            small_distance_matrix = distance_matrix.loc[smiles, smiles].copy()
            metric = compute_Diversity(small_distance_matrix)
            proportion_all_selected = len(smiles)/len(tmap_df)

            metrics.append(metric)
            proportions_all_selected.append(proportion_all_selected)
            iterations.append(iteration)

        tmp_df = pd.DataFrame()
        tmp_df['iterations'] = iterations
        tmp_df['exploration_metric'] = metrics
        tmp_df['proportions_all_selected'] = proportions_all_selected
        tmp_df['strategy'] = strategy_name
        tmp_df['replicate'] = replicate_id
        
        results_df = pd.concat([results_df, tmp_df])

results_df.to_parquet(f'{data_path}/exploration_diversity.parquet')

### SumDiversity
all_strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')
if os.path.exists(f'{data_path}/exploration_sumdiversity.parquet'):
    results_df = pd.read_parquet(f'{data_path}/exploration_sumdiversity.parquet')
    strategy_paths = []
    for strategy_path in all_strategy_paths:
        strategy_name = strategy_path.split('/')[-1]
        if strategy_name not in results_df['strategy'].unique():
            strategy_paths.append(strategy_path)
else:
    results_df = pd.DataFrame()
    strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')

strategy_paths = natsorted(strategy_paths)

print(f"[{get_time()}]  ⚙️  Exploration (SumDiversity) : {len(strategy_paths)} experiments to process (out of {len(all_strategy_paths)})")

for strategy_path in tqdm(strategy_paths, leave=False, desc='Strategies', disable=not use_tqdm):

    strategy_name = strategy_path.split('/')[-1]
    replicate_paths = natsorted(glob.glob(os.path.join(strategy_path, f'replicate*')))
    nb_replicate = len(replicate_paths)

    for replicate_path in tqdm(replicate_paths, leave=False, desc='Replicates', disable=not use_tqdm):

        replicate_id = int(replicate_path.split('_')[-1])

        iteration_paths = natsorted(glob.glob(os.path.join(replicate_path, '*')))
        train_df = pd.read_csv(os.path.join(iteration_paths[-1], 'train_df.csv'))
        last_df = pd.read_csv(os.path.join(iteration_paths[-1], 'selected_train_df.csv'))
        df = pd.concat([train_df, last_df], axis=0).sort_values(by=['iteration'])

        df = df.merge(tmap_df[['SMILES', 'x', 'y', 'DATE']], on='SMILES').sort_values(by=['iteration', 'DATE'])

        metrics = []
        proportions_all_selected = []
        iterations = []

        for iteration in range(0, df['iteration'].max() + 1):
            smiles = df[df['iteration'] <= iteration]['SMILES'].values

            small_distance_matrix = distance_matrix.loc[smiles, smiles].copy()
            metric = compute_SumDiversity(small_distance_matrix)
            proportion_all_selected = len(smiles)/len(tmap_df)

            metrics.append(metric)
            proportions_all_selected.append(proportion_all_selected)
            iterations.append(iteration)

        tmp_df = pd.DataFrame()
        tmp_df['iterations'] = iterations
        tmp_df['exploration_metric'] = metrics
        tmp_df['proportions_all_selected'] = proportions_all_selected
        tmp_df['strategy'] = strategy_name
        tmp_df['replicate'] = replicate_id
        
        results_df = pd.concat([results_df, tmp_df])

results_df.to_parquet(f'{data_path}/exploration_sumdiversity.parquet')

### Diameter
all_strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')
if os.path.exists(f'{data_path}/exploration_diameter.parquet'):
    results_df = pd.read_parquet(f'{data_path}/exploration_diameter.parquet')
    strategy_paths = []
    for strategy_path in all_strategy_paths:
        strategy_name = strategy_path.split('/')[-1]
        if strategy_name not in results_df['strategy'].unique():
            strategy_paths.append(strategy_path)
else:
    results_df = pd.DataFrame()
    strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')

strategy_paths = natsorted(strategy_paths)

print(f"[{get_time()}]  ⚙️  Exploration (Diameter) : {len(strategy_paths)} experiments to process (out of {len(all_strategy_paths)})")

for strategy_path in tqdm(strategy_paths, leave=False, desc='Strategies', disable=not use_tqdm):

    strategy_name = strategy_path.split('/')[-1]
    replicate_paths = natsorted(glob.glob(os.path.join(strategy_path, f'replicate*')))
    nb_replicate = len(replicate_paths)

    for replicate_path in tqdm(replicate_paths, leave=False, desc='Replicates', disable=not use_tqdm):

        replicate_id = int(replicate_path.split('_')[-1])

        iteration_paths = natsorted(glob.glob(os.path.join(replicate_path, '*')))
        train_df = pd.read_csv(os.path.join(iteration_paths[-1], 'train_df.csv'))
        last_df = pd.read_csv(os.path.join(iteration_paths[-1], 'selected_train_df.csv'))
        df = pd.concat([train_df, last_df], axis=0).sort_values(by=['iteration'])

        df = df.merge(tmap_df[['SMILES', 'x', 'y', 'DATE']], on='SMILES').sort_values(by=['iteration', 'DATE'])

        metrics = []
        proportions_all_selected = []
        iterations = []

        for iteration in range(0, df['iteration'].max() + 1):
            smiles = df[df['iteration'] <= iteration]['SMILES'].values

            small_distance_matrix = distance_matrix.loc[smiles, smiles].copy()
            metric = compute_Diameter(small_distance_matrix)
            proportion_all_selected = len(smiles)/len(tmap_df)

            metrics.append(metric)
            proportions_all_selected.append(proportion_all_selected)
            iterations.append(iteration)

        tmp_df = pd.DataFrame()
        tmp_df['iterations'] = iterations
        tmp_df['exploration_metric'] = metrics
        tmp_df['proportions_all_selected'] = proportions_all_selected
        tmp_df['strategy'] = strategy_name
        tmp_df['replicate'] = replicate_id
        
        results_df = pd.concat([results_df, tmp_df])

results_df.to_parquet(f'{data_path}/exploration_diameter.parquet')

### SumDiameter
all_strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')
if os.path.exists(f'{data_path}/exploration_sumdiameter.parquet'):
    results_df = pd.read_parquet(f'{data_path}/exploration_sumdiameter.parquet')
    strategy_paths = []
    for strategy_path in all_strategy_paths:
        strategy_name = strategy_path.split('/')[-1]
        if strategy_name not in results_df['strategy'].unique():
            strategy_paths.append(strategy_path)
else:
    results_df = pd.DataFrame()
    strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')

strategy_paths = natsorted(strategy_paths)

print(f"[{get_time()}]  ⚙️  Exploration (SumDiameter) : {len(strategy_paths)} experiments to process (out of {len(all_strategy_paths)})")

for strategy_path in tqdm(strategy_paths, leave=False, desc='Strategies', disable=not use_tqdm):

    strategy_name = strategy_path.split('/')[-1]
    replicate_paths = natsorted(glob.glob(os.path.join(strategy_path, f'replicate*')))
    nb_replicate = len(replicate_paths)

    for replicate_path in tqdm(replicate_paths, leave=False, desc='Replicates', disable=not use_tqdm):

        replicate_id = int(replicate_path.split('_')[-1])

        iteration_paths = natsorted(glob.glob(os.path.join(replicate_path, '*')))
        train_df = pd.read_csv(os.path.join(iteration_paths[-1], 'train_df.csv'))
        last_df = pd.read_csv(os.path.join(iteration_paths[-1], 'selected_train_df.csv'))
        df = pd.concat([train_df, last_df], axis=0).sort_values(by=['iteration'])

        df = df.merge(tmap_df[['SMILES', 'x', 'y', 'DATE']], on='SMILES').sort_values(by=['iteration', 'DATE'])

        metrics = []
        proportions_all_selected = []
        iterations = []

        for iteration in range(0, df['iteration'].max() + 1):
            smiles = df[df['iteration'] <= iteration]['SMILES'].values

            small_distance_matrix = distance_matrix.loc[smiles, smiles].copy()
            metric = compute_SumDiameter(small_distance_matrix)
            proportion_all_selected = len(smiles)/len(tmap_df)

            metrics.append(metric)
            proportions_all_selected.append(proportion_all_selected)
            iterations.append(iteration)

        tmp_df = pd.DataFrame()
        tmp_df['iterations'] = iterations
        tmp_df['exploration_metric'] = metrics
        tmp_df['proportions_all_selected'] = proportions_all_selected
        tmp_df['strategy'] = strategy_name
        tmp_df['replicate'] = replicate_id
        
        results_df = pd.concat([results_df, tmp_df])

results_df.to_parquet(f'{data_path}/exploration_sumdiameter.parquet')

### Bottleneck
all_strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')
if os.path.exists(f'{data_path}/exploration_bottleneck.parquet'):
    results_df = pd.read_parquet(f'{data_path}/exploration_bottleneck.parquet')
    strategy_paths = []
    for strategy_path in all_strategy_paths:
        strategy_name = strategy_path.split('/')[-1]
        if strategy_name not in results_df['strategy'].unique():
            strategy_paths.append(strategy_path)
else:
    results_df = pd.DataFrame()
    strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')

strategy_paths = natsorted(strategy_paths)

print(f"[{get_time()}]  ⚙️  Exploration (Bottleneck) : {len(strategy_paths)} experiments to process (out of {len(all_strategy_paths)})")

for strategy_path in tqdm(strategy_paths, leave=False, desc='Strategies', disable=not use_tqdm):

    strategy_name = strategy_path.split('/')[-1]
    replicate_paths = natsorted(glob.glob(os.path.join(strategy_path, f'replicate*')))
    nb_replicate = len(replicate_paths)

    for replicate_path in tqdm(replicate_paths, leave=False, desc='Replicates', disable=not use_tqdm):

        replicate_id = int(replicate_path.split('_')[-1])

        iteration_paths = natsorted(glob.glob(os.path.join(replicate_path, '*')))
        train_df = pd.read_csv(os.path.join(iteration_paths[-1], 'train_df.csv'))
        last_df = pd.read_csv(os.path.join(iteration_paths[-1], 'selected_train_df.csv'))
        df = pd.concat([train_df, last_df], axis=0).sort_values(by=['iteration'])

        df = df.merge(tmap_df[['SMILES', 'x', 'y', 'DATE']], on='SMILES').sort_values(by=['iteration', 'DATE'])

        metrics = []
        proportions_all_selected = []
        iterations = []

        for iteration in range(0, df['iteration'].max() + 1):
            smiles = df[df['iteration'] <= iteration]['SMILES'].values

            small_distance_matrix = distance_matrix.loc[smiles, smiles].copy()
            metric = compute_Bottleneck(small_distance_matrix)
            proportion_all_selected = len(smiles)/len(tmap_df)

            metrics.append(metric)
            proportions_all_selected.append(proportion_all_selected)
            iterations.append(iteration)

        tmp_df = pd.DataFrame()
        tmp_df['iterations'] = iterations
        tmp_df['exploration_metric'] = metrics
        tmp_df['proportions_all_selected'] = proportions_all_selected
        tmp_df['strategy'] = strategy_name
        tmp_df['replicate'] = replicate_id
        
        results_df = pd.concat([results_df, tmp_df])

results_df.to_parquet(f'{data_path}/exploration_bottleneck.parquet')

### SumBottleneck
all_strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')
if os.path.exists(f'{data_path}/exploration_sumbottleneck.parquet'):
    results_df = pd.read_parquet(f'{data_path}/exploration_sumbottleneck.parquet')
    strategy_paths = []
    for strategy_path in all_strategy_paths:
        strategy_name = strategy_path.split('/')[-1]
        if strategy_name not in results_df['strategy'].unique():
            strategy_paths.append(strategy_path)
else:
    results_df = pd.DataFrame()
    strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')

strategy_paths = natsorted(strategy_paths)

print(f"[{get_time()}]  ⚙️  Exploration (SumBottleneck) : {len(strategy_paths)} experiments to process (out of {len(all_strategy_paths)})")

for strategy_path in tqdm(strategy_paths, leave=False, desc='Strategies', disable=not use_tqdm):

    strategy_name = strategy_path.split('/')[-1]
    replicate_paths = natsorted(glob.glob(os.path.join(strategy_path, f'replicate*')))
    nb_replicate = len(replicate_paths)

    for replicate_path in tqdm(replicate_paths, leave=False, desc='Replicates', disable=not use_tqdm):

        replicate_id = int(replicate_path.split('_')[-1])

        iteration_paths = natsorted(glob.glob(os.path.join(replicate_path, '*')))
        train_df = pd.read_csv(os.path.join(iteration_paths[-1], 'train_df.csv'))
        last_df = pd.read_csv(os.path.join(iteration_paths[-1], 'selected_train_df.csv'))
        df = pd.concat([train_df, last_df], axis=0).sort_values(by=['iteration'])

        df = df.merge(tmap_df[['SMILES', 'x', 'y', 'DATE']], on='SMILES').sort_values(by=['iteration', 'DATE'])

        metrics = []
        proportions_all_selected = []
        iterations = []

        for iteration in range(0, df['iteration'].max() + 1):
            smiles = df[df['iteration'] <= iteration]['SMILES'].values

            small_distance_matrix = distance_matrix.loc[smiles, smiles].copy()
            metric = compute_SumBottleneck(small_distance_matrix)
            proportion_all_selected = len(smiles)/len(tmap_df)

            metrics.append(metric)
            proportions_all_selected.append(proportion_all_selected)
            iterations.append(iteration)

        tmp_df = pd.DataFrame()
        tmp_df['iterations'] = iterations
        tmp_df['exploration_metric'] = metrics
        tmp_df['proportions_all_selected'] = proportions_all_selected
        tmp_df['strategy'] = strategy_name
        tmp_df['replicate'] = replicate_id
        
        results_df = pd.concat([results_df, tmp_df])

results_df.to_parquet(f'{data_path}/exploration_sumbottleneck.parquet')

### Circles
all_strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')
if os.path.exists(f'{data_path}/exploration_circles.parquet'):
    results_df = pd.read_parquet(f'{data_path}/exploration_circles.parquet')
    strategy_paths = []
    for strategy_path in all_strategy_paths:
        strategy_name = strategy_path.split('/')[-1]
        if strategy_name not in results_df['strategy'].unique():
            strategy_paths.append(strategy_path)
else:
    results_df = pd.DataFrame()
    strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')

strategy_paths = natsorted(strategy_paths)

print(f"[{get_time()}]  ⚙️  Exploration (Circles) : {len(strategy_paths)} experiments to process (out of {len(all_strategy_paths)})")

for threshold in tqdm(np.arange(0, 1.05, 0.05), leave=False, desc='Similarity thresholds', disable=not use_tqdm):
        
    for strategy_path in tqdm(strategy_paths, leave=False, desc='Strategies', disable=not use_tqdm):

        strategy_name = strategy_path.split('/')[-1]
        replicate_paths = natsorted(glob.glob(os.path.join(strategy_path, f'replicate*')))
        nb_replicate = len(replicate_paths)

        for replicate_path in tqdm(replicate_paths, leave=False, desc='Replicates', disable=not use_tqdm):

            replicate_id = int(replicate_path.split('_')[-1])

            iteration_paths = natsorted(glob.glob(os.path.join(replicate_path, '*')))
            train_df = pd.read_csv(os.path.join(iteration_paths[-1], 'train_df.csv'))
            last_df = pd.read_csv(os.path.join(iteration_paths[-1], 'selected_train_df.csv'))
            df = pd.concat([train_df, last_df], axis=0).sort_values(by=['iteration'])

            df = df.merge(tmap_df[['SMILES', 'x', 'y', 'DATE']], on='SMILES').sort_values(by=['iteration', 'DATE'])

            metrics = []
            proportions_all_selected = []
            iterations = []

            for iteration in range(0, df['iteration'].max() + 1):
                smiles = df[df['iteration'] <= iteration]['SMILES'].values

                small_distance_matrix = distance_matrix.loc[smiles, smiles].copy()
                metric = compute_Circles(small_distance_matrix, threshold)
                proportion_all_selected = len(smiles)/len(tmap_df)

                metrics.append(metric)
                proportions_all_selected.append(proportion_all_selected)
                iterations.append(iteration)

            tmp_df = pd.DataFrame()
            tmp_df['iterations'] = iterations
            tmp_df['exploration_metric'] = metrics
            tmp_df['proportions_all_selected'] = proportions_all_selected
            tmp_df['strategy'] = strategy_name
            tmp_df['replicate'] = replicate_id
            tmp_df['threshold'] = threshold
            
            results_df = pd.concat([results_df, tmp_df])

results_df.to_parquet(f'{data_path}/exploration_circles.parquet')

### Bemis-Murcko Scaffold
all_strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')
if os.path.exists(f'{data_path}/exploration_bm_scaffold_coverage.parquet'):
    results_df = pd.read_parquet(f'{data_path}/exploration_bm_scaffold_coverage.parquet')
    strategy_paths = []
    for strategy_path in all_strategy_paths:
        strategy_name = strategy_path.split('/')[-1]
        if strategy_name not in results_df['strategy'].unique():
            strategy_paths.append(strategy_path)
else:
    results_df = pd.DataFrame()
    strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')

strategy_paths = natsorted(strategy_paths)

print(f"[{get_time()}]  ⚙️  Exploration (Bemis-Murcko Scaffold Coverage) : {len(strategy_paths)} experiments to process (out of {len(all_strategy_paths)})")

def process_replicate(task):
    """
    Process one replicate: reads CSV files, merges data, and computes
    BM scaffold coverage for each iteration.
    
    Parameters:
        task (tuple): Contains:
            - strategy_path (str): Path to the strategy folder.
            - replicate_path (str): Path to the replicate folder.
            - tmap_df (pd.DataFrame): DataFrame with mapping information.
            - all_smiles (iterable): List/array of all SMILES strings.
    
    Returns:
        pd.DataFrame: A DataFrame with the computed metrics for each iteration.
    """
    strategy_path, replicate_path, tmap_df, all_smiles = task

    # Get the strategy name and replicate id.
    strategy_name = os.path.basename(strategy_path)
    replicate_id = int(os.path.basename(replicate_path).split('_')[-1])

    # Find all iteration folders in this replicate.
    iteration_paths = natsorted(glob.glob(os.path.join(replicate_path, '*')))

    # Read CSV files from the last iteration folder.
    train_csv = os.path.join(iteration_paths[-1], 'train_df.csv')
    selected_csv = os.path.join(iteration_paths[-1], 'selected_train_df.csv')
    train_df = pd.read_csv(train_csv)
    last_df = pd.read_csv(selected_csv)

    # Concatenate and sort the data.
    df = pd.concat([train_df, last_df], axis=0).sort_values(by=['iteration'])
    df = df.merge(tmap_df[['SMILES', 'x', 'y', 'DATE']], on='SMILES')
    df = df.sort_values(by=['iteration', 'DATE'])

    coverages = []
    proportions_all_selected = []
    iterations = []

    max_iter = int(df['iteration'].max())
    for iteration in range(max_iter + 1):
        # Get SMILES for all iterations up to the current iteration.
        smiles = df[df['iteration'] <= iteration]['SMILES'].values
        # Compute BM scaffold coverage.
        coverage = compute_BMScaffoldCoverage(all_smiles, smiles)
        # Compute the proportion of selected molecules.
        proportion_all_selected = len(smiles) / len(tmap_df)

        coverages.append(coverage)
        proportions_all_selected.append(proportion_all_selected)
        iterations.append(iteration)

    # Build a DataFrame with the results.
    tmp_df = pd.DataFrame({
        'iterations': iterations,
        'exploration_metric': coverages,
        'proportions_all_selected': proportions_all_selected,
        'strategy': strategy_name,
        'replicate': replicate_id
    })

    return tmp_df

tasks = []
for strategy_path in strategy_paths:
    replicate_paths = natsorted(glob.glob(os.path.join(strategy_path, 'replicate*')))
    for replicate_path in replicate_paths:
        tasks.append((strategy_path, replicate_path, tmap_df, all_smiles))

with Pool(processes=cpu_count()) as pool:
    new_results = list(tqdm(pool.imap(process_replicate, tasks),
                            total=len(tasks),
                            desc="Processing replicates",
                            disable=not use_tqdm))

if results_df.empty:
    results_df = pd.concat(new_results, ignore_index=True)
else:
    results_df = pd.concat([results_df] + new_results, ignore_index=True)

results_df.to_parquet(f'{data_path}/exploration_bm_scaffold_coverage.parquet')

### Functional Groups Coverage
all_strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')
if os.path.exists(f'{data_path}/exploration_fg_coverage.parquet'):
    results_df = pd.read_parquet(f'{data_path}/exploration_fg_coverage.parquet')
    strategy_paths = []
    for strategy_path in all_strategy_paths:
        strategy_name = strategy_path.split('/')[-1]
        if strategy_name not in results_df['strategy'].unique():
            strategy_paths.append(strategy_path)
else:
    results_df = pd.DataFrame()
    strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')

strategy_paths = natsorted(strategy_paths)

print(f"[{get_time()}]  ⚙️  Exploration (Functional Groups Coverage) : {len(strategy_paths)} experiments to process (out of {len(all_strategy_paths)})")

def process_replicate(task):
    """
    Process a single replicate folder by reading CSV files,
    merging with tmap_df, and computing the Functional Groups Coverage
    metric for every iteration.
    
    Parameters:
        task (tuple): Contains:
            - strategy_path (str): Path to the strategy directory.
            - replicate_path (str): Path to the replicate directory.
            - tmap_df (pd.DataFrame): DataFrame containing mapping information.
            - all_smiles (iterable): Array or list of all SMILES strings.
    
    Returns:
        pd.DataFrame: A DataFrame with columns for iteration, coverage,
                      proportion of selected molecules, strategy, and replicate.
    """
    strategy_path, replicate_path, tmap_df, all_smiles = task

    # Extract the strategy name and replicate id.
    strategy_name = os.path.basename(strategy_path)
    replicate_id = int(os.path.basename(replicate_path).split('_')[-1])

    # Get all iteration directories within the replicate.
    iteration_paths = natsorted(glob.glob(os.path.join(replicate_path, '*')))
    
    # Read CSV files from the last iteration.
    train_csv = os.path.join(iteration_paths[-1], 'train_df.csv')
    selected_csv = os.path.join(iteration_paths[-1], 'selected_train_df.csv')
    train_df = pd.read_csv(train_csv)
    last_df = pd.read_csv(selected_csv)

    # Concatenate the data and sort by 'iteration'.
    df = pd.concat([train_df, last_df], axis=0).sort_values(by='iteration')
    
    # Merge with tmap_df on SMILES and sort by iteration and DATE.
    df = df.merge(tmap_df[['SMILES', 'x', 'y', 'DATE']], on='SMILES')
    df = df.sort_values(by=['iteration', 'DATE'])

    coverages = []
    proportions_all_selected = []
    iterations = []
    
    max_iter = int(df['iteration'].max())
    for iteration in range(max_iter + 1):
        # Select all molecules up to the current iteration.
        subset = df[df['iteration'] <= iteration]
        smiles = subset['SMILES'].values
        
        # Compute the Functional Groups Coverage metric.
        coverage = compute_FunctionalGroupsCoverage(all_smiles, smiles)
        # Compute the proportion of molecules selected so far.
        proportion_all_selected = len(smiles) / len(tmap_df)
        
        coverages.append(coverage)
        proportions_all_selected.append(proportion_all_selected)
        iterations.append(iteration)
    
    # Create a temporary DataFrame with the results.
    tmp_df = pd.DataFrame({
        'iterations': iterations,
        'exploration_metric': coverages,
        'proportions_all_selected': proportions_all_selected,
        'strategy': strategy_name,
        'replicate': replicate_id
    })
    
    return tmp_df

tasks = []
for strategy_path in strategy_paths:
    replicate_paths = natsorted(glob.glob(os.path.join(strategy_path, 'replicate*')))
    for replicate_path in replicate_paths:
        tasks.append((strategy_path, replicate_path, tmap_df, all_smiles))

with Pool(processes=cpu_count()) as pool:
    results_list = list(tqdm(pool.imap(process_replicate, tasks),
                                total=len(tasks),
                                desc="Replicates",
                                disable=not use_tqdm))

if results_df.empty:
    results_df = pd.concat(results_list, ignore_index=True)
else:
    results_df = pd.concat([results_df] + results_list, ignore_index=True)

results_df.to_parquet(f'{data_path}/exploration_fg_coverage.parquet')

### Ring Systems Coverage
all_strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')
if os.path.exists(f'{data_path}/exploration_rs_coverage.parquet'):
    results_df = pd.read_parquet(f'{data_path}/exploration_rs_coverage.parquet')
    strategy_paths = []
    for strategy_path in all_strategy_paths:
        strategy_name = strategy_path.split('/')[-1]
        if strategy_name not in results_df['strategy'].unique():
            strategy_paths.append(strategy_path)
else:
    results_df = pd.DataFrame()
    strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')

strategy_paths = natsorted(strategy_paths)

print(f"[{get_time()}]  ⚙️  Exploration (Ring Systems Coverage) : {len(strategy_paths)} experiments to process (out of {len(all_strategy_paths)})")

def process_replicate(task):
    """
    Process one replicate: read CSV files from the last iteration folder,
    merge with tmap_df, and compute the ring systems coverage for every iteration.
    
    Parameters:
        task (tuple): Contains:
            - strategy_path (str): Path to the strategy folder.
            - replicate_path (str): Path to the replicate folder.
            - tmap_df (pd.DataFrame): DataFrame with mapping data.
            - all_smiles (iterable): Collection of all SMILES strings.
    
    Returns:
        pd.DataFrame: A DataFrame with columns for iteration, computed coverage,
                      proportion of selected molecules, strategy, and replicate.
    """
    strategy_path, replicate_path, tmap_df, all_smiles = task

    # Extract strategy name and replicate id.
    strategy_name = os.path.basename(strategy_path)
    replicate_id = int(os.path.basename(replicate_path).split('_')[-1])
    
    # Get all iteration folders within the replicate.
    iteration_paths = natsorted(glob.glob(os.path.join(replicate_path, '*')))
    
    # Read CSV files from the last iteration folder.
    train_csv = os.path.join(iteration_paths[-1], 'train_df.csv')
    selected_csv = os.path.join(iteration_paths[-1], 'selected_train_df.csv')
    train_df = pd.read_csv(train_csv)
    last_df = pd.read_csv(selected_csv)
    
    # Concatenate and sort the data by 'iteration'.
    df = pd.concat([train_df, last_df], axis=0).sort_values(by='iteration')
    
    # Merge with tmap_df on SMILES and sort by iteration and DATE.
    df = df.merge(tmap_df[['SMILES', 'x', 'y', 'DATE']], on='SMILES')
    df = df.sort_values(by=['iteration', 'DATE'])
    
    coverages = []
    proportions_all_selected = []
    iterations = []
    
    max_iter = int(df['iteration'].max())
    for iteration in range(max_iter + 1):
        # Get SMILES up to and including the current iteration.
        smiles = df[df['iteration'] <= iteration]['SMILES'].values
        
        # Compute the Ring Systems Coverage metric.
        coverage = compute_RingSystemsCoverage(all_smiles, smiles)
        
        # Compute the proportion of selected molecules.
        proportion_all_selected = len(smiles) / len(tmap_df)
        
        coverages.append(coverage)
        proportions_all_selected.append(proportion_all_selected)
        iterations.append(iteration)
    
    # Build a DataFrame for this replicate.
    tmp_df = pd.DataFrame({
        'iterations': iterations,
        'exploration_metric': coverages,
        'proportions_all_selected': proportions_all_selected,
        'strategy': strategy_name,
        'replicate': replicate_id
    })
    
    return tmp_df

tasks = []
for strategy_path in strategy_paths:
    replicate_paths = natsorted(glob.glob(os.path.join(strategy_path, 'replicate*')))
    for replicate_path in replicate_paths:
        tasks.append((strategy_path, replicate_path, tmap_df, all_smiles))

with Pool(processes=cpu_count()) as pool:
    results_list = list(tqdm(pool.imap(process_replicate, tasks),
                                total=len(tasks),
                                desc="Processing replicates",
                                disable=not use_tqdm))

if results_df.empty:
    results_df = pd.concat(results_list, ignore_index=True)
else:
    results_df = pd.concat([results_df] + results_list, ignore_index=True)

results_df.to_parquet(f'{data_path}/exploration_rs_coverage.parquet')

### KDE TMAP Coverage
all_strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')
if os.path.exists(f'{data_path}/exploration_tmap_coverage.parquet'):
    results_df = pd.read_parquet(f'{data_path}/exploration_tmap_coverage.parquet')
    strategy_paths = []
    for strategy_path in all_strategy_paths:
        strategy_name = strategy_path.split('/')[-1]
        if strategy_name not in results_df['strategy'].unique():
            strategy_paths.append(strategy_path)
else:
    results_df = pd.DataFrame()
    strategy_paths = glob.glob(f'../../experiments/{DATASET}/*')

strategy_paths = natsorted(strategy_paths)

print(f"[{get_time()}]  ⚙️  Exploration (KDE TMAP Coverage) : {len(strategy_paths)} experiments to process (out of {len(all_strategy_paths)})")

def process_replicate(task):
    """
    Process one replicate folder.
    
    Parameters:
        task (tuple): A tuple containing (replicate_path, strategy_name, tmap_df, project_kde)
    
    Returns:
        pd.DataFrame: A DataFrame with the computed metrics for this replicate.
    """
    replicate_path, strategy_name, tmap_df, project_kde = task
    
    # Extract replicate id from the path (assumes replicate folder ends with an underscore and a number)
    replicate_id = int(replicate_path.split('_')[-1])
    
    # Get iteration folders and load data from the last iteration
    iteration_paths = natsorted(glob.glob(os.path.join(replicate_path, '*')))
    last_iteration_path = iteration_paths[-1]
    train_df = pd.read_csv(os.path.join(last_iteration_path, 'train_df.csv'))
    last_df = pd.read_csv(os.path.join(last_iteration_path, 'selected_train_df.csv'))
    
    # Merge the training data and sort
    df = pd.concat([train_df, last_df], axis=0).sort_values(by=['iteration'])
    df = df.merge(tmap_df[['SMILES', 'x', 'y', 'DATE']], on='SMILES') \
           .sort_values(by=['iteration', 'DATE'])
    
    # Initialize lists for results
    overlap_scores = []
    proportions_all_selected = []
    iterations = []
    
    max_iteration = df['iteration'].max()
    for iteration in range(max_iteration + 1):
        # Filter data up to the current iteration
        df_iter = df[df['iteration'] <= iteration]
        x = df_iter['x'].values
        y = df_iter['y'].values

        # Compute the overlap score (assuming compute_TMAPOverlap is compute‐intensive)
        overlap_score = compute_TMAPOverlap(x, y, project_kde)
        proportion_all_selected = len(x) / len(tmap_df)

        # Append results for this iteration
        overlap_scores.append(overlap_score)
        proportions_all_selected.append(proportion_all_selected)
        iterations.append(iteration)
    
    # Create a DataFrame with the results for this replicate
    tmp_df = pd.DataFrame({
        'iterations': iterations,
        'exploration_metric': overlap_scores,
        'proportions_all_selected': proportions_all_selected,
        'strategy': strategy_name,
        'replicate': replicate_id,
    })
    return tmp_df

# Prepare a list of tasks (one per replicate)
tasks = []
for strategy_path in tqdm(strategy_paths, leave=False, desc='Strategies', disable=not use_tqdm):
    strategy_name = os.path.basename(strategy_path)  # cleaner than splitting by '/'
    replicate_paths = natsorted(glob.glob(os.path.join(strategy_path, 'replicate*')))
    for replicate_path in tqdm(replicate_paths, leave=False, desc='Replicates', disable=not use_tqdm):
        tasks.append((replicate_path, strategy_name, tmap_df, project_kde))

# Process all tasks in parallel using a pool of worker processes
results_list = []
with mp.Pool(processes=mp.cpu_count()) as pool:
    # Using imap_unordered to show progress with tqdm
    for tmp_df in tqdm(pool.imap_unordered(process_replicate, tasks), 
                       total=len(tasks), 
                       desc="Processing replicates",
                       disable=not use_tqdm):
        results_list.append(tmp_df)

if results_df.empty:
    results_df = pd.concat(results_list, ignore_index=True)
else:
    results_df = pd.concat([results_df] + results_list, ignore_index=True)

results_df.reset_index(drop=True, inplace=True)
results_df.sort_values(by=['strategy', 'replicate', 'iterations'], inplace=True)
results_df.to_parquet(f'{data_path}/exploration_tmap_coverage.parquet')

### Project
strategy_name = 'project'
metrics = []
proportions_all_selected = []

iterations = []
exploration_bm_scaffold_coverages = []
exploration_fg_coverages = []
exploration_rs_coverages = []
exploration_neighborhood_coverage_aucs = []
exploration_tmap_coverages = []
exploration_diversitys = []
exploration_diameters = []
exploration_bottlenecks = []
exploration_sumdiversitys = []
exploration_sumdiameters = []
exploration_sumbottlenecks = []

for iteration in tqdm(range(0, tmap_df['iteration'].max() + 1), leave=False, disable=not use_tqdm):
    df_iter = tmap_df[tmap_df['iteration'] <= iteration].copy()
    smiles = df_iter['SMILES'].values
    small_distance_matrix = distance_matrix.loc[smiles, smiles].copy()
    proportion_all_selected = len(smiles)/len(tmap_df)

    x = df_iter['x'].values
    y = df_iter['y'].values

    exploration_bm_scaffold_coverages.append(compute_BMScaffoldCoverage(all_smiles, smiles))
    exploration_fg_coverages.append(compute_FunctionalGroupsCoverage(all_smiles, smiles))
    exploration_rs_coverages.append(compute_RingSystemsCoverage(all_smiles, smiles))
    exploration_neighborhood_coverage_aucs.append(compute_DistanceCoverageAUC(all_smiles, smiles, distance_matrix))
    exploration_tmap_coverages.append(compute_TMAPOverlap(x, y, project_kde))
    exploration_diversitys.append(compute_Diversity(small_distance_matrix))
    exploration_diameters.append(compute_Diameter(small_distance_matrix))
    exploration_bottlenecks.append(compute_Bottleneck(small_distance_matrix))
    exploration_sumdiversitys.append(compute_SumDiversity(small_distance_matrix))
    exploration_sumdiameters.append(compute_SumDiameter(small_distance_matrix))
    exploration_sumbottlenecks.append(compute_SumBottleneck(small_distance_matrix))

    proportions_all_selected.append(proportion_all_selected)
    iterations.append(iteration)

results_df = pd.DataFrame()
results_df['iterations'] = iterations
results_df['proportions_all_selected'] = proportions_all_selected
results_df['exploration_bm_scaffold_coverage'] = exploration_bm_scaffold_coverages
results_df['exploration_fg_coverage'] = exploration_fg_coverages
results_df['exploration_rs_coverage'] = exploration_rs_coverages
results_df['exploration_neighborhood_coverage_auc'] = exploration_neighborhood_coverage_aucs
results_df['exploration_tmap_coverage'] = exploration_tmap_coverages
results_df['exploration_diversity'] = exploration_diversitys
results_df['exploration_diameter'] = exploration_diameters
results_df['exploration_bottleneck'] = exploration_bottlenecks
results_df['exploration_sumdiversity'] = exploration_sumdiversitys
results_df['exploration_sumdiameter'] = exploration_sumdiameters
results_df['exploration_sumbottleneck'] = exploration_sumbottlenecks

for threshold in tqdm(np.arange(0., 1.05, 0.05), leave=False, desc='Similarity thresholds', disable=not use_tqdm):
    exploration_distance_coverages = []
    exploration_circles = []
    for iteration in range(0, tmap_df['iteration'].max() + 1):
        df_iter = tmap_df[tmap_df['iteration'] <= iteration].copy()
        smiles = df_iter['SMILES'].values
        small_distance_matrix = distance_matrix.loc[smiles, smiles].copy()

        exploration_distance_coverages.append(compute_DistanceCoverage(all_smiles, smiles, distance_matrix, threshold))
        exploration_circles.append(compute_Circles(small_distance_matrix, threshold))
    results_df[f'exploration_distance_coverage_{round(threshold, 2)}'] = exploration_distance_coverages
    results_df[f'exploration_circles_{round(threshold, 2)}'] = exploration_circles

results_df.to_parquet(f'../../data/{DATASET}/project_exploration.parquet', index=False)

print(f"[{get_time()}]  ✅  Postprocessing (2.3) : Exploration in {time.time() - t_init:.2f} seconds")