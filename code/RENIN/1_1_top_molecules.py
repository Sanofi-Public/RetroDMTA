# Utils
import os
import time

# Data
import numpy as np
import pandas as pd

from utils import sigmoid_HTB, sigmoid_LTB, sigmoid_INT, sum_values, normalize_and_score_property, select_best_smiles, save_pickle, load_pickle, get_time

dict_sigmoid_functions = {
    'H': sigmoid_HTB,
    'L': sigmoid_LTB,
    'V': sigmoid_INT,
}

t_init = time.time()

# Determine the dataset name from the current working directory
DATASET = os.getcwd().split('/')[-1]

# Define base path for dataset
data_path = f'../../data/{DATASET}'

# Load the main and aggregated data
data_df = pd.read_csv(f'{data_path}/data.csv').sort_values(by='DATE').reset_index(drop=True)
df = pd.read_csv(f'{data_path}/data_aggregated.csv').sort_values(by='DATE').reset_index(drop=True)

# Load blueprint data
bp_df = pd.read_csv(f'{data_path}/blueprint.csv')

# Convert 'DATE' column to datetime for proper time handling
data_df['DATE'] = pd.to_datetime(data_df['DATE'])
df['DATE'] = pd.to_datetime(df['DATE'])

# Extract endpoints from blueprint
endpoints = bp_df['PROPERTIES'].values

# --------------------------------------------------
# Setup: Weight Dictionary and Property List
# --------------------------------------------------
# Validate required columns in bp_df
required_cols = ['PROPERTIES', 'WEIGHT']
missing_cols = [col for col in required_cols if col not in bp_df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in bp_df: {missing_cols}")

# Create a dictionary mapping each property to its weight
weight_dict = dict(zip(bp_df['PROPERTIES'], bp_df['WEIGHT']))

# Get a sorted list of unique properties
properties = sorted(bp_df['PROPERTIES'].unique())
# print(f"Processing {len(properties)} unique properties")

# Create a copy of the main dataframe for computations
compute_df = df.copy()

# --------------------------------------------------
# Normalize Properties and Compute Weighted Scores
# --------------------------------------------------
# Track successfully processed properties
processed_properties = []

for prop in properties:
    try:
        normalize_and_score_property(compute_df, prop, bp_df, dict_sigmoid_functions)
        processed_properties.append(prop)
    except Exception as e:
        print(f"Warning: Failed to process property '{prop}': {str(e)}")

# print(f"Successfully processed {len(processed_properties)} out of {len(properties)} properties")

# Compute the total weight per molecule using valid properties
compute_df['total_weight'] = compute_df[properties].apply(
    lambda row: sum_values(row, weight_dict), axis=1
)

# Handle potential zero weights to avoid division errors
compute_df['total_weight'] = compute_df['total_weight'].replace(0, np.nan)

# Calculate overall geometric and arithmetic scores
# Use regex filtering to collect all per-property scores
geom_columns = compute_df.filter(regex='^geometric_')
arith_columns = compute_df.filter(regex='^arithmetic_')

# The exponential geometric score is computed as the product of geometric scores raised 
# to the power of (1/total_weight) for each molecule.
# Handle potentially zero or negative values with np.maximum
safe_geom_values = geom_columns.replace(0, np.nan)
compute_df['exp_geometric_score'] = np.where(
    compute_df['total_weight'].notna() & (compute_df['total_weight'] > 0),
    safe_geom_values.prod(axis=1) ** (1 / compute_df['total_weight'].values),
    np.nan
)

# The exponential arithmetic score is the sum of arithmetic scores divided by total_weight.
compute_df['exp_arithmetic_score'] = np.where(
    compute_df['total_weight'].notna() & (compute_df['total_weight'] > 0),
    arith_columns.sum(axis=1) / compute_df['total_weight'].values,
    np.nan
)

# Fill NaN values in score columns
score_cols = ['exp_geometric_score', 'exp_arithmetic_score']
compute_df[score_cols] = compute_df[score_cols].fillna(0)

# --------------------------------------------------
# Merge Scores and Calculate Documentation Completeness
# --------------------------------------------------
# print("Merging computed scores back into main dataframe")
# Merge computed scores back into the main dataframe using the unique identifier 'SMILES'
result_columns = ['SMILES', 'total_weight', 'exp_geometric_score', 'exp_arithmetic_score']
df = df.merge(
    compute_df[result_columns],
    on='SMILES',
    how='left'  # Use left join to maintain all original rows
)

# --------------------------------------------------
# Documentation Completeness and Product Scores
# --------------------------------------------------
# print("Calculating documentation completeness and final scores")
# Calculate documentation completeness: the fraction of properties that are not missing.
df['documentation'] = df[properties].notna().sum(axis=1) / len(properties)

# Compute final product scores by weighting the exponential scores with documentation completeness.
# These represent the quality of a molecule weighted by how completely it has been tested.
df['geo_DocDScore'] = df['exp_geometric_score'] * df['documentation']
df['ari_DocDScore'] = df['exp_arithmetic_score'] * df['documentation']

# Filter molecules with documentation completeness > 65%
doc_threshold = 0.65
doc_df = df[df['documentation'] > doc_threshold].copy()
# print(f"Filtered to {len(doc_df)} molecules with documentation > {doc_threshold}")

# --------------------------------------------------
# Identify Annotated Molecules
# --------------------------------------------------
if 'ANN' in df.columns and df['ANN'].notna().any():
    annotated_smiles = df.loc[df['ANN'].notna(), 'SMILES'].tolist()
else:
    annotated_smiles = []
# print(f'Number of annotated compounds: {len(annotated_smiles)}')

# Print summary statistics
# print(f"Documentation score stats: min={df['documentation'].min():.2f}, "
#       f"max={df['documentation'].max():.2f}, "
#       f"mean={df['documentation'].mean():.2f}")

# --------------------------------------------------
# Define Best Molecules Based on Scoring Thresholds
# --------------------------------------------------
top_molecules = {
    'geometric': {

        'Fixed (t = 0.5)': list(set(doc_df.loc[doc_df['exp_geometric_score'] > 0.5, 'SMILES'].tolist() + annotated_smiles)),

        'Top DScore (10%)': select_best_smiles(doc_df, 'exp_geometric_score', 0.10, annotated_smiles),
        'Top DScore (5%)': select_best_smiles(doc_df, 'exp_geometric_score', 0.05, annotated_smiles),
        'Top DScore (3%)': select_best_smiles(doc_df, 'exp_geometric_score', 0.03, annotated_smiles),
        'Top DScore (1%)': select_best_smiles(doc_df, 'exp_geometric_score', 0.01, annotated_smiles),
        'Top DScore (0.5%)': select_best_smiles(doc_df, 'exp_geometric_score', 0.005, annotated_smiles),

        'Top DScore (1σ)': list(set(doc_df[doc_df['exp_geometric_score'] > doc_df['exp_geometric_score'].mean() + 1 * doc_df['exp_geometric_score'].std()]['SMILES'].tolist() + annotated_smiles)),
        'Top DScore (2σ)': list(set(doc_df[doc_df['exp_geometric_score'] > doc_df['exp_geometric_score'].mean() + 2 * doc_df['exp_geometric_score'].std()]['SMILES'].tolist() + annotated_smiles)),
        'Top DScore (3σ)': list(set(doc_df[doc_df['exp_geometric_score'] > doc_df['exp_geometric_score'].mean() + 3 * doc_df['exp_geometric_score'].std()]['SMILES'].tolist() + annotated_smiles)),
    },
    'arithmetic': {

        'Fixed (t = 0.5)': list(set(doc_df.loc[doc_df['exp_arithmetic_score'] > 0.5, 'SMILES'].tolist() + annotated_smiles)),
        
        'Top DScore (10%)': select_best_smiles(doc_df, 'exp_arithmetic_score', 0.10, annotated_smiles),
        'Top DScore (5%)': select_best_smiles(doc_df, 'exp_arithmetic_score', 0.05, annotated_smiles),
        'Top DScore (3%)': select_best_smiles(doc_df, 'exp_arithmetic_score', 0.03, annotated_smiles),
        'Top DScore (1%)': select_best_smiles(doc_df, 'exp_arithmetic_score', 0.01, annotated_smiles),
        'Top DScore (0.5%)': select_best_smiles(doc_df, 'exp_arithmetic_score', 0.005, annotated_smiles),

        'Top DScore (1σ)': list(set(doc_df[doc_df['exp_arithmetic_score'] > doc_df['exp_arithmetic_score'].mean() + 1 * doc_df['exp_arithmetic_score'].std()]['SMILES'].tolist() + annotated_smiles)),
        'Top DScore (2σ)': list(set(doc_df[doc_df['exp_arithmetic_score'] > doc_df['exp_arithmetic_score'].mean() + 2 * doc_df['exp_arithmetic_score'].std()]['SMILES'].tolist() + annotated_smiles)),
        'Top DScore (3σ)': list(set(doc_df[doc_df['exp_arithmetic_score'] > doc_df['exp_arithmetic_score'].mean() + 3 * doc_df['exp_arithmetic_score'].std()]['SMILES'].tolist() + annotated_smiles)),
    },
}

# --------------------------------------------------
# Output: Summary of Best Molecule Counts
# --------------------------------------------------
# Determine the maximum length of the scoring method names (for aligned printing)
max_method_length = max(
    len(method)
    for agg in top_molecules.values()
    for method in agg.keys()
)

# Save the best molecules dictionary as a pickle file
output_filepath = f'{data_path}/top_molecules.pkl'
save_pickle(top_molecules, output_filepath)

print(f"[{get_time()}]  ✅  Preprocessing (1.1) : Top molecules in {time.time() - t_init:.2f} seconds")