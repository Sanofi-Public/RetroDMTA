# Utils
import os  
import time

# Data
import numpy as np  
import pandas as pd  

from utils import compute_similarity_matrix, get_time

t_init = time.time()

# Determine the dataset name from the current working directory
DATASET = os.getcwd().split('/')[-1]

# Define base path for dataset
data_path = f'../../data/{DATASET}'

# Load the main and aggregated data
df = pd.read_csv(f'{data_path}/data_aggregated.csv').sort_values(by='DATE').reset_index(drop=True)

# Extract unique SMILES strings
smiles = df['SMILES'].unique()

# Compute the similarity and distance matrices for the SMILES strings
similarity_matrix = compute_similarity_matrix(smiles=smiles)
distance_matrix = 1 - similarity_matrix

# Save the similarity and distance matrices as a parquet file for efficient storage and retrieval
similarity_matrix.to_parquet(f'{data_path}/similarity_matrix.parquet', index=True)
distance_matrix.to_parquet(f'{data_path}/distance_matrix.parquet', index=True)

print(f"[{get_time()}]  ✅  Preprocessing (1.2) : Similarity matrix in {time.time() - t_init:.2f} seconds")