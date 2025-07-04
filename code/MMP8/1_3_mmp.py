# Utils
import os
import glob
import subprocess
from tqdm.auto import tqdm
import subprocess
import time

# Data
import pandas as pd
import numpy as np

# Cheminformatics
from rdkit import Chem

# SQL
import sqlite3

from utils import to_smi, get_r_groups, get_time

use_tqdm = False

t_init = time.time()

# Determine the dataset name from the current working directory
DATASET = os.getcwd().split('/')[-1]

# Define base path for dataset
data_path = f'../../data/{DATASET}'

# Load the main and aggregated data
df = pd.read_csv(f'{data_path}/data_aggregated.csv').sort_values(by='DATE').reset_index(drop=True)

# Extract SMILES and compound IDs, then save to a file in SMILES format
smiles = df[['SMILES', 'CPD_ID']]
smiles.to_csv('molecules.smi', index=False, header=False, sep=' ')

# Run MMPDB fragment command to generate fragments
run_fragment = subprocess.run(
    "mmpdb fragment --num-cuts 1 molecules.smi -o fragments.fragdb",
    shell=True, 
    capture_output=True, 
    text=True
)

# Check if the fragment process ran successfully
if run_fragment.returncode != 0:
    print("Fragment generation failed:")
    print(run_fragment.stderr)

# Run MMPDB index command to create an indexed MMPDB database
run_index = subprocess.run(
    "mmpdb index fragments.fragdb -o mmp.mmpdb",
    shell=True, 
    capture_output=True, 
    text=True
)

# Check if the indexing process ran successfully
if run_index.returncode != 0:
    print("Indexing failed:")
    print(run_index.stderr)

# Clean up temporary files
clean_molecules = subprocess.run("rm molecules.smi", shell=True, capture_output=True, text=True)
clean_fragment = subprocess.run("rm fragments.fragdb", shell=True, capture_output=True, text=True)

# Check if cleanup processes ran successfully
if clean_molecules.returncode != 0:
    print("Failed to delete molecules.smi:")
    print(clean_molecules.stderr)

if clean_fragment.returncode != 0:
    print("Failed to delete fragments.fragdb:")
    print(clean_fragment.stderr)

# Connect to the MMPDB SQLite database
conn = sqlite3.connect("mmp.mmpdb")

# Create a cursor object to interact with the database
cursor = conn.cursor()

# Execute a query to list all table names in the database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

# Fetch and print the results
tables = cursor.fetchall()

# Specify the table name
id = 'compound'

# Query to get the column names and their metadata from the specified table
cursor.execute(f"SELECT name FROM pragma_table_info('{id}');")
columns_info = cursor.fetchall()

# Query to fetch all rows from the specified table
cursor.execute(f"SELECT * FROM {id}")
compound = cursor.fetchall()

# Specify the table name
id = 'rule_smiles'

# Query the column information for the specified table using PRAGMA
cursor.execute(f"PRAGMA table_info('{id}');")
columns_info = cursor.fetchall()

# Query all rows from the 'rule_smiles' table
cursor.execute(f"SELECT * FROM {id}")
rule_smiles = cursor.fetchall()

# Specify the table name
id = 'rule'

# Query the column information for the specified table using PRAGMA
cursor.execute(f"PRAGMA table_info('{id}');")
columns_info = cursor.fetchall()

# Query all rows from the 'rule' table
cursor.execute(f"SELECT * FROM {id}")
rule = cursor.fetchall()

# Check if the table is not empty
if rule:
    # Convert the result to a NumPy array for numerical operations
    rule_array = np.array(rule, dtype=object)  # Use dtype=object for mixed data types
    try:
        # Calculate the maximum value, excluding the first column
        max_value = rule_array[:, 1:].astype(float).max()
        # print(f"Maximum value in the 'rule' table (excluding the first column): {max_value}")
    except ValueError as e:
        print(f"Error converting data to float or finding max: {e}")
else:
    print("The 'rule' table is empty.")

# Specify the table name
id = 'constant_smiles'

# Query the column information for the specified table using PRAGMA
cursor.execute(f"PRAGMA table_info('{id}');")
columns_info = cursor.fetchall()

# Query all rows from the 'constant_smiles' table
cursor.execute(f"SELECT * FROM {id}")
constant_smiles = cursor.fetchall()

# Get the number of rows in the table
row_count = len(constant_smiles)

# Specify the table name
id = 'pair'

# Query the column information for the specified table using PRAGMA
cursor.execute(f"PRAGMA table_info('{id}');")
columns_info = cursor.fetchall()

# Query all rows from the 'pair' table
cursor.execute(f"SELECT * FROM {id}")
pair = np.array(cursor.fetchall())  # Use dtype=object for mixed data types

# Extract unique rows starting from the third column (index 2)
pair = np.unique(pair[:, 2:], axis=0)

# # Get the count of unique rows
unique_count = len(pair)

# Cache molecules for faster access
cache_mols = {}
for smi in np.unique(np.array(compound)[:, 3]):
    cache_mols[smi] = Chem.MolFromSmiles(smi)

# Cache scaffolds, replacing the placeholder in SMILES
cache_scaffolds = {}
for smi in np.array(constant_smiles)[:, 1]:
    cache_scaffolds[smi.replace('[*:1]', '[H]')] = Chem.MolFromSmiles(smi.replace('[*:1]', '[H]'))

# Initialize results dictionary
results = {
    'SMILES1': [],
    'SMILES2': [],
    'scaffold': [],
    'R1': [],
    'R2': [],
    'n_SMILES1': [],
    'n_SMILES2': [],
    'n_scaffold': [],
    'n_R1': [],
    'n_R2': []
}

# Process each pair and extract R-groups
for index, p in enumerate(tqdm(pair, desc="Processing pairs", disable=not use_tqdm)):
    compound1_id, compound2_id, constant_id = p
    smiles1 = compound[compound1_id - 1][3]
    smiles2 = compound[compound2_id - 1][3]
    scaffold_smiles = constant_smiles[constant_id - 1][1].replace('[*:1]', '[H]')
    r1, r2 = get_r_groups(smiles1, smiles2, scaffold_smiles, cache_mols, cache_scaffolds)

    n1 = cache_mols[smiles1].GetNumHeavyAtoms()
    n2 = cache_mols[smiles2].GetNumHeavyAtoms()
    nscaffold = cache_scaffolds[scaffold_smiles].GetNumHeavyAtoms()
    nr1 = abs(n1 - nscaffold)
    nr2 = abs(n2 - nscaffold)
    
    # Populate the results dictionary
    results['SMILES1'].append(smiles1)
    results['SMILES2'].append(smiles2)
    results['scaffold'].append(scaffold_smiles)
    results['R1'].append(to_smi(r1) if r1 else None)
    results['R2'].append(to_smi(r2) if r2 else None)
    results['n_SMILES1'].append(n1)
    results['n_SMILES2'].append(n2)
    results['n_scaffold'].append(nscaffold)
    results['n_R1'].append(nr1)
    results['n_R2'].append(nr2)

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Add derived columns
results_df['n_R1+n_R2'] = results_df['n_R1'] + results_df['n_R2']
results_df['valid_mmp'] = results_df['n_R1+n_R2'] <= 15

# Validate matching rows
mask = (
    (results_df['n_scaffold'] + results_df['n_R1'] == results_df['n_SMILES1']) &
    (results_df['n_scaffold'] + results_df['n_R2'] == results_df['n_SMILES2'])
)

# Rename columns in the original DataFrame for the first merge
df = df.rename(columns={'SMILES': 'SMILES1', 'CPD_ID': 'CPD_1'})

# Merge results_df with compound information for SMILES1
results_df = results_df.merge(df[['SMILES1', 'CPD_1']], on='SMILES1', how='left')

# Rename columns in the original DataFrame for the second merge
df = df.rename(columns={'SMILES1': 'SMILES2', 'CPD_1': 'CPD_2'})

# Merge results_df with compound information for SMILES2
results_df = results_df.merge(df[['SMILES2', 'CPD_2']], on='SMILES2', how='left')

# Save the merged DataFrame to a parquet file
output_path = f'{data_path}/mmp.parquet'
results_df.to_parquet(output_path, index=False)

# Clean up the temporary MMPDB database file
clean_fragment = subprocess.run("rm mmp.mmpdb", shell=True, capture_output=True, text=True)

print(f"[{get_time()}]  ✅  Preprocessing (1.3) : MMPs in {time.time() - t_init:.2f} seconds")