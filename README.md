# RetroDMTA: Setup and Usage Guide

This guide provides step-by-step instructions for setting up Python environments, running simulations, and analyzing results for the paper **Back to the Future of Lead Optimization: Benchmarking Compound Prioritization Strategies**.

## Python Environment Setup

The code has been tested using Rocky Linux release 8.8 (Green Obsidian).

Follow these instructions to set up the required Python environments using Conda:

### 1. Environment for Simulations

```bash
conda create -n RetroDMTA python=3.11 -y
conda activate RetroDMTA
pip install --no-deps -r RetroDMTA.txt
```

### 2. Environment for Generating TMAPs
**Note**: TMAP requires Python ≤ 3.9.
```bash
conda create -n RetroDMTA_TMAP python=3.9 -y
conda activate RetroDMTA_TMAP
pip install --no-deps -r TMAP.txt
```

## Running Simulations: Step-by-Step

## 0. [SCRIPT] Infrastructure setup
Run the initialization script to set up necessary folders and files. Replace $DATASET with your actual dataset name:
```bash
./scripts/0_initialize.sh $DATASET
```
After running, the folder structure will be created. Place your dataset and blueprint files accordingly:

```bash
RetroDMTA/
├── app/
├── code/
│   └── $DATASET/
├── config/
├── data/
│   └── $DATASET/
│       ├── blueprint.csv
│       ├── data_aggregated.csv
│       └── data.csv
├── experiments/
├── figures/
├── scripts/
├── src/
└── README.md
```

## 1. [MANUAL] Dataset: configuration
Manually run the parameters notebook (`./code/$DATASET/0_2_simulation_parameters.ipynb`) to set dataset parameters (initial date, final date, timestep). 

Parameters are auto-calculated but can be manually adjusted and saved in `./data/common/dataset_config.json`.

## 2. [SCRIPT] Preprocessing
Run preprocessing to generate necessary files and figures:
```bash
./scripts/1_preprocessing.sh $DATASET
```
This script will automatically:
- Define best molecules (saved in `./data/$DATASET/top_molecules.pkl`).
- Compute similarity and distance matrices (saved in `./data/similarity_matrix.parquet` and `./data/$DATASET/distance_matrix.parquet`).
- Compute the Matched Molecular Pairs (saved in `./data/$DATASET/mmp.parquet`).
- Generate all exploratory data analysis figures (saved in `./figures/$DATASET/`).

## 3. [MANUAL] Generate TMAP coordinates
Manually run the `./code/$DATASET/0_3_tmap_manual.ipynb` notebook to:
- Determine the TMAP coordinates (saved in `./data/$DATASET/tmap_coordinates.pkl`).
- Generate the interactive TMAP (saved in `./data/$DATASET/tmap.html`).

## 4. [MANUAL] Simulation: configuration
Create a simulation config file in `./config/`. Look at `./config/template.json` for an example file. This file will be noted `$PATH_TO_SIMULATION_CONFIG`.

## 5. [SCRIPT] Simulation: run
Execute the run script to launch the simulation:
```bash
./scripts/2_simulation.sh $PATH_TO_SIMULATION_CONFIG
```
This will launch the simulations and store the results in `./experiments/$DATASET/`

## 6. [SCRIPT] Postprocessing
Execute the postprocessing script:
```bash
./scripts/3_postprocessing.sh $DATASET
```
This automatically:
- Compute exploitation metrics (saved in `./data/$DATASET/exploitation.parquet`).
- Compute exploration metrics (saved in `./data/$DATASET/exploration_*.parquet`).
- Generate all TMAPs figures (saved in `./figures/$DATASET/`).

## 7. [MANUAL] Visualize results
To visualize results, you can launch a local app with the following command be sure to have the correct conda env enabled:
```bash
./scripts/4_app.sh
```