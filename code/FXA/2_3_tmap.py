# Utils
import gc
import glob
import os
import time
from dateutil.relativedelta import relativedelta
from natsort import natsorted
from multiprocessing import Pool, cpu_count
from itertools import product
from faerun import Faerun
import time

# Data
import numpy as np
import pandas as pd

# Viz
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

pd.set_option('future.no_silent_downcasting', True)

from utils import load_common_config, load_tmap_coord, get_time, get_pmas_palette, load_pickle
from utils import get_clean_batchsize, get_clean_model, get_clean_type, get_clean_strategy
pmas = get_pmas_palette()
pmas_cmap = ListedColormap(pmas)
binary_cmap = ListedColormap(['#000000', pmas[0]])

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

SAVE_FIGURES = True

if SAVE_FIGURES:
    os.makedirs(f'{figures_path}/tmap_overview/png', exist_ok=True)
    os.makedirs(f'{figures_path}/tmap_overview/svg', exist_ok=True)
    os.makedirs(f'{figures_path}/tmap_iterative/png', exist_ok=True)
    os.makedirs(f'{figures_path}/tmap_iterative/svg', exist_ok=True)

GRIDSIZE = 200
FIGSIZE = (10, 10)

# Load TMAP coordinates and create the connections between molecules
x, y, s, t = load_tmap_coord(dataset=DATASET)
connections = [(a, b) for a, b in zip(s, t)]

# Load data, convert to datetime the date and add the index as column
tmap_df = pd.read_csv(f'{data_path}/data_aggregated.csv')
tmap_df['DATE'] = pd.to_datetime(tmap_df['DATE'])
tmap_df = tmap_df.reset_index()

bp_df = pd.read_csv(f'{data_path}/blueprint.csv')
endpoints = list(bp_df['PROPERTIES'].unique())

# Load the definitions of good molecules and create binary columns based on this
best_molecules =  load_pickle(f'{data_path}/top_molecules.pkl')
for key in best_molecules['geometric'].keys():
    tmp_smiles = best_molecules['geometric'][key]
    tmap_df[key] = tmap_df['SMILES'].apply(lambda x: 1 if x in tmp_smiles else 0)

top_categories = list(best_molecules['geometric'].keys())

# Store all the smiles and the initial smiles
all_smiles = tmap_df['SMILES'].values
initial_smiles = tmap_df[tmap_df['DATE'] < pd.to_datetime(INITIAL_DATE)]['SMILES'].values
initial_index = tmap_df[tmap_df['SMILES'].isin(initial_smiles)]['index'].values

### Initial TMAPs
print(f"[{get_time()}]  ⚙️  Generation of initial TMAPs")

for top_category in top_categories:

    t0 = time.time()
    # Define index and smiles of the top molecules considered
    top_index = tmap_df[tmap_df[top_category] == 1]['index'].values
    top_smiles = tmap_df[tmap_df[top_category] == 1]['SMILES'].values

    # Initialize figure
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.axis('off')

    # Add the project chemical space kde
    sns.kdeplot(x=x, y=y, fill=True, color=pmas[7], levels=4, bw_adjust=0.1, zorder=0, alpha=0.5, gridsize=GRIDSIZE)

    # Add the initial molecules chemical space kde
    sns.kdeplot(x=x[initial_index], y=y[initial_index], fill=True, color=pmas[2], alpha=0.5, levels=4, bw_adjust=0.1, zorder=1, gridsize=GRIDSIZE)

    # Add the top molecules to the plot
    sns.scatterplot(x=x[top_index], y=y[top_index], zorder=2, marker='X', color=pmas[5], linewidth=1, s=120, label='Top retrieved', edgecolor='white')

    # Generate legend hanfles
    legend_handles = [
    mpatches.Rectangle((0, 0), 1, 1, color=pmas[7], alpha=0.5, label='Project (all compounds)'),
    mpatches.Rectangle((0, 0), 1, 1, color=pmas[2], alpha=0.5, label='Initial compounds'),
    Line2D([0], [0], color=pmas[5], lw=1, label=f'Top compounds | {top_category}', alpha=1, marker='X', linestyle='None', markersize=12, markeredgewidth=1, markeredgecolor='white'),
    ]

    # Create the legend with frame properties
    legend = ax.legend(
        handles=legend_handles,
        borderpad=0.8,
        loc='upper center',
        frameon=True,
        facecolor='white',
        edgecolor='black',
        fontsize=10,
        bbox_to_anchor=(0.5, 0),
        ncol=3
    )

    # Set legend title properties
    legend_title = legend.get_title()
    legend_title.set(fontsize=12, color='black', fontweight='bold')


    # Set axis limits and title
    ax.set(xlim=(-0.55, 0.55), ylim=(-0.55, 0.55))
    ax.set_title(f'{DATASET} | {top_category} | Initial overview', color='black', fontsize=12, pad=10)

    for text in legend.get_texts():
        text.set_color('black')

    fig.tight_layout()

    fig.savefig(f'{figures_path}/tmap_overview/png/{top_category}_initial.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{figures_path}/tmap_overview/svg/{top_category}_initial.svg', bbox_inches='tight')

    plt.close(fig)

    del fig, ax, top_index, top_smiles
    gc.collect()

### Iterative TMAPs
def process_strategy_top_category(args):
    top_category, strategy_path = args
    try:
        t0 = time.time()
        
        # Filter indices and SMILES for the current top_category
        top_indices = tmap_df[tmap_df[top_category] == 1]['index'].values
        top_smiles = tmap_df[tmap_df[top_category] == 1]['SMILES'].values

        experiment = os.path.basename(strategy_path)
        batchsize, model, strategy_type, strategy_name = experiment.split('-')
        
        current_date = pd.to_datetime(INITIAL_DATE)
        iteration = 0

        while current_date <= pd.to_datetime(FINAL_DATE):
            iteration += 1
            next_date = current_date + relativedelta(months=TIMESTEP)

            fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            ax.axis('off')

            sns.kdeplot(
                x=x, y=y, fill=False, color=pmas[9], levels=2,
                bw_adjust=0.1, zorder=0, alpha=0.5,
                label='All compounds (beige)', gridsize=GRIDSIZE
            )

            # Load training data
            train_df_path = os.path.join(strategy_path, f'replicate_1/{iteration}/train_df.csv')
            if not os.path.exists(train_df_path):
                print(f"Train data not found: {train_df_path}")
                current_date = next_date
                plt.close(fig)
                gc.collect()
                continue
            train_smiles = pd.read_csv(train_df_path)['SMILES'].values
            train_index = tmap_df[tmap_df['SMILES'].isin(train_smiles)]['index'].values

            sns.kdeplot(
                x=x[train_index], y=y[train_index], zorder=2,
                fill=True, color=pmas[2], alpha=0.5,
                bw_adjust=0.1, levels=4, label='Train (blue)', gridsize=GRIDSIZE
            )

            try:
                # Load selected pool data
                selected_pool_path = os.path.join(strategy_path, f'replicate_1/{iteration}/selected_pool_df.csv')
                selected_smiles = pd.read_csv(selected_pool_path)['SMILES'].values
                selected_index = tmap_df[tmap_df['SMILES'].isin(selected_smiles)]['index'].values
            except:
                selected_smiles = []
                selected_index = []
            
            try:
                # Load pool data
                pool_path = os.path.join(strategy_path, f'replicate_1/{iteration}/pool_df.csv')
                pool_all_smiles = pd.read_csv(pool_path)['SMILES'].values
                pool_all_index = tmap_df[tmap_df['SMILES'].isin(pool_all_smiles)]['index'].values

            except:
                pool_all_smiles = []
                pool_all_index = []

            try:
                pool_smiles = pool_all_smiles[~np.isin(pool_all_smiles, selected_smiles)]
                pool_index = tmap_df[tmap_df['SMILES'].isin(pool_smiles)]['index'].values
            except:
                pool_smiles = []
                pool_index = []

            simulation_smiles = np.concatenate([selected_smiles, train_smiles])

            try:
                top_selected_smiles = np.intersect1d(top_smiles, selected_smiles)
                top_selected_index = tmap_df[tmap_df['SMILES'].isin(top_selected_smiles)]['index'].values
            except:
                top_selected_smiles = []
                top_selected_index = []

            try:
                top_retrieved_smiles = np.intersect1d(top_smiles, train_smiles)
                top_retrieved_index = tmap_df[tmap_df['SMILES'].isin(top_retrieved_smiles)]['index'].values
            except:
                top_retrieved_smiles = []
                top_retrieved_index = []

            try:
                top_unretrieved_smiles = np.intersect1d(top_smiles, pool_smiles)
                top_unretrieved_index = tmap_df[tmap_df['SMILES'].isin(top_unretrieved_smiles)]['index'].values
            except:
                top_unretrieved_smiles = []
                top_unretrieved_index = []

            project_smiles = all_smiles[
                np.isin(all_smiles, pool_all_smiles) | np.isin(all_smiles, train_smiles)
            ]
            project_index = tmap_df[tmap_df['SMILES'].isin(project_smiles)]['index'].values

            unavailable_smiles = all_smiles[
                ~(
                    np.isin(all_smiles, pool_all_smiles) |
                    np.isin(all_smiles, train_smiles)
                )
            ]
            unavailable_index = tmap_df[tmap_df['SMILES'].isin(unavailable_smiles)]['index'].values

            sns.scatterplot(
                    x=x[selected_index], y=y[selected_index], zorder=3,
                    marker='.', color=pmas[4], linewidth=1, s=300,
                    label='Selected (purple)'
                )

            sns.kdeplot(
                x=x[pool_all_index], y=y[pool_all_index], zorder=1,
                fill=True, color=pmas[8], alpha=0.5,
                bw_adjust=0.1, levels=4, label='All pool (yellow)', gridsize=GRIDSIZE
                )
            
            sns.scatterplot(
                x=x[top_selected_index], y=y[top_selected_index], zorder=4,
                marker='X', color=pmas[4], linewidth=1, s=120,
                label='Top selected (purple)', edgecolor='white'
            )

            sns.scatterplot(
                x=x[top_retrieved_index], y=y[top_retrieved_index], zorder=4,
                marker='X', color=pmas[5], linewidth=1, s=120,
                label='Top retrieved (green)', edgecolor='white'
            )

            sns.scatterplot(
                x=x[top_unretrieved_index], y=y[top_unretrieved_index], zorder=4,
                marker='X', color=pmas[3], linewidth=1, s=120,
                label='Top unselected (red)', edgecolor='white'
            )

            legend_handles = [
            Line2D([0], [0], color=pmas[9], lw=2, label='Project', alpha=0.5),
            mpatches.Rectangle((0, 0), 1, 1, color=pmas[2], alpha=0.5, label='Train'),
            mpatches.Rectangle((0, 0), 1, 1, color=pmas[8], alpha=0.5, label='Pool'),
            Line2D([0], [0], color=pmas[4], lw=1, label='Selected', alpha=1, marker='.', linestyle='None', markersize=16, markeredgewidth=1, markeredgecolor='white'),
            Line2D([0], [0], color=pmas[4], lw=1, label='Top selected (in selection)', alpha=1, marker='X', linestyle='None', markersize=12, markeredgewidth=1, markeredgecolor='white'),
            Line2D([0], [0], color=pmas[5], lw=1, label='Top retrieved (in train)', alpha=1, marker='X', linestyle='None', markersize=12, markeredgewidth=1, markeredgecolor='white'),
            Line2D([0], [0], color=pmas[3], lw=1, label='Top not retrieved (in pool)', alpha=1, marker='X', linestyle='None', markersize=12, markeredgewidth=1, markeredgecolor='white'),
            ]     


            # Create the legend with frame properties
            legend = ax.legend(
                handles=legend_handles,
                borderpad=0.8,
                loc='upper center',
                frameon=True,
                facecolor='white',
                edgecolor='black',
                fontsize=10,
                bbox_to_anchor=(0.5, 0),
                ncol=3
            )

            # Set legend title properties
            legend_title = legend.get_title()
            legend_title.set(fontsize=12, color='black', fontweight='bold')

            # Set axis limits
            ax.set(xlim=(-0.55, 0.55), ylim=(-0.55, 0.55))
            ax.set_title(f'{DATASET} | {top_category} | Iteration {iteration}\n {get_clean_strategy(strategy_name)} | {get_clean_type(strategy_type)} | {get_clean_model(model)} | {get_clean_batchsize(batchsize)}', color='black', fontsize=12, pad=10)

            for text in legend.get_texts():
                text.set_color('black')

            fig.tight_layout()

            # Define paths
            png_dir = os.path.join(
                '../../figures', DATASET, 'tmap_iterative', 'png',
                top_category, experiment
            )
            svg_dir = os.path.join(
                '../../figures', DATASET, 'tmap_iterative', 'svg',
                top_category, experiment
            )
            os.makedirs(png_dir, exist_ok=True)
            os.makedirs(svg_dir, exist_ok=True)

            # Save figures
            png_path = os.path.join(png_dir, f'{iteration}.png')
            svg_path = os.path.join(svg_dir, f'{iteration}.svg')
            fig.savefig(png_path, dpi=300)
            fig.savefig(svg_path, dpi=300)

            plt.close(fig)
            del fig, ax
            gc.collect()

            current_date = next_date

        gc.collect()
        # print(f"[{get_time()}]  ✅ {top_category} | {get_clean_strategy(strategy_name)} | {get_clean_type(strategy_type)} | {get_clean_model(model)} | {get_clean_batchsize(batchsize)} generated in {round(time.time() - t0, 2)}s")
    except Exception as e:
        print(f"❌  Error processing {top_category} | {get_clean_strategy(strategy_name)} | {get_clean_type(strategy_type)} | {get_clean_model(model)} | {get_clean_batchsize(batchsize)}. Error: {e}")

print(f"[{get_time()}]  ⚙️  Generation of iterative TMAPs")

strategy_paths = sorted(glob.glob(f'../../experiments/{DATASET}/*'))
tasks = list(product(top_categories, strategy_paths))

# Determine the number of processes to use
num_processes = cpu_count()  # Leave one CPU free

with Pool(processes=num_processes) as pool:
    pool.map(process_strategy_top_category, tasks)

### Final TMAPs
print(f"[{get_time()}]  ⚙️  Generation of final TMAPs")

for top_category in top_categories[:1]:

    top_index = tmap_df[tmap_df[top_category] == 1]['index'].values
    top_smiles = tmap_df[tmap_df[top_category] == 1]['SMILES'].values

    strategy_paths = sorted(glob.glob(f'../../experiments/{DATASET}/*'))

    for strategy_path in strategy_paths[:1]:

        t0 = time.time()

        experiment = strategy_path.split('/')[-1]
        batchsize, model, strategy_type, strategy_name = experiment.split('-')

        t0 = time.time()

        final_train_smiles = pd.read_csv(natsorted(glob.glob(os.path.join(strategy_path, f'replicate_1/*/train_df.csv')))[-1])['SMILES'].values
        final_selected_smiles = pd.read_csv(natsorted(glob.glob(os.path.join(strategy_path, f'replicate_1/*/selected_train_df.csv')))[-1])['SMILES'].values

        final_smiles = np.concatenate([final_train_smiles, final_selected_smiles])
        final_index = tmap_df[tmap_df['SMILES'].isin(final_smiles)]['index'].values

        top_index = tmap_df[tmap_df[top_category] == 1]['index'].values
        top_smiles = tmap_df[tmap_df[top_category] == 1]['SMILES'].values

        unselected_smiles = all_smiles[~np.isin(all_smiles, final_smiles)]
        unselected_index = tmap_df[tmap_df['SMILES'].isin(unselected_smiles)]['index'].values

        top_retrieved_smiles = np.intersect1d(top_smiles, final_smiles)
        top_retrieved_index = tmap_df[tmap_df['SMILES'].isin(top_retrieved_smiles)]['index'].values

        top_unretrieved_smiles = top_smiles[~np.isin(top_smiles, top_retrieved_smiles)]
        top_unretrieved_index = tmap_df[tmap_df['SMILES'].isin(top_unretrieved_smiles)]['index'].values

        selected_smiles = final_smiles[~np.isin(final_smiles, top_smiles)]
        selected_index = tmap_df[tmap_df['SMILES'].isin(selected_smiles)]['index'].values

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.axis('off')

        sns.kdeplot(x=x[unselected_index], y=y[unselected_index], fill=True, color=pmas[8], levels=4, bw_adjust=0.1, zorder=1, alpha=0.5, label='Unselected compounds', gridsize=GRIDSIZE)

        sns.kdeplot(x=x[selected_index], y=y[selected_index], fill=True, color=pmas[2], levels=4, bw_adjust=0.1, zorder=2, alpha=0.5, label='Selected compounds', gridsize=GRIDSIZE)

        sns.scatterplot(x=x[top_unretrieved_index], y=y[top_unretrieved_index], zorder=4, marker='X', color=pmas[3], linewidth=1, s=120, label='Top unretrieved', edgecolor='white')

        sns.scatterplot(x=x[top_retrieved_index], y=y[top_retrieved_index], zorder=4, marker='X', color=pmas[5], linewidth=1, s=120, label='Top retrieved', edgecolor='white')

        legend_handles = [
        mpatches.Rectangle((0, 0), 1, 1, color=pmas[2], alpha=0.5, label='Selected compounds'),
        mpatches.Rectangle((0, 0), 1, 1, color=pmas[8], alpha=0.5, label='Unselected compounds'),
        Line2D([0], [0], color=pmas[5], lw=1, label='Retrieved top compounds', alpha=1, marker='X', linestyle='None', markersize=12, markeredgewidth=1, markeredgecolor='white'),
        Line2D([0], [0], color='#e74c3c', lw=1, label='Unretrieved top compounds', alpha=1, marker='X', linestyle='None', markersize=12, markeredgewidth=1, markeredgecolor='white'),
        ]     

        # Create the legend with frame properties
        legend = ax.legend(
            handles=legend_handles,
            borderpad=0.8,
            loc='upper center',
            frameon=True,
            facecolor='white',
            edgecolor='black',
            fontsize=10,
            bbox_to_anchor=(0.5, 0),
            ncol=2
        )

        # Set legend title properties
        legend_title = legend.get_title()
        legend_title.set(fontsize=12, color='black', fontweight='bold')

        # Set axis limits
        ax.set(xlim=(-0.55, 0.55), ylim=(-0.55, 0.55))
        ax.set_title(f'{DATASET} | {top_category} | Final overview\n {get_clean_strategy(strategy_name)} | {get_clean_type(strategy_type)} | {get_clean_model(model)} | BS {get_clean_batchsize(batchsize)}', color='black', fontsize=12, pad=10)

        for text in legend.get_texts():
            text.set_color('black')

        fig.tight_layout()

        fig.savefig(f'{figures_path}/tmap_iterative/png/{top_category}/{experiment}/final.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{figures_path}/tmap_iterative/svg/{top_category}/{experiment}/final.svg', bbox_inches='tight')

        plt.close()

        # print(f"[{get_time()}]      ✅ {top_category} | {get_clean_strategy(strategy_name)} | {get_clean_type(strategy_type)} | {get_clean_model(model)} | {get_clean_batchsize(batchsize)} generated in {round(time.time() - t0)}s")

### Timeline TMAP
print(f"[{get_time()}]  ⚙️  Generation of timeline TMAP")

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

connections = [(a, b) for a, b in zip(s, t)]

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.axis('off')

sns.scatterplot(x=x, y=y, hue=tmap_df['iteration'], palette='RdYlGn', s=25, edgecolor='white', linewidth=0.3, alpha=0.8, zorder=1, hue_order=[0, 32])

iterations = list(range(tmap_df['iteration'].min()*TIMESTEP, tmap_df['iteration'].max()*TIMESTEP, 6))
if tmap_df['iteration'].max()*TIMESTEP not in iterations:
    iterations.append(tmap_df['iteration'].max()*TIMESTEP)

cmap = plt.cm.RdYlGn  # Use the same colormap as the plot
norm = plt.Normalize(vmin=0, vmax=tmap_df['iteration'].max())
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(norm(val)), markersize=10)
           for val in iterations]

for start, end in connections:
    ax.plot([x[start], x[end]], [y[start], y[end]], color='black', linewidth=1, alpha=0.5, zorder=0)

ax.set_title(f'{DATASET} | Project timeline', color='black', fontsize=14, pad=10)

# Create the legend with frame properties
legend = ax.legend(
    handles=handles,
    labels=[str(val) for val in iterations],
    title="Months after simulation start",
    borderpad=0.8,
    loc='upper center',
    frameon=True,
    facecolor='white',
    edgecolor='black',
    fontsize=10,
    bbox_to_anchor=(0.5, 0),
    ncol=5
)

fig.tight_layout()

fig.savefig(f'{figures_path}/tmap_overview/png/timeline.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{figures_path}/tmap_overview/svg/timeline.svg', bbox_inches='tight')

## Series TMAP
print(f"[{get_time()}]  ⚙️ Generation of series TMAP")

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.axis('off')

legend_handles = []

if tmap_df['SERIES_ID'].nunique() > 10:
    top9 = list(tmap_df['SERIES_ID'].value_counts().nlargest(9).index)
    tmap_df['SERIES_ID'] = np.where(tmap_df['SERIES_ID'].isin(top9),
                        tmap_df['SERIES_ID'],
                        'Others (smaller series aggregated)')
    top9.append('Others (smaller series aggregated)')
    ranked_series = top9
else:     
    ranked_series = list(tmap_df['SERIES_ID'].value_counts().index)

for i, series in enumerate(ranked_series):
    
    series_smiles = tmap_df[tmap_df['SERIES_ID'] == series]['SMILES'].values
    series_index = tmap_df[tmap_df['SMILES'].isin(series_smiles)]['index'].values

    sns.kdeplot(ax=ax, x=x[series_index], y=y[series_index], fill=True, alpha=0.5, levels=4, bw_adjust=0.1, zorder=1, gridsize=100, label=series, color=pmas[i])

    legend_handles.append(mpatches.Rectangle((0, 0), 1, 1, alpha=0.5, label=series, color=pmas[i]))

ax.legend(
    handles=legend_handles,
    borderpad=0.8,
    loc='upper center',
    frameon=True,
    facecolor='white',
    edgecolor='black',
    bbox_to_anchor=(0.5, -0.),
    fontsize=10,
    ncol=5
)

ax.set_title(f'{DATASET} | Chemical series', color='black', pad=10)

ax.set(xlim=(-0.55, 0.55), ylim=(-0.55, 0.55))

fig.tight_layout()


fig.savefig(f'{figures_path}/tmap_overview/png/series.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{figures_path}/tmap_overview/svg/series.svg', bbox_inches='tight')