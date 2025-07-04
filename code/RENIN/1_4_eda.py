# Utils
import os
from natsort import natsorted
from tqdm.auto import tqdm
import time

# Data
import numpy as np
import pandas as pd

# Viz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from utils import load_pickle, load_common_config, sum_values, get_pmas_palette, extract_year_month, sigmoid_HTB, sigmoid_INT, sigmoid_LTB, get_time

# Set Seaborn theme
sns.set_theme(style='ticks', context='paper')

# Update Matplotlib configuration
plt.rcParams.update({
    'axes.titlesize': 12,         # Title font size
    'axes.labelsize': 10,         # Axis label font size
    'xtick.labelsize': 10,        # X-axis tick label font size
    'ytick.labelsize': 10,        # Y-axis tick label font size
    'legend.fontsize': 8,        # Legend font size
    'lines.linewidth': 1,         # Line width
    'lines.markersize': 8,        # Marker size
    'axes.grid': True,            # Show grid
})

dict_sigmoid_functions = {
    'H': sigmoid_HTB,
    'L': sigmoid_LTB,
    'V': sigmoid_INT
    }

pmas = get_pmas_palette()

use_tqdm = False

t_init = time.time()

# Determine the dataset name from the current working directory
DATASET = os.getcwd().split('/')[-1]

# Define base paths for dataset
data_path = f'../../data/{DATASET}'
figures_path = f'../../figures/{DATASET}'

SAVE_FIGURES = True
if SAVE_FIGURES:
    os.makedirs(f'{figures_path}/eda/png', exist_ok=True)
    os.makedirs(f'{figures_path}/eda/svg', exist_ok=True)

# Load common config file
config = load_common_config(f'../../data/common/datasets_config.json')

# Extract date parameters from config
INITIAL_DATE = pd.to_datetime(config[DATASET]['initial_date'])
FINAL_DATE = pd.to_datetime(config[DATASET]['final_date'])
TIMESTEP = config[DATASET]['timestep']

# Load data files
data_df = pd.read_csv(f'{data_path}/data.csv')
datagg_df = pd.read_csv(f'{data_path}/data_aggregated.csv')
bp_df = pd.read_csv(f'{data_path}/blueprint.csv')

# Convert date columns to datetime format
data_df['DATE'] = pd.to_datetime(data_df['DATE'])
datagg_df['DATE'] = pd.to_datetime(datagg_df['DATE'])

# Load the weights from blueprint
weight_dict = dict(zip(bp_df['PROPERTIES'], bp_df['WEIGHT']))

df = data_df.copy()
df_agg = datagg_df.copy()

endpoints = natsorted(bp_df['PROPERTIES'].values)

df = extract_year_month(df)
df_agg = extract_year_month(df_agg)

### Documented molecules per endpoint
fig, ax = plt.subplots(1, 1, figsize=(7, 4))
index = df_agg[endpoints].count().index
values = df_agg[endpoints].count().values
sns.barplot(y=index, x=values, color=pmas[0])

ax.set_xlabel('Number of documented molecules', labelpad=10)
ax.set_ylabel('Endpoint', labelpad=10)
ax.set_title(f'{DATASET} | Documented molecules per endpoint', pad=10, fontsize=12)

plt.tight_layout()

if SAVE_FIGURES:
    fig.savefig(f'{figures_path}/eda/png/molecules_per_endpoint.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{figures_path}/eda/svg/molecules_per_endpoint.svg', bbox_inches='tight')

plt.close()

### Distribution of experimental assays over time
if len(endpoints) >= 3:
    n_col = 3
    n_row = int(np.ceil(len(endpoints) / n_col))

    fig, ax = plt.subplots(n_row, n_col, figsize=(7*n_col, 4*n_row), sharex=True, sharey=True)
    ax = np.ravel(ax)

    for i, prop in tqdm(enumerate(endpoints), total=len(endpoints), disable=not use_tqdm):
        tmp_df = df[[prop, 'DATE']].dropna()

        sns.histplot(data=tmp_df, x='DATE', ax=ax[i], color=pmas[0], bins=50)
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax[i].xaxis.set_major_locator(mdates.YearLocator())
        ax[i].set_title(prop, pad=5)

    fig.suptitle(f'{DATASET} | Distribution of assays', y=0.99, fontsize=12)

elif len(endpoints) == 2:
    n_col = 2
    n_row = 1

    fig, ax = plt.subplots(n_row, n_col, figsize=(7*n_col, 4*n_row), sharex=True, sharey=True)
    ax = np.ravel(ax)

    for i, prop in tqdm(enumerate(endpoints), total=len(endpoints), disable=not use_tqdm):
        tmp_df = df[[prop, 'DATE']].dropna()

        sns.histplot(data=tmp_df, x='DATE', ax=ax[i], color=pmas[0], bins=50)
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax[i].xaxis.set_major_locator(mdates.YearLocator())
        ax[i].set_title(prop, pad=5)

    fig.suptitle(f'{DATASET} | Distribution of assays', y=0.99, fontsize=12)

elif len(endpoints) == 1:
    n_col = 1
    n_row = 1

    fig, ax = plt.subplots(n_row, n_col, figsize=(7*n_col, 4*n_row), sharex=True, sharey=True)

    for i, prop in tqdm(enumerate(endpoints), total=len(endpoints), disable=not use_tqdm):
        tmp_df = df[[prop, 'DATE']].dropna()

        sns.histplot(data=tmp_df, x='DATE', ax=ax, color=pmas[0], bins=50)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.set_title(prop, pad=5)

    fig.suptitle(f'{DATASET} | Distribution of assays', y=0.99, fontsize=12)

fig.tight_layout()   

if SAVE_FIGURES:
    fig.savefig(f'{figures_path}/eda/png/distribution_of_assays.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{figures_path}/eda/svg/distribution_of_assays.svg', bbox_inches='tight')

plt.close()

### Distribution of experimental assays values and blueprint range
fig, ax = plt.subplots(n_row, n_col, figsize=(7*n_col, 4*n_row))
ax = np.ravel(ax)

for i, prop in tqdm(enumerate(endpoints), total=len(endpoints), disable=not use_tqdm):
    tmp_df = df_agg[[prop, 'DATE']].dropna()

    LTV = bp_df[bp_df['PROPERTIES'] == prop]['LTV'].values[0]
    HTV = bp_df[bp_df['PROPERTIES'] == prop]['HTV'].values[0]
    TREND = bp_df[bp_df['PROPERTIES'] == prop]['TREND'].values[0]

    sns.histplot(data=tmp_df, x=prop, ax=ax[i], color=pmas[0], bins=50, zorder=1)
    xmin, xmax = ax[i].get_xlim()
    ymin, ymax = ax[i].get_ylim()
    if TREND == 'H':
        ax[i].fill_betweenx(x1=LTV, x2=xmax, y=ax[i].get_ylim(), color=pmas[4], alpha=0.2, label='Optimal range', zorder=0)
    elif TREND == 'L':
        ax[i].fill_betweenx(x1=xmin, x2=HTV, y=ax[i].get_ylim(), color=pmas[4], alpha=0.2, label='Optimal range', zorder=0)
    else:
        ax[i].fill_betweenx(x1=LTV, x2=HTV, y=ax[i].get_ylim(), color=pmas[4], alpha=0.2, label='Optimal range', zorder=0)

    ax[i].axvline(x=LTV, color=pmas[4], linestyle='--', linewidth=2, zorder=2, label='LTV & HTV')
    ax[i].axvline(x=HTV, color=pmas[4], linestyle='--', linewidth=2, zorder=2)

    ax[i].axvline(x=tmp_df[prop].median(), color=pmas[1], linestyle='--', label='Median', linewidth=2, zorder=2)
    ax[i].axvline(x=tmp_df[prop].mean(), color=pmas[2], linestyle='--', label='Mean', linewidth=2, zorder=2)
    ax[i].legend(fontsize=7)

    
    ax[i].set_xlabel(f"{prop} experimental values")
    ax[i].set_title(f"{prop} ({TREND})")

    ax[i].set_xlim([xmin, xmax])
    ax[i].set_ylim([ymin, ymax])

fig.suptitle(f'{DATASET} | Distribution of endpoints values', y=0.99, fontsize=12)
fig.tight_layout()

if SAVE_FIGURES:
    fig.savefig(f'{figures_path}/eda/png/distribution_of_endpoints.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{figures_path}/eda/svg/distribution_of_endpoints.svg', bbox_inches='tight')

plt.close()

### Endpoints correlation
cmap = LinearSegmentedColormap.from_list("pmas_coolwarm", [pmas[2], 'white', pmas[3]])

corr = df_agg[endpoints].corr(method='pearson')

# Create a mask to hide the upper triangle of the correlation matrix
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(len(endpoints), len(endpoints)))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, annot=True, fmt=".2f", cbar=False)
plt.xticks(rotation=45, ha='right')
plt.xlabel("")
plt.ylabel("")
plt.title(f"{DATASET} | Endpoints Pearson Correlation Matrix", pad=10, fontsize=12)

plt.tight_layout()

if SAVE_FIGURES:
    fig.savefig(f'{figures_path}/eda/png/pearson_correlation_matrix.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{figures_path}/eda/svg/pearson_correlation_matrix.svg', bbox_inches='tight')


corr = df_agg[endpoints].corr(method='spearman')

# Create a mask to hide the upper triangle of the correlation matrix
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(len(endpoints), len(endpoints)))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, annot=True, fmt=".2f", cbar=False)
plt.xticks(rotation=45, ha='right')
plt.xlabel("")
plt.ylabel("")
plt.title(f"{DATASET} | Endpoints Spearman Correlation Matrix", pad=10, fontsize=12)

plt.tight_layout()

if SAVE_FIGURES:
    fig.savefig(f'{figures_path}/eda/png/spearman_correlation_matrix.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{figures_path}/eda/svg/spearman_correlation_matrix.svg', bbox_inches='tight')


corr = df_agg[endpoints].corr(method='kendall')

# Create a mask to hide the upper triangle of the correlation matrix
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(len(endpoints), len(endpoints)))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, annot=True, fmt=".2f", cbar=False)
plt.xticks(rotation=45, ha='right')
plt.xlabel("")
plt.ylabel("")
plt.title(f"{DATASET} | Endpoints Kendall Correlation Matrix", pad=10, fontsize=12)

plt.tight_layout()

if SAVE_FIGURES:
    fig.savefig(f'{figures_path}/eda/png/kendall_correlation_matrix.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{figures_path}/eda/svg/kendall_correlation_matrix.svg', bbox_inches='tight')

plt.close()

### Experimental desirability scores
compute_df = df_agg.copy()

for prop in endpoints:
    trend, ltv, htv, weight = np.squeeze(bp_df[bp_df['PROPERTIES'] == prop][['TREND', 'LTV', 'HTV', 'WEIGHT']].values)
    sigmoid_func = dict_sigmoid_functions[trend]
    compute_df[f'normalized_{prop}'] = compute_df[prop].apply(lambda x: sigmoid_func(x, ltv, htv, q=1))
    compute_df[f"geometric_{prop}"] = compute_df[f'normalized_{prop}']**weight
    compute_df[f"arithmetic_{prop}"] = compute_df[f'normalized_{prop}']*weight

compute_df['total_weight'] = compute_df[endpoints].apply(lambda row: sum_values(row, weight_dict), axis=1)
compute_df['exp_geometric_score'] = compute_df.filter(regex='^geometric_').prod(axis=1)**(1/compute_df['total_weight'].values)
compute_df['exp_arithmetic_score'] = compute_df.filter(regex='^arithmetic_').sum(axis=1)/compute_df['total_weight'].values

score_df = df_agg.merge(compute_df[['SMILES', 'total_weight', 'exp_geometric_score', 'exp_arithmetic_score']], on='SMILES')
score_df['documentation'] = score_df[endpoints].notna().sum(axis=1)/len(endpoints)
score_df['documentation'] = score_df['documentation'].round(2)

fig, ax = plt.subplots(1, 1, figsize=(7, 4))

sns.histplot(data=score_df, x='exp_geometric_score', label='Desirability score', color=pmas[0], kde=False)

ax.set_xlabel('Desirability score')
ax.legend(fontsize=10)
ax.set_title(f'{DATASET} | Experimental desirability scores distribution', pad=10, fontsize=12)
ax.set_xlim([0, 1])

plt.tight_layout()

if SAVE_FIGURES:
    fig.savefig(f'{figures_path}/eda/png/desirability_scores_distribution.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{figures_path}/eda/svg/desirability_scores_distribution.svg', bbox_inches='tight')

plt.close()

fig, ax = plt.subplots(1, 1, figsize=(7, 4))

sns.histplot(data=score_df, x='documentation', label='Documentation', color=pmas[0], kde=False)

ax.set_xlabel('Documentation')
ax.legend(fontsize=10)
ax.set_title(f'{DATASET} | Documentation distribution', pad=10, fontsize=12)
ax.set_xlim([0, 1])

plt.tight_layout()

if SAVE_FIGURES:
    fig.savefig(f'{figures_path}/eda/png/documentation_distribution.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{figures_path}/eda/svg/documentation_distribution.svg', bbox_inches='tight')

plt.close()

### Top molecules 
top_molecules = load_pickle(f'{data_path}/top_molecules.pkl')

top_molecules_count = pd.DataFrame()

for key in top_molecules['geometric']:
    tmp_df = pd.DataFrame()
    tmp_df['scoring'] = ['geometric']
    tmp_df['top'] = [key]
    tmp_df['size'] = len(top_molecules['geometric'][key])
    top_molecules_count = pd.concat([top_molecules_count, tmp_df])

fig, ax = plt.subplots(1, 1, figsize=(7, 4))
sns.barplot(data=top_molecules_count, x='top', y='size', ax=ax, color=pmas[0])

# Add y-values on top of each bar
for p in ax.patches:
    height = p.get_height()
    ax.text(
        p.get_x() + p.get_width() / 2.,  # x-coordinate
        height + 5,  # y-coordinate
        f'{int(height)}',  # text to display
        ha='center',  # horizontal alignment
        va='bottom',  # vertical alignment
        rotation=0,  # rotation angle
        fontweight='bold',
        color=pmas[0],
        bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0')
    )

    ax.set_title(f'{DATASET} | Number of molecules in each top category', pad=10, fontsize=12)
    ax.set_xlabel('')
    ax.set_ylabel('Size', labelpad=10)

ymax = top_molecules_count['size'].max()

plt.xticks(rotation=45, ha='right')
ax.tick_params(axis='x', which='both', length=0, pad=5)

ax.set_ylim([0, ymax + 50])

plt.tight_layout()

if SAVE_FIGURES:
    fig.savefig(f'{figures_path}/eda/png/top_categories_both.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{figures_path}/eda/svg/top_categories_both.svg', bbox_inches='tight')

plt.close()

### Good, potentially good and bad molecules
score_df['DocDScore'] = score_df['exp_geometric_score']*score_df['documentation']

top_categories = [x for x in top_molecules['geometric'].keys()]

thresholds = {}

for top_category in top_categories:

    if 'Top DocDScore' in top_category:
        percentile = float(top_category.split('(')[-1].split('%')[0]) / 100
        thresholds[top_category] = score_df['DocDScore'].quantile(1-percentile)
    elif 'σ' in top_category:
        mean = score_df[score_df['documentation'] > 0.65]['exp_geometric_score'].mean()
        std = score_df[score_df['documentation'] > 0.65]['exp_geometric_score'].std()
        thresholds[top_category] = mean + float(top_category.split('(')[-1].split('σ')[0]) * std
    elif 'Top DScore' in top_category:
        percentile = float(top_category.split('(')[-1].split('%')[0]) / 100
        thresholds[top_category] = score_df[score_df['documentation'] > 0.65]['exp_geometric_score'].quantile(1-percentile)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))

xticks = [round(i, 2) for i in np.linspace(0, 1, len(endpoints)+1)]

# Bad molecules
bad_rect = Rectangle((-1, 0), 1000, 0.5, color=pmas[3], alpha=0.3, label='Bad molecules', zorder=0)
ax.add_patch(bad_rect)

# Potentially good molecules
pot_rect = Rectangle((-1, 0.5), 1 + 0.65 * (len(xticks) - 1), 0.55, color=pmas[1], alpha=0.3, label='Potentially good molecules', zorder=0)
ax.add_patch(pot_rect)

# Good molecules
good_rect = Rectangle((0.65 * (len(xticks) - 1), 0.5), 1000, 0.55, color=pmas[5], alpha=0.3, label='Good molecules', zorder=0)
ax.add_patch(good_rect)

sns.boxplot(data=score_df, x='documentation', y='exp_geometric_score', order=xticks, ax=ax, color=pmas[2], zorder=2)

ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, zorder=1)
ax.axvline(x=0.65 * (len(xticks) - 1), color='gray', linestyle='--', linewidth=1.5, zorder=1)

ax.set_ylim([0, 1.05])

ax.set_title(f"{DATASET} | Experimental Desirability Score vs Documentation distribution", pad=10, fontsize=12)
ax.set_xlabel('Documentation')
ax.set_ylabel('Desirability score')

ax.grid(False)
ax.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    fontsize=10
    )
plt.xticks(rotation=45)
plt.tight_layout()

if SAVE_FIGURES:
    fig.savefig(f'{figures_path}/eda/png/desirability_documentation_distribution.png', dpi=300)
    fig.savefig(f'{figures_path}/eda/svg/desirability_documentation_distribution.svg')

fig, ax = plt.subplots(1, 1, figsize=(9, 6))

xticks = [round(i, 2) for i in np.linspace(0, 1, len(endpoints)+1)]
ax.set_xticks(xticks)

# Bad molecules
rect = Rectangle((0, 0), 2, 0.5, color=pmas[3], alpha=0.3, label='Bad molecules', zorder=0)
ax.add_patch(rect)

# Potentially good molecules
rect = Rectangle((0, 0.5), 0.65, 0.55, color=pmas[1], alpha=0.3, label='Potentially good molecules', zorder=0)
ax.add_patch(rect)

# Good molecules
rect = Rectangle((0.65, 0.5), 2, 0.55, color=pmas[5], alpha=0.3, label='Good molecules', zorder=0)
ax.add_patch(rect)

sns.scatterplot(data=score_df, x='documentation', y='exp_geometric_score', ax=ax, color=pmas[-1], s=20, zorder=1)

x = np.linspace(0.0001, 1.05, 1000)

for i, top_category in enumerate(natsorted([x for x in thresholds.keys() if 'DocDScore' in x], reverse=True)):
    threshold = thresholds[top_category]
    y = threshold / x
    sns.lineplot(x=x, y=y, linestyle='--', color=pmas[i], linewidth=1.5, zorder=2)
    sns.scatterplot(x=score_df[score_df['DocDScore'] > threshold]['documentation'], y=score_df[score_df['DocDScore'] > threshold]['exp_geometric_score'], color=pmas[i], s=20, zorder=2, label=top_category)

if 'ANN' in score_df.columns:
    ann_df = score_df[~score_df['ANN'].isna()].copy()
    sns.scatterplot(x=ann_df['documentation'], y=ann_df['exp_geometric_score'], color=pmas[-1], s=50, marker='s', linewidth=1, zorder=2, label='Annotated molecules')

if 'ANN' in score_df.columns:
    for i, top_category in enumerate(natsorted([x for x in thresholds.keys() if 'DocDScore' in x], reverse=True)):
        threshold = thresholds[top_category]
        subset_df = score_df[(score_df['documentation'] > 0.65) & (score_df['exp_geometric_score'] > threshold)].copy()
        ann_df = subset_df[~subset_df['ANN'].isna()].copy()
        sns.scatterplot(x=ann_df['documentation'], y=ann_df['exp_geometric_score'], color=pmas[-1], s=50, marker='s', linewidth=1, zorder=0)

ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, zorder=1)
ax.axvline(x=0.65, color='gray', linestyle='--', linewidth=1.5, zorder=1)

ax.set_xlim([0, 1.05])
ax.set_ylim([0, 1.05])

ax.set_title(f"{DATASET} | Top DocDScore categories", pad=10, fontsize=12)
ax.set_xlabel('Documentation')
ax.set_ylabel('Experimental desirability score')

ax.grid(False)
ax.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    fontsize=10
    )

plt.xticks(rotation=45)
plt.tight_layout()

if SAVE_FIGURES:
    fig.savefig(f'{figures_path}/eda/png/DocDScore_definition.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{figures_path}/eda/svg/DocDScore_definition.svg', bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(9, 6))

xticks = [round(i, 2) for i in np.linspace(0, 1, len(endpoints)+1)]
ax.set_xticks(xticks)

# Bad molecules
rect = Rectangle((0, 0), 2, 0.5, color=pmas[3], alpha=0.3, label='Bad molecules', zorder=0)
ax.add_patch(rect)

# Potentially good molecules
rect = Rectangle((0, 0.5), 0.65, 0.55, color=pmas[1], alpha=0.3, label='Potentially good molecules', zorder=0)
ax.add_patch(rect)

# Good molecules
rect = Rectangle((0.65, 0.5), 2, 0.55, color=pmas[5], alpha=0.3, label='Good molecules', zorder=0)
ax.add_patch(rect)

sns.scatterplot(data=score_df, x='documentation', y='exp_geometric_score', ax=ax, color=pmas[-1], s=20, zorder=1)

for i, top_category in enumerate(natsorted([x for x in thresholds.keys() if 'Top DScore' in x and '%' in x], reverse=True)):
    threshold = thresholds[top_category]
    sns.lineplot(x=[0.65, 1.05], y=[threshold, threshold], color=pmas[i], linewidth=1.5, linestyle='--', zorder=2)
    sns.scatterplot(x=score_df[(score_df['documentation'] > 0.65) & (score_df['exp_geometric_score'] > threshold)]['documentation'], y=score_df[(score_df['documentation'] > 0.65) & (score_df['exp_geometric_score'] > threshold)]['exp_geometric_score'], color=pmas[i], s=20, zorder=2, label=top_category)

if 'ANN' in score_df.columns:
    ann_df = score_df[~score_df['ANN'].isna()].copy()
    sns.scatterplot(x=ann_df['documentation'], y=ann_df['exp_geometric_score'], color=pmas[-1], s=50, marker='s', linewidth=1, zorder=2, label='Annotated molecules')

if 'ANN' in score_df.columns:
    for i, top_category in enumerate(natsorted([x for x in thresholds.keys() if 'Top DScore' in x], reverse=True)):
        threshold = thresholds[top_category]
        subset_df = score_df[(score_df['documentation'] > 0.65) & (score_df['exp_geometric_score'] > threshold)].copy()
        ann_df = subset_df[~subset_df['ANN'].isna()].copy()
        sns.scatterplot(x=ann_df['documentation'], y=ann_df['exp_geometric_score'], color=pmas[-1], s=50, marker='s', linewidth=1, zorder=0)


ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, zorder=1)
ax.axvline(x=0.65, color='gray', linestyle='--', linewidth=1.5, zorder=1)

ax.set_xlim([0, 1.05])
ax.set_ylim([0, 1.05])

ax.set_title(f"{DATASET} | Top DScore (%) categories", pad=10, fontsize=12)
ax.set_xlabel('Documentation')
ax.set_ylabel('Desirability score')

ax.grid(False)
ax.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    fontsize=10
    )

plt.xticks(rotation=45)
plt.tight_layout()

if SAVE_FIGURES:
    fig.savefig(f'{figures_path}/eda/png/DScore_prct_definition.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{figures_path}/eda/svg/DScore_prct_definition.svg', bbox_inches='tight')

plt.close()

fig, ax = plt.subplots(1, 1, figsize=(9, 6))

xticks = [round(i, 2) for i in np.linspace(0, 1, len(endpoints)+1)]

# Bad molecules
rect = Rectangle((0, 0), 2, 0.5, color=pmas[3], alpha=0.3, label='Bad molecules', zorder=0)
ax.add_patch(rect)

# Potentially good molecules
rect = Rectangle((0, 0.5), 0.65, 0.55, color=pmas[1], alpha=0.3, label='Potentially good molecules', zorder=0)
ax.add_patch(rect)

# Good molecules
rect = Rectangle((0.65, 0.5), 2, 0.55, color=pmas[5], alpha=0.3, label='Good molecules', zorder=0)
ax.add_patch(rect)

sns.scatterplot(data=score_df, x='documentation', y='exp_geometric_score', ax=ax, color=pmas[-1], s=20, zorder=1)

for i, top_category in enumerate(natsorted([x for x in thresholds.keys() if 'Top DScore' in x and 'σ' in x], reverse=False)):
    threshold = thresholds[top_category]
    sns.lineplot(x=[0.65, 1.05], y=[threshold, threshold], color=pmas[i], linewidth=1.5, linestyle='--', zorder=2)
    sns.scatterplot(x=score_df[(score_df['documentation'] > 0.65) & (score_df['exp_geometric_score'] > threshold)]['documentation'], y=score_df[(score_df['documentation'] > 0.65) & (score_df['exp_geometric_score'] > threshold)]['exp_geometric_score'], color=pmas[i], s=20, zorder=2, label=top_category)

if 'ANN' in score_df.columns:
    ann_df = score_df[~score_df['ANN'].isna()].copy()
    sns.scatterplot(x=ann_df['documentation'], y=ann_df['exp_geometric_score'], color=pmas[-1], s=50, marker='s', linewidth=1, zorder=2, label='Annotated molecules')

if 'ANN' in score_df.columns:
    for i, top_category in enumerate(natsorted([x for x in thresholds.keys() if 'Top DScore' in x], reverse=True)):
        threshold = thresholds[top_category]
        subset_df = score_df[(score_df['documentation'] > 0.65) & (score_df['exp_geometric_score'] > threshold)].copy()
        ann_df = subset_df[~subset_df['ANN'].isna()].copy()
        sns.scatterplot(x=ann_df['documentation'], y=ann_df['exp_geometric_score'], color=pmas[-1], s=50, marker='s', linewidth=1, zorder=0)


ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, zorder=1)
yticks = list(ax.get_yticks())
if 0.5 not in yticks:
    yticks.append(0.5)
yticks.sort()
ax.set_yticks(yticks)

ax.axvline(x=0.65, color='gray', linestyle='--', linewidth=1.5, zorder=1)
if 0.65 not in xticks:
    xticks.append(0.65)
xticks.sort()
ax.set_xticks(xticks)

ax.set_xlim([0, 1.05])
ax.set_ylim([0, 1.05])

ax.set_title(f"{DATASET} | Top DScore (σ) categories", pad=10, fontsize=12)
ax.set_xlabel('Documentation')
ax.set_ylabel('Desirability score')

ax.grid(False)
ax.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,
    fontsize=10
    )


plt.xticks(rotation=45)
plt.tight_layout()

if SAVE_FIGURES:
    fig.savefig(f'{figures_path}/eda/png/DScore_sigma_definition.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{figures_path}/eda/svg/DScore_sigma_definition.svg', bbox_inches='tight')

plt.close()

### Pairwise Tanimoto Similarity distribution
similarity_matrix = pd.read_parquet(f'{data_path}/similarity_matrix.parquet')

def get_pairwise_similarities(matrix):
    np.fill_diagonal(matrix, np.nan)
    matrix[np.triu_indices_from(matrix, k=1)] = np.nan
    pairwise_similarities = np.ravel(matrix)
    pairwise_similarities = pairwise_similarities[~np.isnan(pairwise_similarities)]
    return pairwise_similarities

pairwise_similarities = get_pairwise_similarities(similarity_matrix.values)

fig, ax = plt.subplots(1, 1, figsize=(7, 4))
sns.histplot(pairwise_similarities, bins=50, color=pmas[0], ax=ax)
ax.get_yaxis().set_visible(False)
ax.set_xlabel('Tanimoto Similarity')
ax.set_title(f'{DATASET} | Pairwise Tanimoto Similarity Distribution', pad=10, fontsize=12)
plt.tight_layout()

if SAVE_FIGURES:
    fig.savefig(f'{figures_path}/eda/png/pairwise_tanimoto_similarity_distribution.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{figures_path}/eda/svg/pairwise_tanimoto_similarity_distribution.svg', bbox_inches='tight')

plt.close()

### Molecules over time
if 'ANN' in df_agg:
    agg_dict = {
        **{col: 'mean' for col in data_df.columns 
        if col not in ['SMILES', 'CPD_ID', 'SERIES_ID', 'ANN', 'DATE']},
        'DATE': 'min'
    }

    na_df = data_df[data_df['ANN'].isna()].copy()
    na_df['ANN'] = 'NAN'

    na_df = na_df.groupby(by=['SMILES', 'CPD_ID', 'SERIES_ID', 'ANN']).agg(agg_dict).reset_index()
    na_df['ANN'] = np.nan

    agg_dict = {
        **{col: 'mean' for col in data_df.columns 
        if col not in ['SMILES', 'CPD_ID', 'SERIES_ID', 'ANN', 'DATE']},
        'DATE': 'max'
    }

    notna_df = data_df[data_df['ANN'].notna()].copy()

    notna_df = notna_df.groupby(by=['SMILES', 'CPD_ID', 'SERIES_ID', 'ANN']).agg(agg_dict).reset_index()

    ann_df = pd.concat([na_df, notna_df]).sort_values(by='DATE').reset_index(drop=True)
    tmp_df = ann_df[['SMILES', 'DATE', 'ANN']].copy()
else:
    tmp_df = df_agg[['SMILES', 'DATE']].copy()

start_date = tmp_df['DATE'].min()
end_date = tmp_df['DATE'].max()
days_difference = (end_date - start_date).days

fig, ax = plt.subplots(1, 1, figsize=(10, 4))

sns.histplot(data=tmp_df, x='DATE', color=pmas[0], bins=round(days_difference/7), label='New molecules')

ax.set_xlabel('Date')
ax.set_ylabel('Number of new molecules')
ax.set_title(f'{DATASET} | New molecules over time', pad=10, fontsize=12)

ax.axvline(x=INITIAL_DATE, color=pmas[-1], linestyle='--', linewidth=2, label=f"Simulation starting date ({str(INITIAL_DATE).split(' ')[0]})")

if 'ANN' in tmp_df.columns:

    for i, ann_category in enumerate(natsorted(tmp_df['ANN'].unique())):

        ann_df = tmp_df[tmp_df['ANN'] == ann_category].copy()
        ann_df = ann_df.reset_index()

        for j, row in ann_df.iterrows():
            ax.axvline(x=row['DATE'], color=pmas[i+1], linestyle='--', linewidth=2, label=f"{ann_category}")
            
ax.legend(
    fontsize=9
    )

plt.tight_layout()

if SAVE_FIGURES:
    fig.savefig(f'{figures_path}/eda/png/molecules_overtime.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{figures_path}/eda/svg/molecules_overtime.svg', bbox_inches='tight')

plt.close()

tmp_df.sort_values(by='DATE', inplace=True)
tmp_df['cumulative_count'] = range(1, len(tmp_df) + 1)

fig, ax = plt.subplots(1, 1, figsize=(21, 7))

sns.lineplot(data=tmp_df, x='DATE', y='cumulative_count', color=pmas[0], linewidth=2, label='Cumulative sum of new molecules')

ax.set_xlabel('Date')
ax.set_ylabel('Cumulative sum of new molecules')
ax.set_title(f'[{DATASET}] Cumulative sum of new molecules over time', pad=10, fontsize=12)

ax.axvline(x=INITIAL_DATE, color=pmas[-1], linestyle='--', linewidth=2, label=f"Simulation starting date ({str(INITIAL_DATE).split(' ')[0]})")

if 'ANN' in tmp_df.columns:

    for i, ann_category in enumerate(natsorted(tmp_df['ANN'].unique())):

        ann_df = tmp_df[tmp_df['ANN'] == ann_category].copy()
        ann_df = ann_df.reset_index()
        for j, row in ann_df.iterrows():
            ax.axvline(x=row['DATE'], color=pmas[i+1], linestyle='--', linewidth=2, label=f"{ann_category}")

ax.legend(fontsize=10)

plt.tight_layout()

if SAVE_FIGURES:
    fig.savefig(f'{figures_path}/eda/png/cumsum_molecules_overtime.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{figures_path}/eda/svg/cumsum_molecules_overtime.svg', bbox_inches='tight')

print(f"[{get_time()}]  ✅  Preprocessing (1.4) : EDA Figures in {time.time() - t_init:.2f} seconds")