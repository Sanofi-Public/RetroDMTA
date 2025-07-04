import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

from utils import get_clean_top_category, get_clean_strategy_type, get_clean_strategy_name, get_color_palette

pmas = get_color_palette()


def add_missing_iterations(df):
    # Get the unique replicates
    replicates = df['replicate'].unique()
    n_iterations = df['iteration'].max()
    # Create a full set of iterations (1 to 32)
    full_iterations = pd.DataFrame({'iteration': range(1, n_iterations + 1)})
    
    # Initialize an empty dataframe to store the new data
    new_df = pd.DataFrame()
    
    # Iterate over each replicate
    for rep in replicates:
        # Filter the data for the current replicate
        rep_data = df[df['replicate'] == rep]
        
        # Merge the full iteration range with the current replicate's data
        merged_data = pd.merge(full_iterations, rep_data, on='iteration', how='left')
        
        # Fill missing values with 0, except for 'selected_size', which should be 24
        merged_data['replicate'] = rep
        merged_data['selected_size'] = merged_data['selected_size'].fillna(24)
        merged_data.fillna(0, inplace=True)
        
        # Append the processed replicate data to te new dataframe
        new_df = pd.concat([new_df, merged_data], ignore_index=True)
        print(new_df.info())
    return new_df.sort_values(by=['replicate', 'iteration']), n_iterations

def generate_figure_1(df, dataset, strategy_type, strategy_name, endpoint):

    tmp_df = df[(df['strategy'] == f'{strategy_type}-{strategy_name}') & (df['endpoint'] == endpoint)].copy()
    tmp_df, n_iterations = add_missing_iterations(tmp_df)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    
    sns.barplot(ax=ax, data=tmp_df, x='iteration', y='selected_accuracy', errorbar='sd', color=pmas[0])
    ax.set_title(f'[{dataset}] [{get_clean_strategy_type(strategy_type)}] [{get_clean_strategy_name(strategy_name)}] [{endpoint}] Accuracy for selected batch at each iteration', weight='semibold', fontsize=10, pad=10)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.set_xlim([-1, n_iterations])
    ax.set_ylim([0, 1.15])

    ax.axhline(y=tmp_df['selected_accuracy'].mean(), label='Mean accuracy', c=pmas[2], linestyle='--')
    ax.axhline(y=tmp_df['selected_accuracy'].median(), label='Median accuracy', c=pmas[3], linestyle='--')

    legend_handles = [
    mpatches.Rectangle((0, 0), 1, 1, color=pmas[0], alpha=1, label='Accuracy on selected compounds'),
    Line2D([0], [0], marker='|', color='k', markeredgecolor='k', markersize=10, linewidth=0, label='Standard deviation of accuracy over all replicates', markeredgewidth=2),
    Line2D([0], [0], color=pmas[2], linewidth=2, linestyle='--', label='Mean accuracy'),
    Line2D([0], [0], color=pmas[3], linewidth=2, linestyle='--', label='Median accuracy')
    ]     

    legend = ax.legend(
        handles=legend_handles,
        borderpad=0.8,
        frameon=True,
        facecolor='white',
        edgecolor='black',
        bbox_to_anchor=(0.75, -0.22),
        ncol=2,
    )

    plt.tight_layout()
    
    plt.close()

    return fig

def generate_figure_2(df, dataset, strategy_type, strategy_name, endpoint):

    tmp_df = df[(df['strategy'] == f'{strategy_type}-{strategy_name}') & (df['endpoint'] == endpoint)].copy()
    tmp_df, n_iterations = add_missing_iterations(tmp_df)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    sns.barplot(ax=ax, data=tmp_df, x='iteration', y='selected_accuracy', errorbar='sd', color=pmas[0])

    ax2 = ax.twinx()
    sns.scatterplot(x=tmp_df[['iteration', 'selected_endpoint_size']].groupby(by='iteration').mean().reset_index()['iteration']-1, y=tmp_df[['iteration', 'selected_endpoint_size']].groupby(by='iteration').mean().reset_index()['selected_endpoint_size'], marker='_', ax=ax2, color=pmas[1], linewidth=3) # Replace 'your_lineplot_column' with the column you want to plot
    ax2.axhline(y=24, linestyle='--', color=pmas[1])
    ax.set_title(f'[{dataset}] [{get_clean_strategy_type(strategy_type)}] [{get_clean_strategy_name(strategy_name)}] [{endpoint}] Accuracy for selected batch at each iteration', weight='semibold', fontsize=10, pad=10)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy', color=pmas[0])
    ax2.set_ylabel('Selected documented molecules', color=pmas[1])
    ax.set_xlim([-1, n_iterations])
    ax.set_ylim([0, 1.15])
    ax2.set_ylim([0, 25])
    ax2.grid(False)

    ax.axhline(y=tmp_df['selected_accuracy'].mean(), label='Mean accuracy', c=pmas[2], linestyle='--')
    ax.axhline(y=tmp_df['selected_accuracy'].median(), label='Median accuracy', c=pmas[3], linestyle='--')

    legend_handles = [
    mpatches.Rectangle((0, 0), 1, 1, color=pmas[0], alpha=1, label='Accuracy on selected compounds'),
    Line2D([0], [0], marker='|', color='k', markeredgecolor='k', markersize=10, linewidth=0, label='Standard deviation of accuracy over all replicates', markeredgewidth=2),
    Line2D([0], [0], marker='_', color=pmas[1], markeredgecolor=pmas[1], markersize=5, linewidth=0, label='Number of selected molecules with experimental values', markeredgewidth=3),
    Line2D([0], [0], linestyle='--', color=pmas[1], linewidth=2, label='Batch size = 24', markeredgewidth=3),
    Line2D([0], [0], color=pmas[2], linewidth=2, linestyle='--', label='Mean accuracy'),
    Line2D([0], [0], color=pmas[3], linewidth=2, linestyle='--', label='Median accuracy')
    ]    

    legend = ax.legend(
        handles=legend_handles,
        borderpad=0.8,
        frameon=True,
        facecolor='white',
        edgecolor='black',
        bbox_to_anchor=(0.8, -0.22),
        ncol=2,
    )

    plt.tight_layout()
    
    plt.close()

    return fig

def generate_figure_3(df, dataset, strategy_type, strategy_name, endpoint):

    tmp_df = df[(df['strategy'] == f'{strategy_type}-{strategy_name}') & (df['endpoint'] == endpoint)].copy()
    tmp_df, n_iterations = add_missing_iterations(tmp_df)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    sns.boxplot(ax=ax, data=tmp_df, x='iteration', y='selected_accuracy', color=pmas[0])
    ax.set_title(f'[{dataset}] [{get_clean_strategy_type(strategy_type)}] [{get_clean_strategy_name(strategy_name)}] [{endpoint}] Accuracy for selected batch at each iteration', weight='semibold', fontsize=10, pad=10)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.set_xlim([-1, n_iterations])
    ax.set_ylim([0, 1.15])

    ax.axhline(y=tmp_df['selected_accuracy'].mean(), label='Mean accuracy', c=pmas[2], linestyle='--')
    ax.axhline(y=tmp_df['selected_accuracy'].median(), label='Median accuracy', c=pmas[3], linestyle='--')

    legend_handles = [
    mpatches.Rectangle((0, 0), 1, 1, color=pmas[0], alpha=1, label='Accuracy on selected compounds'),
    Line2D([0], [0], color=pmas[2], linewidth=2, linestyle='--', label='Mean accuracy'),
    Line2D([0], [0], color=pmas[3], linewidth=2, linestyle='--', label='Median accuracy')
    ]  

    legend = ax.legend(
        handles=legend_handles,
        borderpad=0.8,
        frameon=True,
        facecolor='white',
        edgecolor='black',
        bbox_to_anchor=(0.8, -0.22),
        ncol=3,
    ) 

    plt.tight_layout()
    
    plt.close()

    return fig

def generate_figure_4(df, dataset, strategy_type, strategy_name, endpoint):

    tmp_df = df[(df['strategy'] == f'{strategy_type}-{strategy_name}') & (df['endpoint'] == endpoint)]
    tmp_df, n_iterations = add_missing_iterations(tmp_df)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    sns.boxplot(ax=ax, data=tmp_df, x='iteration', y='selected_accuracy', color=pmas[0])

    ax2 = ax.twinx()
    sns.scatterplot(x=tmp_df[['iteration', 'selected_endpoint_size']].groupby(by='iteration').mean().reset_index()['iteration']-1, y=tmp_df[['iteration', 'selected_endpoint_size']].groupby(by='iteration').mean().reset_index()['selected_endpoint_size'], marker='_', ax=ax2, color=pmas[1], linewidth=3) # Replace 'your_lineplot_column' with the column you want to plot
    ax2.axhline(y=24, linestyle='--', color=pmas[1])
    ax.set_title(f'[{dataset}] [{get_clean_strategy_type(strategy_type)}] [{get_clean_strategy_name(strategy_name)}] [{endpoint}] Accuracy for selected batch at each iteration', weight='semibold', fontsize=10, pad=10)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy', color=pmas[0])
    ax2.set_ylabel('Selected documented molecules', color=pmas[1])
    ax.set_xlim([-1, n_iterations])
    ax.set_ylim([0, 1.15])
    # ax2.set_ylim([0, 25])
    ax2.grid(False)

    ax.axhline(y=tmp_df['selected_accuracy'].mean(), label='Mean accuracy', c=pmas[2], linestyle='--')
    ax.axhline(y=tmp_df['selected_accuracy'].median(), label='Median accuracy', c=pmas[3], linestyle='--')

    legend_handles = [
    mpatches.Rectangle((0, 0), 1, 1, color=pmas[0], alpha=1, label='Accuracy on selected compounds'),
    Line2D([0], [0], marker='_', color=pmas[1], markeredgecolor=pmas[1], markersize=5, linewidth=0, label='Number of selected molecules with experimental values', markeredgewidth=3),
    Line2D([0], [0], linestyle='--', color=pmas[1], linewidth=1, label='Batch size = 24', markeredgewidth=3),
    Line2D([0], [0], color=pmas[2], linewidth=2, linestyle='--', label='Mean accuracy'),
    Line2D([0], [0], color=pmas[3], linewidth=2, linestyle='--', label='Median accuracy')
    ]     

    legend = ax.legend(
        handles=legend_handles,
        borderpad=0.8,
        frameon=True,
        facecolor='white',
        edgecolor='black',
        bbox_to_anchor=(0.93, -0.22),
        ncol=3,
    )

    plt.tight_layout()
    
    plt.close()

    return fig