import io
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import glob
from natsort import natsorted
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from utils import get_pmas_palette, get_clean_type, get_clean_strategy, get_clean_model, get_clean_batchsize, get_clean_exploration_metric

sns.set_theme(context='paper', style='ticks')
plt.rcParams.update({
    'axes.titlesize': 12,           # Title font size
    'axes.labelsize': 10,           # Axis label font size
    'xtick.labelsize': 10,          # X-axis tick label font size
    'ytick.labelsize': 10,          # Y-axis tick label font size
    'legend.fontsize': 8,           # Legend font size
    'lines.linewidth': 1,           # Line width
    'lines.markersize': 8,          # Marker size
    'axes.grid': False,             # Show grid
})

pmas = get_pmas_palette()

def show(base_path, dataset):
        
    col1, col2 = st.columns(2)

    with col1:
        data_exploitation = pd.read_parquet(os.path.join(base_path, f'data/{dataset}/exploitation.parquet'))
        top_categories = natsorted(data_exploitation['top_category'].unique())
        top_category = st.selectbox(
            label = 'Top category',
            options = top_categories,
            index=0
        )

    with col2:
        metrics = natsorted([x.split('/')[-1].split('.parquet')[0] for x in glob.glob(os.path.join(base_path, f'data/{dataset}/exploration_*.parquet'))], key=lambda x: get_clean_exploration_metric(x))
        metric = st.selectbox(
            label = 'Exploration metric',
            options = metrics,
            format_func = get_clean_exploration_metric
            )

        data_exploration = pd.read_parquet(os.path.join(base_path, f'data/{dataset}/{metric}.parquet'))

        if metric == 'exploration_neighborhood_coverage':
            data_exploration['distance_threshold'] = data_exploration['distance_threshold'].round(2)
            threshold = st.select_slider('Distance threshold', options=data_exploration['distance_threshold'].unique(), value=0.3)
            data_exploration = data_exploration[data_exploration['distance_threshold'] == threshold].copy()
        elif metric == 'exploration_circles':
            data_exploration['threshold'] = data_exploration['threshold'].round(2)
            threshold = st.select_slider('Distance threshold', options=data_exploration['threshold'].unique(), value=0.75)
            data_exploration = data_exploration[data_exploration['threshold'] == threshold].copy()
    
    experiments_exploitation = np.unique([x for x in data_exploitation[data_exploitation['top_category'] == top_category]['strategy'].unique()])
    experiments_exploration = data_exploration['strategy'].unique()

    experiments = np.intersect1d(experiments_exploitation, experiments_exploration)

    col1, col2, col3, col4 = st.columns([1, 1, 1, 4])

    with col1:
        # Initialize session_state variables
        if 'rows' not in st.session_state:
            st.session_state['rows'] = [0]  # List to keep track of row IDs
            st.session_state['next_row_id'] = 1  # To assign unique IDs to new rows

        # Button to add a new row
        if st.button('Add curve'):
            st.session_state.rows.append(st.session_state.next_row_id)
            st.session_state.next_row_id += 1

    with col2:
        grid = st.toggle('Display grid', value=False)

    with col3:
        scatter = st.toggle('Display scatter', value=False)

    with col4:
        display_project = st.toggle('Plot real scenario', value=False)
        if display_project:
            project_data_exploration = pd.read_parquet(os.path.join(base_path, f'data/{dataset}/project_exploration.parquet'))
            project_data_exploitation = pd.read_parquet(os.path.join(base_path, f'data/{dataset}/project_exploitation.parquet'))
            project_data_exploitation = project_data_exploitation[project_data_exploitation['top_category'] == top_category].copy()
            if metric == 'exploration_neighborhood_coverage':
                project_data_exploration['exploration_metric'] = project_data_exploration[f'exploration_neighborhood_coverage_{threshold}'].copy()
            elif metric == 'exploration_circles':
                project_data_exploration['exploration_metric'] = project_data_exploration[f'exploration_circles_{threshold}'].copy()
            else:
                project_data_exploration['exploration_metric'] = project_data_exploration[metric].copy()

            project_data = project_data_exploitation.merge(project_data_exploration, on='iterations')

    def create_row(row_id):
        cols = st.columns([0.5, 1, 1, 0.5, 0.5, 0.2, 1])

        strategy_types = natsorted(np.unique([x.split('-')[2] for x in experiments]))
        strategy_type = cols[0].selectbox(
            label = 'Strategy type',
            options = strategy_types,
            key = f'strategy_type_{row_id}',
            format_func = get_clean_type,
        )

        strategy_names = natsorted(np.unique([x.split('-')[3] for x in experiments if strategy_type in x]), key=lambda x: get_clean_strategy(x))
        strategy_name = cols[1].selectbox(
            label = 'Strategy name',
            options = strategy_names,
            key = f'strategy_name_{row_id}',
            format_func = get_clean_strategy
        )

        models = natsorted(np.unique([x.split('-')[1] for x in experiments if strategy_type in x and strategy_name in x]))
        index = models.index('MODEL_rf_std') if "MODEL_rf_std" in models else 0
        model = cols[2].selectbox(
            label = 'QSAR/QSPR Model',
            options = models,
            key = f'model_{row_id}',
            format_func = get_clean_model,
            index = index
        )

        batchsizes = natsorted(np.unique([x.split('-')[0] for x in experiments if strategy_type in x and strategy_name in x and model in x]))
        strategy_name = cols[3].selectbox(
            label = 'Batch size',
            options = batchsizes,
            key = f'batchsize_{row_id}',
            format_func = get_clean_batchsize
        )

        linestyle = cols[4].selectbox(
            'Linestyle',
            options=['-', '--', ':', '-.'],
            key=f'linestyle_{row_id}'
        )
    
        color = cols[5].color_picker(
            'Color',
            value=pmas[min(row_id, len(pmas)-1)],
            key=f'color_{row_id}'
        )

        remove = cols[6].button('Remove', key=f'remove_{row_id}')

        if remove:
            # Remove the row ID from the list
            st.session_state.rows.remove(row_id)
            # Delete the select box states for the removed row
            del st.session_state[f'strategy_type_{row_id}']
            # Rerun the app to update the UI
            st.rerun()

    # Create the select boxes and remove buttons for each row
    for row_id in st.session_state.rows:
        create_row(row_id)

    def plot_curve(fig1, ax1, fig2, ax2, dataset, top_category, strategy_type, strategy_name, model, batchsize, color, linestyle):
        
        exploitation_df = data_exploitation[(data_exploitation['top_category'] == top_category) & (data_exploitation['strategy'] == f'{batchsize}-{model}-{strategy_type}-{strategy_name}')][['iterations', 'proportions_top_selected', 'proportions_all_selected']].copy()
        exploration_df = data_exploration[(data_exploration['strategy'] == f'{batchsize}-{model}-{strategy_type}-{strategy_name}')][['iterations', 'exploration_metric', 'proportions_all_selected']].copy()

        df1 = exploitation_df.groupby(by='iterations').mean().reset_index()
        df2 = exploration_df.groupby(by='iterations').mean().reset_index()

        df1['proportions_all_selected'] = df1['proportions_all_selected'].round(4)
        df2['proportions_all_selected'] = df2['proportions_all_selected'].round(4)

        plot_df = df1.merge(df2, on=['iterations'], how='inner')

        # label = f"{get_clean_type(strategy_type)} {get_clean_strategy(strategy_name)} | {get_clean_model(model)} | {get_clean_batchsize(batchsize)}"
        label = f"{get_clean_strategy(strategy_name)} | {get_clean_model(model)} | {get_clean_batchsize(batchsize)}"
        sns.lineplot(data=plot_df, ax=ax1, x='exploration_metric', y='proportions_top_selected', label=label, color=color, linestyle=linestyle, linewidth=2, errorbar=None)
        sns.lineplot(data=plot_df, ax=ax2, x='proportions_top_selected', y='exploration_metric', label=label, color=color, linestyle=linestyle, linewidth=2, errorbar=None)

        if scatter:
            sns.scatterplot(data=plot_df, ax=ax1, x='exploration_metric', y='proportions_top_selected', color=color, s=25)
            sns.scatterplot(data=plot_df, ax=ax2, x='proportions_top_selected', y='exploration_metric', color=color, s=25)

        ax1.legend(fontsize=6, ncol=1)
        
        # ax1bis.set_xlabel("Proportion of all molecules selected")
        ax1.set_ylabel(f"PTMR from {top_category}")
        if get_clean_exploration_metric(metric) == 'Coverage Distance':
            ax1.set_xlabel(f"{get_clean_exploration_metric(metric)} (τ = {threshold})")
            ax1.set_title(f'{dataset} | PTMR from {top_category} / {get_clean_exploration_metric(metric)} (τ = {threshold}) Trade-off', pad=10)
        else:
            ax1.set_xlabel(f"{get_clean_exploration_metric(metric)}")
            ax1.set_title(f'{dataset} | PTMR from {top_category} / {get_clean_exploration_metric(metric)} Trade-off', pad=10)
            
        ax2.legend(fontsize=8, ncol=1)
        ax2.set_ylabel(f"{get_clean_exploration_metric(metric)}")
        # ax2bis.set_ylabel("Proportion of all molecules selected")
        ax2.set_xlabel(f"PTMR from {top_category}")
        ax2.set_title(f'{dataset} | PTMR from {top_category} / {get_clean_exploration_metric(metric)} Trade-off', pad=10)

        return fig1, fig2

    if st.button('Plot'):
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 7), dpi=300)
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7), dpi=300)
        
        if display_project:
            sns.lineplot(data=project_data, ax=ax1, x='exploration_metric', y='proportions_top_selected', label='Real scenario', color='black', linestyle='--', linewidth=2)
            
            if scatter:
                sns.scatterplot(data=project_data, ax=ax1, x='exploration_metric', y='proportions_top_selected', color='black', s=25)

        if grid:
            ax1.grid(True, which='major', axis='y')
            ax2.grid(True, which='major', axis='y')

        for row_id in st.session_state.rows:
            
            strategy_type = st.session_state.get(f'strategy_type_{row_id}', 'Strategy type')
            strategy_name = st.session_state.get(f'strategy_name_{row_id}', 'Strategy name')
            model = st.session_state.get(f'model_{row_id}', 'Model')
            batchsize = st.session_state.get(f'batchsize_{row_id}', 'Batch size')
            color = st.session_state.get(f'color_{row_id}', 'Color')
            linestyle= st.session_state.get(f'linestyle_{row_id}', 'Linestyle')

            fig1, fig2 = plot_curve(fig1, ax1, fig2, ax2, dataset, top_category, strategy_type, strategy_name, model, batchsize, color, linestyle)

        col1, col2 = st.columns(2)

        with col1:
            st.pyplot(fig1)
            buf1_png = io.BytesIO()
            fig1.savefig(buf1_png, format='png', dpi=300)
            buf1_png.seek(0)
            buf1_svg = io.BytesIO()
            fig1.savefig(buf1_svg, format='svg', dpi=300)
            buf1_svg.seek(0)

            # Create a download button
            st.download_button(
                label="Download PNG",
                data=buf1_png,
                file_name="ee_tradeoff_1.png",
                mime="image/png",
                key="download_0"
            )
            st.download_button(
                label="Download SVG",
                data=buf1_svg,
                file_name="ee_tradeoff_1.svg",
                mime="image/svg",
                key="download_1"
            )

        # with col2:
        #     st.pyplot(fig2)
        #     buf1_png = io.BytesIO()
        #     fig2.savefig(buf1_png, format='png', dpi=300)
        #     buf1_png.seek(0)
        #     buf1_svg = io.BytesIO()
        #     fig2.savefig(buf1_svg, format='svg', dpi=300)
        #     buf1_svg.seek(0)

        #     # Create a download button
        #     st.download_button(
        #         label="Download PNG",
        #         data=buf1_png,
        #         file_name="ee_tradeoff_2.png",
        #         mime="image/png",
        #         key="download_2"
        #     )
        #     st.download_button(
        #         label="Download SVG",
        #         data=buf1_svg,
        #         file_name="ee_tradeoff_2.svg",
        #         mime="image/svg",
        #         key="download_3"
        #     )   




