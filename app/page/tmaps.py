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

    toggle = st.toggle('Display annex', key='toggle_annex')

    if toggle:
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:

            top_categories = natsorted([x.split('/')[-1] for x in glob.glob(os.path.join(base_path, f'figures/{dataset}/tmap_iterative/svg/*'))])
            top_category_1 = col1.selectbox(
                label = 'Top category', 
                options = natsorted(top_categories), 
                key='selectbox_top_categories_1',
                index=5)

            experiments = np.unique([x.split('/')[-1] for x in glob.glob(os.path.join(base_path, f'figures/{dataset}/tmap_iterative/svg/{top_category_1}/*'))])

            col11, col12 = st.columns([1, 1])
        
            strategy_types = np.unique([x.split('-')[2] for x in experiments])
            strategy_type_1 = col11.selectbox(
                label = 'Strategy type', 
                options = strategy_types, 
                key = 'selectbox_strategy_types_1',
                format_func = get_clean_type)

            strategy_names = np.unique([x.split('-')[3] for x in experiments if strategy_type_1 in x])
            strategy_name_1 = col12.selectbox(
                label = 'Strategy', 
                options = natsorted(strategy_names), 
                key = 'selectbox_strategies_1',
                format_func = get_clean_strategy)
            
            col11, col12 = st.columns([1, 1])

            models = natsorted(np.unique([x.split('-')[1] for x in experiments if strategy_type_1 in x and strategy_name_1 in x]))
            index_1 = models.index('MODEL_rf_std') if "MODEL_rf_std" in models else 0
            model_1 = col11.selectbox(
                label = 'QSAR/QSPR Model',
                options = models,
                key = f'model_1',
                format_func = get_clean_model,
                index = index_1
            )

            batchsizes = natsorted(np.unique([x.split('-')[0] for x in experiments if strategy_type_1 in x and strategy_name_1 in x and model_1 in x]))
            batchsize_1 = col12.selectbox(
                label = 'Batch size',
                options = batchsizes,
                key = f'batchsize_1',
                format_func = get_clean_batchsize
            )

            iterations_1 = natsorted([x.split('/')[-1].split('.')[0] for x in glob.glob(os.path.join(base_path, f'figures/{dataset}/tmap_iterative/svg/{top_category_1}/{batchsize_2}-{model_2}-{strategy_type_1}-{strategy_name_1}/*.svg'))])
            iterations_1.remove('final')
            iterations_1 = [int(x) for x in iterations_1]

        with col2:

            top_categories = natsorted([x.split('/')[-1] for x in glob.glob(os.path.join(base_path, f'figures/{dataset}/tmap_iterative/svg/*'))])
            top_category_2 = col2.selectbox(
                label = 'Top category', 
                options = natsorted(top_categories), 
                key='selectbox_top_categories_2',
                index=5)

            experiments = np.unique([x.split('/')[-1] for x in glob.glob(os.path.join(base_path, f'figures/{dataset}/tmap_iterative/svg/{top_category_2}/*'))])

            col21, col22 = st.columns([1, 1])
        
            strategy_types = np.unique([x.split('-')[2] for x in experiments])
            strategy_type_2 = col21.selectbox(
                label = 'Strategy type', 
                options = strategy_types, 
                key = 'selectbox_strategy_types_2',
                format_func = get_clean_type)

            strategy_names = np.unique([x.split('-')[3] for x in experiments if strategy_type_2 in x])
            strategy_name_2 = col22.selectbox(
                label = 'Strategy', 
                options = natsorted(strategy_names), 
                key = 'selectbox_strategies_2',
                format_func = get_clean_strategy)
            
            col21, col22 = st.columns([1, 1])

            models = natsorted(np.unique([x.split('-')[1] for x in experiments if strategy_type_2 in x and strategy_name_2 in x]))
            index_2 = models.index('MODEL_rf_std') if "MODEL_rf_std" in models else 0
            model_2 = col21.selectbox(
                label = 'QSAR/QSPR Model',
                options = models,
                key = f'model_2',
                format_func = get_clean_model,
                index = index_2
            )

            batchsizes = natsorted(np.unique([x.split('-')[0] for x in experiments if strategy_type_2 in x and strategy_name_2 in x and model_2 in x]))
            batchsize_2 = col22.selectbox(
                label = 'Batch size',
                options = batchsizes,
                key = f'batchsize_2',
                format_func = get_clean_batchsize
            )

            iterations_2 = natsorted([x.split('/')[-1].split('.')[0] for x in glob.glob(os.path.join(base_path, f'figures/{dataset}/tmap_iterative/svg/{top_category_2}/{batchsize_2}-{model_2}-{strategy_type_2}-{strategy_name_2}/*.svg'))])
            iterations_2.remove('final')
            iterations_2 = [int(x) for x in iterations_2]

        assert iterations_1 == iterations_2

        with col3:

            col31, col32 = st.columns([1, 1])

            with col31:
                annex = st.selectbox(label='Annex display', options=['Initial overview', 'Chemical series', 'Timeline'], key='selectbox_annex', index=1)

            if annex == ('Initial overview'):

                with col32:
                    top_categories = natsorted([x.split('/')[-1].replace('_initial.svg', '') for x in glob.glob(os.path.join(base_path, f'figures/{dataset}/tmap_overview/svg/*_initial.svg'))])
                    top_category_3 = st.selectbox(
                        label='Top category', 
                        options=top_categories, 
                        key='selectbox_best_categories_3', 
                        index=0)
                
        slider_iteration = st.number_input(label='Iteration', min_value=0, max_value=np.max(iterations_1)+1, key='slider_iteration', value=0)

        col1, col2, col3 = st.columns([1, 1, 1])

        if slider_iteration == 0:
            col1.image(os.path.join(base_path, f'figures/{dataset}/tmap_overview/svg/{top_category_1}_initial.svg'))
            col2.image(os.path.join(base_path, f'figures/{dataset}/tmap_overview/svg/{top_category_2}_initial.svg'))
        elif slider_iteration == np.max(iterations_1)+1:
            col1.image(os.path.join(base_path, f'figures/{dataset}/tmap_iterative/svg/{top_category_1}/{batchsize_1}-{model_1}-{strategy_type_1}-{strategy_name_1}/final.svg'))
            col2.image(os.path.join(base_path, f'figures/{dataset}/tmap_iterative/svg/{top_category_2}/{batchsize_2}-{model_2}-{strategy_type_2}-{strategy_name_2}/final.svg'))
        else:
            col1.image(os.path.join(base_path, f'figures/{dataset}/tmap_iterative/svg/{top_category_1}/{batchsize_1}-{model_1}-{strategy_type_1}-{strategy_name_1}/{slider_iteration}.svg'))
            col2.image(os.path.join(base_path, f'figures/{dataset}/tmap_iterative/svg/{top_category_2}/{batchsize_2}-{model_2}-{strategy_type_2}-{strategy_name_2}/{slider_iteration}.svg'))

        if annex == ('Initial overview'):
            col3.image(os.path.join(base_path, f'figures/{dataset}/tmap_overview/svg/{top_category_3}_initial.svg'))
        elif annex == ('Chemical series'):
            col3.image(os.path.join(base_path, f'figures/{dataset}/tmap_overview/svg/chemical_series.svg'))
        elif annex == ('Timeline'):
            col3.image(os.path.join(base_path, f'figures/{dataset}/tmap_overview/svg/timeline.svg'))
    
    
    if not toggle:

        col1, col2 = st.columns([1, 1])

        with col1:

            top_categories = natsorted([x.split('/')[-1] for x in glob.glob(os.path.join(base_path, f'figures/{dataset}/tmap_iterative/svg/*'))])
            top_category_1 = col1.selectbox(
                label = 'Top category', 
                options = natsorted(top_categories), 
                key='selectbox_top_categories_1',
                index=5)

            experiments = np.unique([x.split('/')[-1] for x in glob.glob(os.path.join(base_path, f'figures/{dataset}/tmap_iterative/svg/{top_category_1}/*'))])

            col11, col12 = st.columns([1, 1])
        
            strategy_types = np.unique([x.split('-')[2] for x in experiments])
            strategy_type_1 = col11.selectbox(
                label = 'Strategy type', 
                options = strategy_types, 
                key = 'selectbox_strategy_types_1',
                format_func = get_clean_type)

            strategy_names = np.unique([x.split('-')[3] for x in experiments if strategy_type_1 in x])
            strategy_name_1 = col12.selectbox(
                label = 'Strategy', 
                options = natsorted(strategy_names), 
                key = 'selectbox_strategies_1',
                format_func = get_clean_strategy)
            
            col11, col12 = st.columns([1, 1])

            models = natsorted(np.unique([x.split('-')[1] for x in experiments if strategy_type_1 in x and strategy_name_1 in x]))
            model_1 = col11.selectbox(
                label = 'QSAR/QSPR Model',
                options = models,
                key = f'model_1',
                format_func = get_clean_model
            )

            batchsizes = natsorted(np.unique([x.split('-')[0] for x in experiments if strategy_type_1 in x and strategy_name_1 in x and model_1 in x]))
            batchsize_1 = col12.selectbox(
                label = 'Batch size',
                options = batchsizes,
                key = f'batchsize_1',
                format_func = get_clean_batchsize
            )

            iterations_1 = natsorted([x.split('/')[-1].split('.')[0] for x in glob.glob(os.path.join(base_path, f'figures/{dataset}/tmap_iterative/svg/{top_category_1}/{batchsize_1}-{model_1}-{strategy_type_1}-{strategy_name_1}/*.svg'))])
            if 'final' in iterations_1:
                iterations_1.remove('final')
            iterations_1 = [int(x) for x in iterations_1]

        with col2:

            top_categories = natsorted([x.split('/')[-1] for x in glob.glob(os.path.join(base_path, f'figures/{dataset}/tmap_iterative/svg/*'))])
            top_category_2 = col2.selectbox(
                label = 'Top category', 
                options = natsorted(top_categories), 
                key='selectbox_top_categories_2',
                index=5)

            experiments = np.unique([x.split('/')[-1] for x in glob.glob(os.path.join(base_path, f'figures/{dataset}/tmap_iterative/svg/{top_category_2}/*'))])

            col21, col22 = st.columns([1, 1])
        
            strategy_types = np.unique([x.split('-')[2] for x in experiments])
            strategy_type_2 = col21.selectbox(
                label = 'Strategy type', 
                options = strategy_types, 
                key = 'selectbox_strategy_types_2',
                format_func = get_clean_type)

            strategy_names = np.unique([x.split('-')[3] for x in experiments if strategy_type_2 in x])
            strategy_name_2 = col22.selectbox(
                label = 'Strategy', 
                options = natsorted(strategy_names), 
                key = 'selectbox_strategies_2',
                format_func = get_clean_strategy)
            
            col21, col22 = st.columns([1, 1])

            models = natsorted(np.unique([x.split('-')[1] for x in experiments if strategy_type_2 in x and strategy_name_2 in x]))
            model_2 = col21.selectbox(
                label = 'QSAR/QSPR Model',
                options = models,
                key = f'model_2',
                format_func = get_clean_model
            )

            batchsizes = natsorted(np.unique([x.split('-')[0] for x in experiments if strategy_type_2 in x and strategy_name_2 in x and model_2 in x]))
            batchsize_2 = col22.selectbox(
                label = 'Batch size',
                options = batchsizes,
                key = f'batchsize_2',
                format_func = get_clean_batchsize
            )

            iterations_2 = natsorted([x.split('/')[-1].split('.')[0] for x in glob.glob(os.path.join(base_path, f'figures/{dataset}/tmap_iterative/svg/{top_category_2}/{batchsize_2}-{model_2}-{strategy_type_2}-{strategy_name_2}/*.svg'))])
            if 'final' in iterations_2:
                iterations_2.remove('final')
            iterations_2 = [int(x) for x in iterations_2]

        assert iterations_1 == iterations_2

        slider_iteration = st.number_input(label='Iteration', min_value=0, max_value=np.max(iterations_1)+1, key='slider_iteration', value=0)

        col1, col2 = st.columns([1, 1])

        if slider_iteration == 0:
            col1.image(os.path.join(base_path, f'figures/{dataset}/tmap_overview/svg/{top_category_1}_initial.svg'))
            col2.image(os.path.join(base_path, f'figures/{dataset}/tmap_overview/svg/{top_category_2}_initial.svg'))
        elif slider_iteration == np.max(iterations_1)+1:
            col1.image(os.path.join(base_path, f'figures/{dataset}/tmap_iterative/svg/{top_category_1}/{batchsize_1}-{model_1}-{strategy_type_1}-{strategy_name_1}/final.svg'))
            col2.image(os.path.join(base_path, f'figures/{dataset}/tmap_iterative/svg/{top_category_2}/{batchsize_2}-{model_2}-{strategy_type_2}-{strategy_name_2}/final.svg'))
        else:
            col1.image(os.path.join(base_path, f'figures/{dataset}/tmap_iterative/svg/{top_category_1}/{batchsize_1}-{model_1}-{strategy_type_1}-{strategy_name_1}/{slider_iteration}.svg'))
            col2.image(os.path.join(base_path, f'figures/{dataset}/tmap_iterative/svg/{top_category_2}/{batchsize_2}-{model_2}-{strategy_type_2}-{strategy_name_2}/{slider_iteration}.svg'))