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

from utils import get_pmas_palette, get_clean_type, get_clean_strategy, get_clean_model, get_clean_batchsize

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Timeline', 'Assays distributions', 'Correlations', 'Endpoints distributions', 'Experimental desirability'])

    with tab1:
        st.image(os.path.join(base_path, f'figures/{dataset}/eda/svg/molecules_overtime.svg'), use_container_width=True)
        st.image(os.path.join(base_path, f'figures/{dataset}/eda/svg/cumsum_molecules_overtime.svg'), use_container_width=True)

    with tab2:
        st.image(os.path.join(base_path, f'figures/{dataset}/eda/svg/distribution_of_assays.svg'), use_container_width=True)

    with tab3:
        col1, col2, col3 = st.columns([1, 1, 1])
        col2.image(os.path.join(base_path, f'figures/{dataset}/eda/svg/pearson_correlation_matrix.svg'), use_container_width=True)
        col1.image(os.path.join(base_path, f'figures/{dataset}/eda/svg/spearman_correlation_matrix.svg'), use_container_width=True)
        col3.image(os.path.join(base_path, f'figures/{dataset}/eda/svg/kendall_correlation_matrix.svg'), use_container_width=True)

    with tab4:
        col1, col2 = st.columns([1, 4])
        col1.image(os.path.join(base_path, f'figures/{dataset}/eda/svg/molecules_per_endpoint.svg'), use_container_width=True)
        col2.image(os.path.join(base_path, f'figures/{dataset}/eda/svg/distribution_of_endpoints.svg'), use_container_width=True)

    with tab5:
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        col1.image(os.path.join(base_path, f'figures/{dataset}/eda/svg/pairwise_tanimoto_similarity_distribution.svg'), use_container_width=True)
        col2.image(os.path.join(base_path, f'figures/{dataset}/eda/svg/documentation_distribution.svg'), use_container_width=True)
        col3.image(os.path.join(base_path, f'figures/{dataset}/eda/svg/desirability_scores_distribution.svg'), use_container_width=True)
        col4.image(os.path.join(base_path, f'figures/{dataset}/eda/svg/top_categories_both.svg'), use_container_width=True)
        
        col1, col2 = st.columns([1, 1])
        col1.image(os.path.join(base_path, f'figures/{dataset}/eda/svg/desirability_documentation_distribution.svg'), use_container_width=True)
        col2.image(os.path.join(base_path, f'figures/{dataset}/eda/svg/DocDScore_definition.svg'), use_container_width=True)

        col1, col2 = st.columns([1, 1])
        col1.image(os.path.join(base_path, f'figures/{dataset}/eda/svg/DScore_prct_definition.svg'), use_container_width=True)
        col2.image(os.path.join(base_path, f'figures/{dataset}/eda/svg/DScore_sigma_definition.svg'), use_container_width=True)





