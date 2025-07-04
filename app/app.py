import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from glob import glob
import os

from page import exploitation, exploration, ee, eda, tmaps, interactive_tmaps

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

def main():

    st.set_page_config('Visualization App', layout="wide")

    base_path = f".."

    dataset = st.sidebar.selectbox(
        label = 'Dataset',
        options = natsorted([path.split('/')[-1] for path in glob(os.path.join(base_path, 'data/*')) if 'raw' not in path and 'common' not in path])
        )

    tab = st.sidebar.radio(
        label = "**Tab**", 
        options = [
            'Exploratory Data Analysis', 
            'Exploitation', 
            'Exploration', 
            'Exploitation / Exploration trade-off', 
            'TMAPs', 
            'Interactive TMAP', 
            ]
            )

    if tab == 'Exploratory Data Analysis':
        eda.show(base_path, dataset)
    elif tab == 'Exploitation':
        exploitation.show(base_path, dataset)
    elif tab == 'Exploration':
        exploration.show(base_path, dataset)
    elif tab == 'Exploitation / Exploration trade-off':
        ee.show(base_path, dataset)
    elif tab == 'TMAPs':
        tmaps.show(base_path, dataset)
    elif tab == 'Interactive TMAP':
        interactive_tmaps.show(base_path, dataset)

if __name__ == "__main__":
    main()