import streamlit.components.v1 as components
import streamlit as st 
import os

def show(base_path, dataset):

    st.write(os.path.join(base_path, f'data/{dataset}/tmap.html'))
    with open(os.path.join(base_path, f'data/{dataset}/tmap.html'), "r", encoding="utf-8") as f:
        html_string = f.read()
    components.html(html_string, width=1200, height=700, scrolling=True)