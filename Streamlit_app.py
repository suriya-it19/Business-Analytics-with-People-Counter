import streamlit as st
import streamlit.components.v1 as components

from mylib import cam
from mylib import graph
from mylib import config

st.set_page_config(layout="wide")

st.title('Enterprice AI')
"""
# Business Analysis of Restautant's profit
"""

st.sidebar.title('Tools')

sidebar_expander = st.sidebar.beta_expander("Model")
with sidebar_expander:
   _, slider_col, _ = st.beta_columns([0.02, 0.96, 0.02])
   with slider_col:
        mod = st.text_input('[INFO] Yolo')

sidebar_expander = st.sidebar.beta_expander("Confidence Threshold")
with sidebar_expander:
   _, slider_col, _ = st.beta_columns([0.02, 0.96, 0.02])
   with slider_col:
        conf = st.slider("[INFO] Default - 0.4", 0.0, 1.0, value=0.4)

sidebar_expander1 = st.sidebar.beta_expander("Webcam Url")
with sidebar_expander1:
   _, slider_col1, _ = st.beta_columns([0.02, 0.96, 0.02])
   with slider_col1:
        id1 = st.number_input('[INFO] 0 for webcam',  step=0, min_value=0, max_value=2)
        id2 = st.text_input('[INFO] IP Link')
        id3 = st.text_input('[INFO] Video path')

st.sidebar.title('Navigation')
PAGES = {
    "Trend Analysis": graph,
    "People Counting": cam
}
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.first(mod)

