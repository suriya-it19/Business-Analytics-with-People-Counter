import streamlit as st
import streamlit.components.v1 as components

from mylib import cam
from mylib import graph

st.set_page_config(layout="wide")

st.title('Analytics')
"""
# Business Analysis of Restautant's profit
"""
PAGES = {
    "Trend Analysis": graph,
    "People Counting": cam
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.first()

