from random import randrange

import numpy as np
import pandas as pd
import streamlit as st

from mylib import Run

def first():
    """
    # Using People Counter and Machine learning
    """
    if st.button('Start People Counting?'):
        Run.run()

