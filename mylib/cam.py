from random import randrange

import numpy as np
import pandas as pd
import streamlit as st

from mylib import Run
from mylib import box

def first(mod):
    """
    # Using People Counter and Machine learning
    """
    if st.button('Specify URL and Click here'):
        if mod == 'Yolo' or 'yolo' or 'YOLO':
                    box.run()

        elif mod == 'SSD' or 'Ssd' or 'ssd':
                    Run.run()


