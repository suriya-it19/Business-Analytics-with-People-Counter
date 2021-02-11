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
    def table():
            week = 1
            month = 1
            year = randrange(5)
            date = f'{month}/2021'

            df = pd.DataFrame({
                'Date': f'{date}',
                'Week': f'{week}',
                'Price': box.price.predict([[year, month, week]]),
                'chettinad_mutton_plates' : box.chettinad_mutton_plates.predict([[year, month, week]]),
                'chettinad_mutton_rating' : box.chettinad_mutton_rating.predict([[year, month, week]]),
                'chettinad_mutton_price' : box.chettinad_mutton_price.predict([[year, month, week]]),
                'Discount value' : 'prediction'
            }, index=[0])

            st.table(df)#,width = 10000)

    table()
    if st.button('Specify URL and Click here'):
        if mod == 'Yolo' or 'yolo' or 'YOLO':
                    box.run()

        elif mod == 'SSD' or 'Ssd' or 'ssd':
                    Run.run()


