from random import randrange
import time, schedule

import numpy as np
import pandas as pd
import streamlit as st

from mylib import config
from mylib import Run
from mylib import box

def first(mod, area):
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

    #table()
    if st.button('Specify URL and Click here'):
        if mod == 'Yolo' or 'yolo' or 'YOLO':
            if config.Scheduler:
                ##Runs for every 1 second
                #schedule.every(1).seconds.do(run)
                ##Runs at every day (9:00 am). You can change it.
                schedule.every().week.at("9:00").do(box.run)

                while 1:
                    schedule.run_pending()
            else:
                    box.run(area)

        elif mod == 'SSD' or 'Ssd' or 'ssd':
            if config.Scheduler:
                ##Runs for every 1 second
                #schedule.every(1).seconds.do(run)
                ##Runs at every day (9:00 am). You can change it.
                schedule.every().week.at("9:00").do(Run.run)

                while 1:
                    schedule.run_pending()
            else:
                    Run.run(area)


