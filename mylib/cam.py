from random import randrange
import time, schedule

import numpy as np
import pandas as pd
import streamlit as st

from mylib import config
from mylib import Run
from mylib import box
from mylib.box import client
from mylib.predict import predict


def first(mod, area):
    """
    # Using People Counter and Machine learning
    """

    def table():
        week = 1
        month = 1
        year = randrange(5)
        date = f"{month}/2021"

        df = pd.DataFrame(
            {
                "Date": f"{date}",
                "Week": f"{week}",
                "Price": predict(
                    client,
                    "total_price",
                    ["year", "month", "week"],
                    [year, month, week],
                ),
                "chettinad_mutton_plates": predict(
                    client,
                    "chettinad_mutton_plates_sold",
                    ["year", "month", "week"],
                    [year, month, week],
                ),
                "chettinad_mutton_rating": predict(
                    client,
                    "chettinad_mutton",
                    ["year", "month", "week"],
                    [year, month, week],
                ),
                "chettinad_mutton_price": predict(
                    client,
                    "chettinad_mutton_price_price",
                    ["year", "month", "week"],
                    [year, month, week],
                ),
                "Discount value": "prediction",
            },
            index=[0],
        )

        st.table(df)  # ,width = 10000)

    # table()
    if st.button("Specify URL and Click here"):
        if mod == "Yolo" or "yolo" or "YOLO":
            if config.Scheduler:
                ##Runs for every 1 second
                # schedule.every(1).seconds.do(run)
                ##Runs at every day (9:00 am). You can change it.
                schedule.every().week.at("9:00").do(box.run)

                while 1:
                    schedule.run_pending()
            else:
                box.run(area)

        elif mod == "SSD" or "Ssd" or "ssd":
            if config.Scheduler:
                ##Runs for every 1 second
                # schedule.every(1).seconds.do(run)
                ##Runs at every day (9:00 am). You can change it.
                schedule.every().week.at("9:00").do(Run.run)

                while 1:
                    schedule.run_pending()
            else:
                Run.run(area)
