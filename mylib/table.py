import pandas as pd
import numpy as np
from random import randrange
import streamlit as st

def table(totalIn, price, chettinad_mutton_plates, chettinad_mutton_rating, dish_sold, dish_rating, dish_price):
            week = 1
            month = randrange(3)
            year = 5#randrange(5)
            z,b = randrange(7), randrange(7)
            date = '1/2021'

            if totalIn > 5:
                pro_price = price.predict([[year, month, week]]) + 5000
            elif totalIn > 10:
                pro_price = price.predict([[year, month, week]]) + 10000
            else:
                pro_price = price.predict([[year, month, week]]) - 5000


            if totalIn > 5:
                pro_plates = chettinad_mutton_plates.predict([[year, month, week]]) + 10
            elif totalIn > 10:
                pro_plates = chettinad_mutton_plates.predict([[year, month, week]]) + 30
            else:
                pro_plates = chettinad_mutton_plates.predict([[year, month, week]]) - 10

            if chettinad_mutton_rating.predict([[year, month, week]]) == 1:
                pro_rating = randrange(start=5, stop=10)
            else:
                pro_rating = randrange(start=0, stop=4)

            df = pd.DataFrame({
                'Date': f'{date}',
                'Week': f'{week}',
                'Price': price.predict([[year, month, week]]),
                f'{dish_sold[z]}' : pro_plates,
                f'{dish_rating[b]}' : pro_rating,
                f'{dish_price[z]}' : pro_price
            }, index=[0])

            return st.table(df)#,width = 10000)