import pandas as pd
import numpy as np
from random import randrange
import streamlit as st

from mylib.predict import predict
from mylib.box import client


def table(
    df,
    totalIn,
    price,
    chettinad_mutton_plates,
    chettinad_mutton_rating,
    dish_sold,
    dish_rating,
    dish_price,
):
    week = 1
    month = randrange(3)
    year = 5  # randrange(5)
    z, b = randrange(7), randrange(7)
    date = "1/2021"
    arr, arr1 = [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]

    if totalIn > 5:
        pro_price = predict(
            client,
            "total_price",
            ["year", "month", "week"],
            [year, month, week],
        ) + 5000

    elif totalIn > 10:
        pro_price = predict(
            client,
            "total_price",
            ["year", "month", "week"],
            [year, month, week],
        ) + 10000
    else:
        pro_price = predict(
            client,
            "total_price",
            ["year", "month", "week"],
            [year, month, week],
        ) - 5000

    if totalIn > 5:
        pro_plates = predict(
            client,
            "chettinad_mutton_plates_sold",
            ["year", "month", "week"],
            [year, month, week],
        ) + 10
    elif totalIn > 10:
        pro_plates = predict(
            client,
            "chettinad_mutton_plates_sold",
            ["year", "month", "week"],
            [year, month, week],
        ) + 30
    else:
        pro_plates = predict(
            client,
            "chettinad_mutton_plates_sold",
            ["year", "month", "week"],
            [year, month, week],
        ) - 10

    if predict(
            client,
            "chettinad_mutton",
            ["year", "month", "week"],
            [year, month, week],
        ) == 1:
        pro_rating = randrange(start=5, stop=10)
    else:
        pro_rating = randrange(start=0, stop=4)

    arr[z] = pro_plates
    arr1[b] = pro_rating
    df1 = pd.DataFrame(
        {
            "Date": f"{date}",
            "Week": f"{week}",
            "Price": predict(
                client,
                "total_price",
                ["year", "month", "week"],
                [year, month, week],
            ),
            "dishes": [
                "chettinad_mutton",
                "chicken_curry",
                "chicken_nuggets",
                "dhal_makni",
                "mutton_chops",
                "paneer_tikka",
                "prawn_fry",
                "veg_manchurian",
            ],
            "dish_sold": list(map(round, arr)),
            "dish_rating": list(map(round, arr1)),
            "dish_price": pro_price,
        }
    ).set_index("Date")

    return df1
