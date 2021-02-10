import os
import pickle

#from sklearn.externals import joblib
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import mean, std
from sklearn import datasets, ensemble, metrics
from sklearn.datasets import (make_classification,
                              make_multilabel_classification)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

input = "input"

data = pd.read_excel(os.path.join(input , "hf.xlsx"))

data.replace({
    2015 : 0,
    2016 : 1,
    2017 : 2,
    2018 : 3,
    2019 : 4,
    2020 : 5
    }, inplace=True)

for i in range(3, 25, 3):
    data.iloc[:, i].replace([0,1,2,3,4], 0, inplace=True)
    data.iloc[:, i].replace([5,6,7,8,9,10], 1, inplace=True)


def total_price(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)
    
    params = {'n_estimators': 2000,
          'max_depth': 4,
          'min_samples_split': 6,
          'learning_rate': 0.01,
          'loss': 'ls'}

    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(X_train, y_train)
    acc = r2_score(y_test, reg.predict(X_test))
    joblib.dump(reg, "models/cost_price/price.pkl")
    return {'model_total_price':reg,'acc':acc}

cost_price = []

X = data[['year', 'month', 'week']]
y = data['pure total']
cost_price.append(total_price(X, y))
    


def dish_price(X, y, i):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)
    
    params = {'n_estimators': 2000,
          'max_depth': 4,
          'min_samples_split': 6,
          'learning_rate': 0.01,
          'loss': 'ls'}

    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(X_train, y_train)
    acc = r2_score(y_test, reg.predict(X_test))
    joblib.dump(reg, f"models/final_price/{i}.pkl")
    return {f'model_{i}_price':reg,'acc':acc}

price = ['chicken_curry_price ', 'mutton_chops_price', 'prawn_fry_price', 'chicken_nuggets_price',
       'chettinad_mutton_price',
       'dhal_makni_price',
       'veg_manchurian_price',
       'paneer_tikka_price']
#price = ['chicken_curry', 'mutton_chops', 'prawn_fry', 'chicken_nuggets', 'chettinad_mutton', 'dhal_makni', 'veg_manchurian', 'paneer_tikka']
final_price = []

for i in price:
    X = data[['year','month','week']]
    y = data[[i]]
    final_price.append(dish_price(X, y, i))



def solds(X, y, i):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)
    
    params = {'n_estimators': 2000,
          'max_depth': 4,
          'min_samples_split': 6,
          'learning_rate': 0.01,
          'loss': 'ls'}

    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(X_train, y_train)
    acc = r2_score(y_test, reg.predict(X_test))
    joblib.dump(reg, f"models/final_sold/{i}_plates.pkl")
    return {f'model_{i}_plates_sold':reg,'acc':acc}

sold = ['chicken_curry', 'mutton_chops', 'prawn_fry', 'chicken_nuggets', 'chettinad_mutton', 'dhal_makni', 'veg_manchurian', 'paneer_tikka']
final_sold = []

for i in sold:
    X = data[['year','month','week']]
    y = data[[f'{i}_plates_sold']]
    final_sold.append(solds(X, y, i))


def rating(X, y, i):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=13)

    clf = AdaBoostClassifier(n_estimators=1500, random_state=13)
    clf.fit(X_train, y_train)
    acc = metrics.accuracy_score(y_test, clf.predict(X_test).reshape(-1, 1))
    joblib.dump(clf, f"models/final_rating/{i}_rating.pkl")
    return {f'model_{i}':clf,'acc':acc}

ratings = ['chicken_curry', 'mutton_chops', 'prawn_fry', 'chicken_nuggets', 'chettinad_mutton', 'dhal_makni', 'veg_manchurian', 'paneer_tikka']
final_rating = []

for i in ratings:
    X = data[['year','month','week']]
    y = data[[f'{i}']]
    final_rating.append(rating(X, y, i))

