import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import calendar
import statistics

import os
import warnings
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from joblib import dump, load
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
start_time = time.time()
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit


data = pd.read_csv('processed_data.csv')
# take only original information
data = data.drop(['Season_heavy rain', 'Season_little rain', 'Period_am', 'Period_pm'], axis = 1)
quarter = 6 * 31 * 24
print(data['D mm'].mean(), data['D mm'].std())
# avgD = data.loc[0,'D mm']
# data['prevD'] = data['D mm'].shift(periods=quarter,fill_value=avgD)

print(data.describe())
X = data.drop(['D mm'], axis = 1).values
y = data['D mm'].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, stratify=None)
splitThreshold = -6000
X_train = X[0:splitThreshold,:]
y_train = y[0:splitThreshold]

X_test = X[splitThreshold:,:]
y_test = y[splitThreshold:]

print(np.mean(y_test), np.std(y_test))

# dummy model, same config
# criterion='squared_error', 'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 100
rf = RandomForestRegressor(criterion='squared_error', max_depth=5, min_samples_split=2, n_estimators=20);
# rf = RandomForestRegressor(n_estimators=50)
svr = SVR()
model = rf;
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f"R-squared (R2) Score: {r2}")

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

delta = np.abs(y_test - y_pred)
print(rmse)
figure, axis = plt.subplots(2, 2)
axis[0,0].hist(delta)
axis[1,0].hist(y_test)
axis[0,1].boxplot(y_test)
axis[1,1].boxplot(y_pred)

plt.show()

