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

from sklearn import metrics
data = pd.read_csv('processed_data.csv')

X = data.drop(['D mm'], axis = 1).values
y = data['D mm'].values
duration = 3 * 24 * 30
splitThreshold = -6000
X_train = X[0:splitThreshold,:]
y_train = y[0:splitThreshold]

X_test = X[splitThreshold+duration:splitThreshold+duration*2 ,:]
y_test = y[splitThreshold+duration:splitThreshold+duration*2 ]
plt.hist(y_test,5)
plt.show()

<<<<<<< HEAD
print(np.mean(y_test), np.std(y_test))

=======
>>>>>>> c4023bd253bb5c7962dec278e7f25a7aeecc6aab
best_rf = load('ann.joblib')

y_pred = best_rf.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R2) Score: {r2}")
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)

sqer = np.sqrt(np.square(np.subtract(y_pred, y_test)))
<<<<<<< HEAD
values, base = np.histogram(sqer)
pdf = values / sum(values) * 100
cumulative = np.cumsum(pdf)
# plot the cumulative function
plt.plot(base[:-1], cumulative, c='blue')
plt.title('CDF')
plt.xlabel("Sai số dự báo biên độ khe giãn nở (mm)")
plt.ylabel("Phân bố sai số tích lũy (%)")
plt.grid(visible=True)
plt.show()
=======
plt.hist(sqer)
plt.show()
>>>>>>> c4023bd253bb5c7962dec278e7f25a7aeecc6aab
