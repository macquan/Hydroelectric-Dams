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
start_time = time.time()


data = pd.read_csv('processed_data.csv')
X = data.drop(['D mm'], axis = 1).values
y = data['D mm'].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, stratify=None)

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# Initialize the Random Forest model
svr_model = SVR()
param_svr = {
    'kernel' : ['rbf', 'sigmoid'],
    'gamma' : ['scale', 'auto'],
    'C' : [0.5,1,1.5,2.0],
    'epsilon': [0.001,0.01, 0.1, 0.5]
}
from joblib import dump, load
# Create the Grid Search model with the Random Forest model and hyperparameter grid
grid_search = GridSearchCV(estimator=svr_model, param_grid=param_svr, cv=5, scoring='r2', n_jobs=-1)

# Train the model on the training data
grid_search.fit(X_train, y_train)

result = pd.DataFrame(grid_search.cv_results_)
best_model = grid_search.best_estimator_
print("Grid search - best param: ", grid_search.best_params_)
print("Grid search - best score: ", grid_search.best_score_)

dump(best_model, 'svr.joblib') 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f"R-squared (R2) Score: {r2}")

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: ", rmse)
print("end time: ", time.time() - start_time)