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

data = pd.read_excel("data.xlsx")
data["Time"] = pd.to_datetime(data["Time"])
data['Day_of_week'] = data['Time'].dt.day_name()
data['Day'] = data['Time'].dt.day
data['Month'] = data['Time'].dt.month
data['Year'] = data['Time'].dt.year
data['Hour'] = data['Time'].dt.hour
data['Date'] = [d.date() for d in data['Time']]

# Toal_seconds
data["Time_today"] = datetime.today()
data['Seconds'] = data["Time_today"] - data["Time"]
data['Seconds'] = data['Seconds'].dt.total_seconds()

# Rainy season in the North: May, June, July, August, September; rainy season with mùa khô: the remaining months
data["Season"] = ["mùa mưa" if month in [5,6, 7, 8, 9] else "mùa khô" for  month in data["Month"]]

data['Period'] = ['am' if hour in range(5, 17) else 'pm' for hour in data['Hour']]

# Display
data = data.drop("Time_today", axis = 1)

def delete_missing(data):
    for col in data.columns:
        miss_ind = data[col][data[col].isnull()].index
        data = data.drop(miss_ind, axis = 0)
    return data

def data_by_season(df):
    condition1 = df['Season'] == 'mùa mưa'
    condition2 = df['Season'] == 'mùa khô'
    data1 = df[condition1]
    data2 = df[condition2]
    return data1, data2

def plot_hist_by_season(df):
    data1, data2 = data_by_season(df)
    plt.rcParams.update({'font.size': 14})
    # print(data1.describe())
    # print(data2.describe())
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot Biều đồ tần suấts for the 'T' column in both seasons
    axes[0, 0].hist(data1['T'], bins=5, color='red', alpha=0.7, range=(20, 29))
    axes[0, 0].set_title(f"Nhiệt độ T - mùa mưa")
    axes[0, 0].set_xlabel("Giá trị (°C)")
    axes[0, 0].set_ylabel("Tần suất")

    axes[1, 0].hist(data2['T'], bins=5, color='red', alpha=0.7, range=(20, 29))
    axes[1, 0].set_title(f"Nhiệt độ T - mùa khô")
    axes[1, 0].set_xlabel("Giá trị (°C)")
    axes[1, 0].set_ylabel("Tần suất")

    # Plot Biều đồ tần suấts for the 'Lever water' column in both seasons
    axes[0, 1].hist(data1['Lever water'], bins=5, color='blue', alpha=0.7, range=(175,220))
    axes[0, 1].set_title(f"Mực nước - mùa mưa")
    axes[0, 1].set_xlabel("Giá trị (cm)")
    axes[0, 1].set_ylabel("Tần suất")

    axes[1, 1].hist(data2['Lever water'], bins=5, color='blue', alpha=0.7, range=(175,220))
    axes[1, 1].set_title(f"Mực nước - mùa khô")
    axes[1, 1].set_xlabel("Giá trị (cm)")
    axes[1, 1].set_ylabel("Tần suất")

    # Plot Biều đồ tần suấts for the 'D mm' column in both seasons
    axes[0, 2].hist(data1['D mm'], bins=5, color='green', alpha=0.7, range=(-0.5,2))
    axes[0, 2].set_title(f"Độ rộng khe co giãn - mùa mưa")
    axes[0, 2].set_xlabel("Giá trị (mm)")
    axes[0, 2].set_ylabel("Tần suất")

    axes[1, 2].hist(data2['D mm'], bins=5, color='green', alpha=0.7, range=(-0.5,2))
    axes[1, 2].set_title(f"Độ rộng khe co giãn - mùa khô")
    axes[1, 2].set_xlabel("Giá trị (mm)")
    axes[1, 2].set_ylabel("Tần suất")

    plt.tight_layout()

    plt.show()
data = delete_missing(data)


data['Date'] = pd.to_datetime(data['Date'])

start_date = '2015-01-01'
end_test_date = '2022-09-30'
mask = (data['Date'] >= start_date) & (data['Date'] <= end_test_date)
data = data[mask]
data = data.loc[(data['T']>0) & (data['D mm']>-10) & (data['T'] < 40)]
plot_hist_by_season(data)
data = data.drop(['Time','Day_of_week', 'Day', 'Year','Hour', 'Date','Seconds'], axis = 1)

# print(data.describe())
data = pd.get_dummies(data, columns=['Season', 'Period'])
for column in data.columns:
    if column == 'D mm':
        pass
    data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
data.to_csv('processed_data.csv', index=False)


