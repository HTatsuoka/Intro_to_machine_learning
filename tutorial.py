import pandas as pd

iowa_file_path = "./train.csv"

home_data = pd.read_csv(iowa_file_path)

print(home_data.head())

print(home_data.describe())

print(home_data.columns)

from sklearn.tree import DecisionTreeRegressor
