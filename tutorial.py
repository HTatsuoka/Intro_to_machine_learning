import pandas as pd

iowa_file_path = "./train.csv"

home_data = pd.read_csv(iowa_file_path)

print(home_data.head())

print(home_data.describe())

print(home_data.columns)
##
y = home_data.SalePrice

feature_names = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]

X = home_data[feature_names]

X.describe()
##
X.head()
##
from sklearn.tree import DecisionTreeRegressor

iowa_model = DecisionTreeRegressor(random_state=1)

iowa_model.fit(X, y)

predictions = iowa_model.predict(X)
print(predictions)
##
y.head()
##
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

print(len(train_X))

print(len(val_X))
