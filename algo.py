import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('c:/users/mueez/OneDrive/Desktop/House Prediction/housing.csv')
df.dropna(inplace=True) # removes any rows in your DataFrame df that contain missing (NaN) values
x = df.drop(['median_house_value'], axis=1) # 1 tells pandas to drop a column and 0 tells it to drop a row.
y = df['median_house_value']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
train_data = x_train.join(y_train)
train_data.hist(bins=30, figsize=(15, 10), edgecolor='black')
plt.suptitle("Histograms of Train Data Features", fontsize=16)
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 4))
sns.heatmap(train_data.corr(numeric_only=True), annot=True, cmap='YlGnBu')
plt.title("Correlation Matrix")
plt.show()

# covert a non numeric into a numeric
train_data = train_data.join(pd.get_dummies(train_data['ocean_proximity']).astype(int))
train_data = train_data.drop(['ocean_proximity'], axis=1)

# Making distributions less skewed
# Helping models learn better by transforming data into a more manageable range
# These four were right skewed so we made then less skewed ( normalized it )
train_data['total_rooms'] = np.log(train_data['total_rooms'] + 1)
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] + 1)
train_data['population'] = np.log(train_data['population'] + 1)
train_data['households'] = np.log(train_data['households'] + 1)


# Feature Engineering 
train_data['bedroom_ratio'] = train_data['total_bedrooms']/ train_data['total_rooms']
train_data['houshold_rooms'] = train_data['total_rooms']/ train_data['households']


x_train , y_train = train_data.drop(['median_house_value'], axis=1), train_data['median_house_value']
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

# Test data
test_data = x_test.join(y_test)
test_data = test_data.join(pd.get_dummies(test_data['ocean_proximity']).astype(int))
test_data = test_data.drop(['ocean_proximity'], axis=1)

test_data['total_rooms'] = np.log(test_data['total_rooms'] + 1)
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms'] + 1)
test_data['population'] = np.log(test_data['population'] + 1)
test_data['households'] = np.log(test_data['households'] + 1)
test_data['bedroom_ratio'] = test_data['total_bedrooms']/ test_data['total_rooms']
test_data['houshold_rooms'] = test_data['total_rooms']/ test_data['households']

x_test , y_test = test_data.drop(['median_house_value'], axis=1), test_data['median_house_value']
x_test = scaler.transform(x_test)

# Model Linear Regression
model = LinearRegression()
model.fit(x_train,y_train)
r2  = model.score(x_train,y_train) # R square score It measures how well your model explains the variance in the target.
print(f"R² on training data with Linear Regression : {r2:.2f}")
r2 = model.score(x_test, y_test)
print(f"R2 on testing data with Linear Regression : {r2:.2f}")


# Model Random Forest Regressor
forest = RandomForestRegressor()
forest.fit(x_train,y_train)
print(f"R2 on testing data score with Random Forest {forest.score(x_test,y_test)}")


from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100,200,300],
    'min_samples_split':[2,4],
    'max_depth':[None,4,8]
}

gridSearch = GridSearchCV(forest, param_grid, cv=5, scoring='neg_mean_squared_error' ,return_train_score=True)
print(gridSearch.fit(x_train,y_train))