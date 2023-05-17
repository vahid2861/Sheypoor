# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:23:22 2023

@author: Namafar
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_excel('C:/Users/Namafar/Downloads/house.xlsx')
data['region']=np.where(data['region'].isnull(),data['county'],data['region'])


X=data.loc[:,['elevator','storage','rooms','parking','age','meter','type','region']]


categorical_cols = ['elevator', 'storage', 'parking', 'type','region']
continuous_cols = ['rooms', 'age', 'meter']

# Creating dummy variables for the categorical columns
df_dummies = pd.get_dummies(X[categorical_cols])

# Concatenating the dummy variables and continuous columns
X = pd.concat([df_dummies, X[continuous_cols]], axis=1)
y = data['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the Random Forest Regression model

model = RandomForestRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the root mean squared error (RMSE)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse}")

y_pred_series = pd.Series(y_pred, name='Predicted')
y_test_series = pd.Series(y_test, name='Actual')

y_pred_series = y_pred_series.reset_index(drop=True)
y_test_series = y_test_series.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

# Concatenate the predicted and actual values
concatenated_df = pd.concat([y_pred_series, y_test_series,X_test], axis=1)


new_data=pd.DataFrame(X_test.loc[0:])
