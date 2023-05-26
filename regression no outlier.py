# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:23:22 2023

@author: Vahid Aliakbar
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR



#%% Load Data
data = pd.read_excel('D:/Personal/me/project/final project/house.xlsx')
# Handle Missing Values
data['region']=np.where(data['region'].isnull(),data['county'],data['region'])

#%% Remove regions that manipulate for getting attention
fakepriceRegion=[' دریاچه شهدای خلیج فارس ',' چیتگر شمالی ',' چیتگر شمالی ',' پردیس ']
for region in fakepriceRegion:
    temp=data[data['region']==region]
    indexoutlire = temp.index
    data.drop(indexoutlire,inplace=True)

#%% remove regions that their data are under 100 records

frequentRegion=data['region'].value_counts()>300
for region in frequentRegion[frequentRegion==False].index:
    temp=data[data['region']==region]
    indexoutlire = temp.index
    data.drop(indexoutlire,inplace=True)
    
    
#%% Remove outliers

for region in data.region.unique():
    temp=data[data['region']==region]
    mean=temp['ppm'].mean()
    std=temp['ppm'].std()
    indexoutlire = temp[ (temp['ppm'] > mean+1*std) | (temp['ppm'] < mean-1*std) ].index
    data.drop(indexoutlire,inplace=True)



#%% Select Columns For Model
X=data.loc[:,['rooms','age','meter','region']]

#%% Creating dummy variables for the categorical columns
categorical_cols = ['region']
continuous_cols = ['rooms', 'age', 'meter']
df_dummies = pd.get_dummies(X[categorical_cols])

#%% Concatenating the dummy variables and continuous columns
X = pd.concat([df_dummies, X[continuous_cols]], axis=1)
y = data['price']





#%% Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Initialize the Random Forest Regression model
model = RandomForestRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the root mean squared error (RMSE)
rf_rmse = mean_squared_error(y_test, y_pred)
rf_mae=mean_absolute_error(y_test, y_pred,)
y_pred_series = pd.Series(y_pred, name='Predicted')
y_test_series = pd.Series(y_test, name='Actual')

y_pred_series = y_pred_series.reset_index(drop=True)
y_test_series = y_test_series.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)



#%% Linear Regression
# Initialize the Linear Regression model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train, y_train)

# Make predictions on the test set
linear_y_pred = linear_model.predict(X_test)

# Calculate the root mean squared error (RMSE)
linear_rmse = mean_squared_error(y_test, linear_y_pred)
linear_mae=mean_absolute_error(y_test, linear_y_pred)

# You can perform the same steps as before to concatenate the predicted and actual values
linear_y_pred_series = pd.Series(linear_y_pred, name='Predicted')
linear_concatenated_df = pd.concat([linear_y_pred_series, y_test_series, X_test], axis=1)

#%% Gradiant Boosting

# Initialize the Gradient Boosting Regression model
gb_model = GradientBoostingRegressor(random_state=42)

# Train the model
gb_model.fit(X_train, y_train)

# Make predictions on the test set
gb_y_pred = gb_model.predict(X_test)

# Calculate the root mean squared error (RMSE)
gb_rmse = mean_squared_error(y_test, gb_y_pred)
gb_mae=mean_absolute_error(y_test, gb_y_pred)

# You can perform the same steps as before to concatenate the predicted and actual values
gb_y_pred_series = pd.Series(gb_y_pred, name='Predicted')
gb_concatenated_df = pd.concat([gb_y_pred_series, y_test_series, X_test], axis=1)
gb_concatenated_df.to_clipboard()
#%% Support Vector Machine model

# Initialize the Support Vector Regression model
svr_model = SVR()

# Train the model
svr_model.fit(X_train, y_train)

# Make predictions on the test set
svr_y_pred = svr_model.predict(X_test)

# Calculate the root mean squared error (RMSE)
svr_rmse = mean_squared_error(y_test, svr_y_pred)
print(f"Support Vector Regression RMSE: {svr_rmse}")

# You can perform the same steps as before to concatenate the predicted and actual values
svr_y_pred_series = pd.Series(svr_y_pred, name='Predicted')
svr_concatenated_df = pd.concat([svr_y_pred_series, y_test_series, X_test], axis=1)



# Concatenate the predicted and actual values
concatenated_df = pd.concat([y_pred_series, y_test_series,X_test], axis=1)


new_data=pd.DataFrame(X_test.loc[0:])

# Initialize the Support Vector Regression model
svr_model = SVR()

# Train the model
svr_model.fit(X_train, y_train)

# Make predictions on the test set
svr_y_pred = svr_model.predict(X_test)

# Calculate the root mean squared error (RMSE)
svr_rmse = mean_squared_error(y_test, svr_y_pred)
svr_mae=mean_absolute_error(y_test, svr_y_pred)

# You can perform the same steps as before to concatenate the predicted and actual values
svr_y_pred_series = pd.Series(svr_y_pred, name='Predicted')
svr_concatenated_df = pd.concat([svr_y_pred_series, y_test_series, X_test], axis=1)



#%% Compare Model Result


rmse={
      'randomForest': int(rf_rmse),
      'GradiantBoosting': int(gb_rmse),
      'Linear Regression': int(linear_rmse),
      'Support Vector Machine': int(svr_rmse) 
      }



mae={
     'randomForest': int(rf_mae),
     'GradiantBoosting': int(gb_mae),
     'Linear Regression': int(linear_mae),
     'Support Vector Machine': int(svr_mae) 
     
     }