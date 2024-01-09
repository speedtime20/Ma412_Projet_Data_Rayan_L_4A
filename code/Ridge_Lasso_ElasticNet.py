#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 4 22:54:28 2024
@author: rayan
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge



# Data
data = np.load('data.npy')

# Assuming data has shape (3879, 18)
# Assuming the first two columns are features
X = data[:, :2] 
# Assuming the third column is the target variable 
y = data[:, 2]   

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the model coefficients and evaluation metrics
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize the predictions
plt.scatter(X_test[:, 0], y_test, color='black', label='True values')
plt.scatter(X_test[:, 0], y_pred, color='blue', label='Predicted values')
plt.xlabel('Feature 1')
plt.ylabel('Target Variable')
plt.legend()
plt.show()







# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Ridge regression model
# Regularization strength
alpha = 1.0  
ridge_model = Ridge(alpha=alpha)

# Fit the Ridge model on the training data
ridge_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = ridge_model.predict(X_test)

# Evaluate the Ridge model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the model coefficients and evaluation metrics
print(f'Coefficients: {ridge_model.coef_}')
print(f'Intercept: {ridge_model.intercept_}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize the predictions
plt.scatter(X_test[:, 0], y_test, color='black', label='True values')
plt.scatter(X_test[:, 0], y_pred, color='red', label='Predicted values (Ridge)')
plt.xlabel('Feature 1')
plt.ylabel('Target Variable')
plt.legend()
plt.show()





# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Lasso regression model
# Regularization strength
alpha = 0.1  
lasso_model = Lasso(alpha=alpha)

# Fit the Lasso model on the training data
lasso_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = lasso_model.predict(X_test)

# Evaluate the Lasso model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the model coefficients and evaluation metrics
print(f'Coefficients: {lasso_model.coef_}')
print(f'Intercept: {lasso_model.intercept_}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize the predictions
plt.scatter(X_test[:, 0], y_test, color='black', label='True values')
plt.scatter(X_test[:, 0], y_pred, color='blue', label='Predicted values (Lasso)')
plt.xlabel('Feature 1')
plt.ylabel('Target Variable')
plt.legend()
plt.show()





# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an Elastic Net regression model
# Regularization strength (similar to Lasso)
alpha = 0.1        
# Ratio of L1 penalty in the regularization term
l1_ratio = 0.5     
elastic_net_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

# Fit the Elastic Net model on the training data
elastic_net_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = elastic_net_model.predict(X_test)

# Evaluate the Elastic Net model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the model coefficients and evaluation metrics
print(f'Coefficients: {elastic_net_model.coef_}')
print(f'Intercept: {elastic_net_model.intercept_}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize the predictions
plt.scatter(X_test[:, 0], y_test, color='black', label='True values')
plt.scatter(X_test[:, 0], y_pred, color='green', label='Predicted values (Elastic Net)')
plt.xlabel('Feature 1')
plt.ylabel('Target Variable')
plt.legend()
plt.show()
