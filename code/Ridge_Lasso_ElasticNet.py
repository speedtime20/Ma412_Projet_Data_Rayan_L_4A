#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 22:54:28 2024

@author: rayan
"""

####################################################################II.1

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

# Load the data
data = np.load('data.npy')

# Assuming data has shape (3879, 18)
X = data

# Question 1: Regression function
def regression(X, Y):
    X = np.c_[np.ones(X.shape[0]), X]
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ Y
    return beta_hat

# Chargement des données Boston House Prices
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Application de la fonction de régression
beta_hat = regression(data, target)

print("Least Squares Estimator (Intercept, Coefficients):", beta_hat)

# Question 2: Comparison with scikit-learn
data = np.c_[np.ones(data.shape[0]), data]

X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

beta_sklearn = np.concatenate(([model.intercept_], model.coef_[1:]))
alpha_sklearn = model.coef_

print("Coefficients from LinearRegression:", beta_sklearn)
print("Coefficients from your regression function:", beta_hat)

Y_pred_sklearn = model.predict(X_test)
Y_pred_hat = data @ beta_hat
r2_sklearn = r2_score(Y_test, Y_pred_sklearn)
r2_hat = 1 - (mean_squared_error(target, Y_pred_hat) / np.var(target))

print("R^2 from LinearRegression:", r2_sklearn)
print("R^2 from your regression function:", r2_hat)

# Question 3: Regression function with predictions
def regress(X, alpha, beta):
    Y_hat = alpha + np.dot(X, beta)
    return Y_hat

X_example = np.array([[1, 2], [2, 3], [3, 4]])
alpha_example = 1.0
beta_example = np.array([0.5, 0.75])
Y_hat_example = regress(X_example, alpha_example, beta_example)
print(Y_hat_example)

# Question 4: Least Squares Error
Y = np.array([3.5, 2.7, 5.1]) 
Y_hat = np.array([2.5, 4, 5.5]) 

squared_errors = (Y - Y_hat) ** 2
lse = np.sum(squared_errors)

print("Least Squares Error (LSE):", lse)



####################################################################II.2
# Question 1: Ridge regression function
def ridge_regression(X, Y, lambda_value):
    n = X.shape[1] - 1
    lmbda_matrix = lambda_value * np.eye(n + 1)
    
    coefficients = np.linalg.inv(X.T @ X + lmbda_matrix) @ X.T @ Y
    
    alpha_hat = coefficients[:-1]
    beta_hat = coefficients[-1]
    
    return alpha_hat, beta_hat

# Question 2: Comparison with scikit-learn Ridge

lambda_value = 1
alpha_hat_ridge, beta_hat_ridge = ridge_regression(X_train, Y_train, lambda_value)

ridge_regressor = Ridge(alpha=lambda_value)
ridge_regressor.fit(X_train, Y_train)

print("Custom Function - alpha_hat_ridge:", alpha_hat_ridge)
print("Custom Function - beta_hat_ridge:", beta_hat_ridge)

print("Scikit-Learn Ridge - alpha_hat:", ridge_regressor.coef_[:-1])
print("Scikit-Learn Ridge - beta_hat:", ridge_regressor.intercept_)

# Question 3: Plot the evolution of coefficients with Ridge Regression
def plot_coefficients_evolution(X, Y, lambda_values):
    n = X.shape[1] - 1
    
    alpha_evolution = np.zeros((len(lambda_values), n))

    for i, lambda_value in enumerate(lambda_values):
        alpha_hat, _ = ridge_regression(X, Y, lambda_value)
        alpha_evolution[i, :] = alpha_hat

    plt.figure(figsize=(10, 6))
    for j in range(n):
        plt.plot(lambda_values, alpha_evolution[:, j], label=f'Feature {j}')

    plt.xscale('log')
    plt.xlabel('Regularization Parameter (lambda)')
    plt.ylabel('Coefficient Value')
    plt.title('Evolution of Coefficients with Ridge Regression')
    plt.legend()
    plt.show()

lambda_values = np.logspace(-3, 3, 100)
plot_coefficients_evolution(X_train, Y_train, lambda_values)

# Question 4: Find the best value for lambda and evaluate the regressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 100), store_cv_values=True)
ridge_cv.fit(X_train, Y_train)

best_lambda = ridge_cv.alpha_

ridge_final = Ridge(alpha=best_lambda)
ridge_final.fit(data, target)

Y_pred_final = ridge_final.predict(data)

mse_final = mean_squared_error(target, Y_pred_final)

print(f"Best lambda: {best_lambda}")
print(f"Mean Squared Error on the entire dataset: {mse_final}")


####################################################################II.3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the data
data = np.load('data.npy')

# Considering only the first 12 features
X = data[:, :12]
Y = data[:, -1]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Plot the evolution of coefficients with LASSO regularization for the first 12 features
alphas = np.logspace(-3, 3, num=100)

coefficients = []

for alpha in alphas:
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train, Y_train)
    coefficients.append(lasso_model.coef_)

coefficients = np.array(coefficients)

plt.figure(figsize=(12, 6))
for i in range(12):  
    plt.plot(alphas, coefficients[:, i], label=f'Feature {i+1}')

plt.xscale('log')
plt.xlabel('Regularization Parameter (λ)')
plt.ylabel('Coefficient Value (𝛼̂)')
plt.title('Coefficient Evolution with LASSO Regularization (First 12 Features)')
plt.legend()
plt.grid()
plt.savefig('lasso_regression_plot_12_features.png')  
plt.show()

# Find the best λ value using LassoCV
lasso_cv_model = LassoCV(alphas=alphas, cv=5)
lasso_cv_model.fit(X_train, Y_train)

best_alpha = lasso_cv_model.alpha_

# Train LASSO regression model with the best λ value
lasso_best_model = Lasso(alpha=best_alpha)
lasso_best_model.fit(X, Y)

# Predict on the entire dataset
Y_pred = lasso_best_model.predict(X)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(Y, Y_pred)

print("Best Lambda Value:", best_alpha)
print("Mean Squared Error (MSE) on Entire Dataset:", mse)





####################################################################II.4
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data
data = np.load('data.npy')

# Assuming data has shape (3879, 18)
X = data[:, :-1]
Y = data[:, -1]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Apply Elastic Net Regression
elastic_net_model = ElasticNet(alpha=1.0, l1_ratio=0.5)

# Train the model
elastic_net_model.fit(X_train, Y_train)

# Make predictions
Y_pred = elastic_net_model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(Y_test, Y_pred)

print("Mean Squared Error (MSE) on Test Set:", mse)

# Plot the coefficients
coefficients = elastic_net_model.coef_

plt.figure(figsize=(12, 6))
plt.plot(coefficients, marker='o', linestyle='-', color='b')
plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Value (𝛼̂)')
plt.title('Elastic Net Regression Coefficients')
plt.grid(True)
plt.show()

