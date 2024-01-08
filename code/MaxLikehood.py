#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 19:17:47 2024

@author: rayan
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the data
data = np.load('data.npy')

# Assuming data has shape (3879, 18)
X = data

####################################################################I.1
# Data preprocessing
# 1. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Choosing the number of clusters using the Elbow Method
# In this example, let's consider a range of clusters from 2 to 10
wcss = []  # Within-Cluster-Sum-of-Squares

for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method
plt.plot(range(2, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster-Sum-of-Squares (WCSS)')
plt.show()

# Based on the Elbow Method, choose the optimal number of clusters (k)
# For demonstration, let's assume k=4
k_optimal = 4

# Fit the KMeans model with the optimal number of clusters
kmeans_optimal = KMeans(n_clusters=k_optimal, random_state=42)
labels = kmeans_optimal.fit_predict(X_scaled)

# Evaluate the clustering using Silhouette Score
silhouette_avg = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {silhouette_avg}")


# If you didn't apply PCA:
# Visualize two randomly selected features (adjust as needed)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('Cluster Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

#########I.2


# Load the data
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the data
data = np.load('data.npy')

# Assuming data has shape (3879, 18)
X = data

# Visualize the relationship between 'Word count' and '# Shares'
f1 = X[:, 0]  # Assuming 'Word count' is the first column in the new data
f2 = X[:, 17]  # Assuming '# Shares' is the last column in the new data

plt.scatter(f1, f2, c='b')
plt.xlabel('Word count')
plt.ylabel('# Shares')
plt.title('Word count vs. # Shares')
plt.show()

# Apply a cut to the original dataset
filtered_data = X[(X[:, 0] <= 3500) & (X[:, 17] <= 80000)]

f1 = filtered_data[:, 0]
f2 = filtered_data[:, 17]

plt.scatter(f1, f2, c='b')
plt.xlabel('Word count')
plt.ylabel('# Shares')
plt.title('Filtered Word count vs. # Shares')
plt.show()

# Gradient Descent
# Assuming you have X and y defined

# Cost function
def computeCost(X, y, theta):
    m = y.shape[0] 
    h = np.dot(X, theta)
    error = h - y
    squared_error = error ** 2
    J = (1 / (2 * m)) * np.sum(squared_error)
    return J

# Gradient Descent function
def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    J_history = []

    for i in range(num_iters):
        hypothesis = np.dot(X, theta)
        error = hypothesis - y
        gradient = (1 / m) * np.dot(X.T, error)
        theta = theta - alpha * gradient
        J_history.append(computeCost(X, y, theta))

    return theta, J_history

# Feature scaling
def featureNormalize(X):
    X_norm = X.copy()
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

# Assuming you have X and y defined
y = filtered_data[:, 17]
m = y.shape[0]

# Add intercept term to X
X = np.concatenate([np.ones((m, 1)), filtered_data[:, :17]], axis=1)

# Feature scaling
X_norm, mu, sigma = featureNormalize(X)

# Gradient Descent settings
alpha = 0.01
num_iters = 1000
theta = np.zeros(X_norm.shape[1])

# Run gradient descent
theta, J_history = gradientDescent(X_norm, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.plot(np.arange(1, num_iters + 1), J_history, '-b', linewidth=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.title('Convergence of Gradient Descent')
plt.show()

# Display the final theta
print('Theta computed from gradient descent: \n', theta)

# Predict the value for a new input x = [515, 7]
x = np.array([1, 515, 7, 0, 0, 18, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])  

# Print shapes and values for debugging
print('x shape:', x.shape)
print('mu shape:', mu.shape)
print('sigma shape:', sigma.shape)
print('x:', x)
print('mu:', mu)
print('sigma:', sigma)

# Mise à l'échelle des caractéristiques pour la prédiction
x_normalized = (x - mu) / sigma
shares = np.dot(x_normalized, theta)
print('Predicted number of shares: \n', shares)








####################################################################II.1






