#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 18:00:35 2024

@author: rayan
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#Dataset
data = np.load('data.npy')[:18, :2]

#Number of clusters 
K = 10

#Initialize GMM parameters
means = np.zeros((K, 2))
covariances = np.zeros((K, 2, 2))
weights = np.ones((K,)) / K

#Initialize means randomly
for k in range(K):
    means[k] = np.random.normal(size=(2,))

#Initialize covariance matrices to identity matrices
for k in range(K):
    covariances[k] = np.eye(2)

#Display the initial mean vectors
print("Initial mean vectors (one per row):\n" + str(means))

#Create a meshgrid for visualization
X, Y = np.meshgrid(np.linspace(-8, 8, 100), np.linspace(-8, 8, 100))
pos = np.dstack((X, Y))

#Define a GMM using scipy.stats.multivariate_normal
gmm = np.zeros_like(X)

for k in range(K):
    mix_comp = multivariate_normal(means[k, :].ravel(), covariances[k, :, :])
    gmm += weights[k] * mix_comp.pdf(pos)

#Visualize the dataset
plt.scatter(data[:, 0], data[:, 1], c='k', marker='o', alpha=0.3)
plt.title("Generated Dataset")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()

#Visualize the GMM components
plt.title("GMM Components")
plt.scatter(data[:, 0], data[:, 1], c='k', marker='o', alpha=0.3)

for k in range(K):
    mvn = multivariate_normal(means[k, :].ravel(), covariances[k, :, :])
    xx = mvn.pdf(pos)
    plt.contour(X, Y, xx, alpha=1.0, zorder=10)

plt.show()

#Build and visualize the GMM
plt.title("GMM")
plt.scatter(data[:, 0], data[:, 1], c='k', marker='o', alpha=0.3)

#Plot contours
plt.contour(X, Y, gmm, alpha=1.0, zorder=10)
plt.show()















