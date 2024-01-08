#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 17:20:55 2024

@author: rayan
"""

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from scipy.stats import multivariate_normal

# Load the dataset
data = np.load('data.npy')[:18, :2]

# Standardize the data
data_normalized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Set the number of clusters
K = 3

# Apply Fuzzy C-Means clustering
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data_normalized.T, K, 2, error=0.005, maxiter=1000, init=None)

# Extract cluster labels
labels = np.argmax(u, axis=0)

# Visualize the results
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.8)
plt.scatter(cntr[:, 0], cntr[:, 1], c='red', marker='X', s=100, label='Centroids')
plt.title('Fuzzy C-Means Clustering')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.show()
