#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 17:30:22 2024

@author: rayan
"""

from sklearn_extra.cluster import KMedoids
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset (replace 'data' with your actual data)
data = np.load('data.npy')[:18, :2]

# Number of clusters (K=3 for example)
K = 3

# Initialize K-Medoids
kmedoids = KMedoids(n_clusters=K)

# Fit and predict the labels
kmedoids_labels = kmedoids.fit_predict(data)

# Visualize the results
plt.scatter(data[:, 0], data[:, 1], c=kmedoids_labels, cmap='viridis', marker='o', alpha=0.8)
plt.scatter(kmedoids.cluster_centers_[:, 0], kmedoids.cluster_centers_[:, 1], c='red', marker='x', s=200)
plt.title("K-Medoids Clustering")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()
