#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 17:33:59 2024

@author: rayan
"""

from sklearn.cluster import AffinityPropagation
import numpy as np
import matplotlib.pyplot as plt

# Dataset
data = np.load('data.npy')[:18, :2]

# Initialize Affinity Propagation
affinity_propagation = AffinityPropagation()

# Fit and predict the labels
affinity_propagation_labels = affinity_propagation.fit_predict(data)

# Visualize the results
plt.scatter(data[:, 0], data[:, 1], c=affinity_propagation_labels, cmap='viridis', marker='o', alpha=0.8)
plt.scatter(affinity_propagation.cluster_centers_[:, 0], affinity_propagation.cluster_centers_[:, 1], c='red', marker='x', s=200)
plt.title("Affinity Propagation Clustering")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()
