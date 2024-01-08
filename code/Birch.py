#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 17:28:32 2024

@author: rayan
"""

from sklearn.cluster import Birch
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data = np.load('data.npy')[:18, :2]

# Apply Birch clustering
brc = Birch(branching_factor=50, n_clusters=3, threshold=0.5, compute_labels=True)
brc.fit(data)

# Visualize the results
plt.scatter(data[:, 0], data[:, 1], c=brc.labels_, cmap='viridis', marker='o', alpha=0.7)
plt.scatter(brc.subcluster_centers_[:, 0], brc.subcluster_centers_[:, 1], c='red', marker='X', s=100, label='Subcluster Centers')
plt.title("Birch Clustering")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend()
plt.show()
