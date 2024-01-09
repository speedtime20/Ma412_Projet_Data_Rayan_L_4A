#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 17:24:27 2024

@author: rayan
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler

# Dataset
data = np.load('data.npy')[:18, :2]

# Standardize the data
data_standardized = StandardScaler().fit_transform(data)

# Apply OPTICS clustering
optics = OPTICS(min_samples=5, xi=.05, min_cluster_size=.05)
optics.fit(data_standardized)

# Visualize the results
plt.figure(figsize=(8, 6))

# Plot all points
plt.scatter(data_standardized[:, 0], data_standardized[:, 1], c='gray', alpha=0.5)

# Plot the clusters
colors = [plt.cm.nipy_spectral(cluster / float(max(optics.labels_) + 1)) for cluster in optics.labels_]
plt.scatter(data_standardized[:, 0], data_standardized[:, 1], c=colors, alpha=0.8)

plt.title('OPTICS Clustering')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()
