#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 17:25:42 2024

@author: rayan
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = np.load('data.npy')[:18, :2]

# Standardize the data
data_standardized = StandardScaler().fit_transform(data)

# Estimate bandwidth (you can adjust quantile to control the bandwidth)
bandwidth = estimate_bandwidth(data_standardized, quantile=0.2, n_samples=len(data_standardized))

# Apply Mean-Shift clustering
meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
meanshift.fit(data_standardized)

# Visualize the results
plt.figure(figsize=(8, 6))

# Plot all points
plt.scatter(data_standardized[:, 0], data_standardized[:, 1], c='gray', alpha=0.5)

# Plot the clusters
colors = [plt.cm.nipy_spectral(cluster / float(len(set(meanshift.labels_)))) for cluster in meanshift.labels_]
plt.scatter(data_standardized[:, 0], data_standardized[:, 1], c=colors, alpha=0.8)

plt.title('Mean-Shift Clustering')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()
