#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 17:17:29 2024

@author: rayan
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Dataset
data = np.load('data.npy')[:18, :2]

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply DBSCAN with different parameters for the first plot
dbscan_params1 = {'eps': 0.3, 'min_samples': 4}
dbscan1 = DBSCAN(**dbscan_params1)
labels1 = dbscan1.fit_predict(data_scaled)

# Apply DBSCAN with different parameters for the second plot
dbscan_params2 = {'eps': 0.5, 'min_samples': 3}
dbscan2 = DBSCAN(**dbscan_params2)
labels2 = dbscan2.fit_predict(data_scaled)

# Visualize the results in two separate plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot for the first DBSCAN configuration
axes[0].set_title('DBSCAN Clustering (Parameters 1)')
axes[0].set_xlabel('$x_1$')
axes[0].set_ylabel('$x_2$')

for label in set(labels1):
    if label == -1:
        col = 'gray'
        markersize = 6
    else:
        col = plt.cm.Spectral(label / len(set(labels1)))
        markersize = 10

    class_member_mask = (labels1 == label)
    xy = data[class_member_mask]
    axes[0].plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markersize=markersize)

# Plot for the second DBSCAN configuration
axes[1].set_title('DBSCAN Clustering (Parameters 2)')
axes[1].set_xlabel('$x_1$')
axes[1].set_ylabel('$x_2$')

for label in set(labels2):
    if label == -1:
        col = 'gray'
        markersize = 6
    else:
        col = plt.cm.Spectral(label / len(set(labels2)))
        markersize = 10

    class_member_mask = (labels2 == label)
    xy = data[class_member_mask]
    axes[1].plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markersize=markersize)

plt.show()



