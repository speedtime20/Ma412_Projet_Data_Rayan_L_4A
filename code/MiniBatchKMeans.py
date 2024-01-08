#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 17:29:10 2024

@author: rayan
"""

from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data = np.load('data.npy')[:18, :2]

# Apply MiniBatchKMeans clustering
mbk = MiniBatchKMeans(n_clusters=3, batch_size=100, random_state=42)
mbk.fit(data)

# Visualize the results
plt.scatter(data[:, 0], data[:, 1], c=mbk.labels_, cmap='viridis', marker='o', alpha=0.7)
plt.scatter(mbk.cluster_centers_[:, 0], mbk.cluster_centers_[:, 1], c='red', marker='X', s=100, label='Cluster Centers')
plt.title("MiniBatchKMeans Clustering")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend()
plt.show()
