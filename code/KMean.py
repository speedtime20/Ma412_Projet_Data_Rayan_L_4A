#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 17:27:11 2024

@author: rayan
"""

# ============================================================================== 2 affichages !
import os
import numpy as np
import utils
import re
import matplotlib.pyplot as plt


from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

from IPython.display import HTML, display, clear_output

try:
    pyplot.rcParams["animation.html"] = "jshtml"
except ValueError:
    pyplot.rcParams["animation.html"] = "html5"

from scipy import optimize
from scipy.io import loadmat
import utils

def findClosestCentroids(X, centroids):
    # The number of clusters
    K = centroids.shape[0]
    
    # The centroids 
    idx = np.zeros(X.shape[0], dtype=int)

    # Iterate 
    for i in range(X.shape[0]):
        # Calculate the Euclidean distance 
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        
        # Find the index of the centroid with the smallest distance 
        closest_centroid_index = np.argmin(distances)
        
        # Assign the index of the closest centroid 
        idx[i] = closest_centroid_index
    
    return idx

# Data
data = np.load('data.npy')

# Assuming data has shape (3879, 18)
X = data[:, :2] 

# The number of clusters 
K = 3

# Initialize centroids
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Find the closest centroids 
idx = findClosestCentroids(X, initial_centroids)

# Display the results
print('Closest centroids for the examples:')
print(idx)







def computeCentroids(X, idx, K):
    m, n = X.shape
    # The centroids are in the following variable
    centroids = np.zeros((K, n))
    
    # Iterate over each centroid
    for k in range(K):
        
        # On sélectionne les exemples qui ont été assignés au centroid actuel
        # Elle utilise un masque booléen idx == k pour sélectionner les exemples associés à ce centroid
        examples_in_centroid = X[idx == k]
        
        # Check if there are examples assigned to the centroid
        if len(examples_in_centroid) > 0:
            # On calcule la moyenne des exemples associés au centroid actuel le long 
            # de l'axe axis=0. Cela donne la nouvelle position du centroid
            centroid_mean = np.mean(examples_in_centroid, axis=0)
            
            # On met à jour les coordonnées du centroid actuel avec la nouvelle moyenne calculée
            centroids[k] = centroid_mean
    
    return centroids


# Assuming K is the number of centroids
K = initial_centroids.shape[0]

# Compute centroids means
centroids = computeCentroids(X, idx, K)

print('Centroids computed after initial finding of closest centroids:')
print(centroids)




# Settings for running K-Means
K = 3
max_iters = 10

# For consistency, here we set centroids to specific values
# but in practice you want to generate them automatically, such as by
# settings them to be random examples (as can be seen in
# kMeansInitCentroids).
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])


# Run K-Means algorithm. The 'true' at the end tells our function to plot
# the progress of K-Means
centroids, idx, anim = utils.runkMeans(X, initial_centroids,
                                       findClosestCentroids, computeCentroids, max_iters, True)
#Save the animation as a GIF file
anim.save('kmeans_animation.gif', writer='imagemagick', fps=2)

#Display a message indicating where the animation is saved
print("Animation saved as 'kmeans_animation.gif'")


def kMeansInitCentroids(X, K):

    m, n = X.shape 
    
    centroids = np.zeros((K, n))

    indices = np.random.choice(m, K, replace=False)
    
    centroids = X[indices, :]
    
    return centroids

# Assuming data has shape (3879, 18)
X = data[:, :2]  

# Set the number of centroids and maximum iterations
K = 16
max_iters = 10

# When using K-Means, it is important to randomly initialize centroids
# You should complete the code in kMeansInitCentroids above before proceeding
initial_centroids = kMeansInitCentroids(X, K)

# Run K-Means
centroids, idx = utils.runkMeans(X, initial_centroids,
                                 findClosestCentroids,
                                 computeCentroids,
                                 max_iters)


# Visualize the results
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=idx, cmap='viridis', marker='o', edgecolors='w', s=30)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()





