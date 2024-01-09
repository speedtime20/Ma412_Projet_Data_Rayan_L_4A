#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 17:05:26 2024

@author: rayan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import utils

# Data
data = np.load('data.npy')

# Assuming data has shape (3879, 18)
# Assuming the first two columns are features
X = data[:, :2]  

# Visualize the data
plt.figure()
plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors='b')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


def feature_normalize(X):
    
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    
    return X_norm, mu, sigma

def pca(X):
    
    Sigma = np.dot(X.T, X) / X.shape[0]

    U, S, V = np.linalg.svd(Sigma)
    
    return U, S, V

def draw_line(ax, p1, p2, dash=False, label=None, color='k'):

    if dash:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', color=color, label=label)
    else:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, label=label)


# Feature normalization
X_norm, mu, sigma = feature_normalize(X)

# Run PCA
U, S, V = pca(X_norm)

# Visualize the principal components
plt.figure()
draw_line(plt.gca(), mu, mu + 1.5 * S[0] * U[:, 0].T, label='Principal Component 1', color='blue')
draw_line(plt.gca(), mu, mu + 1.5 * S[1] * U[:, 1].T, label='Principal Component 2', color='red')
plt.legend()
plt.show()

print('Top eigenvector:')
print ('U = ', U[:, 0])




def project_data(X, U, K):

    U_reduce = U[:, :K]

    Z = np.dot(X, U_reduce)

    return Z

def recover_data(Z, U, K):
 
    X_rec = Z.dot(U[:, 0:K].T)
    return X_rec

# Assuming data has shape (3879, 18)
# Assuming the first two columns are features
X = data[:, :2] 

# Feature normalization
X_norm, mu, sigma = feature_normalize(X)

# Run PCA
U, S, V = pca(X_norm)

# Plot the normalized dataset
plt.figure()
plt.scatter(X_norm[:, 0], X_norm[:, 1], facecolors='none', edgecolors='b')
plt.gca().set_aspect('equal', adjustable='box')

# Project the data onto K = 1 dimension
K = 1
Z = project_data(X_norm, U, K)
print('Projection of the first example:', Z[0, ])
print('(this value should be about 1.48127391)')

# Recover the data from the projected data
X_rec = recover_data(Z, U, K)
print('Approximation of the first example:', X_rec[0, ])
print('(this value should be about  -1.04741883 -1.04741883)')

# Draw lines connecting the projected points to the original points
plt.scatter(X_rec[:, 0], X_rec[:, 1], facecolors='none', edgecolors='r')
for i in range(X_norm.shape[0]):
    draw_line(plt.gca(), X_norm[i, :], X_rec[i, :], dash=True)

plt.show()



# Data
data = np.load('data.npy')
# Assuming the first two columns are features
X = data[:, :2]  

# Visualize the original data
plt.figure()
plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors='b')
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Original Data')
plt.show()

# Feature normalization
X_norm, mu, sigma = feature_normalize(X)

# Run PCA
U, S, V = pca(X_norm)

# Reduce dimensionality to K = 1
K = 1
Z = project_data(X_norm, U, K)

# Display the reduced data
plt.figure()
plt.scatter(Z, np.zeros_like(Z), facecolors='none', edgecolors='r')
plt.title('Reduced Data (Projection onto K = 1)')
plt.show()

# Recover the data from the reduced dimensionality
X_rec = recover_data(Z, U, K)

# Display original and reconstructed data
plt.figure()
plt.scatter(X_norm[:, 0], X_norm[:, 1], facecolors='none', edgecolors='b')
plt.title('Original Data')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

plt.figure()
plt.scatter(X_rec[:, 0], X_rec[:, 1], facecolors='none', edgecolors='r')
plt.title('Reconstructed Data from K = 1')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

