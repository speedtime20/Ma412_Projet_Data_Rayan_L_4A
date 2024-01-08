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

# Load your data
data = np.load('data.npy')

# Assuming data has shape (3879, 18)
X = data[:, :2]  # Assuming the first two columns are features

# Visualize the data
plt.figure()
plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors='b')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()



#EDIT THIS CELL
def feature_normalize(X):
    """
    Normalizes the features in X.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.

    Returns
    -------
    X_norm : ndarray, shape (n_samples, n_features)
        Normalized training vectors.
    mu : ndarray, shape (n_feature, )
        Mean value of each feature.
    sigma : ndarray, shape (n_feature, )
        Standard deviation of each feature.
    """
    # ============================================================
    # Your code here ...  
    
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    
    # ============================================================
    
    return X_norm, mu, sigma


#EDIT THIS CELL
def pca(X):
    """
    Run principal component analysis on the dataset X.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.

    Returns
    -------
    U : ndarray, shape (n_features, n_features)
        Unitary matrices.
    S : ndarray, shape (n_features,)
        The singular values for every matrix.
    V : ndarray, shape (n_features, n_features)
        Unitary matrices.
    """
    # ============================================================
    # Your code here ...
    
    Sigma = np.dot(X.T, X) / X.shape[0]

    U, S, V = np.linalg.svd(Sigma)
    
    # ============================================================
    return U, S, V

def draw_line(ax, p1, p2, dash=False, label=None, color='k'):
    """
    Draws a line from point p1 to point p2.

    Parameters
    ----------
    ax : matplotlib AxesSubplot
        The axes on which to draw the line.
    p1 : ndarray
        Point 1.
    p2 : ndarray
        Point 2.
    dash : bool, optional
        True to plot a dashed line.
    label : str, optional
        Label for the line.
    color : str, optional
        Line color.
    """
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




   
#=============================================================================2

def project_data(X, U, K):
    """
    Computes the reduced data representation when projecting only on to the top K eigenvectors.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.
    U : ndarray, shape (n_features, n_features)
        Unitary matrices.
    K : int
        Reduced dimension.

    Returns
    -------
    Z : ndarray, shape (n_samples, K)
        The projection of X into the reduced dimensional space spanned by the first K columns of U.
    """
    # ============================================================
    # Your code here ...
    
    U_reduce = U[:, :K]

    Z = np.dot(X, U_reduce)
    
    # ============================================================
    return Z

def recover_data(Z, U, K):
    """
    Recovers an approximation of the original data when using the projected data.
    
    Parameters
    ----------
    Z : ndarray, shape (n_samples, K)
        The projected data, where n_samples is the number of samples and K is the number of reduced dimensions.
    U : ndarray, shape (n_features, n_features)
        Unitary matrices, where n_features is the number of features.
    K : int
        Reduced dimension.

    Returns
    -------
    X_rec : ndarray, shape (n_samples, n_features)
        The recovered samples.
    """
    X_rec = Z.dot(U[:, 0:K].T)
    return X_rec

# Assuming data has shape (3879, 18)
X = data[:, :2]  # Assuming the first two columns are features

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



 # ============================================================ EXEMPLE
# Load your data
data = np.load('data.npy')
X = data[:, :2]  # Assuming the first two columns are features

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

