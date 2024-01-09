#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 23:23:30 2024

@author: rayan
"""

import os
import numpy as np
from matplotlib import pyplot
import utils

# Data
data = np.load('data.npy')

# Assuming data has shape (3879, 18)
# Assuming the first two columns are features
X = data[:, :2] 
# Assuming the third column is the label 
y = data[:, 2]   

# Visualize the data
pyplot.scatter(X[:, 0], X[:, 1])
utils.plotData(X, y)  # Overlay the points
pyplot.show()

# Set the regularization parameter C
C = 1

# Train the SVM model with C=1
model = utils.svmTrain(X, y, C, utils.linearKernel, tol=1e-3, max_passes=20)

# Visualize the decision boundary
pyplot.figure()
pyplot.scatter(X[:, 0], X[:, 1])
utils.visualizeBoundaryLinear(X, y, model)
pyplot.title(f'SVM Decision Boundary with C={C}')
pyplot.show()

# Try different values of C, e.g., C=1000
# Set the regularization parameter C
C = 1000

# Train the SVM model with C=1000
model_high_c = utils.svmTrain(X, y, C, utils.linearKernel, tol=1e-3, max_passes=20)

# Visualize the decision boundary for the higher C value
pyplot.figure()
pyplot.scatter(X[:, 0], X[:, 1])
utils.visualizeBoundaryLinear(X, y, model_high_c)
pyplot.title(f'SVM Decision Boundary with C={C}')
pyplot.show()





def gaussianKernel(x1, x2, sigma):
   
    sim = 0

    diff = x1 - x2
    distance_squared = np.dot(diff, diff)

    sim = np.exp(-distance_squared / (2 * sigma**2))

    return sim

# Gaussian Kernel parameters
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 450

# Test the Kernel
similarity = gaussianKernel(x1, x2, sigma)
print(f'Gaussian Kernel similarity: {similarity}')

# The Gaussian Kernel for SVM with the data
C = 1543
model = utils.svmTrain(X, y, C, gaussianKernel, args=(sigma,))
pyplot.figure()
utils.visualizeBoundary(X, y, model)
pyplot.title(f'SVM Decision Boundary with Gaussian Kernel, C={C}, sigma={sigma}')
pyplot.show()