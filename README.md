# AERO 4 - Mathematical Tools for Data Science (2023/2024)

## Project Overview

This repository contains the implementation of unsupervised clustering techniques applied to a dataset of aircraft trajectories. The goal is to group the examples in the dataset based on their features. 

## Dataset

The dataset, 'data.npy', consists of 3879 examples with 18 features each. Each example represents an aircraft trajectory with its position in the sky and other significant features. The data are clean, and no pre-processing is required.

## Repository Structure

- `code/`: Contains the Python source code file.
  - `Affinity Propagation.py`: Python code implementing the Affinity Propagation clustering method on the dataset.
  - `Birch.py`: Python code applying the Birch clustering method on the dataset.
  - `DBSCAN.py`: Python code using the DBScan clustering method on the dataset.
  - `Fuzzy C-Means clustering.py`: Python code implementing the Fuzzy C-Means clustering method on the dataset.
  - `GMM.py`: Python code of the Gaussian mixtures model method on the dataset.
  - `KMean.py`: Python code of the K-means method on the dataset.
  - `KMedoids.py`: Python code applying the K-Medoids clustering method on the dataset.
  - `MaxLikehood.py`: Python code of the maximum likelihood method on the dataset.
  - `Mean-Shift Clustering.py`: Python code applying the Mean-Shift clustering method on the dataset.
  - `MiniBatchKMeans`: Python code implementing the Mini-Batch K-Means clustering method on the dataset.
  - `OPTICS.py`: Python code of the Ordering Points To Identify the Clustering Structure on the dataset.
  - `PCA.py`: Python code of the Principal component analysis method and dimension reduction on the dataset.
  - `Ridge_Lasso_ElasticNet.py`: Python code of the Ridge regression method on the dataset.
  - `Spectral Clustering.py`: Python code implementing the Spectral Clustering method on the dataset.
  - `SVM.py`: Python code of the Support vector machine method on the dataset.
  - `utils.py`: Library Python code of the function used for the other method.

- `data.npy`: The dataset file.
- `Ma412_Final_Project_LAKARDI.pdf`: Document explaining the problem, possible solutions, chosen methods, and algorithm explanations.

## Getting Started

To get started with the project, follow these steps:

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/speedtime20/Ma412_Projet_Data_Rayan_L_4A.git
    ```

2. **Navigate to the Cloned Repository:**
    ```bash
    cd speedtime20-Ma412_Projet_Data_Rayan_L_4A
    ```
