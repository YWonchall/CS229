import numpy as np


def kMeansInitCentroids(X, K):
    """returns K initial centroids to be
    used with the K-Means on the dataset X
    """

# You should return this values correctly
    centroids = np.zeros((K, X.shape[1]))

# ====================== YOUR CODE HERE ======================
# Instructions: You should set centroids to randomly chosen examples from
#               the dataset X
#
    
    index = np.random.randint(0,X.shape[0],size=K)
    centroids = X[index]

# =============================================================
    return centroids
