import numpy as np
def computeCostMulti(X, y, theta):
    """
     Compute cost for linear regression with multiple variables
       J = computeCost(X, y, theta) computes the cost of using theta as the
       parameter for linear regression to fit the data points in X and y
    """
    m = y.size
    J = 0
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.
    m = y.size
    z = np.dot(X,theta)
    J = np.sum((y-z)**2)*0.5/m

# =========================================================================

    return J
