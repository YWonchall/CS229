import numpy as np
def linearRegCostFunction(X, y, theta, Lambda):
    """computes the
    cost of using theta as the parameter for linear regression to fit the
    data points in X and y. Returns the cost in J and the gradient in grad
    """
# Initialize some useful values

    m = y.size # number of training examples

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost and gradient of regularized linear 
#               regression for a particular choice of theta.
#
#               You should set J to the cost and grad to the gradient.
#
    grad = np.zeros(len(theta))
    error= np.dot(X,theta)-y
    J = np.sum(error**2)*0.5/m + Lambda*np.sum(theta[1:]**2)/2/m
    grad[0] += np.dot(error,X[:,0])
    for j in range(1,len(theta)):
        grad[j] += np.dot(error,X[:,j]) + Lambda*theta[j]
    grad /= m
# =========================================================================

    return J, grad