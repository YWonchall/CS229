import numpy as np

def computeCost(X, y, theta):
    """
       computes the cost of using theta as the parameter for linear 
       regression to fit the data points in X and y
    """
    m = y.size
    J = 0

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.
    for i in range(m):
        J+= (np.dot(theta,X[i])-y[i])**2
    J /=(2*m)    

# =========================================================================

    return J


def computeCost2(X,y,theta):
    # 矩阵运算
    m = y.size
    z = np.dot(X,theta)
    J = np.sum((y-z)**2)*0.5/m
    return J

