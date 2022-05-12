import numpy as np
from sigmoid import sigmoid

def costFunction(theta, X,y):
    """ computes the cost of using theta as the
    parameter for logistic regression and the
    gradient of the cost w.r.t. to the parameters."""

# Initialize some useful values
    m = y.size # number of training examples


# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
    J = 0
    for i in range(m):
        J += y[i]*np.log(sigmoid(np.dot(X[i],theta)))+(1-y[i])*np.log(1-sigmoid(np.dot(X[i],theta)))
    
    J = -J/m
#
# Note: grad should have the same dimensions as theta
#
    return J

def costFunction2(theta, X,y):
    # 矩阵运算
    m = y.size
    z = np.dot(X,theta)
    z = sigmoid(z)
    # 解决log(0)的问题
    z[z==1] = 0.999
    z[z==0] = 0.001
    J = np.dot(y,np.log(z))+np.dot(1-y,np.log(1-z))

    J/= -m
#
#
    return J
