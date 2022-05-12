from computeCost import computeCost
import numpy as np

def gradientDescent(X, y, theta, alpha, num_iters):
    """
     Performs gradient descent to learn theta
       theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    J_history = []
    m = y.size  # number of training examples

    for i in range(num_iters):
        #   ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #
        temp0 = 0
        temp1 = 0
        for j in range(m):
            temp0 += (np.dot(theta,X[j])-y[j])
            temp1 += (np.dot(theta,X[j])-y[j])*X[j][1]
        theta[0] -= alpha*temp0/m
        theta[1] -= alpha*temp1/m
        # Simultaneously update
        # ============================================================

        # Save the cost J in every iteration
        J_temp = computeCost(X, y, theta)
        J_history.append(J_temp)

    return theta, J_history
