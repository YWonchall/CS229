from costFunctionReg import costFunctionReg
import numpy as np
from sigmoid import sigmoid
def gradientDescentReg(X, y, theta, alpha,Lambda, num_iters):
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
        # 对每个样本先求导，再累加
        grad = np.zeros(X.shape[1])
        error = sigmoid(np.dot(X,theta))-y
        grad[0] += np.dot(error,X[:,0])
        for j in range(1,len(theta)):
            grad[j] += np.dot(error,X[:,j]) + Lambda*theta[j]
        grad /= m
        theta -= grad*alpha
             
            


        # ============================================================

        # Save the cost J in every iteration
        cost = costFunctionReg(theta,X, y,Lambda)
        J_history.append(cost)
        print("iters = %d, cost = %f" % (i,cost))

    return theta, J_history