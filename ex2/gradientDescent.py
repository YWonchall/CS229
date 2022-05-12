from costFunction import costFunction2
import numpy as np
from sigmoid import sigmoid
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
        # 对每个样本先求导，再累加
        temp = np.array([0 for j in range(len(theta))])
        error = sigmoid(np.dot(X,theta))-y
        for j in range(len(theta)):
            temp[j] += np.dot(error,X[:,j])
        theta -= temp*alpha*1/m
             
            


        # ============================================================

        # Save the cost J in every iteration
        cost = costFunction2(theta,X, y)
        J_history.append(cost)
        print("iters = %d, cost = %f" % (i,cost))

    return theta, J_history