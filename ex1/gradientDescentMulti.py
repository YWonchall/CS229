from computeCostMulti import computeCostMulti
import numpy as np

def gradientDescentMulti(X, y, theta, alpha, num_iters):
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
        for k in range(m):
            for j in range(len(theta)):
                temp[j] += (np.dot(theta,X[k])-y[k])*X[k][j]
        theta -= temp*alpha*1/m
             
            


        # ============================================================

        # Save the cost J in every iteration
        J_history.append(computeCostMulti(X, y, theta))

    return theta, J_history