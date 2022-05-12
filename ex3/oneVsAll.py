import sys
sys.path.append('../ex2/')
import numpy as np
from scipy.optimize import minimize
from costFunctionReg import costFunctionReg


def oneVsAll(X, y, num_labels, Lambda):
    """trains multiple logistic regression classifiers and returns all
        the classifiers in a matrix all_theta, where the i-th row of all_theta
        corresponds to the classifier for label i
    """

# Some useful variables
    m, n = X.shape

# You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))

# Add ones to the X data matrix
    X = np.column_stack((np.ones((m, 1)), X))

# ====================== YOUR CODE HERE ======================
# Instructions: You should complete the following code to train num_labels
#               logistic regression classifiers with regularization
#               parameter lambda. 
#
# Hint: theta(:) will return a column vector.
#
# Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
#       whether the ground truth is true/false for this class.
#
# Note: For this assignment, we recommend using fmincg to optimize the cost
#       function. It is okay to use a for-loop (for c = 1:num_labels) to
#       loop over the different classes.

    # Set Initial theta
    initial_theta = np.zeros((n + 1, 1))

    # This function will return theta and the cost
    for cf in range(1,num_labels+1):
        y_ = np.zeros(m)
        y_[y==cf] =1
        res = minimize(costFunctionReg, initial_theta, method='SLSQP',
               jac=False, args=(X, y_,Lambda), options={'gtol': 1e-3, 'disp': True, 'maxiter': 1000})
        if(cf!=10):
            all_theta[cf] = res.x
        else:
            all_theta[0] = res.x
        print("classfier %d Done!" % cf)
        



# =========================================================================

    return all_theta

