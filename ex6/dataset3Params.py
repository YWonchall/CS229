import numpy as np
from sklearn import svm


def dataset3Params(X, y, Xval, yval):
    """returns your choice of C and sigma. You should complete
    this function to return the optimal C and sigma based on a
    cross-validation set.
    """

# You need to return the following variables correctly.
    C_best = 1
    sigma_best = 0.3
    #gamma = 1.0 / (2.0 * sigma ** 2)
    acc_best = 0
# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return the optimal C and sigma
#               learning parameters found using the cross validation set.
#               You can use svmPredict to predict the labels on the cross
#               validation set. For example, 
#                   predictions = svmPredict(model, Xval)
#               will return the predictions on the cross validation set.
#
#  Note: You can compute the prediction error using 
#        mean(double(predictions ~= yval))
#
    for C in [0.1,0.3,0.5,1,5,10,50,100,1000]:
        for sigma in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
            gamma = 1.0 / (2.0 * sigma ** 2)
            clf = svm.SVC(C=C, kernel='rbf', tol=1e-3, max_iter=200, gamma=gamma)
            model = clf.fit(X, y)
            acc = model.score(Xval,yval)
            if(acc>acc_best):
                acc_best = acc
                C_best = C
                sigma_best = sigma
            
    print("C=%d,sigma=%f,acc=%f"%(C_best,sigma_best,acc_best))
# =========================================================================
    return C_best, sigma_best
