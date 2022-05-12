import numpy as np
from matplotlib import pyplot as plt
from pandas import Series
from mpl_toolkits.mplot3d import axes3d
from sklearn.preprocessing import PolynomialFeatures  #这用于生成多项式

def plotData(X,y):
    pos = X[np.where(y==1,True,False).flatten()]
    neg = X[np.where(y==0,True,False).flatten()]
    plt.plot(pos[:,0], pos[:,1], '+', markersize=7, markeredgecolor='black', markeredgewidth=2)
    plt.plot(neg[:,0], neg[:,1], 'o', markersize=7, markeredgecolor='black', markerfacecolor='yellow')

def plotDecisionBoundary(theta, X, y):
    """
    Plots the data points X and y into a new figure with the decision boundary defined by theta
      PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
      positive examples and o for the negative examples. X is assumed to be
      a either
      1) Mx3 matrix, where the first column is an all-ones column for the
         intercept.
      2) MxN, N>3 matrix, where the first column is all-ones
    """

    # Plot Data
    plt.figure()
    plotData(X[:,1:], y)

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([min(X[:, 2]),  max(X[:, 2])])

        # Calculate the decision boundary line
        plot_y = (-1./theta[2])*(theta[1]*plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)

    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = [
                np.array([mapFeature([[u[i],v[j]]])[0].dot(theta) for i in range(len(u))])
                for j in range(len(v))
            ]
        plt.contour(u,v,z, levels=[0.0])

    # Legend, specific for the exercise
    # axis([30, 100, 30, 100])


def mapFeature(X,degree=6):
    
    reg = PolynomialFeatures(degree=degree) #这个3看下面的解释
    return reg.fit_transform(X)
    