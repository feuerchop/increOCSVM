__author__ = 'LT'

#print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import ocsvm
import kernel
import itertools
import sys

def plot(predictor, X_train, X_test, X_outliers, grid_size, incremental):

    y_min = -5
    y_max = 5
    x_min = -5
    x_max = 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size),
                         indexing='ij')
    result = []
    for (i, j) in itertools.product(range(grid_size), range(grid_size)):
        point = np.array([xx[i, j], yy[i, j]]).reshape(1, 2)
        result.append(predictor.decision_function(point))

    Z = np.array(result).reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')
    if incremental:
        plt.scatter(X_train[:-1, 0], X_train[:-1, 1], c='white')
        plt.scatter(X_train[-1:, 0], X_train[-1:, 1], c='yellow')
    else: plt.scatter(X_train[:, 0], X_train[:, 1], c='white')
    plt.scatter(X_test[:, 0], X_test[:, 1], c='green')
    plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

def standardExample():
    # Generate train data
    X = 0.3 * np.random.randn(60, 2)
    X_train = np.r_[X + 2, X-2]

    # Generate some regular novel observations
    X = 0.3 * np.random.randn(15, 2)
    X_test = np.r_[X + 2,X-2]

    # Generate some abnormal novel observations
    X_outliers = np.random.uniform(low=-4, high=4, size=(15, 2))

    # Train the data
    clf = ocsvm.OCSVM("rbf", nu=0.5, gamma=3.1625)
    clf.train(X_train)


    #Plot the data
    plot(clf, X_train, X_test, X_outliers, 100, False)

    plt.show()
    #plt.savefig('test.pdf')

    # new point
    X = 0.3 * np.random.randn(1, 2)
    X_new = np.r_[X + 2, X-2]
    print X_new[0]

def incrementExample():
    # Generate train data
    X = 0.3 * np.random.randn(5, 2)
    X_train = np.r_[X + 2, X-2]
    X_train = np.array([[ 2.05014156, 1.57874676], [ 1.77364132, 1.79817027], [ 2.168239, 1.79832986], [ 1.75084617, 1.97693673], [ 2.22726199, 2.54527565],
            [-1.94985844, -2.42125324], [-2.22635868, -2.20182973], [-1.831761, -2.20167014], [-2.24915383, -2.02306327], [-1.77273801, -1.45472435]])


    # Generate some regular novel observations
    X = 0.3 * np.random.randn(1, 2)
    X_test = np.r_[X + 2,X-2]

    # Generate some abnormal novel observations
    X_outliers = np.random.uniform(low=-4, high=4, size=(2, 2))

    clf1 = ocsvm.OCSVM("rbf", nu=0.5, gamma=3.1625)
    clf1.train(X_train)
    #plot(clf1, X_train, X_test, X_outliers, 100, False)

    # Train the data
    clf = ocsvm.OCSVM("rbf", nu=0.5, gamma=3.1625)
    clf.train(X_train[:-1])

    #Plot the data
    plt.figure()
    #plot(clf1, X_train[:-1], X_test, X_outliers, 100, False)
    #plt.show()
    #print "point to increment"
    clf.increment(X_train[-1:])
    #plt.figure()
    #plot(clf, X_train, X_test, X_outliers, 100, True)

    #plt.savefig('test.pdf')

if __name__ == "__main__":
    incrementExample()




