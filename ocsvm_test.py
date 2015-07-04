__author__ = 'LT'

#print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import ocsvm
import kernel
import itertools
import sys

def plot(predictor, X_train, X_test, X_outliers, grid_size):

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

    plt.scatter(X_train[:, 0], X_train[:, 1], c='white')
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

    # Plot the data
    #plot(clf, X_train, X_test, X_outliers, 100)

    #plt.show()
    #plt.savefig('test.pdf')

    # new point
    X = 0.3 * np.random.randn(1, 2)
    X_new = np.r_[X + 2, X-2]
    print X_new[0]

def incrementExample():
    # Generate train data
    X = 0.3 * np.random.randn(5, 2)
    X_train = np.r_[X + 2, X-2]


    # Generate some regular novel observations
    X = 0.3 * np.random.randn(5, 2)
    X_test = np.r_[X + 2,X-2]

    # Generate some abnormal novel observations
    X_outliers = np.random.uniform(low=-4, high=4, size=(5, 2))

    # Train the data
    clf = ocsvm.OCSVM("rbf", nu=0.5, gamma=3.1625)
    clf.train(X_train[0:8])

    # Plot the data
    plot(clf, X_train[0:9], X_test, X_outliers, 100)
    print "point to increment"
    clf.increment(X_train[9])
    plt.figure()
    plot(clf, X_train[0:10], X_test, X_outliers, 100)
    plt.show()
    #plt.savefig('test.pdf')

if __name__ == "__main__":
    incrementExample()




