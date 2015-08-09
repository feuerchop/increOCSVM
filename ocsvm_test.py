__author__ = 'LT'

#print(__doc__)
import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt
import ocsvm
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
    #print "Z: %s" % Z
    #print Z.max()
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
    clf = ocsvm.OCSVM("rbf", nu=0.1, gamma=5.1625)
    clf.train(X_train)

    print "alpha_s: %s" % clf._data.alpha_s()


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
    X = 0.3 * np.random.randn(10, 2)
    #X_train = np.r_[X + 2, X-2]
    X_train = X
    pickle.dump(X_train, open("/Users/LT/Documents/Uni/MA/increOCSVM/Xtrain.p", "w+"))
    #X_train = pickle.load(open("/Users/LT/Documents/Uni/MA/increOCSVM/Xtrain.p", 'r+'))
    #print X_train
    # Generate some regular novel observations
    X = 0.3 * np.random.randn(5, 2)
    #X_test = np.r_[X + 2,X-2]
    X_test = X
    # Generate some abnormal novel observations
    X_outliers = np.random.uniform(low=-4, high=4, size=(5, 2))
    pickle.dump(X_outliers, open("/Users/LT/Documents/Uni/MA/increOCSVM/Xoutliers.p", "w+"))

    #X_outliers = pickle.load(open("/Users/LT/Documents/Uni/MA/increOCSVM/Xoutliers.p", 'r+'))

    #print X_outliers
    clf1 = ocsvm.OCSVM("rbf", nu=0.1, gamma=0.1)

    #clf1.train(X_train[0:1])
    clf1.train(np.vstack((X_train,X_outliers[0])))
    plot(clf1, X_train, X_test, X_outliers, 100, False)
    print "standard alpha: %s" %clf1._data.alpha()
    print "standard alpha_s: %s" %clf1._data.alpha_s()
    print "standard X_s: %s "%clf1._data.Xs()
    #plt.show()

    # Train the data
    clf = ocsvm.OCSVM("rbf", nu=0.1, gamma=0.1)
    #clf.train(np.vstack((X_train[1:],X_outliers[1:3]))) # testing with outliers when training
    clf.train(X_train)
    #plt.figure()
    #plot(clf, X_train, X_test, X_outliers[1:], 100, False)

    #plot(clf, X_train[1:], X_test, X_outliers[-1:], 100, False)
    #
    clf.increment(X_outliers[0])

    #Plot the data
    plt.figure()
    plot(clf, X_train, X_test, X_outliers, 100, False)

    # Train the data
    #clf2 = ocsvm.OCSVM("rbf", nu=0.1, gamma=0.1)
    #clf.train(np.vstack((X_train[1:],X_outliers[1:3]))) # testing with outliers when training
    #clf2.train(np.vstack((X_train[1:],X_outliers[0])))
    #plt.figure()
    #plot(clf, X_train, X_test, X_outliers[1:], 100, False)

    #plot(clf, X_train[1:], X_test, X_outliers[-1:], 100, False)
    #
    #clf2.increment(X_train[0])

    #Plot the data
    #plt.figure()
    #plot(clf2, X_train, X_test, X_outliers, 100, False)

    #plt.draw()
    #print "point to increment"
    #
    #plt.figure()
    #plot(clf, X_train, X_test, X_outliers, 100, True)
    plt.show()
    #plt.savefig('test.pdf')


if __name__ == "__main__":
    incrementExample()




