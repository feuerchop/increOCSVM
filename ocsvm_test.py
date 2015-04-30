__author__ = 'LT'

#print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import ocsvm
import kernel
import itertools

def plot(predictor, X_train, X_test, X_outliers, grid_size, filename):

    y_min = -5
    y_max = 5
    x_min = -5
    x_max = 5
    result = []
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size),
                         indexing='ij')
    result = []
    for (i, j) in itertools.product(range(grid_size), range(grid_size)):
        point = np.array([xx[i, j], yy[i, j]]).reshape(1, 2)
        result.append(predictor.predict(point))

    Z = np.array(result).reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=[-1.5,0], colors='blue')
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
    plt.contour(xx, yy, Z, levels=[0,1.5], colors='orange')

    plt.scatter(X_train[:, 0], X_train[:, 1], c='white')
    plt.scatter(X_test[:, 0], X_test[:, 1], c='green')
    plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()
    plt.savefig(filename)

if __name__ == "__main__":

    # Generate train data
    X = 0.3 * np.random.randn(20, 2)
    X_train = np.r_[X + 2, X-2]

    # Generate some regular novel observations
    X = 0.3 * np.random.randn(5, 2)
    X_test = np.r_[X + 2,X-2]

    # Generate some abnormal novel observations
    X_outliers = np.random.uniform(low=-4, high=4, size=(5, 2))

    # Train the data
    clf = ocsvm.OCSVM(kernel.Kernel.gaussian(0.1),nu=0.07, sigma=0.1)
    predictor = clf.train(X_train)

    # Plot the data

    for x_i in X_train:

        if predictor.predict(x_i) == 1.0:
            plt.scatter(x_i[0],x_i[1],c="red")
        else:
            plt.scatter(x_i[0],x_i[1],c="blue")

    for x_i in X_test:
        if predictor.predict(x_i) == 1.0:
            plt.scatter(x_i[0],x_i[1],c="red")
        else:
            plt.scatter(x_i[0],x_i[1],c="blue")

    for x_i in X_outliers:
        if predictor.predict(x_i) == 1.0:
            plt.scatter(x_i[0],x_i[1],c="red")
        else:
            plt.scatter(x_i[0],x_i[1],c="blue")

    # Plot prediction
    plt.show()

    # plot(predictor, X_train, X_test, X_outliers, 50, 'test.pdf')
