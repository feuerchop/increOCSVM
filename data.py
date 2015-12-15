__author__ = 'LT'
import numpy as np
import sys
class Data(object):

    _X = None
    _alpha = None
    _C = None
    _e = 1e-5
    _K_X = None

    def __init__(self):
        self._X = None
        self._alpha = None
        self._K_X = None


    def set_X(self, X):
        self._X = X

    def set_alpha(self, alpha):
        self._alpha = alpha

    def alpha(self):
        return self._alpha

    def X(self):
        return self._X

    # return data points corresponding to support vector
    def Xs(self):
        return self._X[self.get_sv()]

    def set_e(self, e):
        self._e = e

    # returns support vector
    def alpha_s(self):
        return self._alpha[self.get_sv()]
    # sets regularization value
    def set_C(self, C):
        self._C = C
    # returns regularization value
    def C(self):
        return self._C

    def add(self, x_c, alpha_c):
        X = np.zeros((self._X.shape[0] + 1, self._X.shape[1]))
        X[0,:] = x_c
        X[1:,:] = self._X
        self._X = X
        alpha = np.zeros(len(self._alpha)+1)
        alpha[0] = alpha_c
        alpha[1:] = self._alpha
        self._alpha = alpha
        #self._X = np.vstack((self._X, x_c))
        #self._alpha = np.hstack((self._alpha, alpha_c))
    # get indecis of support vector
    def get_sv(self):

        if len(self.alpha()[np.all([self._alpha > self._e, self._alpha < self._C - self._e], axis=0)]) < 1:
            e = self._e
            while len(self.alpha()[np.all([self._alpha > e, self._alpha < self._C - e], axis=0)]) < 1:
                e /= 10
            return np.all([self._alpha > e, self._alpha < self._C - e], axis=0)
        else:
            return np.all([self._alpha > self._e, self._alpha < self._C - self._e], axis=0)
    # return gram matrix
    def set_K_X(self, K_X):
        self._K_X = K_X
    # sets gram matrix
    def K_X(self):
        return self._K_X


