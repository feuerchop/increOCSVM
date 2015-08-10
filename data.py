__author__ = 'LT'
import numpy as np
class Data(object):
    _X = None
    _alpha = None
    _C = None

    def __init__(self):
        self._X = None
        self._alpha = None

    def set_X(self, X):
        self._X = X

    def set_alpha(self, alpha):
        self._alpha = alpha

    def set_alpha_s(self,alpha_s):
        self._alpha[self.get_sv()] = alpha_s

    def alpha(self):
        return self._alpha

    def X(self):
        return self._X

    # return data points corresponding to support vector
    def Xs(self):
        return self._X[np.all([self._alpha > 1e-5, self._alpha < self.C() - 1e-5], axis=0)]

    # returns support vector
    def alpha_s(self):
        return self._alpha[self.get_sv()]

    def set_C(self, C):
        self._C = C

    def C(self):
        return self._C

    def add(self, x_c, alpha_c):
        self._X = np.vstack((self._X, x_c))
        self._alpha = np.hstack((self._alpha, alpha_c))

    def get_sv(self):
        return np.all([self._alpha > 1e-5, self._alpha < self.C() - 1e-5], axis=0)


