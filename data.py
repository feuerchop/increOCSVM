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

    def get_alpha(self):
        return self._alpha

    def get_X(self):
        return self._X

    # return data points corresponding to support vector
    def get_Xs(self):
        return self._X[np.all([self._alpha > 1e-5, self._alpha < self.get_C()], axis=0)]

    def get_Xr(self):
        return self._X[np.any([self._alpha < 1e-5, self._alpha == self.get_C()], axis=0)]

    # returns support vector
    def get_alpha_s(self):
        return self._alpha[self.get_sv()]

    # returns error vector and non-support vectors
    def get_alpha_r(self):
        return self._alpha[np.any([self._alpha <= 1e-5, self._alpha == self.get_C()], axis=0)]

    def set_C(self, C):
        self._C = C

    def get_C(self):
        return self._C

    def update_alpha_s(self, update_alpha):
        self._X[np.all([self._alpha > 1e-5, self._alpha < self.get_C()], axis=0)] = update_alpha

    def add(self, x_c, alpha_c):
        self._X = np.concatenate((self._X, x_c), axis=0)
        self._alpha = np.concatenate((self._alpha, alpha_c), axis=1)
    def get_sv(self):
        return np.all([self._alpha > 1e-5, self._alpha < self.get_C()], axis=0)
    def get_alpha_s_ind(self):
        return [ind for ind, i in enumerate(self.get_sv()) if i]