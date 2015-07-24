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

    def alpha(self):
        return self._alpha

    def X(self):
        return self._X

    # return data points corresponding to support vector
    def Xs(self):
        return self._X[np.all([self._alpha > 1e-5, self._alpha < self.C()], axis=0)]

    def get_Xr(self):
        return self._X[np.any([self._alpha < 1e-5, self._alpha == self.C()], axis=0)]

    # returns support vector
    def alpha_s(self):
        return self._alpha[self.get_sv()]

    # returns error vector and non-support vectors
    def get_alpha_r(self):
        return self._alpha[np.any([self._alpha <= 1e-5, self._alpha == self.C()], axis=0)]

    def set_C(self, C):
        self._C = C

    def C(self):
        return self._C

    def update_alpha_s(self, update_alpha):
        self._alpha[np.all([self._alpha > 1e-5, self._alpha < self.C()], axis=0)] = update_alpha

    def add(self, x_c, alpha_c):
        self._X = np.vstack((self._X, x_c))
        self._alpha = np.hstack((self._alpha, alpha_c))
    def get_sv(self):
        return np.all([self._alpha > 1e-5, self._alpha < self.C()], axis=0)
    def get_alpha_s_ind(self):
        return [ind for ind, i in enumerate(self.get_sv()) if i]

    def get_alpha_e(self):
        return self._alpha[self._alpha == self.C()]

    def get_alpha_o(self):
        return self._alpha[self._alpha <= 1e-5]

    def X_e(self):
        return self._X[self._alpha == self.C()]

    def get_X_o(self):
        return self._X[self._alpha <= 1e-5]

    def get_ind_S(self):
        ind = np.all([self._alpha > 1e-5, self._alpha < self.C()], axis=0)
        return [i for i,bool in enumerate(ind) if bool]

    def get_ind_R(self):
        ind = np.any([self._alpha <= 1e-5, self._alpha == self.C()], axis=0)
        return [i for i,bool in enumerate(ind) if bool]
