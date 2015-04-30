__author__ = 'LT'

import numpy as np
import numpy.linalg as la

# implements lists of kernels
class Kernel(object):

    @staticmethod
    def gaussian(c):
        return lambda x, y: \
            np.exp(-(la.norm(x-y)) ** 2 / c)