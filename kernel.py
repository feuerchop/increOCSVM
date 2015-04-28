__author__ = 'LT'

import numpy as np
import numpy.linalg as la

# implements lists of kernels
class Kernel(object):

    @staticmethod
    def gaussian(sigma):
        return lambda x, y: \
            np.exp(-np.sqrt(la.norm(x-y) ** 2 / (2 * sigma ** 2)))