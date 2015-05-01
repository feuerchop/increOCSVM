__author__ = 'LT'

import numpy as np
import numpy.linalg as la

# implements lists of kernels
class Kernel(object):

    @staticmethod
    def gaussian(gamma):
        return lambda x, y: \
            np.exp(- gamma * (la.norm(x-y) ** 2))