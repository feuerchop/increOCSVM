__author__ = 'LT'
import numpy
import cvxopt.solvers
import kernel
from time import gmtime, strftime
from sklearn.metrics.pairwise import pairwise_kernels
#print(__doc__)

#Trains an SVM
class OCSVM(object):
    # define global variables
    _rho = None
    _alpha = None
    _sv = None
    _kernel = None
    _nu = None
    _gamma = None

    #Class constructor: kernel function & nu & sigma
    def __init__(self, metric, nu, gamma):
        if metric == "rbf":
            self._kernel = kernel.Kernel.gaussian(gamma)
        self._nu = nu
        self._gamma = gamma

    #returns trained SVM predictor given features (X)
    # TODO: we need to store the key properties of model after training
    # Please check libsvm what they provide for output, e.g., I need to access to the sv_idx all the time
    def train(self, X):
        # get lagrangian multiplier
        alpha = self.alpha(X)
        # defines necessary parameter for prediction
        self.predictor(X, alpha)

    #returns SVM prediction with given X and langrange mutlipliers
    def predictor(self, X, alpha):
        # define support vector and weights/alpha
        sv_index = alpha > 1e-5
        self._alpha = alpha[sv_index]
        self._sv = X[sv_index]

        #for computing rho we need an x_i with corresponding a_i < 1/nu and a > 0
        rho_x = 0
        for a_i, x_i in zip(self._alpha, self._sv):
            if a_i > 0 and a_i < 1/self._nu:
                rho_x = x_i
                break
        #compute error assuming non zero rho
        self._rho = numpy.sum([a_i * self._kernel(x_i,rho_x) for a_i, x_i in zip(self._alpha,self._sv)])

    #compute Gram matrix
    def gram(self, X):
        ## pairwise_kernels:
        ## K(x, y) = exp(-gamma ||x-y||^2)
        return pairwise_kernels(X, None, "rbf", gamma=self._gamma)

    #compute Lagrangian multipliers
    # TODO: I'd rather this part directly goes in train()
    def alpha(self, X):
        n_samples, n_features = X.shape
        K = self.gram(X)

        P = cvxopt.matrix(K)
        q = cvxopt.matrix(numpy.zeros(n_samples))
        A = cvxopt.matrix(numpy.ones((n_samples,1)),(1,n_samples))
        b = cvxopt.matrix(1.0)

        G_1 = cvxopt.matrix(numpy.diag(numpy.ones(n_samples) * -1))
        h_1 = cvxopt.matrix(numpy.zeros(n_samples))

        G_2 = cvxopt.matrix(numpy.diag(numpy.ones(n_samples)))
        h_2 = cvxopt.matrix(numpy.ones(n_samples) * 1/(self._nu*len(X)))

        G = cvxopt.matrix(numpy.vstack((G_1, G_2)))
        h = cvxopt.matrix(numpy.vstack((h_1, h_2)))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        return numpy.ravel(solution['x'])

    #Returns SVM predicton given feature vector
    def predict(self, x):
        result = -1 * self._rho
        for w_i, x_i in zip(self._alpha, self._sv):
            result += w_i * self._kernel(x_i, x)
        return numpy.sign(result)

    # Returns distance to boundary
    # TODO: optimize, still slow with a greater number of grid points
    def decision_function(self, x):
        result = -1 * self._rho
        for w_i, x_i in zip(self._alpha, self._sv):
            result += w_i * self._kernel(x_i, x)
        return result


