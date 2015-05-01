__author__ = 'LT'
import numpy
import cvxopt.solvers
#print(__doc__)

#Trains an SVM
class OCSVM(object):

    #Class constructor: kernel function & nu & sigma
    def __init__(self, kernel, nu, c):
        self._kernel = kernel
        self._nu = nu
        self._c = c

    #returns trained SVM predictor given features (X)
    # TODO: we need to store the key properties of model after training
    # Please check libsvm what they provide for output, e.g., I need to access to the sv_idx all the time
    def train(self, X):
        lagrange_multipliers = self.lagrangian_multipliers(X)
        return self.predictor(X, lagrange_multipliers)

    #returns SVM prediction with given X and langrange mutlipliers
    def predictor(self, X, lagrange_multipliers):
        sv_index = lagrange_multipliers > 1e-5
        sv_mult = lagrange_multipliers[sv_index]
        sv = X[sv_index]

        #for computing rho we need an x_i with corresponding a_i < 1/nu and a > 0
        rho_x = 0
        for a_i, x_i in zip(sv_mult,sv):
            if a_i > 0 and a_i < 1/self._nu:
                rho_x = x_i
                break

        #compute error assuming non zero rho
        rho = numpy.sum([a_i * self._kernel(x_i,rho_x) for a_i, x_i in zip(sv_mult,sv)])
        return OCSVMPrediction(self._kernel, rho, sv_mult,sv)

    #compute Gram matrix
    # TODO: you can use built-in kernel functions in sklearn, i.e., pairwise_kernels
    # As this might be more efficient than a double-for-loop
    def gram(self, X):
        n_samples, n_features = X.shape
        K = numpy.zeros((n_samples, n_samples))
        for i in range(0, n_samples):
            for j in range(0, n_samples):
                K[i, j] = self._kernel(X[i], X[j])
        return K

    #compute Lagrangian multipliers
    # TODO: I'd rather this part directly goes in train()
    # TODO: using alpha=xxx instead of lagrangian_multipliers
    def lagrangian_multipliers(self, X):
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

#SVM prediction

# TODO: why we need two classes here?
class OCSVMPrediction(object):
    #Class constructor
    def __init__(self, kernel, rho, weights, sv):
        self._kernel = kernel
        self._rho = rho
        self._weights = weights
        self._sv = sv

    #Returns SVM predicton given feature vector
    def predict(self, x):

        result = -1 * self._rho
        for w_i, x_i in zip(self._weights, self._sv):
            result += w_i * self._kernel(x_i, x)
        #print "------- predict " + str(x) + " with w*k(x_i,x) = " + str(result) + " => " + str(numpy.sign(result))
        return numpy.sign(result)

    def decision_function(self, x):
        result = -1 * self._rho
        for w_i, x_i in zip(self._weights, self._sv):
            result += w_i * self._kernel(x_i, x)
        return result
