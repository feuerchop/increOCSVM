__author__ = 'LT'
import numpy as np
import cvxopt.solvers
import kernel
from time import gmtime, strftime
from numpy.linalg import inv
from sklearn.metrics.pairwise import pairwise_kernels
import data

#print(__doc__)

#Trains an SVM
class OCSVM(object):
    # define global variables
    _rho = None
    _kernel = None
    _nu = None
    _gamma = None

    #Class constructor: kernel function & nu & sigma
    def __init__(self, metric, nu, gamma):
        if metric == "rbf":
            self._kernel = kernel.Kernel.gaussian(gamma)
        self._nu = nu
        self._gamma = gamma
        self._data = data.Data()

    #returns trained SVM predictor given features (X)
    #TODO: we need to store the key properties of model after training
    # Please check libsvm what they provide for output, e.g., I need to access to the sv_idx all the time
    def train(self, X):
        self._data.set_X(X)
        self._data.set_C(1/(self._nu*len(X)))
        # get lagrangian multiplier
        self._data.set_alpha(self.alpha(self._data.get_X()))
        # defines necessary parameter for prediction
        self.predictor(self._data.get_X(), self._data.get_alpha())

    #returns SVM predictors with given X and langrange mutlipliers
    def predictor(self, X, alpha):

        # define support vector and weights/alpha
        alpha = self._data.get_alpha_s()
        sv = self._data.get_Xs()

        #for computing rho we need an x_i with corresponding a_i < 1/nu and a > 0
        rho_x = 0
        for a_i, x_i in zip(alpha, sv):
            if a_i > 0 and a_i < 1/self._nu:
                rho_x = x_i
                break
        #compute error assuming non zero rho
        self._rho = np.sum([a_i * self._kernel(x_i,rho_x) for a_i, x_i in zip(alpha,sv)])

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
        q = cvxopt.matrix(np.zeros(n_samples))
        A = cvxopt.matrix(np.ones((n_samples,1)),(1,n_samples))
        b = cvxopt.matrix(1.0)

        G_1 = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_1 = cvxopt.matrix(np.zeros(n_samples))

        G_2 = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_2 = cvxopt.matrix(np.ones(n_samples) * 1/(self._nu*len(X)))

        G = cvxopt.matrix(np.vstack((G_1, G_2)))
        h = cvxopt.matrix(np.vstack((h_1, h_2)))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        return np.ravel(solution['x'])

    #Returns SVM predicton given feature vector
    def predict(self, x):
        result = -1 * self._rho

        return np.sign(result)

    # Returns distance to boundary
    #TODO: optimize, still slow with a greater number of grid points
    def decision_function(self, x):
        result = -1 * self._rho
        for w_i, x_i in zip(self._data.get_alpha_s(), self._data.get_Xs()):
            result += w_i * self._kernel(x_i, x)
        return result


    ### incremental

    def increment(self, x_c):
        epsilon = 0.001
         # calculate mu according to KKT-conditions
        mu = 1 - sum([w_i * self._kernel(x_i, self._data.get_Xs()[0])
                         for w_i, x_i
                         in zip(self._data.get_alpha(), self._data.get_X())])

        #calculate gradient of alpha (g_i)
        kernel_matrix = self.gram(self._data.get_X())

        grad_alpha = - kernel_matrix.diagonal().reshape(len(self._data.get_X()),1) \
                     + self.gram(self._data.get_X()).dot(np.transpose(self._data.get_alpha()))\
                                .reshape(len(self._data.get_X()),1)\
                     + mu * np.ones(len(self._data.get_alpha())).reshape(len(self._data.get_alpha()),1)

        # set alpha of x_c zero
        # calculate gradient of alpha_c
        alpha_c = 0
        grad_alpha_c = self.gram(x_c) + sum([a_i * self._kernel(x_i, x_c)
                        for a_i, x_i in zip(self._data.get_alpha(), self._data.get_X())]) \
                        + mu

        #while grad_alpha_c[0] < 0 and alpha_c[0] < self._data.get_C():
        # just to test the loop
        while True:

            #TODO: optimize Q because inverse is computationally extensive
            Q = - inv(np.concatenate(
                    (np.concatenate(
                        (np.ones(len(self._data.get_Xs())).reshape(len(self._data.get_Xs()),1),
                        self.gram(self._data.get_Xs())), axis=1
                    ),
                    np.concatenate(
                        ([[0]], np.ones(len(self._data.get_alpha_s())).
                                    reshape(1,len(self._data.get_Xs()))), axis=1)
                    ), axis=0))

            beta = Q.dot(
                    np.concatenate( (pairwise_kernels(self._data.get_Xs(), x_c), [[1]]), axis=0))
            print beta
            return True
            K_cs = pairwise_kernels(x_c, self._data.get_Xs())
            K_rs = pairwise_kernels(self._data.get_Xr(), self._data.get_Xs())
            gamma = np.concatenate(np.ones(len(self._data.get_Xr()) + 1),
                                   np.concatenate(K_cs, K_rs, axis=0),
                                   axis=1) * beta \
                    + np.concatenate(pairwise_kernels(x_c,x_c), pairwise_kernels(x_c, self._data.get_Xr()), axis=0)

            # accounting
            I_Splus = beta > epsilon
            I_Sminus = beta < - epsilon
            alpha_s = self._data.get_alpha_s()
            abs_Xs = np.absolute(self._data.get_Xs())
            grad_alpha_I = np.zeros(len(beta))
            grad_alpha_I[I_Splus] = self._data.get_C() * np.ones(len(I_Splus)) - alpha_s[I_Splus]
            grad_alpha_I[I_Sminus] = self._data.get_C() * np.ones(len(I_Sminus)) - alpha_s[I_Sminus]
            alpha_beta = np.divide(grad_alpha_I, beta)
            grad_alpha_c_S = np.absolute(alpha_beta).min() * cmp(np.absolute(alpha_beta).min(),0)
            grad_alpha_c = 0

            break







