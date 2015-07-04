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
    def gram(self, X, Y=None):
        ## pairwise_kernels:
        ## K(x, y) = exp(-gamma ||x-y||^2)
        return pairwise_kernels(X, Y, "rbf", gamma=self._gamma)

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
            # calculate beta
            #TODO: optimize Q because inverse is computationally extensive
            Q = - inv(np.concatenate(
                    (np.concatenate(
                        (np.ones(len(self._data.get_Xs())).reshape(len(self._data.get_Xs()),1),
                        self.gram(self._data.get_Xs())), axis=1
                    ),
                    np.concatenate(
                        ([[0]], np.ones(len(self._data.get_alpha_s())).
                                    reshape(1,len(self._data.get_alpha_s()))), axis=1)
                    ), axis=0))

            beta = Q.dot(
                    np.concatenate( (self.gram(self._data.get_Xs(), x_c), [[1]]), axis=0))
            # calculate gamma
            K_cs = self.gram(x_c, self._data.get_Xs())


            if len(self._data.get_Xr()) > 0:
                K_rs = self.gram(self._data.get_Xr(), self._data.get_Xs())
                gamma = np.concatenate(
                            (np.concatenate(
                                ([[1]],K_cs),axis=1),
                             np.concatenate(
                                 (np.ones(len(self._data.get_alpha_r())).
                                    reshape(1,len(self._data.get_alpha_r())),
                                  K_rs),axis=1)),
                            axis=0).dot(beta) \
                        + np.concatenate((self.gram(x_c),
                                          self.gram(x_c, self._data.get_Xr())),axis=0)
            else:
                gamma = np.concatenate(([[1]],K_cs),axis=1).dot(beta) + self.gram(x_c)

            # accounting

            #case 1: Some alpha_i in S reaches a bound

            I_Splus = beta[1:] > epsilon

            # possible weight updates
            grad_alpha_I = - self._data.get_alpha_s().reshape(len(self._data.get_alpha_s()),1)
            grad_alpha_I[I_Splus] += self._data.get_C()
            alpha_beta = np.divide(grad_alpha_I, beta[1:])
            grad_alpha_c_S = np.absolute(alpha_beta).min() * cmp(np.absolute(alpha_beta).min(),0)

            #case 2: Some g_i in R reaches zero
            grad_alpha_c_R = 0
            #case 3: g_c becomes zero
            if gamma[0] > epsilon:
                grad_alpha_c_g = np.divide(-grad_alpha_c, gamma[0])
            else:
                grad_alpha_c_g = None
            #case 4
            grad_alpha_c_alpha = self._data.get_C() - alpha_c

            # get smallest gradient of alpha

            grad_alpha_c_max = min(filter(None, [grad_alpha_c_S, grad_alpha_c_R,
                                           grad_alpha_c_g, grad_alpha_c_alpha]))
            alpha_c += grad_alpha_c_max
            self._data.update_alpha_s(beta*grad_alpha_c_max)
            if not grad_alpha_c_S == grad_alpha_c_max and not grad_alpha_c_R == grad_alpha_c_max:
                break








