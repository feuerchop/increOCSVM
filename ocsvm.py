__author__ = 'LT'
import numpy as np
import cvxopt.solvers
import kernel
from time import gmtime, strftime
from numpy.linalg import inv
from numpy import concatenate as cat
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
    def decision_function(self, x):
        return -1 * self._rho + self._data.get_alpha_s().dot(self.gram(self._data.get_Xs(),x))

    ### incremental

    def increment(self, x_c):
        grad_alpha_c_max = None
        print self._data.get_alpha_s()
        epsilon = 0.001
         # calculate mu according to KKT-conditions
        #
        mu = 1 - self.gram(self._data.get_Xs()[0], self._data.get_Xs()).dot(self._data.get_alpha_s())

        # set alpha of x_c zero
        # calculate gradient of alpha_c
        alpha_c = 0
        grad_alpha_c = - self.gram(x_c) + self.gram(x_c, self._data.get_Xs()).dot(self._data.get_alpha_s()) + mu

        #
        # just to test the loop
        len_s = len(self._data.get_alpha_s())
        Q = - inv(cat((cat(([[0]], np.ones((1,len_s))), axis=1),cat((np.ones((len_s,1)),self.gram(self._data.get_Xs())), axis=1
                        )),axis=0))
        loop_count = 0
        while grad_alpha_c < 0 and alpha_c < self._data.get_C():
            print loop_count
            # calculate beta

            beta = Q.dot(cat(([1], self.gram(x_c,self._data.get_Xs())[0]), axis=0))

            # calculate gamma
            K_cs = self.gram(x_c, self._data.get_Xs())
            K_rs = self.gram(self._data.get_Xr(), self._data.get_Xs())


            gamma = cat(
                        (cat(([[1]], K_cs), axis=1),
                         cat(
                             (np.ones((len(self._data.get_alpha_r()),1)),
                              K_rs), axis = 1)),
                        axis=0).dot(beta) \
                    + cat((self.gram(x_c),
                                      self.gram(x_c,self._data.get_Xr())),axis=1)[0]

        # accounting

            #case 1: Some alpha_i in S reaches a bound
            I_Splus = beta[1:] > epsilon
            I_Splus_ind = [c for c,i in enumerate(I_Splus) if i]
            I_Sminus = beta[1:] < - epsilon
            I_Sminus_ind = [c for c,i in enumerate(I_Sminus) if i]

            grad_alpha_I_Splus = np.divide(- self._data.get_alpha_s()[I_Splus], beta[1:][I_Splus])
            grad_alpha_I_Sminus = np.divide(- self._data.get_alpha_s()[I_Sminus] \
                                  + np.ones(len(self._data.get_alpha_s()[I_Sminus])) * self._data.get_C(),\
                                    beta[1:][I_Sminus])

            I_S_ind = I_Splus_ind + I_Sminus_ind
            # possible min S weight update
            alpha_beta = cat((grad_alpha_I_Splus,grad_alpha_I_Sminus))
            abs_min = np.absolute(alpha_beta).min()
            grad_alpha_c_S = abs_min * cmp(alpha_beta[np.where(np.absolute(alpha_beta) == abs_min)],0)

            #case 2: Some g_i in R reaches zero

            alpha_r = self._data.get_alpha_r()

            I_Rplus = np.all([gamma[1:] > epsilon, self._data.get_alpha_r() == self._data.get_C()],axis=0)
            print "I_Rplus: "+ str(I_Rplus)
            I_Rplus_ind = [c for c,i in enumerate(I_Rplus) if i]
            I_Rminus = np.all([gamma[1:] < - epsilon, self._data.get_alpha_r() <= 1e-5],axis=0)
            print "I_Rminus: " + str(I_Rminus)
            I_Rminus_ind = [c for c,i in enumerate(I_Rminus) if i]

            #calculate gradient of alpha (g_r)
            grad_alpha_r = - self.gram(self._data.get_Xr()).diagonal()  \
              + self.gram(self._data.get_Xr(),self._data.get_Xs()).dot(self._data.get_alpha_s()) \
              + mu * np.ones(len(self._data.get_alpha_r()))
            #- self.gram(x_c) + self.gram(x_c, self._data.get_Xs()).dot(self._data.get_alpha_s()) + mu
            largest_increase_r = np.divide(- grad_alpha_r, gamma[1:])
            grad_alpha_I_Rplus = largest_increase_r[I_Rplus]
            grad_alpha_I_Rminus = largest_increase_r[I_Rminus]
            I_R_ind = I_Rplus_ind + I_Rminus_ind
            if len(cat((grad_alpha_I_Rplus, grad_alpha_I_Rminus))) > 0:
                grad_alpha_c_R = cat((grad_alpha_I_Rplus, grad_alpha_I_Rminus)).min()
            else: grad_alpha_c_R = None
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

            print "old alpha_s: " + str(self._data.get_alpha_s())
            print "new alpha_s: " + str(self._data.get_alpha_s() + beta[1:]*grad_alpha_c_max)

            if grad_alpha_c_max == grad_alpha_c_R:
                #TODO: How to update alpha_R???????
                print "move from R to S => increment Q"
                R = -1 * Q
                R = cat((cat((R,np.zeros((1,R.shape[1]))),axis=0),np.zeros((R.shape[0]+1,1))),axis=1) + (1/gamma[0]) * cat((beta, [1]),axis=0)* np.array([cat((beta, [1]),axis=0)]).T
                Q = -1 * R

            else:
                print "move from S to R => decrement Q"
                grad_alpha_c_S_ind = I_S_ind[np.where(alpha_beta == grad_alpha_c_S)[0]] + 1
                R = -1 * Q
                for i in range(R.shape[0]):
                    for j in range(R.shape[1]):
                        if i != grad_alpha_c_S_ind and j != grad_alpha_c_S_ind:
                            R[i][j] = R[i][j] - R[i][grad_alpha_c_S_ind] * R[grad_alpha_c_S_ind][j] / R[grad_alpha_c_S_ind][grad_alpha_c_S_ind]
                R = np.delete(R, grad_alpha_c_S_ind,0)
                R = np.delete(R, grad_alpha_c_S_ind,1)
                Q = -1 * R


            # update alpha after Q update (is better)
            alpha_c += grad_alpha_c_max
            grad_alpha_c = gamma[0] * grad_alpha_c_max
            grad_alpha_r = gamma[1:] * grad_alpha_c_max


            self._data.update_alpha_s(self._data.get_alpha_s() + beta[1:]*grad_alpha_c_max)
            if grad_alpha_c_max == grad_alpha_c_g:
                print "grad_alpha_c_max from c"
                break
            loop_count += 1

        print grad_alpha_c_max
        if grad_alpha_c <= 1e-5:
            self._data.add(x_c, alpha_c)


