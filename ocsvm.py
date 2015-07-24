__author__ = 'LT'
import numpy as np
import cvxopt.solvers
import kernel
from time import gmtime, strftime
from numpy.linalg import inv
from numpy import concatenate as cat
from sklearn.metrics.pairwise import pairwise_kernels
from numpy import vstack, hstack, ones, zeros, absolute, where, divide, inf, delete, outer
import data
import sys

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
        self._data.set_alpha(self.alpha(self._data.X()))
        # defines necessary parameter for prediction
        self.predictor()

    #returns SVM predictors with given X and langrange mutlipliers
    def predictor(self):

        # define support vector and weights/alpha
        alpha = self._data.alpha_s()
        sv = self._data.Xs()

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
        q = cvxopt.matrix(zeros(n_samples))
        A = cvxopt.matrix(ones((n_samples,1)),(1,n_samples))
        b = cvxopt.matrix(1.0)

        G_1 = cvxopt.matrix(np.diag(ones(n_samples) * -1))
        h_1 = cvxopt.matrix(zeros(n_samples))

        G_2 = cvxopt.matrix(np.diag(ones(n_samples)))
        h_2 = cvxopt.matrix(ones(n_samples) * 1/(self._nu*len(X)))

        G = cvxopt.matrix(vstack((G_1, G_2)))
        h = cvxopt.matrix(vstack((h_1, h_2)))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        return np.ravel(solution['x'])

    # Returns distance to boundary
    def decision_function(self, x):
        #print self._rho
        return -1 * self._rho + self._data.alpha_s().dot(self.gram(self._data.Xs(),x))

    ### incremental

    def increment(self, xc, xo=None):
        e = 1e-5

        # initialize X, a, C, g, indeces, kernel values

        X = self._data.X()          # data points
        a = self._data.alpha()      # alpha
        ac = 0                      # alpha of new point c
        C = 1/(self._nu*len(X))

        inds = np.all([a > e, a < C - e], axis=0)           # support vectors indeces
        indr = np.any([a <= e, a >= C - e], axis=0)         # error and non-support vectors indeces
        inde = a[indr] >= C - e                             # error vectors indeces in R
        indo = a[indr] <= e                                 # non-support vectors indeces in R

        l = len(a)
        ls = len(a[inds])                               # support vectors length
        lr = len(a[indr])                               # error and non-support vectors length
        le = len(a[inde])                               # error vectors lenght
        lo = len(a[indo])                               # non-support vectors

        Kss = self.gram(X[inds]) # kernel of support vectors
        Krs = self.gram(X[indr], X[inds]) # kernel of error vectors, support vectors
        Kcs = self.gram(xc, X[inds])[0]
        Kcr = self.gram(xc, X[indr])[0]
        Kcc = 1

        # calculate mu according to KKT-conditions
        mu = 1 - self.gram(X[inds][0], X[inds]).dot(a[inds])
        # calculate gradient
        gc = - 1 + Kcs.dot(a[inds]) + mu
        g = ones(l)
        g[inds] = zeros(ls)
        g[indr] = - ones(lr) + Krs.dot(a[inds]) + ones((lr,1)) * mu

        # initial calculation for beta
        Q = inv(vstack([hstack([0,ones(ls)]),hstack([ones((ls,1)), Kss])]))

        loop_count = 1
        while gc < e and ac < C - e:
            ac0 = ac
            print "--------------------------" + "increment/decrement loop " + str(loop_count) + "--------------------------"
            print "a[inds]: %s" %a[inds]
            print "a[indr]: %s" %a[indr]
            loop_count += 1
            # calculate beta
            n = hstack([1, Kcs])
            beta = - Q.dot(n)
            betas = beta[1:]

            # calculate gamma
            if lr > 0:
                gamma = vstack([hstack([1, Kcs]), hstack([ones((lr,1)), Krs])]).dot(beta) + hstack([Kcc, Kcr])
                gammac = gamma[0]
                gammar = gamma[1:]
            else:
                gammac =hstack([1, Kcs]).dot(beta) + Kcc

            # accounting
            #case 1: Some alpha_i in S reaches a bound
            if ls > 0:
                IS_plus = betas > e
                IS_minus = betas < - e
                IS_zero = np.any([betas <= e, betas >= -e], axis=0)

                gsmax = zeros(ls)
                gsmax[IS_zero] = ones(len(betas[IS_zero])) * inf
                gsmax[IS_plus] = ones(len(betas[IS_plus]))*C-a[inds][IS_plus]
                gsmax[IS_minus] = - a[inds][IS_minus]

                gsmax = divide(gsmax, betas)
                gsmin = absolute(gsmax).min()
                ismin = where(absolute(gsmax) == gsmin)
            else: gsmin = inf
            #case 2: Some g_i in R reaches zero
            if le > 0:
                Ie_plus = gammar[inde] > e
                Ie_inf = gammar[inde] <= e
                gec = zeros(le)
                gec[Ie_plus] = divide(-g[indr][inde][Ie_plus], gammar[inde][Ie_plus])
                gec[Ie_inf] = inf
                gemin = gec.min()
                iemin = where(gec == gemin)
            else: gemin = inf
            if lo > 0:
                Io_minus = gammar[indo] < - e
                Io_inf = gammar[indo] >= - e
                goc = zeros(lo)
                goc[Io_minus] = divide(-g[indr][indo][Io_minus], gammar[indo][Io_minus])
                goc[Io_inf] = inf
                gomin = goc.min()
                iomin = where(goc == gomin)

            else: gomin = inf

            # case 3: gc becomes zero
            if gammac > e: gcmin = - gc/gammac
            else: gcmin = inf
            # case 4
            gacmin = C - ac

            # determine minimum largest increment
            gmin = min([gsmin, gemin, gomin, gcmin, gacmin])
            imin = where([gsmin, gemin, gomin, gcmin, gacmin] == gmin)[0][0]

            # update a, g,
            ac += gmin
            a[inds] = a[inds] + betas*gmin
            g[indr] = g[indr] + gammar * gmin
            gc = gc + gammac * gmin

            if imin == 0: # min = gsmin => move k from s to r

                print "move k from s to r"
                #update indeces

                # get x, a and g
                Xk = X[inds][ismin]
                ak = a[inds][ismin]
                gk = g[inds][ismin]
                # delete the elements from X,a and g => add it to the end of X,a,g
                inds_ind = [c for c,val in enumerate(inds) if val]
                ind_del = inds_ind[ismin[0]]
                X = delete(X, ind_del, axis=0)
                a = delete(a, ind_del)
                g = delete(g, ind_del)
                X = vstack((X, Xk))
                a = hstack((a, ak))
                g = hstack((g, gk))

                # set indeces new
                indr = delete(indr, ind_del)
                indr = hstack((indr,True))

                inds = delete(inds, ind_del)
                inds = hstack((inds, False))

                if ak < e:
                    indo = hstack((indo, True))
                    inde = hstack((inde, False))
                else:
                    indo = hstack((indo, False))
                    inde = hstack((inde, True))

                #decrement Q, delete row ismin and column ismin
                ismin = ismin[0][0]
                for i in range(Q.shape[0]):
                    for j in range(Q.shape[1]):
                        if i != ismin and j != ismin:
                            Q[i][j] = Q[i][j] - Q[i][ismin]*Q[ismin][j]/Q[ismin][ismin]
                Q = delete(Q, ismin, 0)
                Q = delete(Q, ismin, 1)

            elif imin == 1:
                print "move k from e (r) to s"
                # get x, a and g
                Xk = X[indr][inde][iemin]
                ak = a[indr][inde][iemin]
                gk = g[indr][inde][iemin]
                gammak = gammar[iemin]

                # delete the elements from X,a and g => add it to the end of X,a,g
                indr_ind = [c for c,val in enumerate(indr) if val]
                ind_del = indr_ind[iemin[0]]
                X = delete(X, ind_del, axis=0)
                a = delete(a, ind_del)
                g = delete(g, ind_del)
                X = vstack((X, Xk))
                a = hstack((a, ak))
                g = hstack((g, gk))

                # set indeces new
                inds = delete(inds, ind_del)
                inds = hstack((inds,True))
                indr = delete(indr, ind_del)
                indr = hstack((indr, False))
                inde = delete(inde, iemin)
                indr = delete(indr, iemin)

                #increment Q
                Q = hstack((vstack((Q, zeros(Q.shape[1]))),zeros((Q.shape[0] + 1,1)))) \
                    + 1/gammak * outer(hstack((beta,1)), hstack((beta,1)))

            elif imin == 2: # min = gemin | gomin => move k from r to s
                print "move k from i (r) to s"
                Xk = X[indr][indo][iomin]
                ak = a[indr][indo][iomin]
                gk = g[indr][indo][iomin]
                gammak = gammar[iomin]

                # delete the elements from X,a and g => add it to the end of X,a,g
                indr_ind = [c for c,val in enumerate(indr) if val]
                ind_del = indr_ind[iomin[0]]
                X = delete(X, ind_del, axis=0)
                a = delete(a, ind_del)
                g = delete(g, ind_del)
                X = vstack((X, Xk))
                a = hstack((a, ak))
                g = hstack((g, gk))

                # set indeces new
                inds = delete(inds, ind_del)
                inds = hstack((inds,True))
                indr = delete(indr, ind_del)
                indr = hstack((indr, False))
                inde = delete(inde, iomin)
                indo = delete(indo, iomin)

                #increment Q
                Q = hstack((vstack((Q, zeros(Q.shape[1]))),zeros((Q.shape[0] + 1,1)))) \
                    + 1/gammak * outer(hstack((beta,1)), hstack((beta,1)))

            else: # k = c => terminate
                print "k = c => terminate"
                break


            #update length of sets
            ls = len(a[inds])                               # support vectors length
            lr = len(a[indr])                               # error and non-support vectors length
            le = len(a[inde])                               # error vectors lenght
            lo = len(a[indo])                               # non-support vectors
            #update kernel
            if ls > 0:
                Kss = self.gram(X[inds])
                Kcs = self.gram(xc, X[inds])[0]                   # kernel of support vectors
            if lr > 0 and ls > 0:
                Krs = self.gram(X[indr], X[inds])               # kernel of error vectors, support vectors
                Kcr = self.gram(xc, X[indr])[0]



            # update
            mu = 1 - self.gram(X[inds][0], X[inds]).dot(a[inds])

        self._data.set_X(X)
        self._data.set_alpha(a)
        self._data.add(xc, ac)
        self.predictor()
        print self._data.alpha_s()
        print sum(self._data.alpha_s())
        print self._data.Xs()