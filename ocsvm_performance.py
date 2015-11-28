__author__ = 'LT'
import numpy as np
import cvxopt.solvers
import kernel
from time import gmtime, strftime
from sklearn.metrics.pairwise import pairwise_kernels
from numpy import vstack, hstack, ones, zeros, absolute, where, divide, inf, \
    delete, outer, transpose, diag, tile, arange, concatenate, empty, unique, round, amin

from numpy.linalg import inv, eig, solve
import data
#from profilehooks import profile

from scipy import linalg
import sys
import math
#print(__doc__)

#Trains an SVM
class OCSVM(object):
    # define global variables
    _rho = None
    _v = None
    _gamma = None
    _a_history = False

    #Class constructor: kernel function & nu & sigma
    def __init__(self, metric, nu, gamma):
        self._v = nu
        self._gamma = gamma
        self._data = data.Data()

    #returns trained SVM rho given features (X)
    # Please check libsvm what they provide for output, e.g., I need to access to the sv_idx all the time
    def fit(self, X, scale=1, v_target=None, rho=True):
        self._data.set_X(X)
        self._data.set_C(1/(self._v * len(X)) * scale)
        # get lagrangian multiplier
        alpha = self.alpha(X, scale, v_target)
        self._data.set_alpha(alpha)
        # defines necessary parameter for prediction
        if rho: self.rho()

    #returns SVM prediction with given X and langrange mutlipliers
    def rho(self):
        # compute rho assuming non zero rho, take average rho!
        Xs = self._data.Xs()
        if self._data.K_X() != None:
            inds = self._data.get_sv()
            K_X_Xs = self._data.K_X()[:,inds]
        else:
            K_X_Xs = self.gram(self._data.X(), self._data.Xs())
        rho_all = self._data.alpha().dot(K_X_Xs)

        self._rho = np.mean(rho_all)

    #compute Gram matrix
    def gram(self, X, Y=None):
        return pairwise_kernels(X, Y, "rbf", gamma=self._gamma)

    #compute Lagrangian multipliers
    def alpha(self, X, scale = 1, v_target = None):
        n_samples, n_features = X.shape
        K = 2 * self.gram(X)

        P = cvxopt.matrix(K)
        q = cvxopt.matrix(zeros(n_samples))
        A = cvxopt.matrix(ones((n_samples, 1)), (1, n_samples))
        if v_target == None:

            b = cvxopt.matrix(self._v * n_samples)
        else:
            b = cvxopt.matrix(v_target * n_samples)

        G_1 = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_1 = cvxopt.matrix(np.zeros(n_samples))

        G_2 = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_2 = cvxopt.matrix(np.ones(n_samples) * 1/(self._v*len(X)) * scale)
        G = cvxopt.matrix(np.vstack((G_1, G_2)))
        h = cvxopt.matrix(np.vstack((h_1, h_2)))

        cvxopt.solvers.options['show_progress'] = False

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        return np.ravel(solution['x'])

    # Returns SVM predicton given feature vector
    def predict(self, x):
        result = np.sign(self.decision_function(x))
        result[result == 0] = 1
        return result

    # Returns distance to boundary
    def decision_function(self, x):
        return - self._rho + self._data.alpha().dot(self.gram(self._data.X(), x))
    #@profile
    def increment(self, Xc, init_ac=0, break_count=-1):
        # epsilon
        e = self._data._e
        mu = 0

        # initialize existing X, coefficients a, C
        X_origin = self._data.X()
        K_X_origin = self._data.K_X()
        n_data = X_origin.shape[0]
        n_feature = X_origin.shape[1]

        C = self._data.C()
        a_origin = self._data.alpha()

        # number of new incremental points
        n_new = Xc.shape[0]

        # number of all (new and existing) points
        n_all = n_data + n_new

        # concatenate all new points with all existing points
        X = empty((n_new + n_data, n_feature))
        X[0:n_new, :] = Xc
        X[n_new:, :] = X_origin

        # create kernel matrix for all new and existing points

        # create of all data points
        if K_X_origin == None:
            K_X = self.gram(X)
        else:
            K_X = empty((n_all, n_all))
            K_X[n_new:, n_new:] = K_X_origin
            K_X_new = self.gram(Xc, X_origin)
            K_X[0:n_new, :] = K_X_new
            K_X[:, 0:n_new] = K_X_new.T

        # creating coefficient vector alpha for all data points
        a = empty(n_all)
        a[n_new:] = a_origin
        a[:n_new] = init_ac

        # creating gradient vector
        g = zeros(n_all)

        # create sensitivity vector
        gamma = empty(n_all)
        check_gradient = False
        # loop through all new points to increment
        for x_count in range(n_new):

            #print "--------- START %s ---------" % x_count

            if x_count == break_count:
                self._data.set_X(X)
                self._data.set_alpha(a)
                self._data.set_C(C)
                self._data.set_K_X(K_X)
                self.rho()
                return False

            # initialize X, a, C, g, indices, kernel values
            start_origin = n_new - x_count
            start_new = start_origin - 1

            if x_count == 0:
                inds = []
                indr = []
                inde = []
                indo = []
                for i in range(n_new, n_all):
                    if e < a[i] < C - e:
                        inds.append(i)
                    else:
                        indr.append(i)
                        if a[i] <= e:
                            indo.append(i)
                        else:
                            inde.append(i)

                ls = len(inds)                               # support vectors length
                lr = len(indr)                               # error and non-support vectors length
                le = len(inde)                               # error vectors lenght
                lo = len(indo)
                #mu_old = mu
                mu = - K_X[inds[0], :][start_origin:].dot(a[start_origin:])
                if lr > 0:
                    g[indr] = K_X[indr, :][:, start_origin:].dot(a[start_origin:]) + mu
                # calculate mu according to KKT-conditions


            c_inds = [start_new] + inds

            # kernel of support vectors
            Kss = K_X[:, inds][inds, :]
            #print "difference indo: %s" % unique(round(K_X[indo, :][:, start_origin:].dot(a[start_origin:]) + mu - g[indo],6))
            #check_gradient = True
            #if check_gradient:
                #g[indr] = K_X[indr, :][:, start_origin:].dot(a[start_origin:]) + mu
                #g[indo] += K_X[indo[0], :][start_origin:].dot(a[start_origin:]) + mu - g[indo[0]]
                #check_gradient = False
                #print "difference indo: %s" % unique(round(K_X[indo, :][:, start_origin:].dot(a[start_origin:]) + mu - g[indo],6))
            if ls > 0:
                gc = K_X[start_new, start_origin:].dot(a[start_origin:]) + mu

            ac = a[start_new]

            if x_count == 0:
                Q = ones((ls+1, ls+1))
                Q[0, 0] = 0
                Q[1:, 1:] = Kss
                try:
                    R = inv(Q)
                except np.linalg.linalg.LinAlgError:
                    x = 1e-11
                    found = False
                    print "singular matrix"
                    while not found:
                        try:
                            R = inv(Q + diag(ones(ls+1) * x))
                            found = True
                        except np.linalg.linalg.LinAlgError:
                            x = x*10
            loop_count = 1
            #print "gc: %s, ac: %s" % (gc, ac)
            while gc < e and ac < C - e:
                if ls == 0: check_gradient = True
                #print "-------------------- incremental %s-%s ---------" % (x_count, loop_count)
                #print "ac: %s" % ac
                #print "len inds: %s" % len(inds)
                if ls > 0:
                    n = K_X[start_new, :][c_inds]
                    #print R.shape
                    #print n.shape

                    beta = - R.dot(n)
                    betas = beta[1:]

                # calculate gamma
                if lr > 0 and ls > 0:
                    gamma_tmp = K_X[:, c_inds][start_new:]
                    gamma_tmp[:, 0] = 1
                    gamma[start_new:] = gamma_tmp.dot(beta) + K_X[start_new, :][start_new:]
                    gammac = gamma[start_new]

                elif ls > 0:
                    # empty R set
                    gammac = K_X[start_new, :][c_inds].dot(beta) + 1

                else:
                    # empty S set
                    gammac = 1
                    gamma[indr] = 1
                    #gamma[indo] = -1

                # accounting
                #case 1: Some alpha_i in S reaches a bound
                if ls > 0:
                    IS_plus = betas > e
                    IS_minus = betas < - e
                    gsmax = ones(ls)*inf
                    #if np.isnan(np.min(gsmax)):
                    #    gsmax = ones(ls)*inf
                    gsmax[IS_plus] = -a[inds][IS_plus]
                    gsmax[IS_plus] += C
                    gsmax[IS_minus] = - a[inds][IS_minus]
                    gsmax = divide(gsmax, betas)
                    gsmin = min(absolute(gsmax))
                    #print where(absolute(gsmax) == gsmin)
                    ismin = where(absolute(gsmax) == gsmin)[0][0]

                else: gsmin = inf

                #case 2: Some g_i in E reaches zero
                if le > 0:

                    gamma_inde = gamma[inde]
                    g_inde = g[inde]
                    Ie_plus = gamma_inde > e

                    if len(g_inde[Ie_plus]) > 0:
                        gec = divide(-g_inde[Ie_plus], gamma_inde[Ie_plus])
                        gec[gec <= 0] = inf
                        gemin = min(gec)
                        if gemin < inf:
                            iemin = where(gec == gemin)[0][0]
                    else: gemin = inf
                else: gemin = inf
                #case 2: Some g_i in O reaches zero
                if lo > 0 and ls > 0:
                    gamma_indo = gamma[indo]
                    g_indo = g[indo]
                    Io_minus = gamma_indo < - e
                    if len(g_indo[Io_minus]) > 0:
                        goc = divide(-g_indo[Io_minus], gamma_indo[Io_minus])
                        goc[goc <= 0] = inf
                        goc[g_indo[Io_minus] < 0] = inf
                        gomin = min(goc)
                        if gomin < inf:
                            iomin = where(goc == gomin)[0][0]
                    else: gomin = inf
                else: gomin = inf

                # case 3: gc becomes zero
                if gammac > e: gcmin = - gc/gammac
                else: gcmin = inf

                # case 4
                if ls > 0: gacmin = C - ac
                else: gacmin = inf

                # determine minimum largest increment
                all_deltas = [gsmin, gemin, gomin, gcmin, gacmin]
                gmin = min(all_deltas)
                imin = where(all_deltas == gmin)[0][0]
                # update a, g
                if ls > 0:
                    mu += beta[0]*gmin
                    ac += gmin
                    a[inds] += betas*gmin
                else:
                    mu += gmin
                if lr > 0:
                    g[indr] += gamma[indr] * gmin
                gc += gammac * gmin
                if imin == 0: # min = gsmin => move k from s to r
                    # if there are more than 1 minimum, just take 1
                    ak = a[inds][ismin]

                    # delete the elements from X,a and g
                    # => add it to the end of X,a,g
                    ind_del = inds[ismin]
                    inds.remove(ind_del)
                    c_inds = [start_new] + inds
                    indr.append(ind_del)
                    if ak < e:
                        indo.append(ind_del)
                        lo += 1
                    else:
                        inde.append(ind_del)
                        le += 1

                    lr += 1
                    #decrement R, delete row ismin and column ismin

                    if ls > 2:
                        ismin += 1
                        R_new = zeros((ls,ls))
                        R_new[0:ismin, 0:ismin] = R[0:ismin, 0:ismin]
                        R_new[ismin:, 0:ismin] = R[ismin+1:,0:ismin]
                        R_new[0:ismin, ismin:] = R[0:ismin, ismin+1:]
                        R_new[ismin:, ismin:] = R[ismin+1:, ismin+1:]
                        betak = zeros(ls)
                        betak[:ismin] = R[ismin, :ismin]
                        betak[ismin:] = R[ismin, ismin+1:]
                        R_new -= outer(betak, betak)/R[ismin,ismin]
                        R = R_new
                    elif ls == 2:
                        R = ones((2, 2))
                        R[1,1] = 0
                        R[0,0] = -1
                    else:
                        R = inf
                    ls -= 1

                elif imin == 1:
                    # delete the elements from X,a and g => add it to the end of X,a,g
                    ### old version find index to delete
                    #Ieplus_l = [i for i,b in enumerate(Ie_plus) if b]
                    #ind_del = inde[Ieplus_l[iemin]]
                    ### old version find index to delete
                    ind_del = np.asarray(inde)[Ie_plus][iemin]
                    if ls > 0:
                        nk = K_X[ind_del, :][[ind_del] + inds]
                        betak = - R.dot(nk)
                        betak1 = ones(ls + 2)
                        betak1[:-1] = betak
                        R_old = R
                        R = zeros((ls +2, ls +2))
                        R[:-1, :-1] = R_old
                        R += 1/(1 - nk.dot(R_old).dot(nk)) * outer(betak1, betak1)
                    else:
                        R = ones((2, 2))
                        R[1,1] = 0
                        R[0,0] = -1
                    inds.append(ind_del)
                    c_inds = [start_new] + inds
                    indr.remove(ind_del)
                    inde.remove(ind_del)
                    ls += 1
                    lr -= 1
                    le -= 1

                elif imin == 2: # min = gemin | gomin => move k from r to s

                    # delete the elements from X,a and g => add it to the end of X,a,g

                    ### old version find index to delete
                    #Io_minus_l = [i for i,b in enumerate(Io_minus) if b]
                    #ind_del = indo[Io_minus_l[iomin]]
                    ### old version find index to delete
                    ind_del = np.asarray(indo)[Io_minus][iomin]
                    if ls > 0:
                        nk = ones(ls+1)
                        nk[1:] = K_X[ind_del,:][inds]
                        betak = - R.dot(nk)
                        k = 1 - nk.dot(R).dot(nk)
                        betak1 = ones(ls+2)
                        betak1[:-1] = betak
                        R_old = R
                        R = zeros((ls+2, ls+2))
                        R[:-1,:-1] = R_old
                        R += 1/k * outer(betak1, betak1)
                    else:
                        R = ones((2, 2))
                        R[1,1] = 0
                        R[0,0] = -1

                    indo.remove(ind_del)
                    indr.remove(ind_del)
                    inds.append(ind_del)
                    c_inds = [start_new] + inds
                    lo -= 1
                    lr -= 1
                    ls += 1
                elif imin == 3:
                    if ls > 0:
                        nk = ones(ls+1)
                        nk[1:] = K_X[start_new, :][inds]

                        betak = - R.dot(nk)
                        k = 1 - nk.dot(R).dot(nk)
                        betak1 = ones(ls + 2)
                        betak1[:-1] = betak
                        R_old = R
                        R = zeros((ls +2, ls +2))
                        R[:-1,:-1] = R_old
                        R += 1/k * outer(betak1, betak1)
                    else:
                        R = ones((2, 2))
                        R[1,1] = 0
                        R[0,0] = -1
                    break
                else:
                    break
                loop_count += 1

            a[start_new] = ac
            g[start_new] = gc
            if ac < e:
                indr.append(start_new)
                indo.append(start_new)
                lr += 1
                lo += 1
            elif ac > C - e:
                indr.append(start_new)
                inde.append(start_new)
                lr += 1
                le += 1
            else:
                inds.append(start_new)
                if R.shape[0] != len(inds) + 1:
                    nk = ones(ls+1)
                    nk[1:] = K_X[start_new, :][inds[:-1]]
                    betak = - R.dot(nk)
                    k = 1 - nk.dot(R).dot(nk)
                    betak1 = ones(ls + 2)
                    betak1[:-1] = betak
                    R_old = R
                    R = zeros((ls +2, ls +2))
                    R[:-1,:-1] = R_old
                    R += 1/k * outer(betak1, betak1)

                ls += 1

         # update X, a
        self._data.set_X(X)
        self._data.set_alpha(a)
        self._data.set_C(C)
        self._data.set_K_X(K_X)
        self.rho()
    def perturb_c(self):
        pass

    def KKT(self, X=None, a=None):
        print "KKT Test---start"
        e = self._data._e
        C = self._data.C()
        if X == None and a == None:

            # initialize X, a, C, g, indeces, kernel values
            a = self._data.alpha()
            #print "a saved: %s" % a
            #print "KKT test a_c: %s" % a[0]
            #print "KKT test x_c: %s" % self._data.X()[0]


            if self._data.K_X() != None:
                K_X = self._data.K_X()
            else:
                X = self._data.X()                                  # data points
                K_X = self.gram(X)
        else:
            a = a
            X = X
            K_X = self.gram(X)
        print "sum a: %s" % sum(a)
        inds = [i for i, bool in enumerate(np.all([a > e, a < C - e], axis=0)) if bool]
        print "inds: %s (%s), a[inds]: %s" % (inds, len(inds), a[inds])
        #print "inds: %s" % inds
        indr = [i for i, bool in enumerate(np.any([a <= e, a >= C - e], axis=0)) if bool]
        #print "indr: %s" % indr
        inde = [i for i, bool in enumerate(a >= C - e) if bool]
        print "inde: %s (%s)" % (inde, len(inde))
        indo = [i for i, bool in enumerate(a <= e) if bool]
        print "indo: %s (%s)" % (indo, len(indo))
        l = len(a)
        ls = len(a[inds])                               # support vectors length
        lr = len(a[indr])                               # error and non-support vectors length
        mu_all = - K_X[inds,:].dot(a)
        print "mu_all: %s" % mu_all
        #print "K_X[inds,:]: %s" % K_X[inds,:]
        #print "a[inds]: %s" % a[inds]
        #print "mu_all: %s" % mu_all
        mu = max(mu_all)
        print "mu: %s" % mu
        g = ones(l) * -1

        g[inds] = zeros(ls)

        if lr > 0:
            Kr = K_X[indr, :]
            g[indr] = Kr.dot(a) + ones((lr,1)) * mu

        if ls > 0:
            Ks = K_X[inds, :]
            g[inds] = Ks.dot(a) + ones((ls,1)) * mu
            print "g[inds]: %s" % g[inds]

        #print "test b: %s" % (K_X[inds,:][:,indr].dot(a[indr]))
        #print "kernel kkt: %s " % K_X[inds,:][:,indr]
        #print "a[0]: %s" % a[0]
        #print "g[0]: %s" % g[0]
        #print "g[inds]: %s" % g[inds]
        KKT = True
        #print "a[54]: %s" % a[54]
        #print "g[55]: %s" % g[55]
        if len(g[inde][g[inde] > 0]) > 0:
            print "g[inde]: %s" % g[inde]
            print g[inde][g[inde] > 0]
            ind = where(g == g[inde][g[inde] > 0])[0]
            print where(g == g[inde][g[inde] > 0])
            print "index (g[inde] > 0): %s" % ind
            print "inde: g[index]: %s, a[index]: %s" % (g[ind], a[ind])
            #print "g[index-1]: %s, a[index-1]: %s" % (g[ind-1], a[ind-1])
            #print "error wrong!"
            #print "g[inde]: %s" %g[inde]
            #print "a[inde]: %s" %a[inde]
            KKT = False
        #print "indo: %s" % indo
        if len(g[indo][g[indo] < 0]) > 0:
            #print "non-support wrong!"
            #print "g[indo]: %s" %g[indo]
            ind = where(g == g[indo][g[indo] < 0])[0]
            print "index (g[indo] > 0): %s" % ind
            print "indo: g[index]: %s, a[index]: %s" % (g[ind], a[ind])
            #print "a[indo]: %s" %a[indo]
            KKT = False
        if not KKT:
            print "KKT not satisfied"
        print "KKT Test---end"
        return KKT