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

import sys

#Trains an SVM
class OCSVM(object):
    # define global variables
    _rho = None
    _v = None
    _gamma = None
    _a_history = False

    #Class constructor: kernel function & nu & sigma
    def __init__(self, metric, nu, gamma, e=None):
        self._v = nu
        self._gamma = gamma
        self._data = data.Data()
        if e is not None:
            self._data.set_e(e)

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
            K_X_Xs = self._data.K_X()[:, inds]
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

    def increment_supervised(self, Xc, labels, init_ac=0):
        #print "semi supervised"
        # epsilon
        e = self._data._e
        drop = 0

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

        # create gram matrix for all new and existing points

        # create of all data points
        if K_X_origin == None:
            K_X = self.gram(X)
        # create gram matrix for new points and add to existing ones
        else:
            K_X = empty((n_all, n_all))
            K_X[n_new:, n_new:] = K_X_origin
            K_X_new = self.gram(Xc, concatenate((Xc,X_origin), axis=0))
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

        if labels is None:
            labels = zeros(n_new)

        restart = False
        save_state = False
        # loop through all new points to increment

        for x_count in range(n_new):
            #print "--------- START %s ---------" % x_count

            #print "dropped: %s" % drop
            # initialize X, a, and kernel matrices

            start_origin = n_new - x_count
            start_new = start_origin - 1
            K_X_start_new = K_X[start_new:]
            K_X_start_origin = K_X[:, start_origin:]
            a_origin = a[start_origin:]
            label = labels[x_count]
            # initalize indices for bookkeeping

            if restart:
                restart = False
                ls = len(inds)                               # support vectors length
                lr = len(indr)                               # error and non-support vectors length
                le = len(inde)                               # error vectors lenght
                lo = len(indo)
                l = ls + lr
                # calculate mu according to KKT-conditions
                mu = - K_X_start_origin[inds[0]].dot(a_origin)

            if x_count == 0:
                r = range(start_origin, n_all)
                inds = [i for i in r if e < a[i] < C - e]
                indr = [i for i in r if i not in inds]
                inde = [i for i in indr if a[i] > e]
                indo = [i for i in indr if a[i] <= e]

                ls = len(inds)                               # support vectors length
                lr = len(indr)                               # error and non-support vectors length
                le = len(inde)                               # error vectors lenght
                lo = len(indo)
                l = ls + lr
                # calculate mu according to KKT-conditions
                mu = - K_X_start_origin[inds[0]].dot(a_origin)
                # calculate gradient of error and non-support vectors
                g[inds] = 0
                if lr > 0:
                    g[indr] = K_X_start_origin[indr].dot(a_origin) + mu
                Qs = ones((l+1, ls+1))
                Qs[:,1:] = K_X_start_new[:, inds]
            else:
                l = ls + lr
                if ls > 0:
                    Qs = concatenate(([K_X_start_new[0, [start_new] + inds]], Qs), axis=0)
                else:
                    Qs = concatenate(([1], Qs), axis=1)

            # calculate gradient of error and non-support vectors

            c_inds = [start_new] + inds
            # only calculate gradient if there are support vectors
            if ls > 0:
                gc = K_X_start_origin[start_new].dot(a_origin) + mu
            else:
                #print "Semisupervised Error: No support vectors to train!"
                return False
            ac = a[start_new]

            if x_count == 0:
                Q = ones((ls+1, ls+1))
                Q[0, 0] = 0
                inds_row = [[i] for i in inds]
                Q[1:, 1:] = K_X[inds_row, inds]
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
            loop_count = 0
            # supervised label
            if label != 0:
                gc = K_X_start_origin[start_new].dot(a_origin) + mu
                # normal data point
                if label == 1:
                    ac_new = 0
                    if gc >= e:
                        #print "drop 1"
                        # all good, nothing to do
                        # but saving the gradient of new data point
                        g[start_new] = gc
                        indr.append(start_new)
                        indo.append(start_new)
                        lr += 1
                        lo += 1
                        continue
                    else:
                        # continue with incremental
                        # learning, save previous state
                        # for unlearning
                        save_state = True
                # anomaly
                else:
                    ac_new = 1
                    if gc < e:
                        # continue with incremental
                        # learning, save previous state
                        # for unlearning
                        save_state = True
                    else:
                        # drop this data point
                        #print "drop 2"
                        drop += 1
                        n_all -= 1
                        X = delete(X, start_new, axis=0)
                        K_X = delete(K_X, start_new, axis=0)
                        K_X = delete(K_X, start_new, axis=1)
                        inds = [i - 1 for i in inds]
                        indr = [i - 1 for i in indr]
                        inde = [i - 1 for i in inde]
                        indo = [i - 1 for i in indo]
                        a = delete(a, start_new)
                        g=delete(g, start_new)
                        gamma=delete(gamma, start_new)
                        Qs = Qs[1:,:]
                        restart = True
                        continue
            # saving necessary variables to undo learning:
            if save_state:
                R_save = R
                a_save = a
                g_save = g
                Qs_save = Qs[1:,:]
                indices = [list(inds), list(indr), list(inde), list(indo)]
            # unsupervised label

            while gc < e and ac < C - e:


                loop_count += 1
                #print "-------------------- incremental %s-%s ---------" % (x_count, loop_count)

                #calculate beta
                if ls > 0:
                    beta = - R.dot(K_X_start_new[0,c_inds])
                    #print R
                    #print K_X_start_new[0,c_inds]
                    betas = beta[1:]

                # calculate gamma
                if lr > 0 and ls > 0:
                    # non-empty R and S set
                    gamma[start_new:] = Qs.dot(beta) + K_X_start_new[:, start_new]
                    gammac = gamma[start_new]
                    ggamma = divide(-g, gamma)
                elif ls > 0:
                    # empty R set
                    gammac = K_X_start_new[0, c_inds].dot(beta) + 1

                else:
                    # empty S set
                    gammac = 1
                    gamma[indr] = 1
                    ggamma = -g

                # accounting
                #case 1: Some alpha_i in S reaches a bound
                if ls > 0:
                    gsmax = - a[inds]
                    gsmax[betas > e] += C
                    #print gsmax
                    gsmax = divide(gsmax, betas)
                    #print betas
                    # only consider positive increment weights
                    gsmax[absolute(betas) <= e] = inf
                    gsmin = min(absolute(gsmax))
                    if gsmin != inf:
                        ismin = where(absolute(gsmax) == gsmin)[0][0]
                    #print "----"

                else: gsmin = inf
                #case 2: Some g_i in E reaches zero
                if le > 0:
                    # only consider positive margin sensitivity for points in E
                    gec = ggamma[inde]
                    # only consider positive increment weights
                    gec[gec <= e] = inf
                    gemin = min(gec)
                    if gemin < inf:
                        iemin = where(gec == gemin)[0][0]

                else: gemin = inf
                #case 2: Some g_i in O reaches zero
                if lo > 0 and ls > 0:
                    # only consider positive margin sensitivity for points in E
                    goc = ggamma[indo]
                    # only consider positive increment weights
                    goc[goc <= e] = inf
                    # find minimum and index of it
                    gomin = min(goc)
                    if gomin < inf:
                        iomin = where(goc == gomin)[0][0]
                else: gomin = inf

                # case 3: gc becomes zero => algorithm converges
                if gammac > e: gcmin = - gc/gammac
                else: gcmin = inf

                # case 4: ac becomes an error vector => algorithm converges
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
                    g += gamma * gmin
                gc += gammac * gmin
                if imin == 0: # min = gsmin => move k from s to r
                    ak = a[inds][ismin]
                    ind_del = inds[ismin]
                    #bookkeeping
                    inds.remove(ind_del)
                    indr.append(ind_del)
                    if ak < e:
                        indo.append(ind_del)
                        lo += 1
                    else:
                        inde.append(ind_del)
                        le += 1
                    lr += 1
                    c_inds = [start_new] + inds

                    ismin += 1
                    #decrement R, delete row ismin and column ismin
                    if ls > 2:
                        ### R update
                        R_new = zeros((ls,ls))
                        R_new[0:ismin, 0:ismin] = R[0:ismin, 0:ismin]
                        R_new[ismin:, 0:ismin] = R[ismin+1:,0:ismin]
                        R_new[0:ismin, ismin:] = R[0:ismin, ismin+1:]
                        R_new[ismin:, ismin:] = R[ismin+1:, ismin+1:]

                        betak = R[ismin,
                                [i for i in range(ls+1) if i != ismin]]
                        R_new -= outer(betak, betak)/R[ismin,ismin]
                        R = R_new

                        # update Qs for gamma
                        Qs_new = ones((l+1, ls))
                        Qs_new[:, :ismin] = Qs[:,:ismin]
                        Qs_new[:, ismin:] = Qs[:,ismin+1:]
                        Qs = Qs_new

                    elif ls == 2:
                        ### R update
                        R = ones((2, 2))
                        R[1,1] = 0
                        R[0,0] = -1
                        # update Qs for gamma
                        Qs_new = ones((l+1, ls))
                        Qs_new[:, :ismin] = Qs[:,:ismin]
                        Qs_new[:, ismin:] = Qs[:,ismin+1:]
                        Qs = Qs_new

                    else:
                        Qs = ones(l+1)
                        R = inf
                    ls -= 1

                elif imin == 1:
                    ind_del = inde[iemin]
                    if ls > 0:

                        nk = K_X[[ind_del], [ind_del] + inds]
                        betak = - R.dot(nk)
                        k = 1 - nk.dot(R).dot(nk)
                        if k == 0:
                            k = 0.001
                        betak1 = ones(ls + 2)
                        betak1[:-1] = betak
                        R_old = R
                        R = 1/k * outer(betak1, betak1)
                        R[:-1,:-1] += R_old
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

                    if ls > 1:
                        Qs_new = ones((l+1, ls+1))
                        Qs_new[:, :-1] = Qs
                        Qs_new[:, ls] = K_X_start_new[:, ind_del]
                        Qs = Qs_new
                    else:
                        Qs_new = ones((l+1, 2))
                        Qs_new[:, 1] = K_X_start_new[:, ind_del]
                        Qs = Qs_new


                elif imin == 2: # min = gemin | gomin => move k from r to s

                    # delete the elements from X,a and g => add it to the end of X,a,g
                    #ind_del = np.asarray(indo)[Io_minus][iomin]
                    ind_del = indo[iomin]
                    if ls > 0:
                        nk = K_X[[ind_del], [ind_del] + inds]
                        betak = - R.dot(nk)
                        k = 1 - nk.dot(R).dot(nk)
                        # if k = 0
                        if k == 0:
                            k = 0.001
                        betak1 = ones(ls+2)
                        betak1[:-1] = betak
                        R_old = R
                        R = 1/k * outer(betak1, betak1)
                        R[:-1,:-1] += R_old

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

                    if ls > 1:

                        Qs_new = ones((l+1, ls+1))
                        Qs_new[:, :-1] = Qs
                        Qs_new[:, ls] = K_X_start_new[:, ind_del]
                        Qs = Qs_new


                    else:
                        Qs_new = ones((l+1, 2))
                        Qs_new[:, 1] = K_X_start_new[:, ind_del]
                        Qs = Qs_new

                elif imin == 3:
                    break
                else:
                    break
                #loop_count += 1
            if save_state:
                save_state = False
                # normal data
                if label == 1:
                    if ac > C - e:
                        restart = True
                # anomaly
                else:
                    if ac <= C - e:
                        restart = True
                if restart:
                    drop += 1
                    n_all -= 1
                    K_X = delete(K_X, start_new, axis=0)
                    K_X = delete(K_X, start_new, axis=1)
                    X = delete(X, start_new, axis=0)
                    gamma = delete(gamma, start_new)
                    R = R_save
                    a = a_save
                    a = delete(a, start_new)
                    g = g_save
                    g = delete(g, start_new)

                    Qs = Qs_save
                    inds, indr, inde, indo = indices
                    inds = [i - 1 for i in inds]
                    indr = [i - 1 for i in indr]
                    inde = [i - 1 for i in inde]
                    indo = [i - 1 for i in indo]
                    continue

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
                g[start_new] = 0
                if len(inds) == 1:
                    R = ones((2, 2))
                    R[1,1] = 0
                    R[0,0] = -1
                else:
                    if R.shape[0] != len(inds) + 1:
                        nk = ones(ls+1)
                        nk[1:] = K_X_start_new[[0], inds[:-1]]
                        betak = - R.dot(nk)
                        k = 1 - nk.dot(R).dot(nk)
                        betak1 = ones(ls + 2)
                        betak1[:-1] = betak
                        R_old = R
                        R = 1/k * outer(betak1, betak1)
                        R[:-1,:-1] += R_old
                if ls > 0:
                    Qs = concatenate((Qs,K_X_start_new[:, start_new].reshape((l+1,1))), axis=1)
                else:
                    Qs = concatenate((Qs.reshape((l+1,1)),K_X_start_new[:, start_new].reshape((l+1,1))), axis=1)
                Qs[:, 1] = 1
                ls += 1


         # update X, a
        self._data.set_X(X)
        self._data.set_alpha(a)
        self._data.set_C(C)
        self._data.set_K_X(K_X)
        self._rho = -1 * mu
        #print "dropped: %s" % drop
        return True


    def increment(self, Xc, init_ac=0):
        #print "increment"
        #print Xc.shape
        # epsilon
        e = self._data._e

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

        # create gram matrix for all new and existing points

        # create of all data points
        if K_X_origin == None:
            K_X = self.gram(X)
        # create gram matrix for new points and add to existing ones
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


        restart = False
        save_state = False
        drop = 0
        # loop through all new points to increment
        for x_count in range(n_new):
            #print "--------- START %s ---------" % x_count

            # initialize X, a, and kernel matrices
            start_origin = n_new - x_count
            start_new = start_origin - 1
            K_X_start_new = K_X[start_new:]
            K_X_start_origin = K_X[:, start_origin:]
            a_origin = a[start_origin:]
            # initalize indices for bookkeeping
            if x_count == 0 or restart:
                r = range(n_new, n_all)
                inds = [i for i in r if e < a[i] < C - e]
                indr = [i for i in r if i not in inds]
                inde = [i for i in indr if a[i] > e]
                indo = [i for i in indr if a[i] <= e]

                ls = len(inds)                               # support vectors length
                lr = len(indr)                               # error and non-support vectors length
                le = len(inde)                               # error vectors lenght
                lo = len(indo)
                l = ls + lr
                # calculate mu according to KKT-conditions
                if ls == 0:
                    return False
                mu = - K_X_start_origin[inds[0]].dot(a_origin)
                # calculate gradient of error and non-support vectors
                g[inds] = 0
                if lr > 0:
                    g[indr] = K_X_start_origin[indr].dot(a_origin) + mu
                Qs = ones((l+1, ls+1))
                Qs[:,1:] = K_X_start_new[:, inds]
                restart = False
            else:
                l += 1
                if ls > 0:
                    Qs = concatenate(([K_X_start_new[0, [start_new] + inds]], Qs), axis=0)

            c_inds = [start_new] + inds
            # only calculate gradient if there are support vectors
            if ls > 0:
                gc = K_X_start_origin[start_new].dot(a_origin) + mu
            else:
                print "Error: No support vectors to train!"
                return False
            ac = a[start_new]

            if x_count == 0:
                Q = ones((ls+1, ls+1))
                Q[0, 0] = 0
                inds_row = [[i] for i in inds]
                Q[1:, 1:] = K_X[inds_row, inds]
                try:
                    R = inv(Q)
                except np.linalg.linalg.LinAlgError:
                    x = 1e-11
                    found = False
                    #print "singular matrix"
                    while not found:
                        try:
                            R = inv(Q + diag(ones(ls+1) * x))
                            found = True
                        except np.linalg.linalg.LinAlgError:
                            x = x*10
            loop_count = 0
            # unsupervised label


            while gc < e and ac < C - e:

                loop_count += 1
                #print "-------------------- incremental %s-%s ---------" % (x_count, loop_count)

                #calculate beta
                if ls > 0:
                    beta = - R.dot(K_X_start_new[0,c_inds])

                    #if x_count == 303:
                    #    print "ls: %s" % ls
                    #    print R
                    #print K_X_start_new[0,c_inds]
                    betas = beta[1:]

                # calculate gamma
                if lr > 0 and ls > 0:
                    # non-empty R and S set
                    gamma[start_new:] = Qs.dot(beta) + K_X_start_new[:, start_new]
                    gammac = gamma[start_new]
                    ggamma = divide(-g, gamma)
                elif ls > 0:
                    # empty R set
                    gammac = K_X_start_new[0, c_inds].dot(beta) + 1

                else:
                    # empty S set
                    gammac = 1
                    gamma[indr] = 1
                    ggamma = -g

                # accounting
                #case 1: Some alpha_i in S reaches a bound
                if ls > 0:
                    gsmax = - a[inds]
                    gsmax[betas > e] += C
                    #print gsmax
                    gsmax = divide(gsmax, betas)
                    #print betas
                    # only consider positive increment weights
                    gsmax[absolute(betas) <= e] = inf
                    gsmin = min(absolute(gsmax))
                    #print "gsmin: %s" % gsmin
                    if gsmin != inf:
                        ismin = where(absolute(gsmax) == gsmin)[0][0]
                    #print "----"

                else: gsmin = inf
                #case 2: Some g_i in E reaches zero
                if le > 0:
                    # only consider positive margin sensitivity for points in E
                    gec = ggamma[inde]
                    # only consider positive increment weights
                    gec[gec <= e] = inf
                    gemin = min(gec)
                    if gemin < inf:
                        iemin = where(gec == gemin)[0][0]

                else: gemin = inf
                #case 2: Some g_i in O reaches zero
                if lo > 0 and ls > 0:
                    # only consider positive margin sensitivity for points in E
                    goc = ggamma[indo]
                    # only consider positive increment weights
                    goc[goc <= e] = inf
                    # find minimum and index of it
                    gomin = min(goc)
                    if gomin < inf:
                        iomin = where(goc == gomin)[0][0]
                else: gomin = inf

                # case 3: gc becomes zero => algorithm converges
                if gammac > e: gcmin = - gc/gammac
                else: gcmin = inf

                # case 4: ac becomes an error vector => algorithm converges
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
                    g += gamma * gmin
                gc += gammac * gmin
                if imin == 0: # min = gsmin => move k from s to r
                    ak = a[inds][ismin]
                    ind_del = inds[ismin]
                    #bookkeeping
                    inds.remove(ind_del)
                    indr.append(ind_del)
                    if ak < e:
                        indo.append(ind_del)
                        lo += 1
                    else:
                        inde.append(ind_del)
                        le += 1
                    lr += 1
                    c_inds = [start_new] + inds

                    ismin += 1
                    #decrement R, delete row ismin and column ismin
                    if ls > 2:
                        ### R update
                        R_new = zeros((ls,ls))
                        R_new[0:ismin, 0:ismin] = R[0:ismin, 0:ismin]
                        R_new[ismin:, 0:ismin] = R[ismin+1:,0:ismin]
                        R_new[0:ismin, ismin:] = R[0:ismin, ismin+1:]
                        R_new[ismin:, ismin:] = R[ismin+1:, ismin+1:]

                        betak = R[ismin,
                                [i for i in range(ls+1) if i != ismin]]
                        R_new -= outer(betak, betak)/R[ismin,ismin]
                        R = R_new

                        # update Qs for gamma
                        Qs_new = ones((l+1, ls))
                        Qs_new[:, :ismin] = Qs[:,:ismin]
                        Qs_new[:, ismin:] = Qs[:,ismin+1:]
                        Qs = Qs_new

                    elif ls == 2:
                        ### R update
                        R = ones((2, 2))
                        R[1,1] = 0
                        R[0,0] = -1
                        # update Qs for gamma
                        Qs_new = ones((l+1, ls))
                        Qs_new[:, :ismin] = Qs[:,:ismin]
                        Qs_new[:, ismin:] = Qs[:,ismin+1:]
                        Qs = Qs_new

                    else:
                        Qs = ones(l+1)
                        R = inf
                    ls -= 1

                elif imin == 1:
                    ind_del = inde[iemin]
                    if ls > 0:

                        nk = K_X[[ind_del], [ind_del] + inds]
                        betak = - R.dot(nk)
                        k = 1 - nk.dot(R).dot(nk)
                        if k == 0:
                            k = 0.001
                        betak1 = ones(ls + 2)
                        betak1[:-1] = betak
                        R_old = R
                        R = 1/k * outer(betak1, betak1)
                        R[:-1,:-1] += R_old
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

                    if ls > 1:
                        Qs_new = ones((l+1, ls+1))
                        Qs_new[:, :-1] = Qs
                        Qs_new[:, ls] = K_X_start_new[:, ind_del]
                        Qs = Qs_new
                    else:
                        Qs_new = ones((l+1, 2))
                        Qs_new[:, 1] = K_X_start_new[:, ind_del]
                        Qs = Qs_new


                elif imin == 2: # min = gemin | gomin => move k from r to s

                    # delete the elements from X,a and g => add it to the end of X,a,g
                    #ind_del = np.asarray(indo)[Io_minus][iomin]
                    ind_del = indo[iomin]
                    if ls > 0:
                        nk = K_X[[ind_del], [ind_del] + inds]
                        betak = - R.dot(nk)
                        k = 1 - nk.dot(R).dot(nk)
                        #work around!
                        if k == 0:
                            k = 0.001
                        betak1 = ones(ls+2)
                        betak1[:-1] = betak
                        R_old = R
                        R = 1/k * outer(betak1, betak1)
                        R[:-1,:-1] += R_old

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

                    if ls > 1:

                        Qs_new = ones((l+1, ls+1))
                        Qs_new[:, :-1] = Qs
                        Qs_new[:, ls] = K_X_start_new[:, ind_del]
                        Qs = Qs_new


                    else:
                        Qs_new = ones((l+1, 2))
                        Qs_new[:, 1] = K_X_start_new[:, ind_del]
                        Qs = Qs_new

                elif imin == 3:
                    break
                else:
                    break
                #loop_count += 1

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
                g[start_new] = 0
                if len(inds) == 1:
                    R = ones((2, 2))
                    R[1,1] = 0
                    R[0,0] = -1
                else:
                    if R.shape[0] != len(inds) + 1:
                        nk = ones(ls+1)
                        nk[1:] = K_X_start_new[[0], inds[:-1]]
                        betak = - R.dot(nk)
                        k = 1 - nk.dot(R).dot(nk)
                        betak1 = ones(ls + 2)
                        betak1[:-1] = betak
                        R_old = R
                        R = 1/k * outer(betak1, betak1)
                        R[:-1,:-1] += R_old

                if ls < 1:
                    Qs = concatenate((Qs.reshape((l+1,1)), K_X_start_new[:, start_new].reshape((l+1,1))), axis=1)
                else:
                    Qs = concatenate((Qs,K_X_start_new[:, start_new].reshape((l+1,1))), axis=1)
                Qs[:, 1] = 1
                ls += 1

         # update X, a
        self._data.set_X(X)
        self._data.set_alpha(a)
        self._data.set_C(C)
        self._data.set_K_X(K_X)
        self._rho = -1 * mu
        return True

    def increment_perturb(self, Xc, C_new, init_ac=0, break_count=-1):
        # epsilon
        e = self._data._e
        mu = 0
        imin = None

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

        # create gram matrix for all new and existing points

        # create of all data points
        if K_X_origin == None:
            K_X = self.gram(X)
        # create gram matrix for new points and add to existing ones
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

            # initialize X, a, C, g, indices, kernel values
            start_origin = n_new - x_count
            start_new = start_origin - 1
            K_X_start_new = K_X[start_new:]
            K_X_start_origin = K_X[:, start_origin:]
            a_origin = a[start_origin:]
            if x_count == 0:
                r = range(n_new, n_all)
                inds = [i for i in r if e < a[i] < C - e]
                indr = [i for i in r if i not in inds]
                inde = [i for i in indr if a[i] > e]
                indo = [i for i in indr if a[i] <= e]

                ls = len(inds)                               # support vectors length
                lr = len(indr)                               # error and non-support vectors length
                le = len(inde)                               # error vectors lenght
                lo = len(indo)
                l = ls + lr
                # calculate mu according to KKT-conditions
                mu = - K_X_start_origin[inds[0]].dot(a_origin)
                # calculate gradient of error and non-support vectors
                if lr > 0:
                    g[indr] = K_X_start_origin[indr].dot(a_origin) + mu
                Qs = ones((l+1, ls+1))
                Qs[:,1:] = K_X_start_new[:, inds]
            else:
                l += 1
                Qs = concatenate(([K_X_start_new[0, [start_new] + inds]], Qs), axis=0)


            c_inds = [start_new] + inds
            # only calculate gradient if there are support vectors
            if ls > 0:
                gc = K_X_start_origin[start_new].dot(a_origin) + mu
            else:
                print "Error: No support vectors to train!"
                sys.exit()
            ac = a[start_new]

            if x_count == 0:
                Q = ones((ls+1, ls+1))
                Q[0, 0] = 0
                inds_row = [[i] for i in inds]
                Q[1:, 1:] = K_X[inds_row, inds]
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
            loop_count = 0
            while gc < e and ac < C - e:
                loop_count += 1
                #print "-------------------- incremental %s-%s ---------" % (x_count, loop_count)

                #calculate beta
                if ls > 0:
                    beta = - R.dot(K_X_start_new[0,c_inds])
                    betas = beta[1:]

                # calculate gamma
                if lr > 0 and ls > 0:
                    # non-empty R and S set
                    gamma[start_new:] = Qs.dot(beta) + K_X_start_new[:, start_new]
                    gammac = gamma[start_new]
                    ggamma = divide(-g, gamma)
                elif ls > 0:
                    # empty R set
                    gammac = K_X_start_new[0, c_inds].dot(beta) + 1

                else:
                    # empty S set
                    gammac = 1
                    gamma[indr] = 1
                    ggamma = -g

                # accounting
                #case 1: Some alpha_i in S reaches a bound
                if ls > 0:
                    gsmax = - a[inds]
                    gsmax[betas > e] += C
                    gsmax = divide(gsmax, betas)
                    # only consider positive increment weights
                    gsmax[absolute(betas) <= e] = inf
                    gsmin = min(absolute(gsmax))
                    if gsmin != inf:
                        ismin = where(absolute(gsmax) == gsmin)[0][0]

                else: gsmin = inf
                #case 2: Some g_i in E reaches zero
                if le > 0:
                    # only consider positive margin sensitivity for points in E
                    gec = ggamma[inde]
                    # only consider positive increment weights
                    gec[gec <= e] = inf
                    gemin = min(gec)
                    if gemin < inf:
                        iemin = where(gec == gemin)[0][0]

                else: gemin = inf
                #case 2: Some g_i in O reaches zero
                if lo > 0 and ls > 0:
                    # only consider positive margin sensitivity for points in E
                    goc = ggamma[indo]
                    # only consider positive increment weights
                    goc[goc <= e] = inf
                    # find minimum and index of it
                    gomin = min(goc)
                    if gomin < inf:
                        iomin = where(goc == gomin)[0][0]
                else: gomin = inf

                # case 3: gc becomes zero => algorithm converges
                if gammac > e: gcmin = - gc/gammac
                else: gcmin = inf

                # case 4: ac becomes an error vector => algorithm converges
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
                    g += gamma * gmin
                gc += gammac * gmin
                if imin == 0: # min = gsmin => move k from s to r
                    ak = a[inds][ismin]
                    ind_del = inds[ismin]
                    #bookkeeping
                    inds.remove(ind_del)
                    indr.append(ind_del)
                    if ak < e:
                        indo.append(ind_del)
                        lo += 1
                    else:
                        inde.append(ind_del)
                        le += 1
                    lr += 1
                    c_inds = [start_new] + inds

                    ismin += 1
                    #decrement R, delete row ismin and column ismin
                    if ls > 2:
                        ### R update
                        R_new = zeros((ls,ls))
                        R_new[0:ismin, 0:ismin] = R[0:ismin, 0:ismin]
                        R_new[ismin:, 0:ismin] = R[ismin+1:,0:ismin]
                        R_new[0:ismin, ismin:] = R[0:ismin, ismin+1:]
                        R_new[ismin:, ismin:] = R[ismin+1:, ismin+1:]

                        betak = R[ismin,
                                [i for i in range(ls+1) if i != ismin]]
                        R_new -= outer(betak, betak)/R[ismin,ismin]
                        R = R_new

                        # update Qs for gamma
                        Qs_new = ones((l+1, ls))
                        Qs_new[:, :ismin] = Qs[:,:ismin]
                        Qs_new[:, ismin:] = Qs[:,ismin+1:]
                        Qs = Qs_new

                    elif ls == 2:
                        ### R update
                        R = ones((2, 2))
                        R[1,1] = 0
                        R[0,0] = -1
                        # update Qs for gamma
                        Qs_new = ones((l+1, ls))
                        Qs_new[:, :ismin] = Qs[:,:ismin]
                        Qs_new[:, ismin:] = Qs[:,ismin+1:]
                        Qs = Qs_new

                    else:
                        Qs = ones(l+1)
                        R = inf
                    ls -= 1

                elif imin == 1:
                    ind_del = inde[iemin]
                    if ls > 0:
                        nk = K_X[[ind_del], [ind_del] + inds]
                        betak = - R.dot(nk)
                        k = 1 - nk.dot(R).dot(nk)
                        betak1 = ones(ls + 2)
                        betak1[:-1] = betak
                        R_old = R
                        R = 1/k * outer(betak1, betak1)
                        R[:-1,:-1] += R_old
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

                    if ls > 1:
                        Qs_new = ones((l+1, ls+1))
                        Qs_new[:, :-1] = Qs
                        Qs_new[:, ls] = K_X_start_new[:, ind_del]
                        Qs = Qs_new
                    else:
                        Qs_new = ones((l+1, 2))
                        Qs_new[:, 1] = K_X_start_new[:, ind_del]
                        Qs = Qs_new


                elif imin == 2: # min = gemin | gomin => move k from r to s

                    # delete the elements from X,a and g => add it to the end of X,a,g
                    #ind_del = np.asarray(indo)[Io_minus][iomin]
                    ind_del = indo[iomin]
                    if ls > 0:
                        nk = K_X[[ind_del], [ind_del] + inds]
                        betak = - R.dot(nk)
                        k = 1 - nk.dot(R).dot(nk)
                        betak1 = ones(ls+2)
                        betak1[:-1] = betak
                        R_old = R
                        R = 1/k * outer(betak1, betak1)
                        R[:-1,:-1] += R_old
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

                    if ls > 1:

                        Qs_new = ones((l+1, ls+1))
                        Qs_new[:, :-1] = Qs
                        Qs_new[:, ls] = K_X_start_new[:, ind_del]
                        Qs = Qs_new


                    else:
                        Qs_new = ones((l+1, 2))
                        Qs_new[:, 1] = K_X_start_new[:, ind_del]
                        Qs = Qs_new

                elif imin == 3:
                    break
                else:
                    break
                #loop_count += 1

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
                g[start_new] = 0
                if len(inds) == 1:
                    R = ones((2, 2))
                    R[1,1] = 0
                    R[0,0] = -1
                else:
                    if R.shape[0] != len(inds) + 1:
                        nk = ones(ls+1)
                        nk[1:] = K_X_start_new[[0], inds[:-1]]
                        betak = - R.dot(nk)
                        k = 1 - nk.dot(R).dot(nk)
                        betak1 = ones(ls + 2)
                        betak1[:-1] = betak
                        R_old = R
                        R = 1/k * outer(betak1, betak1)
                        R[:-1,:-1] += R_old

                Qs = concatenate((Qs,K_X_start_new[:, start_new].reshape((l+1,1))), axis=1)
                Qs[:, 1] = 1
                '''
                Qs_new = ones((l+1, ls+2))
                if ls > 0: Qs_new[:, :-1] = Qs
                Qs_new[:, ls+1] = K_X_start_new[:, start_new]
                Qs = Qs_new
                '''
                ls += 1


                #Qs = concatenate((Qs,K_X_start_new[:, start_new].reshape((l+1,1))), axis=1)

         # update X, a
        self._data.set_X(X)
        self._data.set_alpha(a)
        self._data.set_C(C)
        self._data.set_K_X(K_X)
        #print self.rho()
        self._rho = -1 * mu# epsilon
        e = self._data._e
        mu = 0
        imin = None

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

        # create gram matrix for all new and existing points

        # create of all data points
        if K_X_origin == None:
            K_X = self.gram(X)
        # create gram matrix for new points and add to existing ones
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

            # initialize X, a, C, g, indices, kernel values
            start_origin = n_new - x_count
            start_new = start_origin - 1
            K_X_start_new = K_X[start_new:]
            K_X_start_origin = K_X[:, start_origin:]
            a_origin = a[start_origin:]
            if x_count == 0:
                r = range(n_new, n_all)
                inds = [i for i in r if e < a[i] < C - e]
                indr = [i for i in r if i not in inds]
                inde = [i for i in indr if a[i] > e]
                indo = [i for i in indr if a[i] <= e]

                ls = len(inds)                               # support vectors length
                lr = len(indr)                               # error and non-support vectors length
                le = len(inde)                               # error vectors lenght
                lo = len(indo)
                l = ls + lr
                # calculate mu according to KKT-conditions
                mu = - K_X_start_origin[inds[0]].dot(a_origin)
                # calculate gradient of error and non-support vectors
                if lr > 0:
                    g[indr] = K_X_start_origin[indr].dot(a_origin) + mu
                Qs = ones((l+1, ls+1))
                Qs[:,1:] = K_X_start_new[:, inds]
            else:
                l += 1
                Qs = concatenate(([K_X_start_new[0, [start_new] + inds]], Qs), axis=0)


            c_inds = [start_new] + inds
            # only calculate gradient if there are support vectors
            if ls > 0:
                gc = K_X_start_origin[start_new].dot(a_origin) + mu
            else:
                print "Error: No support vectors to train!"
                sys.exit()
            ac = a[start_new]

            if x_count == 0:
                Q = ones((ls+1, ls+1))
                Q[0, 0] = 0
                inds_row = [[i] for i in inds]
                Q[1:, 1:] = K_X[inds_row, inds]
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
            loop_count = 0
            while gc < e and ac < C - e:
                loop_count += 1
                #print "-------------------- incremental %s-%s ---------" % (x_count, loop_count)

                #calculate beta
                if ls > 0:
                    beta = - R.dot(K_X_start_new[0,c_inds])
                    betas = beta[1:]

                # calculate gamma
                if lr > 0 and ls > 0:
                    # non-empty R and S set
                    gamma[start_new:] = Qs.dot(beta) + K_X_start_new[:, start_new]
                    gammac = gamma[start_new]
                    ggamma = divide(-g, gamma)
                elif ls > 0:
                    # empty R set
                    gammac = K_X_start_new[0, c_inds].dot(beta) + 1

                else:
                    # empty S set
                    gammac = 1
                    gamma[indr] = 1
                    ggamma = -g

                # accounting
                #case 1: Some alpha_i in S reaches a bound
                if ls > 0:
                    gsmax = - a[inds]
                    gsmax[betas > e] += C
                    gsmax = divide(gsmax, betas)
                    # only consider positive increment weights
                    gsmax[absolute(betas) <= e] = inf
                    gsmin = min(absolute(gsmax))
                    if gsmin != inf:
                        ismin = where(absolute(gsmax) == gsmin)[0][0]

                else: gsmin = inf
                #case 2: Some g_i in E reaches zero
                if le > 0:
                    # only consider positive margin sensitivity for points in E
                    gec = ggamma[inde]
                    # only consider positive increment weights
                    gec[gec <= e] = inf
                    gemin = min(gec)
                    if gemin < inf:
                        iemin = where(gec == gemin)[0][0]

                else: gemin = inf
                #case 2: Some g_i in O reaches zero
                if lo > 0 and ls > 0:
                    # only consider positive margin sensitivity for points in E
                    goc = ggamma[indo]
                    # only consider positive increment weights
                    goc[goc <= e] = inf
                    # find minimum and index of it
                    gomin = min(goc)
                    if gomin < inf:
                        iomin = where(goc == gomin)[0][0]
                else: gomin = inf

                # case 3: gc becomes zero => algorithm converges
                if gammac > e: gcmin = - gc/gammac
                else: gcmin = inf

                # case 4: ac becomes an error vector => algorithm converges
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
                    g += gamma * gmin
                gc += gammac * gmin
                if imin == 0: # min = gsmin => move k from s to r
                    ak = a[inds][ismin]
                    ind_del = inds[ismin]
                    #bookkeeping
                    inds.remove(ind_del)
                    indr.append(ind_del)
                    if ak < e:
                        indo.append(ind_del)
                        lo += 1
                    else:
                        inde.append(ind_del)
                        le += 1
                    lr += 1
                    c_inds = [start_new] + inds

                    ismin += 1
                    #decrement R, delete row ismin and column ismin
                    if ls > 2:
                        ### R update
                        R_new = zeros((ls,ls))
                        R_new[0:ismin, 0:ismin] = R[0:ismin, 0:ismin]
                        R_new[ismin:, 0:ismin] = R[ismin+1:,0:ismin]
                        R_new[0:ismin, ismin:] = R[0:ismin, ismin+1:]
                        R_new[ismin:, ismin:] = R[ismin+1:, ismin+1:]

                        betak = R[ismin,
                                [i for i in range(ls+1) if i != ismin]]
                        R_new -= outer(betak, betak)/R[ismin,ismin]
                        R = R_new

                        # update Qs for gamma
                        Qs_new = ones((l+1, ls))
                        Qs_new[:, :ismin] = Qs[:,:ismin]
                        Qs_new[:, ismin:] = Qs[:,ismin+1:]
                        Qs = Qs_new

                    elif ls == 2:
                        ### R update
                        R = ones((2, 2))
                        R[1,1] = 0
                        R[0,0] = -1
                        # update Qs for gamma
                        Qs_new = ones((l+1, ls))
                        Qs_new[:, :ismin] = Qs[:,:ismin]
                        Qs_new[:, ismin:] = Qs[:,ismin+1:]
                        Qs = Qs_new

                    else:
                        Qs = ones(l+1)
                        R = inf
                    ls -= 1

                elif imin == 1:
                    ind_del = inde[iemin]
                    if ls > 0:
                        nk = K_X[[ind_del], [ind_del] + inds]
                        betak = - R.dot(nk)
                        k = 1 - nk.dot(R).dot(nk)
                        betak1 = ones(ls + 2)
                        betak1[:-1] = betak
                        R_old = R
                        R = 1/k * outer(betak1, betak1)
                        R[:-1,:-1] += R_old
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

                    if ls > 1:
                        Qs_new = ones((l+1, ls+1))
                        Qs_new[:, :-1] = Qs
                        Qs_new[:, ls] = K_X_start_new[:, ind_del]
                        Qs = Qs_new
                    else:
                        Qs_new = ones((l+1, 2))
                        Qs_new[:, 1] = K_X_start_new[:, ind_del]
                        Qs = Qs_new


                elif imin == 2: # min = gemin | gomin => move k from r to s

                    # delete the elements from X,a and g => add it to the end of X,a,g
                    #ind_del = np.asarray(indo)[Io_minus][iomin]
                    ind_del = indo[iomin]
                    if ls > 0:
                        nk = K_X[[ind_del], [ind_del] + inds]
                        betak = - R.dot(nk)
                        k = 1 - nk.dot(R).dot(nk)
                        betak1 = ones(ls+2)
                        betak1[:-1] = betak
                        R_old = R
                        R = 1/k * outer(betak1, betak1)
                        R[:-1,:-1] += R_old
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

                    if ls > 1:

                        Qs_new = ones((l+1, ls+1))
                        Qs_new[:, :-1] = Qs
                        Qs_new[:, ls] = K_X_start_new[:, ind_del]
                        Qs = Qs_new


                    else:
                        Qs_new = ones((l+1, 2))
                        Qs_new[:, 1] = K_X_start_new[:, ind_del]
                        Qs = Qs_new

                elif imin == 3:
                    break
                else:
                    break
                #loop_count += 1

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
                g[start_new] = 0
                if len(inds) == 1:
                    R = ones((2, 2))
                    R[1,1] = 0
                    R[0,0] = -1
                else:
                    if R.shape[0] != len(inds) + 1:
                        nk = ones(ls+1)
                        nk[1:] = K_X_start_new[[0], inds[:-1]]
                        betak = - R.dot(nk)
                        k = 1 - nk.dot(R).dot(nk)
                        betak1 = ones(ls + 2)
                        betak1[:-1] = betak
                        R_old = R
                        R = 1/k * outer(betak1, betak1)
                        R[:-1,:-1] += R_old

                Qs = concatenate((Qs,K_X_start_new[:, start_new].reshape((l+1,1))), axis=1)
                Qs[:, 1] = 1
                '''
                Qs_new = ones((l+1, ls+2))
                if ls > 0: Qs_new[:, :-1] = Qs
                Qs_new[:, ls+1] = K_X_start_new[:, start_new]
                Qs = Qs_new
                '''
                ls += 1


                #Qs = concatenate((Qs,K_X_start_new[:, start_new].reshape((l+1,1))), axis=1)

         # update X, a
        self._data.set_X(X)
        self._data.set_alpha(a)
        self._data.set_C(C)
        self._data.set_K_X(K_X)
        #print self.rho()
        self._rho = -1 * mu
        ##### PERTUBATION START #####
        C = 1
        lmbda = C_new - C
        # if there are no error vectors initially...
        if le == 0:
            pass
            delta_p = (a - C) / lmbda
            i = delta_p > e
            p = min(delta_p[i], 1)
            C += lmbda *p
            if p < 1:
                # find index of minimum
                i = where(delta_p == p)[0][0]
                # update R
                ismin = inds.index(i)
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
                # bookkeeping from margin to error
                a[i] = 0
                inds.remove(i)
                indr.append(i)
                inde.append(i)
                ls -= 1
                lr += 1
                le += 1
        # if there are error vectors to adjust...
        disp_p_delta = 0.2
        disp_p_count = 1
        perturbations = 0
        if p < 1:
            SQl = np.sum(K_X[:, inde], axis=1) * lmbda
            Syl = n_all*lmbda
        while p < 1:
            perturbations += 1
            if ls > 0:
                v = zeros(ls + 1)
                if p < 1 - e:
                    v[0] = - Syl - sum(a)/(1-p)
                else:
                    v[0] = -Syl
                v[1:] = - -SQl[inds]
                beta = R * v
                betas = beta[1:]
                gamma = zeros(K_X.shape[0])
                if lr > 0:
                    Q = ones((indr, inds))
                    indr_row = [[i] for i in indr]
                    Q[1:,:] = K_X[indr_row,inds]
                    gamma[indr] = Q.dot(betas) + SQl[indr]
            else:
                beta = 0
                gamma = SQl
            ### minimum increment or decrement
            #accounting

            #upper limit on change in p_c assuming no other examples change status
            delta_p_c = 1 - p;

            #case 1: Some alpha_i in S reaches a bound
            if ls > 0:
                # only consider non-zero coefficient sensitivity betas

                # change in p_c that causes a margin vector to change to a reserve vector
                IS_minus = betas < - e
                gsmax = ones(ls)*inf
                # look for greatest increment according to sensitivity
                if gsmax[IS_minus].shape[0] > 0:
                    gsmax[IS_minus] = - a[inds][IS_minus]
                    gsmax = divide(gsmax, betas)
                    # find minimum and index of it
                    gsmin1 = min(absolute(gsmax))
                    ismin1 = where(gsmax == gsmin1)[0][0]
                else:
                    gsmin = inf

                new_beta = betas-lmbda;
                IS_plus = new_beta > e
                gsmax = ones(ls)*inf
                # look for greatest increment according to sensitivity
                if gsmax[IS_plus].shape[0] > 0:
                    gsmax[IS_plus] = (C - a[inds][IS_plus]) / new_beta
                    gsmin2 = min(absolute(gsmax))
                    ismin2 = where(gsmax == gsmin2)[0][0]
                else:
                    gsmin2 = inf

            #case 2: Some g_i in E reaches zero
            if le > 0:
                gamma_inde = gamma[inde]
                g_inde = g[inde]
                # only consider positive margin sensitivity for points in E
                Ie_plus = gamma_inde > e
                if len(g_inde[Ie_plus]) > 0:
                    gec = divide(-g_inde[Ie_plus], gamma_inde[Ie_plus])
                    # only consider positive increment weights
                    gec[gec <= 0] = inf
                    # find minimum and index of it
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
            # determine minimum largest increment
            all_deltas = [gsmin1, gsmin2, gemin, gomin]
            gmin = min(all_deltas)
            imin = where(all_deltas == gmin)[0][0]

            # update a, b, g and p
            if lr > 0:
                a[indr] += lmbda * gmin
                g[indr] += gamma[indr] * gmin
            if ls > 0:
                mu += beta[0]*gmin
                a[inds] += betas*gmin
            else:
                mu += gmin
            p += gmin
            C += lmbda * gmin

            if imin == 0: # min = gsmin1 => move k from s to o
                # if there are more than 1 minimum, just take 1
                ak = a[inds][ismin1]
                # delete the elements from X,a and g
                # => add it to the end of X,a,g
                ind_del = inds[ismin1]
                inds.remove(ind_del)
                indr.append(ind_del)
                indo.append(ind_del)
                lr += 1
                lo += 1
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
                # if there are more than 1 minimum, just take 1
                ak = a[inds][ismin2]
                # delete the elements from X,a and g
                # => add it to the end of X,a,g
                ind_del = inds[ismin2]
                inds.remove(ind_del)
                indr.append(ind_del)
                inde.append(ind_del)
                lr += 1
                le += 1
                #update SQl and Syl when the status of
                # indss changes from MARGIN to ERROR
                SQl += K_X[:, ind_del] * lmbda
                Syl += lmbda
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
            elif imin == 2:
                # delete the elements from X,a and g => add it to the end of X,a,g
                ind_del = np.asarray(inde)[Ie_plus][iemin]

                #update SQl and Syl when the status of
                # indss changes from ERROR to MARGIN
                SQl -= K_X[:, ind_del] * lmbda
                Syl -= lmbda
                #
                if ls > 0:
                    nk = K_X[ind_del, :][[ind_del] + inds]
                    betak = - R.dot(nk)
                    k = 1 - nk.dot(R).dot(nk)
                    betak1 = ones(ls + 2)
                    betak1[:-1] = betak
                    R_old = R
                    R = 1/k * outer(betak1, betak1)
                    R[:-1,:-1] += R_old
                else:
                    R = ones((2, 2))
                    R[1,1] = 0
                    R[0,0] = -1
                # bookkeeping
                inds.append(ind_del)
                indr.remove(ind_del)
                inde.remove(ind_del)
                ls += 1
                lr -= 1
                le -= 1


            elif imin == 3: # min = gemin | gomin => move k from r to s

                # delete the elements from X,a and g => add it to the end of X,a,g
                ind_del = np.asarray(indo)[Io_minus][iomin]
                if ls > 0:
                    nk = ones(ls+1)
                    nk[1:] = K_X[ind_del,:][inds]
                    betak = - R.dot(nk)
                    k = 1 - nk.dot(R).dot(nk)
                    betak1 = ones(ls+2)
                    betak1[:-1] = betak
                    R_old = R
                    R = 1/k * outer(betak1, betak1)
                    R[:-1,:-1] += R_old
                else:
                    R = ones((2, 2))
                    R[1,1] = 0
                    R[0,0] = -1

                indo.remove(ind_del)
                indr.remove(ind_del)
                inds.append(ind_del)
                lo -= 1
                lr -= 1
                ls += 1
            if p >= disp_p_delta*disp_p_count:
                disp_p_count += 1;
                print 'p = %.2f' % p



        ##### PERTUBATION END #####
        # update X, a

        self._data.set_X(X)
        self._data.set_alpha(a)
        self._data.set_C(C)
        self._data.set_K_X(K_X)
        #TODO: CHECK IF mu == rho,
        # then just set it like that
        self.rho()

    def test_kkt(self, K_X, a, e, C):
        l = a.shape[0]
        r = range(l)
        inds = [i for i in r if e < a[i] < C - e]
        indr = [i for i in r if i not in inds]
        inde = [i for i in indr if a[i] > e]
        indo = [i for i in indr if a[i] <= e]

        ls = len(inds)                               # support vectors length
        lr = len(indr)                               # error and non-support vectors length
        mu = - K_X[inds[0],:].dot(a)
        g = ones(l) * -1

        g[inds] = zeros(ls)

        if lr > 0:
            Kr = K_X[indr, :]
            g[indr] = Kr.dot(a) + ones((lr,1)) * mu

            if len(g[inde][g[inde] > 0]) > 0:
                return False
            if len(g[indo][g[indo] < 0]) > 0:
                return False

        if ls > 0:
            Ks = K_X[inds, :]
            g[inds] = Ks.dot(a) + ones((ls,1)) * mu
            for i in range(len(inds)):
                if g[inds[0]] <= -e or  g[inds[0]] >= C-e:
                    return False

        return True

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