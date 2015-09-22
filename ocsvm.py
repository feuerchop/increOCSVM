__author__ = 'LT'
import numpy as np
import cvxopt.solvers
import kernel
from time import gmtime, strftime
from sklearn.metrics.pairwise import pairwise_kernels
from numpy import vstack, hstack, ones, zeros, absolute, where, divide, inf, delete, outer, transpose, diag, tile, arange, concatenate

from numpy.linalg import inv, eig
import data
from profilehooks import profile

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

    #Class constructor: kernel function & nu & sigma
    def __init__(self, metric, nu, gamma):
        self._v = nu
        self._gamma = gamma
        self._data = data.Data()

    #returns trained SVM rho given features (X)
    # TODO: we need to store the key properties of model after training
    # Please check libsvm what they provide for output, e.g., I need to access to the sv_idx all the time
    @profile
    def train(self, X, scale = 1):
        self._data.set_X(X)
        self._data.set_C(1/(self._v * len(X)) * scale)
        # get lagrangian multiplier
        alpha = self.alpha(X, scale)
        self._data.set_alpha(alpha)
        # defines necessary parameter for prediction
        self.rho()

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

        #test if all rhos are the same!!
        '''
        rho_all = self._data.alpha().dot(self.gram(self._data.X(), self._data.Xs())).tolist()
        print self._data.C()
        print "alpha: %s" % self._data.alpha_s()
        print "rho_all: %s" % rho_all
        avg_rho = sum(rho_all)/len(rho_all)
        print "rho_avg: %s" % (sum(rho_all)/len(rho_all))

        for i in range(len(Xs)):
            print "decision function: %s" % self.decision_function(Xs[i])
            print "avg decision function: %s" % (- avg_rho + self._data.alpha().dot(self.gram(self._data.X(), Xs[i])))
        sys.exit()
        '''

    #compute Gram matrix
    def gram(self, X, Y=None):
        return pairwise_kernels(X, Y, "rbf", gamma=self._gamma)

    #compute Lagrangian multipliers
    def alpha(self, X, scale = 1):
        n_samples, n_features = X.shape
        K = 2 * self.gram(X) * scale

        P = cvxopt.matrix(K)
        q = cvxopt.matrix(np.zeros(n_samples))
        A = cvxopt.matrix(np.ones((n_samples,1)),(1,n_samples))
        b = cvxopt.matrix(1.0 * scale)

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
        distance = -1 * self._rho + self._data.alpha_s().dot(self.gram(self._data.Xs(),x))
        return np.sign(distance)

    # Returns distance to boundary
    def decision_function(self, x):
        return - self._rho + self._data.alpha().dot(self.gram(self._data.X(), x))

    @profile
    def increment(self, xc, init_ac = 0, v = None):

        e = 1e-6
        # initialize X, a, C, g, indeces, kernel values
        X = self._data.X()                                  # data points
        C = self._data.C()
        a = self._data.alpha()
        ac = init_ac


        inds = [i for i, bool in enumerate(np.all([a > e, a < C - e], axis=0)) if bool]
        indr = [i for i, bool in enumerate(np.any([a <= e, a >= C - e], axis=0)) if bool]

        inde = [i for i, bool in enumerate(a[indr] >= C - e) if bool]

        indo = [i for i, bool in enumerate(a[indr] <= e) if bool]
        inde_bool = [True for i, val in enumerate(indr) if val in inde]
        indo_bool = [True for i, val in enumerate(indr) if val in indo]

        l = len(a)
        indices = arange(l)
        indices_all = arange(l+1)
        ls = len(a[inds])                               # support vectors length
        lr = len(a[indr])                               # error and non-support vectors length
        le = len(a[inde])                               # error vectors lenght
        lo = len(a[indo])                               # non-support vectors

        if self._data.K_X() != None:
            K_X_old = self._data.K_X()
            K_xc_X = self.gram(xc, X)[0]
            K_X_all = zeros((K_X_old.shape[0]+1, K_X_old.shape[1]+1))
            K_X_all[1:, 1:] = K_X_old
            K_X_all[0, 0] = 1.0
            K_X_all[0, 1:] = K_xc_X
            K_X_all[1:, 0] = K_xc_X
        else:
            # kernel of all data points including the new one

            K_X_all = self.gram(vstack((xc,X)))
            # kernel of all data points excluding the new one

        K_X = K_X_all[1:, 1:]
        # kernel of support vectors
        Kss = K_X[:,inds]
        Kss = Kss[inds,:]

        # calculate mu according to KKT-conditions
        mu_all = - K_X[inds,:].dot(a)
        mu = np.mean(mu_all)
        g = ones(l) * -1

        g[inds] = zeros(ls)

        if ls > 0:
            Kcs = K_X_all[0, 1:][inds]
        if lr > 0:
            Krs = K_X[:, inds][indr, :]
            Kr = K_X[indr, :]
            Kcr = K_X_all[0, 1:][indr]
            g[indr] = Kr.dot(a) + ones((lr,1)) * mu
        Kcc = 1
        gc = Kcs.dot(a[inds]) + mu
        Q = ones((ls+1, ls+1))
        Q[0, 0] = 0
        Q[1:, 1:] = Kss

        try:
            R = inv(Q)
        except np.linalg.linalg.LinAlgError:
            #print "singular matrix"
            R = inv(Q + diag(ones(ls+1) * 1e-2))

        loop_count = 1
        while gc < e and ac < C - e:
            #print "a[inds]: %s" % a[inds]
            #print "loop count: %s" % loop_count
            # calculate beta
            if ls > 0:
                if ls == 1:
                    Q = ones((ls+1, ls+1))
                    Q[0, 0] = 0
                    Q[1:, 1:] = Kss
                    R = inv(Q)
                n = ones(ls+1)
                n[1:] = Kcs
                beta = - R.dot(n)
                betas = beta[1:]
                ##print "beta: %s" %beta
            # calculate gamma
            if lr > 0 and ls > 0:
                gamma = ones((lr+1,ls+1))
                gamma[0, 1:] = Kcs
                gamma[1:,1:] = Krs
                tmp = ones(lr+1)
                tmp[1:] = Kcr
                gamma = gamma.dot(beta) + tmp
                gammac = gamma[0]
                gammar = gamma[1:]

            elif ls > 0:
                # empty R set
                gammac = ones(ls + 1)
                gammac[1:] = Kcs
                gammac = gammac.dot(beta) + Kcc

            else:
                # empty S set
                gammac = 1
                gammar = ones(lr)

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
                #print "gsmax: %s" % gsmax
                gsmin = absolute(gsmax).min()
                ismin = where(absolute(gsmax) == gsmin)
                #print ismin
            else: gsmin = inf

            #case 2: Some g_i in R reaches zero
            if le > 0:
                Ie_plus = gammar[inde_bool] > e
                Ie_inf = gammar[inde_bool] <= e
                gec = zeros(len(g[inde] > e))

                gec[Ie_plus] = divide(-g[inde][Ie_plus], gammar[inde_bool][Ie_plus])
                gec[Ie_inf] = inf

                for i in range(0, len(gec)):
                    if gec[i] <= e:
                        gec[i] = inf

                gemin = gec.min()
                iemin = where(gec == gemin)

            else: gemin = inf
            if lo > 0:
                #print indr
                #print indo
                Io_minus = gammar[indo_bool] < - e
                Io_inf = gammar[indo_bool] >= - e

                goc = zeros(len(g[indo] > e))
                goc[Io_minus] = divide(-g[indo][Io_minus], gammar[indo_bool][Io_minus])
                goc[Io_inf] = inf

                for i in range(0, len(goc)):
                    if goc[i] <= e:
                        goc[i] = inf
                    if g[indo][i] < 0:
                        goc[i] = inf


                gomin = goc.min()
                iomin = where(goc == gomin)
            else: gomin = inf
            # case 3: gc becomes zero
            if gammac > e: gcmin = - gc/gammac
            else: gcmin = inf
            # case 4
            gacmin = C - ac

            # determine minimum largest increment
            all_deltas = [gsmin, gemin, gomin, gcmin, gacmin]
            gmin = min(all_deltas)

            for i, val in enumerate(all_deltas):
                if val == gmin:
                    imin = i
                    break

            # update a, g,
            if ls > 0:
                ac += gmin
                if imin == 4: a[inds] += betas*gmin
                else: a[inds] += betas*gmin
                if lr > 0: g[indr] += gammar * gmin
                gc = gc + gammac * gmin
            # else??
            if imin == 0: # min = gsmin => move k from s to r
                # if there are more than 1 minimum, just take 1
                if len(ismin[0]) > 1:
                    ismin = [ismin[0][0]]
                ak = a[inds][ismin]

                # delete the elements from X,a and g => add it to the end of X,a,g
                #print "ismin: %s" % ismin
                #print "inds: %s" % inds
                ind_del = inds[ismin[0]]
                inds.remove(ind_del)
                indr.append(ind_del)
                if ak < e:
                    indo.append(ind_del)
                else:
                    inde.append(ind_del)

                #decrement R, delete row ismin and column ismin

                if ls > 1:
                    #if isinstance(ismin[0], )
                    if type(ismin[0]).__name__ == "ndarray":
                        ismin = ismin[0][0] + 1
                    else: ismin = ismin[0] + 1
                    for i in range(R.shape[0]):
                        for j in range(R.shape[1]):
                            if i != ismin and j != ismin:
                                R[i][j] = R[i][j] - R[i][ismin]*R[ismin][j]/R[ismin][ismin]

                    R = delete(R, ismin, 0)
                    R = delete(R, ismin, 1)
                else:
                    R = inf


            elif imin == 1:
                # if there are more than 1 minimum, just take 1
                if len(iemin[0]) > 1:
                    iemin = [iemin[0][0]]

                # delete the elements from X,a and g => add it to the end of X,a,g

                ind_del = inde[iemin[0]]

                if ls > 0:
                    nk = ones(ls+1)
                    nk[1:] = K_X[ind_del][inds]
                    betak = - R.dot(nk)
                    k = 1 - nk.dot(R).dot(nk)

                    betak1 = ones(R.shape[0] + 1)
                    betak1[:-1] = betak
                    R_old = R
                    R = zeros((R.shape[0] + 1, R.shape[1] + 1))
                    R[:-1,:-1] = R_old
                    R += 1/k * outer(betak1, betak1)
                inds.append(ind_del)
                indr.remove(ind_del)
                inde.remove(ind_del)

            elif imin == 2: # min = gemin | gomin => move k from r to s
                if len(iomin[0]) > 1:
                    iomin = [iomin[0][0]]

                # delete the elements from X,a and g => add it to the end of X,a,g
                ind_del = indo[iomin[0]]
                if ls > 0:
                    nk = ones(ls+1)
                    nk[1:] = K_X[ind_del][inds]
                    betak = - R.dot(nk)
                    k = 1 - nk.dot(R).dot(nk)
                    betak1 = ones(R.shape[0] + 1)
                    betak1[:-1] = betak
                    R_old = R
                    R = zeros((R.shape[0] + 1, R.shape[1] + 1))
                    R[:-1,:-1] = R_old
                    R += 1/k * outer(betak1, betak1)

                indo.remove(ind_del)
                indr.remove(ind_del)
                inds.append(ind_del)
            else: # k = c => terminate
                break
            inde_bool = [True for i, val in enumerate(indr) if val in inde]
            indo_bool = [True for i, val in enumerate(indr) if val in indo]
            #update length of sets
            ls = len(a[inds])                               # support vectors length
            lr = len(a[indr])                               # error and non-support vectors length
            le = len(a[inde])                               # error vectors lenght
            lo = len(a[indo])                               # non-support vectors

            #update kernel
            if ls > 0:
                # kernel of support vectors
                Kss = K_X[:,inds]
                Kss = Kss[inds,:]
                Kcs = K_X_all[0, 1:][inds]
            else:
                Kcs = []
            if lr > 0 and ls > 0:
                # kernel of error vectors, support vectors
                Krs = K_X[:, inds][indr, :]
                Kcr = K_X_all[0, 1:][indr]
            loop_count += 1
        # update X, a
        self._data.set_X(X)
        self._data.set_alpha(a)
        # add x_c and a_c
        self._data.add(xc, ac)
        # set C if necessary
        self._data.set_C(C)
        K_col = K_X_all[:, 0]
        K_col = K_col.reshape(len(K_col),1)
        K_X_all = delete(K_X_all, 0, axis=1)
        K_X_all = hstack((K_X_all, K_col))
        K_row = K_X_all[0, :]
        K_X_all = delete(K_X_all, 0, axis=0)
        K_X_all = vstack((K_X_all, K_row))
        #sys.exit()
        self._data.set_K_X(K_X_all)

        # update rho
        self.rho()

    def perturbc(self, C_new, C_old, a, X):
        #print "perturbc"

        e = eps = 1e-5
        print "a: %s" % a
        inds = np.all([a > e, a < C_old - e], axis=0)           # support vectors indeces
        indr = np.any([a <= e, a >= C_old - e], axis=0)         # error and non-support vectors indeces
        inde = a[indr] >= C_old - e                             # error vectors indeces in R
        indo = a[indr] <= e                             # error vectors indeces in R
        # calculate Q and Rs
        print ones((len(a),1)).shape
        print self.gram(X[inds], X).shape
        Q = vstack((hstack((0, ones(len(a)))), hstack((ones((len(a[inds]),1)), self.gram(X[inds], X)))))
        Rs = inv(vstack((hstack((0, ones(len(a[inds])))), hstack((ones((len(a[inds]),1)), self.gram(X[inds]))))))

        # create a vector containing the regularization parameter
        # for each example if necessary
        C_new = C_new*ones(len(a))
        C = C_old * ones(len(a))
        # compute the regularization sensitivities
        l = C_new-C
        # if there are no error vectors initially...
        if len(a[indr][inde]) == 0:
            # find the subset of the above examples that could become error vectors
            delta_p = divide(C-a,l)
            delta_p[delta_p <= 0] = inf

            # determine the minimum acceptable change in p and adjust the regularization parameters
            p = min(delta_p)[0]
            C = C + l*p
            # if one example becomes an error vector, perform the necessary bookkeeping
            if p < 1:
                i = where(delta_p == p)
                a[i] = C[i]
                ai = -1
                # get index of i in inds
                for i, p_del in enumerate(delta_p):
                    if a[i] > e and a[i] < C_old:
                        ai += 1
                    if p_del == p:
                        break
                # decrement Rs
                if Rs.shape[0] > 2:
                    ai += 1
                    for i in range(Rs.shape[0]):
                        for j in range(Rs.shape[1]):
                            if i != ai and j != ai:
                                Q[i][j] = Q[i][j] - Q[i][ai]*Q[ai][j]/Q[ai][ai]
                    Rs = delete(Rs, ai, 0)
                    Rs = delete(Rs, ai, 1)
                else:
                    Rs = inf
                Q = delete(Q, ai, 0)
        else:
            p = 0

        # if there are error vectors to adjust...
        if (p < 1):
            SQl = transpose(self.gram(X, X[indr][inde]).dot(l[indr][inde]))
            Syl = sum(l[indr][inde])

        print 'p = %s' % p

        # change the regularization parameters incrementally
        disp_p_delta = 0.2
        disp_p_count = 1
        num_MVs = len(a[inds])
        perturbations = 0
        while p < 1:
            perturbations = perturbations + 1

            # compute beta and gamma
            if (num_MVs > 0):

                v = zeros(num_MVs+1)
                if (p < 1 - eps):
                    v[1] = - Syl - sum(a)/(1-p)
                else:
                    v[1] = - Syl
                v[1:] = -SQl[inds]
                beta = Rs.dot(v)
                ind_temp = indr
                if len(a[ind_temp]) > 0:
                    print ind_temp
                    print transpose(Q[:, ind_temp]).shape
                    print beta.shape
                    gamma = transpose(Q[:,ind_temp]).dot(beta) + SQl[ind_temp]
            else:
                beta = 0
                gamma = SQl

            # minimum acceptable parameter change
            min_delta_p, indss, cstatus, nstatus = self.min_delta_p_c(p,gamma,beta,l,inds, a, C, indr, inde, g, indo)

            # update a, b, g and p
            if len(a[indr][inde]) > 0:
                a[indr][inde] += l[[indr][inde]]*min_delta_p
            if (num_MVs > 0):
                a[inds] += + beta[1:]*min_delta_p

            b = b + beta[1]*min_delta_p
            g = g + gamma*min_delta_p
            p = p + min_delta_p
            C = C + l*min_delta_p

            # perform bookkeeping
            #start: Bookeeping
            # if the example is currently a margin vector, determine the row
            # in the extended kernel matrix inverse that needs to be removed
            if cstatus == self._MARGIN:
                indco = where(a[inds] == a[indss])[0] + 1
            else:
                indco = -1

            # adjust coefficient to avoid numerical errors if necessary
            if nstatus == self._RESERVE:
                a[indss] = 0
            elif nstatus == self._ERROR:
                a[indss] = C[indss]
            g_new = g[indss]
            a_new = a[indss]
            g = delete(g, indss)
            a = delete(a, indss)
            g = hstack((g, g_new))
            a = hstack((a, a_new))
            #change the status of the example
            inds = np.all([a > e, a < C - e], axis=0)           # support vectors indeces
            indr = np.any([a <= e, a >= C - e], axis=0)         # error and non-support vectors indeces
            inde = a[indr] >= C - e                             # error vectors indeces in R
            indo = a[indr] <= e                                 # error vectors indeces in R

            #end: Bookeeping

            # update SQl and Syl when the status of indss changes from MARGIN to ERROR
            if cstatus == self._MARGIN and nstatus == self._ERROR:
                SQl = SQl + Q[indco,:].dot(l(indss))
                Syl = Syl + l(indss)

            # set g(ind{MARGIN}) to zero
            g[inds] = 0

            # update Rs and Q if necessary
            if nstatus == self._MARGIN:
                num_MVs = num_MVs + 1
                if (num_MVs > 1):

                # compute beta and gamma for indss
                    beta = -Rs*Q[:,indss]
                    gamma = self.gram(X[:,indss], X[:,indss]) + Q[:,indss].dot(beta)

                rows = Rs.shape[0]

                if rows > 1:
                    Rs = vstack((hstack((Rs, zeros(rows))), zeros(rows + 1))) \
                         + 1/gamma * outer(hstack((beta,1)), hstack((beta,1)))

                else:
                    Rs = vstack((hstack(( - self.gram(X[:, indss], X[:, indss], 1))), hstack((1,0))))
                X_new = X[indss]
                X = delete(X, indss, axis=0)
                X = vstack((X, X_new))
                #TODO: optimize Q calculation
                Q = vstack((hstack((0, ones(len(a)))), hstack((ones((len(a[inds]),1)), self.gram(X[inds], X)))))
            else:
                if cstatus == self._MARGIN:
                    # compress Rs and Q
                    num_MVs = num_MVs - 1
                    # decrement Rs
                    if Rs.shape[0] > 2:
                        for i in range(Rs.shape[0]):
                            for j in range(Rs.shape[1]):
                                if i != indco and j != indco:
                                    Q[i][j] = Q[i][j] - Q[i][indco]*Q[indco][j]/Q[indco][indco]
                        Rs = delete(Rs, indco, 0)
                        Rs = delete(Rs, indco, 1)
                    else:
                        Rs = inf
                    Q = delete(Q, indco, 0)

            # update SQl and Syl when the status of indss changes from ERROR to MARGIN
            if cstatus == self._ERROR and nstatus == self._MARGIN:
                SQl = SQl - Q[num_MVs,:].dot(l[indss])
                Syl = Syl - l[indss]

            if p >= disp_p_delta*disp_p_count:
                disp_p_count = disp_p_count + 1
        return a,X

    def min_delta_p_c(self, p_c, gamma, beta, l, inds, a, C, indr, inde, g, indo):
        eps = 1e-5
        indss = zeros(5)
        cstatus = zeros(5)
        nstatus = zeros(5)

        # upper limit on change in p_c assuming no other examples change status
        delta_p_c = 1 - p_c

        # change in p_c that causes a margin vector to change to a reserve vector
        if (len(beta) > 1): # if there are margin vectors
            beta_s = beta[1:]
            flags = beta_s < 0
            delta_mr, i_mr = self.min_delta(flags, a[inds], zeros(len(a[inds])), beta_s)
            if (delta_mr < inf):
                count = -1
                for j, bool in enumerate(inds):
                    if bool:
                        count += 1
                    if count == i_mr:
                        i = j
                        break
                indss[1] = i
                cstatus[1] = self._MARGIN;
                nstatus[1] = self._RESERVE;
        else:
           delta_mr = inf;


        # change in p_c that causes a margin vector to change to an error vector
        if (len(beta) > 1):  # if there are margin vectors
            l_s = l[inds]
            v = beta_s - l_s
            flags = v > eps
            if len(v[flags]) > 0:
                not_z = v > 0
                delta_me = inf*ones(len(v))
                delta_me[not_z] = C[inds][not_z] - divide(a[inds][not_z],v(not_z))
                delta_m = min(delta_me)
                i_s = where(delta_me == delta_m)
                bool = 0

                for j, b in enumerate(inds):
                    if b:
                        b += 1
                    if b == i_s:
                        i = j
                        break
                if (delta_me < inf):
                    indss[2] = i;
                    cstatus[2] = self._MARGIN;
                    nstatus[2] = self._ERROR;
            else:
                delta_me = inf;
        else:
            delta_me = inf;

        # change in p_c that causes an error vector to change to a margin vector
        gamma_e = gamma[indr][inde]
        flags = gamma_e > 0
        delta_em, ie = self._min_delta(flags,g[indr][inde],zeros(g[indr][inde]),gamma_e);
        if (delta_em < inf):
            count = 0
            for j, bool in enumerate(indr):
                if bool:
                    if inde[count]:
                        count += 1
                    if count - 1 == ie:
                        i = j
                        break

            indss[3] = i
            cstatus[3] = self._ERROR
            nstatus[3] = self._MARGIN

        # change in p_c that causes a reserve vector to change to a margin vector
        gamma_r = gamma[indr][indo]
        flags = np.all([g[indr][indo] >= 0, gamma_r < 0])
        delta_rm,io = self.min_delta(flags,g[indr][indo],zeros(len(g[indr[indo]])),gamma_r)
        if (delta_rm < inf):
            ind = ones(len(a[indr][indo])) * -1
            count = 0
            for j,bool in enumerate(indr):
                if bool:
                    if inde[count]:
                        count += 1
                    if count -1 == io:
                        i = j
                        break
            indss[4] = i
            cstatus[4] = self._RESERVE
            nstatus[4] = self._MARGIN

        # minimum acceptable value for p_c
        min_dpc = min([delta_p_c,delta_mr,delta_me,delta_em,delta_rm]);
        min_ind = where([delta_p_c,delta_mr,delta_me,delta_em,delta_rm] == min_dpc)
        indss = indss[min_ind]
        cstatus = cstatus[min_ind]
        nstatus = nstatus[min_ind]
        return min_dpc, indss, cstatus, nstatus

    def min_delta(self, flags, psi_initial, psi_final, psi_sens):
        if len(psi_sens[flags]) > 0:
            # find the parameters to check
            ind = flags
            deltas = divide(psi_final[ind] - psi_initial[ind], psi_sens[ind])
            min_d = min(deltas)
            i = where(deltas == min_d)[0]
            if len(i) > 1:
                max_sens = max(abs(psi_sens(i)))
                k = i[where(abs(psi_sens(i)) == max_sens)][0]
            else:
                k = i[0]
        else:
           min_d = inf
           k = -1
        return min_d,k

    def KKT(self, X, a):
        e = 1e-5
        inds = np.all([a > e, a < self._data.C() - e], axis=0)           # support vectors indeces
        indr = np.any([a <= e, a >= self._data.C() - e], axis=0)         # error and non-support vectors indeces
        inde = a[indr] >= self._data.C() - e                             # error vectors indeces in R
        indo = a[indr] <= e                                              # non-support vectors indeces in R
        mu = - self.gram(X[inds], X[inds]).dot(a[inds])
        for i, m in enumerate(mu):
            if i == 0: continue
            if abs(m - mu[i-1]) > 1e-5:
                return False
        g = self.gram(X,X[inds]).dot(a[inds]) + ones((len(a),1)) * mu[0]
        for i,gi in enumerate(g):
            if gi <= e and gi >= -1:
                if a[i] > self._data.C() or a[i] < e:
                    print "ai: %s, gi: %s" % (a[i], gi)
                    return False
            elif gi > e:
                if a[i] > e:
                    print "ai: %s, gi: %s" % (a[i], gi)
                    return False
            elif gi < - e:
                if a[i] < self._data.C() - e:
                    print "ai: %s, gi: %s" % (a[i], gi)
                    return False