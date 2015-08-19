__author__ = 'LT'
import numpy as np
import cvxopt.solvers
import kernel
from numpy.linalg import inv
from sklearn.metrics.pairwise import pairwise_kernels
from numpy import vstack, hstack, ones, zeros, absolute, where, divide, inf, delete, outer, transpose
import data
import sys
import math

#print(__doc__)

#Trains an SVM
class OCSVM(object):
    # define global variables
    _rho = None
    _kernel = None
    _nu = None
    _gamma = None
    # flags for example state
    _MARGIN    = 1;
    _ERROR     = 2;
    _RESERVE   = 3;
    _UNLEARNED = 4;

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
        self._rho = (np.sum([a_i * self._kernel(x_i,rho_x) for a_i, x_i in zip(alpha,sv)]))

    #compute Gram matrix
    def gram(self, X, Y=None):
        ## pairwise_kernels:
        ## K(x, y) = exp(-gamma ||x-y||^2)
        return pairwise_kernels(X, Y, "rbf", gamma=self._gamma)

    #compute Lagrangian multipliers
    def alpha(self, X):
        n_samples, n_features = X.shape
        K = self.gram(X)
        #K = self.gram(X) * (self._nu*len(X))

        P = cvxopt.matrix(K)
        q = cvxopt.matrix(zeros(n_samples))
        A = cvxopt.matrix(ones((n_samples,1)),(1,n_samples))
        b = cvxopt.matrix(1.0)
        #b = cvxopt.matrix(self._nu*len(X))

        G_1 = cvxopt.matrix(np.diag(ones(n_samples) * -1))
        h_1 = cvxopt.matrix(zeros(n_samples))

        G_2 = cvxopt.matrix(np.diag(ones(n_samples)))
        h_2 = cvxopt.matrix(ones(n_samples) * 1/(self._nu*len(X)))
        #h_2 = cvxopt.matrix(ones(n_samples))

        G = cvxopt.matrix(vstack((G_1, G_2)))
        h = cvxopt.matrix(vstack((h_1, h_2)))
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        return np.ravel(solution['x'])

    # Returns distance to boundary
    def decision_function(self, x):
        #print self._rho
        return -1 * self._rho + self._data.alpha_s().dot(self.gram(self._data.Xs(),x))

    def predict(self, x):
        #print self._rho
        df = -1 * self._rho + self._data.alpha_s().dot(self.gram(self._data.Xs(),x))
        df_copy = df.copy()
        for i, val in enumerate(df_copy):
            if val > 0:
                df[i] = 1
            else: df[i] = -1
        return df


    ### incremental

    def increment(self, xc):

        e = 1e-4
        # initialize X, a, C, g, indeces, kernel values
        X = self._data.X()                                  # data points
        C = 1/(self._nu*(len(X)+1))
        print "C: %s" %C
        a = self._data.alpha()

        ac = 0                      # alpha of new point c

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
        # calculate mu according to KKT-conditions
        mu = 1 - self.gram(X[inds][0], X[inds]).dot(a[inds])
        print "mu: %s" % mu
        # calculate gradient
        g = ones(l) * -1

        g[inds] = zeros(ls)

        if ls > 0:
            Kcs = self.gram(xc, X[inds])[0]
        if lr > 0:
            Krs = self.gram(X[indr], X[inds]) # kernel of error vectors, support vectors
            print "Krs: %s" %Krs
            Kcr = self.gram(xc, X[indr])[0]
            g[indr] = - ones(lr) + Krs.dot(a[inds]) + ones((lr,1)) * mu

        #test
        #mu = - max(a[indr][inde])

        Kcc = 1
        gc = - 1 + Kcs.dot(a[inds]) + mu
        print "gc: %s" %gc
        # initial calculation for beta
        Q = inv(vstack([hstack([0,ones(ls)]),hstack([ones((ls,1)), Kss])]))


        loop_count = 1

        while gc < e and ac < C - e:
            print "--------------------------" + "increment/decrement loop " + str(loop_count) + "--------------------------"
            print "sum(a): %s" % (sum(a) + ac)
            print "a: %s" % a
            print "a[inds]: %s" %a[inds]
            print "a[indr]: %s" %a[indr]
            print "a[indo]: %s" %a[indo]
            print "a[inde]: %s" %a[inde]
            print "ac: %s"%ac
            print "g: %s"%g
            print "gc: %s" %gc


            # calculate beta
            if ls > 0:
                if ls == 1:
                    Q = inv(vstack([hstack([0,ones(ls)]),hstack([ones((ls,1)), Kss])]))
                n = hstack([1, Kcs])
                beta = - Q.dot(n)
                betas = beta[1:]
                print "beta: %s" %beta
            # calculate gamma
            if lr > 0 and ls > 0:
                gamma = vstack([hstack([1, Kcs]), hstack([ones((lr,1)), Krs])]).dot(beta) + hstack([Kcc, Kcr])
                gammac = gamma[0]
                gammar = gamma[1:]
                print "gammar: %s" % gammar
                print "gammac: %s" % gammac
            elif ls > 0:
                # empty R set
                gammac =hstack([1, Kcs]).dot(beta) + Kcc
                print "gammac: %s" %gammac
            else:
                # empty S set
                gammac = 1
                gammar = ones(lr)
                print "gammar: %s" %gammar
                print "gammac: %s" %gammac


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
                gec = zeros(len(g[inde] > e))
                gec[Ie_plus] = divide(-g[indr][inde][Ie_plus], gammar[inde][Ie_plus])
                gec[Ie_inf] = inf

                for i in range(0, len(gec)):
                    if gec[i] <= e:
                        gec[i] = inf

                gemin = gec.min()
                iemin = where(gec == gemin)

            else: gemin = inf
            if lo > 0:
                Io_minus = gammar[indo] < - e
                Io_inf = gammar[indo] >= - e

                goc = zeros(len(g[indo] > e))
                goc[Io_minus] = divide(-g[indr][indo][Io_minus], gammar[indo][Io_minus])
                goc[Io_inf] = inf

                for i in range(0, len(goc)):
                    if goc[i] <= e:
                        goc[i] = inf
                    if g[indr][indo][i] < 0:
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
            gmin = min([gsmin, gemin, gomin, gcmin, gacmin])
            print "gsmin: %s, gemin: %s, gomin: %s, gcmin: %s, gacmin: %s" % (gsmin, gemin, gomin, gcmin, gacmin)
            print "gmin: %s" %gmin
            print where([gsmin, gemin, gomin, gcmin, gacmin] == gmin)
            imin = where([gsmin, gemin, gomin, gcmin, gacmin] == gmin)[0][0]

            # update a, g,
            print "start:================= update =================="
            print "before update sum(a) + ac: %s" % (sum(a) + ac)
            print "betas*gmin: %s and sum(betas*gmin): %s" % (betas*gmin, sum(betas*gmin))
            print "a[inds]: %s" % a[inds]
            print "ac: %s" % ac

            ac += gmin
            a[inds] = a[inds] + betas*gmin
            if lr > 0: g[indr] = g[indr] + gammar * gmin
            gc = gc + gammac * gmin
            print "after update sum(a): %s" % (sum(a) + ac)
            print "a[inds]: %s" % a[inds]
            print "ac: %s" % ac

            print "end:================= update =================="

            if imin == 0: # min = gsmin => move k from s to r
                # if there are more than 1 minimum, just take 1
                if len(ismin[0]) > 1:
                    ismin = [ismin[0][0]]
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
                    a[len(a)-1] = 0
                else:
                    indo = hstack((indo, False))
                    inde = hstack((inde, True))
                    a[len(a)-1] = C

                #decrement Q, delete row ismin and column ismin

                if ls > 1:
                    ismin = ismin[0][0] + 1
                    for i in range(Q.shape[0]):
                        for j in range(Q.shape[1]):
                            if i != ismin and j != ismin:
                                Q[i][j] = Q[i][j] - Q[i][ismin]*Q[ismin][j]/Q[ismin][ismin]
                    #if debug: print "Q after double loop: %s" % Q
                    Q = delete(Q, ismin, 0)
                    Q = delete(Q, ismin, 1)
                else:
                    Q = inf

            elif imin == 1:
                print "move k from e (r) to s"
                # if there are more than 1 minimum, just take 1
                if len(iemin[0]) > 1:
                    iemin = [iemin[0][0]]

                # get x, a and g
                Xk = X[indr][inde][iemin]
                ak = a[indr][inde][iemin]
                gk = g[indr][inde][iemin]
                Xs_old = X[inds]

                # delete the elements from X,a and g => add it to the end of X,a,g
                indr_ind = [c for c,val in enumerate(indr) if val]
                indr_ind = [val for c, val in enumerate(indr_ind) if inde[c]]
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
                indo = delete(indo, iemin)

                if ls > 0:
                    nk = hstack((1, self.gram(Xk, Xs_old)[0]))
                    betak = - Q.dot(nk)
                    k = 1 - nk.dot(Q).dot(nk)
                    Q = hstack((vstack((Q, zeros(Q.shape[1]))),zeros((Q.shape[0] + 1,1)))) \
                        + 1/k * outer(hstack((betak,1)), hstack((betak,1)))

            elif imin == 2: # min = gemin | gomin => move k from r to s
                print "start:============= move k from o (r) to s ============="
                if len(iomin[0]) > 1:
                    iomin = [iomin[0][0]]
                Xk = X[indr][indo][iomin]
                ak = a[indr][indo][iomin]
                gk = g[indr][indo][iomin]
                Xs_old = X[inds]

                # delete the elements from X,a and g => add it to the end of X,a,g
                indr_ind = [c for c,val in enumerate(indr) if val]
                indr_ind = [val for c, val in enumerate(indr_ind) if indo[c]]

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

                if ls > 0:
                    nk = hstack((1, self.gram(Xk, Xs_old)[0]))
                    betak = - Q.dot(nk)
                    k = 1 - nk.dot(Q).dot(nk)
                    Q = hstack((vstack((Q, zeros(Q.shape[1]))),zeros((Q.shape[0] + 1,1)))) \
                        + 1/k * outer(hstack((betak,1)), hstack((betak,1)))
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
            else:
                Kcs = []
            if lr > 0 and ls > 0:
                Krs = self.gram(X[indr], X[inds])               # kernel of error vectors, support vectors
                Kcr = self.gram(xc, X[indr])[0]

            loop_count += 1
        print "----------------- end incremental loop -----------------"
        print "C: %s" % C
        print "ac: %s" %ac
        self._data.set_X(X)
        self._data.set_alpha(a)
        self._data.add(xc, ac)
        self._data.set_C(C)
        self.predictor()
        print "sum(a) after: %s" % sum(self._data.alpha())
        self.predictor()
        print "a: %s" % self._data.alpha()
        print "as: %s" % self._data.alpha_s()
        #print self._data.Xs()
        if len(self._data.alpha_s()) == 0: sys.exit()
    def bookkeeping(self, a, C):
        return 0
    def perturbc(self, C_new, C_old, a, X):
        e = eps = 1e-5

        inds = np.all([a > e, a < C_old - e], axis=0)           # support vectors indeces
        indr = np.any([a <= e, a >= C_old - e], axis=0)         # error and non-support vectors indeces
        inde = a[indr] >= C_old - e                             # error vectors indeces in R

        # calculate Q and Rs
        Q = vstack((hstack((0, ones(len(a)))), hstack((ones(1,len(a)), self.gram(X, X[inds])))))
        Rs = inv(vstack((hstack((0, ones(len(a)))), hstack((ones(1,len(a)), self.gram(X[inds]))))))

        # create a vector containing the regularization parameter
        # for each example if necessary
        if len(C_new) == 1:             # same regularization parameter for all examples
            C_new = C_new*ones(len(a))
        C = C_old * ones(len(a))
        # compute the regularization sensitivities
        l = C_new-C
        # if there are no error vectors initially...
        if (len(a[indr][inde]) == 0):
           # find the subset of the above examples that could become error vectors
           delta_p = divide(C-a,l)
           delta_p[delta_p <= 0] = inf

           # determine the minimum acceptable change in p and adjust the regularization parameters
           p = min(delta_p)[0]
           C = C + l*p

           # if one example becomes an error vector, perform the necessary bookkeeping
           if (p < 1):
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
        while (p < 1):
           perturbations = perturbations + 1

           # compute beta and gamma
           if (num_MVs > 0):

              v = zeros(num_MVs+1)
              if (p < 1 - eps):
                  v[1] = - Syl - sum(a)/(1-p)
              else:
                 v[1] = - Syl
              v[1:] = -SQl[inds]
              beta = Rs*v
              ind_temp = indr
              if (len(a[ind_temp]) > 0):
                 gamma = Q[:,ind_temp].dot(beta) + SQl(ind_temp)
           else:
              beta = 0
              gamma = SQl

           # minimum acceptable parameter change
           min_delta_p, indss, cstatus, nstatus = self.min_delta_p_c(p,gamma,beta,l)

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
           #TODO: Bookeeping
           indco = self.bookkeeping(indss,cstatus,nstatus)

           # update SQl and Syl when the status of indss changes from MARGIN to ERROR
           if cstatus == self._MARGIN and nstatus == self._ERROR:
               SQl = SQl + Q[indco,:].dot(l(indss))
               Syl = Syl + l(indss)

           # set g(ind{MARGIN}) to zero
           g[inds] = 0

           # update Rs and Q if necessary
           if (nstatus == self._MARGIN):

              num_MVs = num_MVs + 1
              if (num_MVs > 1):

                 # compute beta and gamma for indss
                 beta = -Rs*Q[:,indss]
                 gamma = self.gram(X[:,indss], X[:,indss]) + Q[:,indss].dot(beta)


              #updateRQ(beta,gamma,indss)

           else:
               if cstatus == self._MARGIN:

                  # compress Rs and Q
                  num_MVs = num_MVs - 1
                  #TODO: expand Rs and Q
                  #updateRQ(indco)

           # update SQl and Syl when the status of indss changes from ERROR to MARGIN
           if cstatus == self._ERROR and nstatus == self._MARGIN:
              SQl = SQl - Q[num_MVs+1,:].dot(l[indss])
              Syl = Syl - l[indss]

           if p >= disp_p_delta*disp_p_count:
              disp_p_count = disp_p_count + 1
              #s = sprintf('p = #.2f',p)
              #disp(s)

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
