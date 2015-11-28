def increment(self, xc, init_ac=0):

        e = self._data._e
        # initialize X, a, C, g, indeces, kernel values
        X = self._data.X()                                  # data points
        C = self._data.C()
        a = self._data.alpha()
        ac = init_ac
        #if self._a_history:
        #    a_history = []
        #    a_history.append(a)


        inds = [i for i, bool in enumerate(np.all([a > e, a < C - e], axis=0)) if bool]
        indr = [i for i, bool in enumerate(np.any([a <= e, a >= C - e], axis=0)) if bool]

        inde = [i for i, bool in enumerate(a >= C - e) if bool]
        indo = [i for i, bool in enumerate(a <= e) if bool]

        inde_bool = np.asarray([True if val in inde else False for i, val in enumerate(indr) ])
        indo_bool = np.asarray([True if val in indo else False for i, val in enumerate(indr) ])

        l = len(a)
        ls = len(a[inds])                               # support vectors length
        lr = len(a[indr])                               # error and non-support vectors length
        le = len(a[inde])                               # error vectors lenght
        lo = len(a[indo])                               # non-support vectors
        X_all = zeros((l + 1, X.shape[1]))
        X_all[0,] = xc
        X_all[1:,] = X
        if self._data.K_X() != None:
            K_X_old = self._data.K_X()
            K_xc_X = self.gram(xc, X)[0]
            K_X_all = zeros((l+1, l+1))
            K_X_all[1:, 1:] = K_X_old
            K_X_all[0, 0] = 1.0
            K_X_all[0, 1:] = K_xc_X
            K_X_all[1:, 0] = K_xc_X
        else:
            # kernel of all data points including the new one
            K_X_all = self.gram(X_all)

        K_X = K_X_all[1:, 1:]
        # kernel of support vectors
        Kss = K_X[:,inds]
        Kss = Kss[inds,:]

        # calculate mu according to KKT-conditions
        mu_all = - K_X[inds,:].dot(a)
        #print "mu_all: %s" % mu_all
        mu = np.mean(mu_all)
        print "mu: %s "% mu
        g = ones(l) * -1

        g[inds] = zeros(ls)

        if ls > 0:
            Kcs = K_X_all[0, 1:][inds]
            #print "Kcs: %s" % Kcs
        if lr > 0:
            Krs = K_X[:, inds][indr, :]
            Kr = K_X[indr, :]
            Kcr = K_X_all[0, 1:][indr]
            g[indr] = Kr.dot(a) + ones((lr,1)) * mu
        Kcc = 1
        gc = Kcs.dot(a[inds]) + mu
        #print "gc: %s" % gc
        Q = ones((ls+1, ls+1))
        Q[0, 0] = 0
        Q[1:, 1:] = Kss
        #print "Q: %s" % Q
        #print "Q: %s" % Q
        try:
            R = inv(Q)
        except np.linalg.linalg.LinAlgError:
            print "singular matrix"
            R = inv(Q + diag(ones(ls+1) * 1e-2))
            sys.exit()

        loop_count = 1

        while gc < e and ac < C - e:

            print "--------------loop_count: %s" % loop_count
            print "indo: %s" % indo
            #print "indo: %s" % indo
            #print "a[0]: %s" % a[0]
            #print "g[0]: %s" % g[0]
            #print "ac: %s" % ac
            #print "a[inds]: %s" % a[inds]
            #print inds
            # calculate beta
            if ls > 0:
                if ls == 1:

                    R = ones((2, 2))
                    R[1,1] = 0
                    R[0,0] = -1

                n = ones(ls+1)
                n[1:] = Kcs
                beta = - R.dot(n)
                betas = beta[1:]
                KXs = self.gram(X[inds])

                #print KXs
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
                #print "gammar: %s" % gammar
                #print "gr: %s" % g[indr]

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
                #print betas
                IS_plus = betas > e

                IS_minus = betas < - e
                #print "betas[IS_plus]: %s" % betas[IS_plus]
                #print "a[IS_plus]: %s" % a[inds][IS_plus]
                #print "a[IS_minus]: %s" % a[inds][IS_minus]
                #print "betas[IS_minus]: %s" % betas[IS_minus]
                IS_zero = np.any([betas <= e, betas >= -e], axis=0)

                gsmax = zeros(ls)
                gsmax[IS_zero] = ones(len(betas[IS_zero])) * inf
                gsmax[IS_plus] = ones(len(betas[IS_plus]))*C-a[inds][IS_plus]
                gsmax[IS_minus] = - a[inds][IS_minus]
                gsmax = divide(gsmax, betas)
                #print "gsmax: %s" % gsmax
                #print "gsmax[0]: %s" % gsmax[0]
                #print "gsmax[1]: %s" % gsmax[1]
                gsmin = absolute(gsmax).min()
                ismin = where(absolute(gsmax) == gsmin)
                print "ismin: %s" % ismin
            else: gsmin = inf

            #case 2: Some g_i in R reaches zero
            if le > 0:
                Ie_plus = gammar[inde_bool] > e
                Ie_inf = gammar[inde_bool] <= e
                gec = zeros(le)
                gec[Ie_plus] = divide(-g[indr][inde_bool][Ie_plus], gammar[inde_bool][Ie_plus])
                gec[Ie_inf] = inf
                #if lr > 169: print "gec[168]: %s" % (gec[168] <= -0.0)
                for i in range(0, len(gec)):
                    if gec[i] <= 0:
                        gec[i] = inf
                if len(gec[gec < inf]) < 1:
                    gemin = inf
                else:
                    gemin = gec.min()
                    iemin = where(gec == gemin)
                #print "gec[168]: %s" % gec[168]
                #print "what if update: %s" % (g[indr][168] + gammar[168] * gec[168])
            else: gemin = inf
            if lo > 0:
                print "g[99]: %s" % g[99]
                print "a[99]: %s" % a[99]
                ind_99 = where(indr==99)[0]
                print ind_99
                Io_minus = gammar[indo_bool] < - e
                Io_inf = gammar[indo_bool] >= - e

                goc = zeros(lo)
                goc[Io_minus] = divide(-g[indr][indo_bool][Io_minus], gammar[indo_bool][Io_minus])
                goc[Io_inf] = inf

                for i in range(0, len(goc)):
                    if goc[i] <= 0:
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
            print "all_deltas: %s" % all_deltas

            gmin = min(all_deltas)

            for i, val in enumerate(all_deltas):
                if val == gmin:
                    imin = i
                    break
            # update a, g
            #print "g[indo] >= 0: %s"% g[indo][g[indo] > 0]

            if ls > 0:
                ac += gmin
                a[inds] += betas*gmin
            if lr > 0:
                g[indr] += gammar * gmin
                #print "gammar: %s" % gammar
            gc = gc + gammac * gmin
            # else??
            print "imin: %s" % imin
            print "after update g[99]: %s, a[99]: %s" % (g[99], a[99])
            if imin == 0: # min = gsmin => move k from s to r
                # if there are more than 1 minimum, just take 1
                if len(ismin[0]) > 1:
                    ismin = [ismin[0][0]]
                ak = a[inds][ismin]

                # delete the elements from X,a and g => add it to the end of X,a,g
                #print "ismin: %s" % ismin
                #print "inds: %s" % inds
                ind_del = inds[ismin[0]]
                print "ind_del: %s" % ind_del
                inds.remove(ind_del)
                indr.append(ind_del)
                if ak < e:
                    indo.append(ind_del)
                    lo +=1
                else:
                    inde.append(ind_del)
                    le +=1

                lr +=1
                #decrement R, delete row ismin and column ismin

                if ls > 1:
                    #if isinstance(ismin[0], )
                    if type(ismin[0]).__name__ == "ndarray":
                        ismin = ismin[0][0] + 1
                    else: ismin = ismin[0] + 1
                    for i in range(ls + 1):
                        for j in range(ls + 1):
                            if i != ismin and j != ismin:
                                R[i][j] = R[i][j] - R[i][ismin]*R[ismin][j]/R[ismin][ismin]

                    new_index = []
                    R_new = zeros((ls,ls))
                    R_new[0:ismin, 0:ismin] = R[0:ismin, 0:ismin]
                    R_new[ismin:, 0:ismin] = R[ismin+1:,0:ismin]
                    R_new[0:ismin, ismin:] = R[0:ismin, ismin+1:]
                    R_new[ismin:, ismin:] = R[ismin+1:, ismin+1:]
                    R = R_new
                else:
                    R = inf
                ls -= 1

            elif imin == 1:
                # if there are more than 1 minimum, just take 1
                if len(iemin[0]) > 1:
                    iemin = [iemin[0][0]]


                # delete the elements from X,a and g => add it to the end of X,a,g
                try:
                    ind_del = np.asarray(indr)[inde_bool][iemin[0]][0]
                except:
                    ind_del = np.asarray(indr)[inde_bool][iemin[0]]
                print ind_del
                if ls > 0:
                    nk = ones(ls+1)
                    nk[1:] = K_X[ind_del][inds]
                    betak = - R.dot(nk)
                    k = 1 - nk.dot(R).dot(nk)

                    betak1 = ones(ls + 2)
                    betak1[:-1] = betak
                    R_old = R
                    R = zeros((ls +2, ls +2))
                    R[:-1,:-1] = R_old
                    R += 1/k * outer(betak1, betak1)
                inds.append(ind_del)

                indr.remove(ind_del)

                inde.remove(ind_del)
                ls += 1
                lr -= 1
                le -= 1
                #g[ind_del] = 0


            elif imin == 2: # min = gemin | gomin => move k from r to s
                #print "gomin => move k from r to s"
                if len(iomin[0]) > 1:
                    iomin = [iomin[0][0]]

                # delete the elements from X,a and g => add it to the end of X,a,g
                ind_del = np.asarray(indr)[indo_bool][iomin[0]][0]
                if ls > 0:
                    nk = ones(ls+1)
                    nk[1:] = K_X[ind_del][inds]
                    betak = - R.dot(nk)
                    k = 1 - nk.dot(R).dot(nk)
                    betak1 = ones(ls+2)
                    betak1[:-1] = betak
                    R_old = R
                    R = zeros((ls+2, ls+2))
                    R[:-1,:-1] = R_old
                    R += 1/k * outer(betak1, betak1)

                indo.remove(ind_del)
                indr.remove(ind_del)
                inds.append(ind_del)
                #g[ind_del] = 0
                lo -= 1
                lr -= 1
                ls += 1
            else: # k = c => terminate
                #print "before termination g[0]: %s, a[0]: %s" % (g[0], a[0])
                break
            inde_bool = np.asarray([True if val in inde else False for i, val in enumerate(indr) ])
            indo_bool = np.asarray([True if val in indo else False for i, val in enumerate(indr)])

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
            '''
            #if self._a_history: a_history.append(a)
            #if len(g[indo][g[indo] > 0]) > 0:
                #g_nz = g[indo][g[indo] < 0][0]
                #print "ge > 0: %s" % g_nz


                #print "which index? %s -> %s -> %s" % (where(g[indr] == g_nz)[0], indr[where(g[indr] == g_nz)[0]], indr.index(indr[where(g[indr] == g_nz)[0]]))

                print "lr: %s" % lr
                print "g[indr][168]: %s" % g[indr][168]
                print "g[71]: %s" % g[71]
            '''
                #print "index gr > 0: %s" % where(g[indr] == g[indr][g[indr] > 0])
                #sys.exit()
            if len(g[inde][g[inde] > 0]) > 0:
                g_nz = g[inde][g[inde] > 0][0]
                print "ge > 0: %s" % g_nz
                #sys.exit()
                #print "which index? %s -> %s -> %s" % (where(g == g_nz)[0], indr[where(g[indr] == g_nz)[0]], indr.index(71))
            if len(g[indo][g[indo] < 0]) > 0:
                g_nz = g[indo][g[indo] < 0][0]
                print "go < 0: %s" % g_nz
                print "index: %s, %s" % (where(g[indr] == g_nz)[0], indr[where(g[indr] == g_nz)[0]])
                print "g[99]: %s" % g[99]
                print "g[indr][211]: %s " % g[indr][211]
                #sys.exit()
            '''
            check = True
            if check:
                #print "a[inde] < 0.98? %s" % a[inde][a[inde] < 0.98]
                g_nz = g[indo][g[indo] > 0][0]
                print "g[inde] > 0: %s"% g[inde][g[inde] > 0]
                print "g[indo] > 0: %s"% g[indo][g[indo] > 0]
                if len(g[inde][g[inde] > 0]) > 0: sys.exit()
                if len(g[indo][g[indo] > 0]) > 0: sys.exit()
            '''
        if len(g[indo][g[indo] > 0]) < 0:
            #print "after termination g[0]: %s, a[0]: %s" % (g[0], a[0])

            g_nz = g[indo][g[indo] < 0][0]
            #print "ge > 0: %s" % g_nz
            print "indo: %s" % indo
            print "which index? %s -> %s -> %s" % (where(g[indr] == g_nz)[0], indr[where(g[indr] == g_nz)[0]], indr.index(indr[where(g[indr] == g_nz)[0]]))
        '''
        check = True
        if check:
            print "a[inde] < 0.98? %s" % a[inde][a[inde] < 0.98]
            print "g[inde] > 0: %s"% g[inde][g[inde] > 0]
            if len(g[inde][g[inde] > 0]) > 0: sys.exit()
        '''
        if len(g[inde][g[inde] > 0]) > 0:
            g_nz = g[inde][g[inde] > 0][0]
            print "ge > 0: %s" % g_nz
            sys.exit()
            #print "which index? %s -> %s -> %s" % (where(g == g_nz)[0], indr[where(g[indr] == g_nz)[0]], indr.index(71))
        if len(g[indo][g[indo] < 0]) > 0:
            g_nz = g[indo][g[indo] < 0][0]
            print "go > 0: %s" % g_nz

            sys.exit()

    # update X, a
        self._data.set_X(X_all)
        alphas = zeros(len(a) + 1)
        print "gc: %s" % gc
        #print "x_c: %s" % xc
        print "ac: %s" % ac
        #print "ainds: %s" % a[inds]
        alphas[0] = ac
        alphas[1:] = a
        self._data.set_alpha(alphas)
        # add x_c and a_c
        #self._data.add(xc, ac)
        # set C if necessary
        self._data.set_C(C)
        '''
        K_col = K_X_all[:, 0]
        K_col = K_col.reshape(len(K_col),1)
        K_X_all = delete(K_X_all, 0, axis=1)
        K_X_all = hstack((K_X_all, K_col))
        K_row = K_X_all[0, :]
        K_X_all = delete(K_X_all, 0, axis=0)
        K_X_all = vstack((K_X_all, K_row))
        '''
        #sys.exit()
        self._data.set_K_X(K_X_all)
        #print "a saved: %s" % self._data.alpha()