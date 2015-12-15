['../evaluation_tmp.py', '20000']
mnist classes = 2
size: 20000
(5268,)
(14732,)
data size: 20000, nu: 0.2, gamma: 1
============ 1. Fold of CV ============
1) Incremental OCSVM
0 data points processed
1000 data points processed
2000 data points processed
3000 data points processed
4000 data points processed
5000 data points processed
6000 data points processed
7000 data points processed
8000 data points processed
9000 data points processed
10000 data points processed
11000 data points processed
12000 data points processed
None
Confusion matrix:
Prediction      1
Target           
-1           5268
 1          14732
precision: 0.7366, recall: 1.0, f1-score: 0.848324311874
Number of support vectors: 16000
-----------
2) cvxopt-OCSVM
Confusion matrix:
Prediction     -1     1
Target                 
-1           4188  1080
 1          11812  2920
precision: 0.73, recall: 0.198207982623, f1-score: 0.31176596199
Number of support vectors: 16000
---------
3) sklearn-OCSVM
Confusion matrix:
Prediction    -1     1
Target                
-1          3331  1937
 1          9469  5263
Number of support vectors: 16000
precision: 0.730972222222, recall: 0.357249524844, f1-score: 0.479937990151
========================================
Average Incremental OCSVM results:
precision: 0.7366, recall: 1.0, f1-score: 0.848324311874
Average cvxopt-OCSVM results:
precision: 0.73, recall: 0.198207982623, f1-score: 0.31176596199
Average sklearn-OCSVM results:
precision: 0.730972222222, recall: 0.357249524844, f1-score: 0.479937990151
Wrote profile results to evaluation_tmp.py.lprof
Timer unit: 1e-06 s

Total time: 28788.4 s
File: ../ocsvm.py
Function: increment at line 97

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    97                                               @profile
    98                                               def increment(self, Xc, init_ac=0, break_count=-1):
    99                                           
   100                                                   # epsilon
   101         1            7      7.0      0.0          e = self._data._e
   102         1            4      4.0      0.0          mu = 0
   103         1            3      3.0      0.0          imin = None
   104                                           
   105                                                   # initialize existing X, coefficients a, C
   106         1            8      8.0      0.0          X_origin = self._data.X()
   107         1            8      8.0      0.0          K_X_origin = self._data.K_X()
   108         1            7      7.0      0.0          n_data = X_origin.shape[0]
   109         1            4      4.0      0.0          n_feature = X_origin.shape[1]
   110                                           
   111         1            8      8.0      0.0          C = self._data.C()
   112         1            7      7.0      0.0          a_origin = self._data.alpha()
   113                                           
   114                                                   # number of new incremental points
   115         1            4      4.0      0.0          n_new = Xc.shape[0]
   116                                           
   117                                                   # number of all (new and existing) points
   118         1            4      4.0      0.0          n_all = n_data + n_new
   119                                           
   120                                                   # concatenate all new points with all existing points
   121         1           31     31.0      0.0          X = empty((n_new + n_data, n_feature))
   122         1        68206  68206.0      0.0          X[0:n_new, :] = Xc
   123         1        17873  17873.0      0.0          X[n_new:, :] = X_origin
   124                                           
   125                                                   # create kernel matrix for all new and existing points
   126                                           
   127                                                   # create of all data points
   128         1            7      7.0      0.0          if K_X_origin == None:
   129         1     48051946 48051946.0      0.2              K_X = self.gram(X)
   130                                                   else:
   131                                                       K_X = empty((n_all, n_all))
   132                                                       K_X[n_new:, n_new:] = K_X_origin
   133                                                       K_X_new = self.gram(Xc, X_origin)
   134                                                       K_X[0:n_new, :] = K_X_new
   135                                                       K_X[:, 0:n_new] = K_X_new.T
   136                                           
   137                                                   # creating coefficient vector alpha for all data points
   138         1           30     30.0      0.0          a = empty(n_all)
   139         1           28     28.0      0.0          a[n_new:] = a_origin
   140         1           43     43.0      0.0          a[:n_new] = init_ac
   141                                           
   142                                                   # creating gradient vector
   143         1           45     45.0      0.0          g = zeros(n_all)
   144                                           
   145                                                   # create sensitivity vector
   146         1            8      8.0      0.0          gamma = empty(n_all)
   147         1            4      4.0      0.0          check_gradient = False
   148                                                   # loop through all new points to increment
   149     12632        68713      5.4      0.0          for x_count in range(n_new):
   150     12631        63240      5.0      0.0              if x_count % 1000 == 0:
   151        13          202     15.5      0.0                  print "%s data points processed" % x_count
   152                                                       #print "--------- START %s ---------" % x_count
   153                                           
   154     12631        53113      4.2      0.0              if x_count == break_count:
   155                                                           self._data.set_X(X)
   156                                                           self._data.set_alpha(a)
   157                                                           self._data.set_C(C)
   158                                                           self._data.set_K_X(K_X)
   159                                                           self.rho()
   160                                                           return False
   161                                           
   162                                                       # initialize X, a, C, g, indices, kernel values
   163     12631        57581      4.6      0.0              start_origin = n_new - x_count
   164     12631        47570      3.8      0.0              start_new = start_origin - 1
   165                                           
   166     12631        45473      3.6      0.0              if x_count == 0:
   167         1            4      4.0      0.0                  inds = []
   168         1            4      4.0      0.0                  indr = []
   169         1            3      3.0      0.0                  inde = []
   170         1            4      4.0      0.0                  indo = []
   171      3370        12391      3.7      0.0                  for i in range(n_new, n_all):
   172      3369        18848      5.6      0.0                      if e < a[i] < C - e:
   173      3369        13388      4.0      0.0                          inds.append(i)
   174                                                               else:
   175                                                                   indr.append(i)
   176                                                                   if a[i] <= e:
   177                                                                       indo.append(i)
   178                                                                   else:
   179                                                                       inde.append(i)
   180                                           
   181         1            6      6.0      0.0                  ls = len(inds)                               # support vectors length
   182         1            4      4.0      0.0                  lr = len(indr)                               # error and non-support vectors length
   183         1            4      4.0      0.0                  le = len(inde)                               # error vectors lenght
   184         1            4      4.0      0.0                  lo = len(indo)
   185                                                           #mu_old = mu
   186         1          307    307.0      0.0                  mu = - K_X[inds[0], :][start_origin:].dot(a[start_origin:])
   187         1            4      4.0      0.0                  if lr > 0:
   188                                                               g[indr] = K_X[indr, :][:, start_origin:].dot(a[start_origin:]) + mu
   189                                                           # calculate mu according to KKT-conditions
   190                                           
   191                                           
   192     12631      3968688    314.2      0.0              c_inds = [start_new] + inds
   193                                           
   194                                                       # kernel of support vectors
   195                                                       #Kss = K_X[:, inds][inds, :]
   196                                                       #print "difference indo: %s" % unique(round(K_X[indo, :][:, start_origin:].dot(a[start_origin:]) + mu - g[indo],6))
   197                                                       #check_gradient = True
   198                                                       #if check_gradient:
   199                                                           #g[indr] = K_X[indr, :][:, start_origin:].dot(a[start_origin:]) + mu
   200                                                           #g[indo] += K_X[indo[0], :][start_origin:].dot(a[start_origin:]) + mu - g[indo[0]]
   201                                                           #check_gradient = False
   202                                                           #print "difference indo: %s" % unique(round(K_X[indo, :][:, start_origin:].dot(a[start_origin:]) + mu - g[indo],6))
   203     12631        57878      4.6      0.0              if ls > 0:
   204     12631      8968522    710.0      0.0                  gc = K_X[start_new, start_origin:].dot(a[start_origin:]) + mu
   205                                           
   206     12631        68391      5.4      0.0              ac = a[start_new]
   207                                           
   208     12631        48200      3.8      0.0              if x_count == 0:
   209         1        59895  59895.0      0.0                  Q = ones((ls+1, ls+1))
   210         1           11     11.0      0.0                  Q[0, 0] = 0
   211                                                           #Kss = self.gram(X[inds])
   212      3370        16056      4.8      0.0                  inds_row = [[i] for i in inds]
   213         1      1772247 1772247.0      0.0                  Q[1:, 1:] = K_X[inds_row, inds]
   214         1            6      6.0      0.0                  try:
   215         1     14077830 14077830.0      0.0                      R = inv(Q)
   216                                                           except np.linalg.linalg.LinAlgError:
   217                                                               x = 1e-11
   218                                                               found = False
   219                                                               print "singular matrix"
   220                                                               while not found:
   221                                                                   try:
   222                                                                       R = inv(Q + diag(ones(ls+1) * x))
   223                                                                       found = True
   224                                                                   except np.linalg.linalg.LinAlgError:
   225                                                                       x = x*10
   226     12631        49412      3.9      0.0              loop_count = 1
   227                                                       #print "gc: %s, ac: %s" % (gc, ac)
   228     12631       188013     14.9      0.0              while gc < e and ac < C - e:
   229     12631        49006      3.9      0.0                  if ls == 0: check_gradient = True
   230                                                           #print "-------------------- incremental %s-%s ---------" % (x_count, loop_count)
   231                                           
   232     12631        48282      3.8      0.0                  if ls > 0:
   233     12631     32474464   2571.0      0.1                      n = K_X[start_new, :][c_inds]
   234     12631   2166742848 171541.7      7.5                      beta = - R.dot(n)
   235     12631       143105     11.3      0.0                      betas = beta[1:]
   236                                           
   237                                                           # calculate gamma
   238     12631        57510      4.6      0.0                  if lr > 0 and ls > 0:
   239                                                               gamma_tmp = K_X[:, c_inds][start_new:]
   240                                                               gamma_tmp[:, 0] = 1
   241                                                               gamma[start_new:] = gamma_tmp.dot(beta) + K_X[start_new, :][start_new:]
   242                                                               gammac = gamma[start_new]
   243                                           
   244     12631        54075      4.3      0.0                  elif ls > 0:
   245                                                               # empty R set
   246     12631     45750351   3622.1      0.2                      gammac = K_X[start_new, :][c_inds].dot(beta) + 1
   247                                           
   248                                                           else:
   249                                                               # empty S set
   250                                                               gammac = 1
   251                                                               gamma[indr] = 1
   252                                                               #gamma[indo] = -1
   253                                           
   254                                                           # accounting
   255                                                           #case 1: Some alpha_i in S reaches a bound
   256     12631        60679      4.8      0.0                  if ls > 0:
   257     12631       454420     36.0      0.0                      IS_plus = betas > e
   258     12631       327472     25.9      0.0                      IS_minus = betas < - e
   259     12631       809848     64.1      0.0                      gsmax = ones(ls)*inf
   260                                                               #if np.isnan(np.min(gsmax)):
   261                                                               #    gsmax = ones(ls)*inf
   262     12631     26776798   2119.9      0.1                      gsmax[IS_plus] = -a[inds][IS_plus] + C
   263     12631     27234541   2156.2      0.1                      gsmax[IS_minus] = - a[inds][IS_minus]
   264                                                               #gsmax[IS_plus] = -a[inds][IS_plus]
   265                                                               #gsmax[IS_plus] += C
   266                                                               #gsmax[IS_minus] = - a[inds][IS_minus]
   267     12631      1012324     80.1      0.0                      gsmax = divide(gsmax, betas)
   268     12631     16820837   1331.7      0.1                      gsmin = min(absolute(gsmax))
   269                                                               #print where(absolute(gsmax) == gsmin)
   270     12631      1587478    125.7      0.0                      ismin = where(absolute(gsmax) == gsmin)[0][0]
   271                                           
   272                                                           else: gsmin = inf
   273                                           
   274                                                           #case 2: Some g_i in E reaches zero
   275     12631        56296      4.5      0.0                  if le > 0:
   276                                           
   277                                                               gamma_inde = gamma[inde]
   278                                                               g_inde = g[inde]
   279                                                               Ie_plus = gamma_inde > e
   280                                           
   281                                                               if len(g_inde[Ie_plus]) > 0:
   282                                                                   gec = divide(-g_inde[Ie_plus], gamma_inde[Ie_plus])
   283                                                                   gec[gec <= 0] = inf
   284                                                                   gemin = min(gec)
   285                                                                   if gemin < inf:
   286                                                                       iemin = where(gec == gemin)[0][0]
   287                                                               else: gemin = inf
   288     12631        55883      4.4      0.0                  else: gemin = inf
   289                                                           #case 2: Some g_i in O reaches zero
   290     12631        54976      4.4      0.0                  if lo > 0 and ls > 0:
   291                                                               gamma_indo = gamma[indo]
   292                                                               g_indo = g[indo]
   293                                                               Io_minus = gamma_indo < - e
   294                                                               if len(g_indo[Io_minus]) > 0:
   295                                                                   goc = divide(-g_indo[Io_minus], gamma_indo[Io_minus])
   296                                                                   goc[goc <= 0] = inf
   297                                                                   goc[g_indo[Io_minus] < 0] = inf
   298                                                                   gomin = min(goc)
   299                                                                   if gomin < inf:
   300                                                                       iomin = where(goc == gomin)[0][0]
   301                                                               else: gomin = inf
   302     12631        55026      4.4      0.0                  else: gomin = inf
   303                                           
   304                                                           # case 3: gc becomes zero
   305     12631       111752      8.8      0.0                  if gammac > e: gcmin = - gc/gammac
   306                                                           else: gcmin = inf
   307                                           
   308                                                           # case 4
   309     12631        80666      6.4      0.0                  if ls > 0: gacmin = C - ac
   310                                                           else: gacmin = inf
   311                                           
   312                                                           # determine minimum largest increment
   313     12631        73912      5.9      0.0                  all_deltas = [gsmin, gemin, gomin, gcmin, gacmin]
   314     12631       101815      8.1      0.0                  gmin = min(all_deltas)
   315     12631       314064     24.9      0.0                  imin = where(all_deltas == gmin)[0][0]
   316                                                           # update a, g
   317     12631        56831      4.5      0.0                  if ls > 0:
   318     12631        81747      6.5      0.0                      mu += beta[0]*gmin
   319     12631        59526      4.7      0.0                      ac += gmin
   320     12631     48521033   3841.4      0.2                      a[inds] += betas*gmin
   321                                                           else:
   322                                                               mu += gmin
   323     12631        66546      5.3      0.0                  if lr > 0:
   324                                                               g[indr] += gamma[indr] * gmin
   325     12631        75273      6.0      0.0                  gc += gammac * gmin
   326     12631        98730      7.8      0.0                  if imin == 0: # min = gsmin => move k from s to r
   327                                                               # if there are more than 1 minimum, just take 1
   328                                                               ak = a[inds][ismin]
   329                                           
   330                                                               # delete the elements from X,a and g
   331                                                               # => add it to the end of X,a,g
   332                                                               ind_del = inds[ismin]
   333                                                               inds.remove(ind_del)
   334                                                               c_inds = [start_new] + inds
   335                                                               indr.append(ind_del)
   336                                                               if ak < e:
   337                                                                   indo.append(ind_del)
   338                                                                   lo += 1
   339                                                               else:
   340                                                                   inde.append(ind_del)
   341                                                                   le += 1
   342                                           
   343                                                               lr += 1
   344                                                               #decrement R, delete row ismin and column ismin
   345                                           
   346                                                               if ls > 2:
   347                                                                   ismin += 1
   348                                                                   R_new = zeros((ls,ls))
   349                                                                   R_new[0:ismin, 0:ismin] = R[0:ismin, 0:ismin]
   350                                                                   R_new[ismin:, 0:ismin] = R[ismin+1:,0:ismin]
   351                                                                   R_new[0:ismin, ismin:] = R[0:ismin, ismin+1:]
   352                                                                   R_new[ismin:, ismin:] = R[ismin+1:, ismin+1:]
   353                                                                   betak = zeros(ls)
   354                                                                   betak[:ismin] = R[ismin, :ismin]
   355                                                                   betak[ismin:] = R[ismin, ismin+1:]
   356                                                                   R_new -= outer(betak, betak)/R[ismin,ismin]
   357                                                                   R = R_new
   358                                                               elif ls == 2:
   359                                                                   R = ones((2, 2))
   360                                                                   R[1,1] = 0
   361                                                                   R[0,0] = -1
   362                                                               else:
   363                                                                   R = inf
   364                                                               ls -= 1
   365                                           
   366     12631        66657      5.3      0.0                  elif imin == 1:
   367                                                               # delete the elements from X,a and g => add it to the end of X,a,g
   368                                                               ### old version find index to delete
   369                                                               #Ieplus_l = [i for i,b in enumerate(Ie_plus) if b]
   370                                                               #ind_del = inde[Ieplus_l[iemin]]
   371                                                               ### old version find index to delete
   372                                                               ind_del = np.asarray(inde)[Ie_plus][iemin]
   373                                                               if ls > 0:
   374                                                                   nk = K_X[ind_del, :][[ind_del] + inds]
   375                                                                   betak = - R.dot(nk)
   376                                                                   betak1 = ones(ls + 2)
   377                                                                   betak1[:-1] = betak
   378                                                                   R_old = R
   379                                                                   R = 1/k * outer(betak1, betak1)
   380                                                                   R[:-1,:-1] += R_old
   381                                                               else:
   382                                                                   R = ones((2, 2))
   383                                                                   R[1,1] = 0
   384                                                                   R[0,0] = -1
   385                                                               inds.append(ind_del)
   386                                                               c_inds = [start_new] + inds
   387                                                               indr.remove(ind_del)
   388                                                               inde.remove(ind_del)
   389                                                               ls += 1
   390                                                               lr -= 1
   391                                                               le -= 1
   392                                           
   393     12631        63787      5.1      0.0                  elif imin == 2: # min = gemin | gomin => move k from r to s
   394                                           
   395                                                               # delete the elements from X,a and g => add it to the end of X,a,g
   396                                           
   397                                                               ### old version find index to delete
   398                                                               #Io_minus_l = [i for i,b in enumerate(Io_minus) if b]
   399                                                               #ind_del = indo[Io_minus_l[iomin]]
   400                                                               ### old version find index to delete
   401                                                               ind_del = np.asarray(indo)[Io_minus][iomin]
   402                                                               if ls > 0:
   403                                                                   nk = ones(ls+1)
   404                                                                   nk[1:] = K_X[ind_del,:][inds]
   405                                                                   betak = - R.dot(nk)
   406                                                                   k = 1 - nk.dot(R).dot(nk)
   407                                                                   betak1 = ones(ls+2)
   408                                                                   betak1[:-1] = betak
   409                                                                   R_old = R
   410                                                                   R = 1/k * outer(betak1, betak1)
   411                                                                   R[:-1,:-1] += R_old
   412                                                               else:
   413                                                                   R = ones((2, 2))
   414                                                                   R[1,1] = 0
   415                                                                   R[0,0] = -1
   416                                           
   417                                                               indo.remove(ind_del)
   418                                                               indr.remove(ind_del)
   419                                                               inds.append(ind_del)
   420                                                               c_inds = [start_new] + inds
   421                                                               lo -= 1
   422                                                               lr -= 1
   423                                                               ls += 1
   424     12631        64306      5.1      0.0                  elif imin == 3:
   425                                                               '''
   426                                                               if ls > 0:
   427                                                                   nk = ones(ls+1)
   428                                                                   nk[1:] = K_X[start_new, :][inds]
   429                                                                   betak = - R.dot(nk)
   430                                                                   k = 1 - nk.dot(R).dot(nk)
   431                                                                   betak1 = ones(ls + 2)
   432                                                                   betak1[:-1] = betak
   433                                                                   R_old = R
   434                                                                   R = zeros((ls +2, ls +2))
   435                                                                   R[:-1,:-1] = R_old
   436                                                                   R += 1/k * outer(betak1, betak1)
   437                                                               else:
   438                                                                   R = ones((2, 2))
   439                                                                   R[1,1] = 0
   440                                                                   R[0,0] = -1
   441                                                               '''
   442     12631        54883      4.3      0.0                      break
   443                                                           else:
   444                                                               break
   445                                                           loop_count += 1
   446                                           
   447     12631        63794      5.1      0.0              a[start_new] = ac
   448     12631        62100      4.9      0.0              g[start_new] = gc
   449     12631        68462      5.4      0.0              if ac < e:
   450                                                           indr.append(start_new)
   451                                                           indo.append(start_new)
   452                                                           lr += 1
   453                                                           lo += 1
   454     12631        75887      6.0      0.0              elif ac > C - e:
   455                                                           indr.append(start_new)
   456                                                           inde.append(start_new)
   457                                                           lr += 1
   458                                                           le += 1
   459                                                       else:
   460     12631        91298      7.2      0.0                  inds.append(start_new)
   461     12631        58219      4.6      0.0                  g[start_new] = 0
   462     12631        72991      5.8      0.0                  if len(inds) == 1:
   463                                                               R = ones((2, 2))
   464                                                               R[1,1] = 0
   465                                                               R[0,0] = -1
   466                                                           else:
   467     12631        88348      7.0      0.0                      if R.shape[0] != len(inds) + 1:
   468     12631       321401     25.4      0.0                          nk = ones(ls+1)
   469     12631     33557102   2656.7      0.1                          nk[1:] = K_X[start_new, :][inds[:-1]]
   470     12631   2165957337 171479.5      7.5                          betak = - R.dot(nk)
   471     12631      4747101    375.8      0.0                          k = 1 - nk.dot(R).dot(nk)
   472     12631       527810     41.8      0.0                          betak1 = ones(ls + 2)
   473     12631       288450     22.8      0.0                          betak1[:-1] = betak
   474     12631    605657957  47950.1      2.1                          R_old = R
   475     12631  18133147633 1435606.7     63.0                          R = 1/k * outer(betak1, betak1)
   476     12631   5373801698 425445.5     18.7                          R[:-1,:-1] += R_old
   477                                           
   478     12631       106118      8.4      0.0                  ls += 1
   479                                                    # update X, a
   480         1           25     25.0      0.0          self._data.set_X(X)
   481         1           11     11.0      0.0          self._data.set_alpha(a)
   482         1            9      9.0      0.0          self._data.set_C(C)
   483         1           10     10.0      0.0          self._data.set_K_X(K_X)
   484         1     21354637 21354637.0      0.1          print self.rho()


*** PROFILER RESULTS ***
incremental_ocsvm (../evaluation_tmp.py:185)
function called 1 times

         371074 function calls in 28868.202 seconds

   Ordered by: cumulative time, internal time, call count
   List reduced from 149 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000 28868.202 28868.202 evaluation_tmp.py:185(incremental_ocsvm)
        1    0.240    0.240 28793.403 28793.403 line_profiler.py:95(wrapper)
        1 15986.924 15986.924 28793.163 28793.163 ocsvm.py:97(increment)
    12631 8360.712    0.662 8360.982    0.662 numeric.py:740(outer)
    75788 4344.013    0.057 4344.013    0.057 {method 'dot' of 'numpy.ndarray' objects}
        1    0.043    0.043   74.799   74.799 ocsvm.py:35(fit)
        1    1.913    1.913   74.755   74.755 ocsvm.py:62(alpha)
        1    0.019    0.019   69.979   69.979 coneprog.py:4159(qp)
        1    0.007    0.007   69.960   69.960 coneprog.py:1441(coneqp)
        5    0.000    0.000   68.290   13.658 coneprog.py:1984(kktsolver)
        5    0.490    0.098   68.290   13.658 misc.py:1389(factor)
        5   50.092   10.018   50.092   10.018 {cvxopt.base.syrk}
        2    0.000    0.000   49.955   24.978 ocsvm.py:58(gram)
        2    0.000    0.000   49.955   24.978 pairwise.py:1164(pairwise_kernels)
        2    0.000    0.000   49.955   24.978 pairwise.py:949(_parallel_pairwise)
        2    8.100    4.050   49.955   24.978 pairwise.py:740(rbf_kernel)
        2    3.041    1.520   41.802   20.901 pairwise.py:136(euclidean_distances)
        2    0.000    0.000   38.705   19.353 extmath.py:171(safe_sparse_dot)
        2    0.000    0.000   38.705   19.352 extmath.py:129(fast_dot)
        2   37.873   18.937   38.705   19.352 extmath.py:97(_fast_dot)
        1   20.469   20.469   21.192   21.192 ocsvm.py:45(rho)
    25279   16.491    0.001   16.491    0.001 {min}
        1    0.016    0.016   14.078   14.078 linalg.py:404(inv)
        1    0.000    0.000   13.987   13.987 linalg.py:244(solve)
        1   13.299   13.299   13.299   13.299 {numpy.linalg.lapack_lite.dgesv}
       10    8.958    0.896    8.958    0.896 {cvxopt.lapack.potrf}
        5    8.574    1.715    8.574    1.715 {cvxopt.base.gemm}
       56    1.295    0.023    1.295    0.023 {cvxopt.base.gemv}
        9    0.001    0.000    1.098    0.122 misc.py:1489(solve)
    37898    0.245    0.000    1.040    0.000 numeric.py:1791(ones)
        8    0.000    0.000    0.977    0.122 coneprog.py:2333(f4)
        8    0.000    0.000    0.977    0.122 coneprog.py:2291(f4_no_ir)
        8    0.000    0.000    0.921    0.115 validation.py:268(check_array)
        8    0.000    0.000    0.902    0.113 validation.py:43(_assert_all_finite)
        8    0.902    0.113    0.902    0.113 {method 'sum' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.832    0.208 extmath.py:87(_impose_f_order)
        2    0.000    0.000    0.809    0.405 shape_base.py:177(vstack)
        2    0.807    0.403    0.807    0.403 {numpy.core.multiarray.concatenate}
    25262    0.781    0.000    0.781    0.000 {numpy.core.multiarray.where}
        1    0.000    0.000    0.599    0.599 linalg.py:139(_fastCopyAndTranspose)



*** PROFILER RESULTS ***
cvxopt_ocsvm (../evaluation_tmp.py:181)
function called 1 times

         1399 function calls in 6460.395 seconds

   Ordered by: cumulative time, internal time, call count
   List reduced from 123 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000 6460.395 6460.395 evaluation_tmp.py:181(cvxopt_ocsvm)
        1    0.812    0.812 6460.395 6460.395 ocsvm.py:35(fit)
        1   62.778   62.778 6398.068 6398.068 ocsvm.py:62(alpha)
        1    0.382    0.382 6255.415 6255.415 coneprog.py:4159(qp)
        1    0.017    0.017 6255.033 6255.033 coneprog.py:1441(coneqp)
        5    0.000    0.000 6223.788 1244.758 coneprog.py:1984(kktsolver)
        5    9.853    1.971 6223.788 1244.758 misc.py:1389(factor)
        5 5099.822 1019.964 5099.822 1019.964 {cvxopt.base.syrk}
       10  878.570   87.857  878.570   87.857 {cvxopt.lapack.potrf}
        5  231.585   46.317  231.585   46.317 {cvxopt.base.gemm}
        2    0.000    0.000  108.758   54.379 ocsvm.py:58(gram)
        2    0.000    0.000  108.758   54.379 pairwise.py:1164(pairwise_kernels)
        2    0.017    0.008  108.758   54.379 pairwise.py:949(_parallel_pairwise)
        2   15.939    7.969  108.741   54.370 pairwise.py:740(rbf_kernel)
        2    5.796    2.898   92.452   46.226 pairwise.py:136(euclidean_distances)
        2    0.000    0.000   86.516   43.258 extmath.py:171(safe_sparse_dot)
        2    0.000    0.000   86.516   43.258 extmath.py:129(fast_dot)
        2   85.097   42.549   86.516   43.258 extmath.py:97(_fast_dot)
        1    0.000    0.000   61.516   61.516 ocsvm.py:45(rho)
        2    0.000    0.000   28.338   14.169 shape_base.py:177(vstack)
        2   28.338   14.169   28.338   14.169 {numpy.core.multiarray.concatenate}
       56   24.530    0.438   24.530    0.438 {cvxopt.base.gemv}
        9    0.001    0.000   21.567    2.396 misc.py:1489(solve)
        8    0.000    0.000   19.206    2.401 coneprog.py:2333(f4)
        8    0.000    0.000   19.206    2.401 coneprog.py:2291(f4_no_ir)
       10    0.000    0.000    7.929    0.793 coneprog.py:1900(fG)
       10    0.000    0.000    7.929    0.793 misc.py:801(sgemv)
       18    4.963    0.276    4.963    0.276 {cvxopt.blas.trsv}
        5    3.942    0.788    3.942    0.788 {cvxopt.blas.trsm}
        2    0.002    0.001    3.361    1.680 twodim_base.py:220(diag)
        4    3.359    0.840    3.359    0.840 {numpy.core.multiarray.zeros}
       10    0.000    0.000    1.857    0.186 validation.py:268(check_array)
        5    0.000    0.000    1.675    0.335 coneprog.py:1847(fP)
        5    1.675    0.335    1.675    0.335 {cvxopt.base.symv}
       10    0.000    0.000    1.593    0.159 validation.py:43(_assert_all_finite)
       10    1.593    0.159    1.593    0.159 {method 'sum' of 'numpy.ndarray' objects}
        4    0.000    0.000    1.418    0.355 extmath.py:87(_impose_f_order)
        2    0.534    0.267    0.537    0.268 data.py:29(Xs)
        4    0.000    0.000    0.439    0.110 pairwise.py:57(check_pairwise_arrays)
        1    0.396    0.396    0.396    0.396 {method 'dot' of 'numpy.ndarray' objects}



*** PROFILER RESULTS ***
sklearn_ocsvm (../evaluation_tmp.py:177)
function called 1 times

         61 function calls in 3288.371 seconds

   Ordered by: cumulative time, internal time, call count

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000 3288.371 3288.371 evaluation_tmp.py:177(sklearn_ocsvm)
        1    0.005    0.005 3288.371 3288.371 classes.py:941(fit)
        1    0.000    0.000 3288.365 3288.365 base.py:99(fit)
        1    0.000    0.000 3288.247 3288.247 base.py:211(_dense_fit)
        1 3288.247 3288.247 3288.247 3288.247 {sklearn.svm.libsvm.fit}
        1    0.000    0.000    0.118    0.118 validation.py:268(check_array)
        5    0.089    0.018    0.089    0.018 {numpy.core.multiarray.array}
        1    0.000    0.000    0.030    0.030 validation.py:43(_assert_all_finite)
        1    0.030    0.030    0.030    0.030 {method 'sum' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 numeric.py:1791(ones)
        1    0.000    0.000    0.000    0.000 base.py:193(_validate_targets)
        1    0.000    0.000    0.000    0.000 validation.py:126(_shape_repr)
        1    0.000    0.000    0.000    0.000 {method 'fill' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.000    0.000 numeric.py:167(asarray)
        1    0.000    0.000    0.000    0.000 {method 'join' of 'str' objects}
        2    0.000    0.000    0.000    0.000 base.py:553(isspmatrix)
        2    0.000    0.000    0.000    0.000 sputils.py:116(_isinstance)
        2    0.000    0.000    0.000    0.000 numeric.py:237(asanyarray)
        3    0.000    0.000    0.000    0.000 validation.py:153(<genexpr>)
        1    0.000    0.000    0.000    0.000 getlimits.py:234(__init__)
        1    0.000    0.000    0.000    0.000 validation.py:105(_num_samples)
        2    0.000    0.000    0.000    0.000 {numpy.core.multiarray.empty}
        1    0.000    0.000    0.000    0.000 shape_base.py:58(atleast_2d)
        1    0.000    0.000    0.000    0.000 {sklearn.svm.libsvm.set_verbosity_wrap}
        1    0.000    0.000    0.000    0.000 {method 'copy' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 validation.py:503(check_random_state)
        1    0.000    0.000    0.000    0.000 getlimits.py:259(max)
        3    0.000    0.000    0.000    0.000 {hasattr}
        1    0.000    0.000    0.000    0.000 base.py:203(_warn_from_fit_status)
        6    0.000    0.000    0.000    0.000 {len}
        4    0.000    0.000    0.000    0.000 {method 'split' of 'str' objects}
        3    0.000    0.000    0.000    0.000 {isinstance}
        1    0.000    0.000    0.000    0.000 {method 'randint' of 'mtrand.RandomState' objects}
        1    0.000    0.000    0.000    0.000 {method 'index' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
        2    0.000    0.000    0.000    0.000 {callable}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        0    0.000             0.000          profile:0(profiler)


