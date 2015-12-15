['../evaluation_tmp.py', '10000']
mnist classes = 2
size: 10000
(2609,)
(7391,)
data size: 10000, nu: 0.2, gamma: 1
============ 1. Fold of CV ============
1) Incremental OCSVM
0 data points processed
1000 data points processed
2000 data points processed
3000 data points processed
4000 data points processed
5000 data points processed
6000 data points processed
None
Confusion matrix:
Prediction    -1     1
Target                
-1          2085   524
 1          5915  1476
precision: 0.738, recall: 0.199702340685, f1-score: 0.314343520392
Number of support vectors: 8000
-----------
2) cvxopt-OCSVM
Confusion matrix:
Prediction     1
Target          
-1          2609
 1          7391
precision: 0.7391, recall: 1.0, f1-score: 0.849979874648
Number of support vectors: 8000
---------
3) sklearn-OCSVM
Confusion matrix:
Prediction    -1     1
Target                
-1          1677   932
 1          4723  2668
Number of support vectors: 8000
precision: 0.741111111111, recall: 0.360979569747, f1-score: 0.485488126649
Wrote profile results to evaluation_tmp.py.lprof
Timer unit: 1e-06 s

Total time: 4446.27 s
File: ../ocsvm.py
Function: increment at line 97

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    97                                               @profile
    98                                               def increment(self, Xc, init_ac=0, break_count=-1):
    99                                           
   100                                                   # epsilon
   101         1            6      6.0      0.0          e = self._data._e
   102         1            3      3.0      0.0          mu = 0
   103         1            3      3.0      0.0          imin = None
   104                                           
   105                                                   # initialize existing X, coefficients a, C
   106         1            7      7.0      0.0          X_origin = self._data.X()
   107         1            6      6.0      0.0          K_X_origin = self._data.K_X()
   108         1            6      6.0      0.0          n_data = X_origin.shape[0]
   109         1            3      3.0      0.0          n_feature = X_origin.shape[1]
   110                                           
   111         1            6      6.0      0.0          C = self._data.C()
   112         1            6      6.0      0.0          a_origin = self._data.alpha()
   113                                           
   114                                                   # number of new incremental points
   115         1            3      3.0      0.0          n_new = Xc.shape[0]
   116                                           
   117                                                   # number of all (new and existing) points
   118         1            4      4.0      0.0          n_all = n_data + n_new
   119                                           
   120                                                   # concatenate all new points with all existing points
   121         1           21     21.0      0.0          X = empty((n_new + n_data, n_feature))
   122         1        31954  31954.0      0.0          X[0:n_new, :] = Xc
   123         1         8589   8589.0      0.0          X[n_new:, :] = X_origin
   124                                           
   125                                                   # create kernel matrix for all new and existing points
   126                                           
   127                                                   # create of all data points
   128         1            8      8.0      0.0          if K_X_origin == None:
   129         1     10559662 10559662.0      0.2              K_X = self.gram(X)
   130                                                   else:
   131                                                       K_X = empty((n_all, n_all))
   132                                                       K_X[n_new:, n_new:] = K_X_origin
   133                                                       K_X_new = self.gram(Xc, X_origin)
   134                                                       K_X[0:n_new, :] = K_X_new
   135                                                       K_X[:, 0:n_new] = K_X_new.T
   136                                           
   137                                                   # creating coefficient vector alpha for all data points
   138         1           28     28.0      0.0          a = empty(n_all)
   139         1           19     19.0      0.0          a[n_new:] = a_origin
   140         1           25     25.0      0.0          a[:n_new] = init_ac
   141                                           
   142                                                   # creating gradient vector
   143         1           26     26.0      0.0          g = zeros(n_all)
   144                                           
   145                                                   # create sensitivity vector
   146         1            7      7.0      0.0          gamma = empty(n_all)
   147         1            4      4.0      0.0          check_gradient = False
   148                                                   # loop through all new points to increment
   149      6316        34329      5.4      0.0          for x_count in range(n_new):
   150      6315        34922      5.5      0.0              if x_count % 1000 == 0:
   151         7          128     18.3      0.0                  print "%s data points processed" % x_count
   152                                                       #print "--------- START %s ---------" % x_count
   153                                           
   154      6315        25174      4.0      0.0              if x_count == break_count:
   155                                                           self._data.set_X(X)
   156                                                           self._data.set_alpha(a)
   157                                                           self._data.set_C(C)
   158                                                           self._data.set_K_X(K_X)
   159                                                           self.rho()
   160                                                           return False
   161                                           
   162                                                       # initialize X, a, C, g, indices, kernel values
   163      6315        28553      4.5      0.0              start_origin = n_new - x_count
   164      6315        23484      3.7      0.0              start_new = start_origin - 1
   165                                           
   166      6315        22499      3.6      0.0              if x_count == 0:
   167         1            4      4.0      0.0                  inds = []
   168         1            4      4.0      0.0                  indr = []
   169         1            3      3.0      0.0                  inde = []
   170         1            4      4.0      0.0                  indo = []
   171      1686         6578      3.9      0.0                  for i in range(n_new, n_all):
   172      1685         9710      5.8      0.0                      if e < a[i] < C - e:
   173      1685         7113      4.2      0.0                          inds.append(i)
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
   186         1          170    170.0      0.0                  mu = - K_X[inds[0], :][start_origin:].dot(a[start_origin:])
   187         1            4      4.0      0.0                  if lr > 0:
   188                                                               g[indr] = K_X[indr, :][:, start_origin:].dot(a[start_origin:]) + mu
   189                                                           # calculate mu according to KKT-conditions
   190                                           
   191                                           
   192      6315       930130    147.3      0.0              c_inds = [start_new] + inds
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
   203      6315        28325      4.5      0.0              if ls > 0:
   204      6315      2500668    396.0      0.1                  gc = K_X[start_new, start_origin:].dot(a[start_origin:]) + mu
   205                                           
   206      6315        31463      5.0      0.0              ac = a[start_new]
   207                                           
   208      6315        23865      3.8      0.0              if x_count == 0:
   209         1         5438   5438.0      0.0                  Q = ones((ls+1, ls+1))
   210         1           10     10.0      0.0                  Q[0, 0] = 0
   211                                                           #Kss = self.gram(X[inds])
   212      1686         7977      4.7      0.0                  inds_row = [[i] for i in inds]
   213         1       371349 371349.0      0.0                  Q[1:, 1:] = K_X[inds_row, inds]
   214         1            6      6.0      0.0                  try:
   215         1      1887830 1887830.0      0.0                      R = inv(Q)
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
   226      6315        24024      3.8      0.0              loop_count = 1
   227                                                       #print "gc: %s, ac: %s" % (gc, ac)
   228      6315        96477     15.3      0.0              while gc < e and ac < C - e:
   229      6315        24173      3.8      0.0                  if ls == 0: check_gradient = True
   230                                                           #print "-------------------- incremental %s-%s ---------" % (x_count, loop_count)
   231                                           
   232      6315        24031      3.8      0.0                  if ls > 0:
   233      6315      7278480   1152.6      0.2                      n = K_X[start_new, :][c_inds]
   234      6315    276420874  43772.1      6.2                      beta = - R.dot(n)
   235      6315        70854     11.2      0.0                      betas = beta[1:]
   236                                           
   237                                                           # calculate gamma
   238      6315        30666      4.9      0.0                  if lr > 0 and ls > 0:
   239                                                               gamma_tmp = K_X[:, c_inds][start_new:]
   240                                                               gamma_tmp[:, 0] = 1
   241                                                               gamma[start_new:] = gamma_tmp.dot(beta) + K_X[start_new, :][start_new:]
   242                                                               gammac = gamma[start_new]
   243                                           
   244      6315        26280      4.2      0.0                  elif ls > 0:
   245                                                               # empty R set
   246      6315     11314559   1791.7      0.3                      gammac = K_X[start_new, :][c_inds].dot(beta) + 1
   247                                           
   248                                                           else:
   249                                                               # empty S set
   250                                                               gammac = 1
   251                                                               gamma[indr] = 1
   252                                                               #gamma[indo] = -1
   253                                           
   254                                                           # accounting
   255                                                           #case 1: Some alpha_i in S reaches a bound
   256      6315        28753      4.6      0.0                  if ls > 0:
   257      6315       171869     27.2      0.0                      IS_plus = betas > e
   258      6315       115733     18.3      0.0                      IS_minus = betas < - e
   259      6315       297828     47.2      0.0                      gsmax = ones(ls)*inf
   260                                                               #if np.isnan(np.min(gsmax)):
   261                                                               #    gsmax = ones(ls)*inf
   262      6315      6462461   1023.4      0.1                      gsmax[IS_plus] = -a[inds][IS_plus] + C
   263      6315      6554085   1037.9      0.1                      gsmax[IS_minus] = - a[inds][IS_minus]
   264                                                               #gsmax[IS_plus] = -a[inds][IS_plus]
   265                                                               #gsmax[IS_plus] += C
   266                                                               #gsmax[IS_minus] = - a[inds][IS_minus]
   267      6315       284358     45.0      0.0                      gsmax = divide(gsmax, betas)
   268      6315      4342393    687.6      0.1                      gsmin = min(absolute(gsmax))
   269                                                               #print where(absolute(gsmax) == gsmin)
   270      6315       459013     72.7      0.0                      ismin = where(absolute(gsmax) == gsmin)[0][0]
   271                                           
   272                                                           else: gsmin = inf
   273                                           
   274                                                           #case 2: Some g_i in E reaches zero
   275      6315        27912      4.4      0.0                  if le > 0:
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
   288      6315        27717      4.4      0.0                  else: gemin = inf
   289                                                           #case 2: Some g_i in O reaches zero
   290      6315        26844      4.3      0.0                  if lo > 0 and ls > 0:
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
   302      6315        26472      4.2      0.0                  else: gomin = inf
   303                                           
   304                                                           # case 3: gc becomes zero
   305      6315        52912      8.4      0.0                  if gammac > e: gcmin = - gc/gammac
   306                                                           else: gcmin = inf
   307                                           
   308                                                           # case 4
   309      6315        39095      6.2      0.0                  if ls > 0: gacmin = C - ac
   310                                                           else: gacmin = inf
   311                                           
   312                                                           # determine minimum largest increment
   313      6315        37991      6.0      0.0                  all_deltas = [gsmin, gemin, gomin, gcmin, gacmin]
   314      6315        51044      8.1      0.0                  gmin = min(all_deltas)
   315      6315       151241     23.9      0.0                  imin = where(all_deltas == gmin)[0][0]
   316                                                           # update a, g
   317      6315        28142      4.5      0.0                  if ls > 0:
   318      6315        40268      6.4      0.0                      mu += beta[0]*gmin
   319      6315        29448      4.7      0.0                      ac += gmin
   320      6315     11957014   1893.4      0.3                      a[inds] += betas*gmin
   321                                                           else:
   322                                                               mu += gmin
   323      6315        31456      5.0      0.0                  if lr > 0:
   324                                                               g[indr] += gamma[indr] * gmin
   325      6315        35200      5.6      0.0                  gc += gammac * gmin
   326      6315        44916      7.1      0.0                  if imin == 0: # min = gsmin => move k from s to r
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
   366      6315        33341      5.3      0.0                  elif imin == 1:
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
   393      6315        32065      5.1      0.0                  elif imin == 2: # min = gemin | gomin => move k from r to s
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
   424      6315        32323      5.1      0.0                  elif imin == 3:
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
   442      6315        28391      4.5      0.0                      break
   443                                                           else:
   444                                                               break
   445                                                           loop_count += 1
   446                                           
   447      6315        31180      4.9      0.0              a[start_new] = ac
   448      6315        31385      5.0      0.0              g[start_new] = gc
   449      6315        34513      5.5      0.0              if ac < e:
   450                                                           indr.append(start_new)
   451                                                           indo.append(start_new)
   452                                                           lr += 1
   453                                                           lo += 1
   454      6315        39033      6.2      0.0              elif ac > C - e:
   455                                                           indr.append(start_new)
   456                                                           inde.append(start_new)
   457                                                           lr += 1
   458                                                           le += 1
   459                                                       else:
   460      6315        45526      7.2      0.0                  inds.append(start_new)
   461      6315        29069      4.6      0.0                  g[start_new] = 0
   462      6315        37538      5.9      0.0                  if len(inds) == 1:
   463                                                               R = ones((2, 2))
   464                                                               R[1,1] = 0
   465                                                               R[0,0] = -1
   466                                                           else:
   467      6315        43113      6.8      0.0                      if R.shape[0] != len(inds) + 1:
   468      6315       127707     20.2      0.0                          nk = ones(ls+1)
   469      6315      7318330   1158.9      0.2                          nk[1:] = K_X[start_new, :][inds[:-1]]
   470      6315    276033663  43710.8      6.2                          betak = - R.dot(nk)
   471      6315       949917    150.4      0.0                          k = 1 - nk.dot(R).dot(nk)
   472      6315       221603     35.1      0.0                          betak1 = ones(ls + 2)
   473      6315        96065     15.2      0.0                          betak1[:-1] = betak
   474      6315     82876318  13123.7      1.9                          R_old = R
   475      6315   2616448189 414322.8     58.8                          R = 1/k * outer(betak1, betak1)
   476      6315   1114393414 176467.7     25.1                          R[:-1,:-1] += R_old
   477                                           
   478      6315        56172      8.9      0.0                  ls += 1
   479                                                    # update X, a
   480         1           27     27.0      0.0          self._data.set_X(X)
   481         1            9      9.0      0.0          self._data.set_alpha(a)
   482         1            9      9.0      0.0          self._data.set_C(C)
   483         1           10     10.0      0.0          self._data.set_K_X(K_X)
   484         1      4118987 4118987.0      0.1          print self.rho()


*** PROFILER RESULTS ***
incremental_ocsvm (../evaluation_tmp.py:185)
function called 1 times

         186226 function calls in 4458.908 seconds

   Ordered by: cumulative time, internal time, call count
   List reduced from 149 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000 4458.908 4458.908 evaluation_tmp.py:185(incremental_ocsvm)
        1    0.062    0.062 4448.712 4448.712 line_profiler.py:95(wrapper)
        1 2499.493 2499.493 4448.650 4448.650 ocsvm.py:97(increment)
     6315 1372.956    0.217 1373.067    0.217 numeric.py:740(outer)
    37892  554.850    0.015  554.850    0.015 {method 'dot' of 'numpy.ndarray' objects}
        2    0.000    0.000   11.064    5.532 ocsvm.py:58(gram)
        2    0.000    0.000   11.064    5.532 pairwise.py:1164(pairwise_kernels)
        2    0.000    0.000   11.064    5.532 pairwise.py:949(_parallel_pairwise)
        2    2.008    1.004   11.064    5.532 pairwise.py:740(rbf_kernel)
        1    0.013    0.013   10.196   10.196 ocsvm.py:35(fit)
        1    0.386    0.386   10.183   10.183 ocsvm.py:62(alpha)
        1    0.003    0.003    9.108    9.108 coneprog.py:4159(qp)
        1    0.005    0.005    9.104    9.104 coneprog.py:1441(coneqp)
        2    0.890    0.445    9.029    4.515 pairwise.py:136(euclidean_distances)
        5    0.000    0.000    8.759    1.752 coneprog.py:1984(kktsolver)
        5    0.120    0.024    8.759    1.752 misc.py:1389(factor)
        2    0.000    0.000    8.112    4.056 extmath.py:171(safe_sparse_dot)
        2    0.000    0.000    8.112    4.056 extmath.py:129(fast_dot)
        2    7.718    3.859    8.112    4.056 extmath.py:97(_fast_dot)
        5    6.097    1.219    6.097    1.219 {cvxopt.base.syrk}
    12647    4.239    0.000    4.239    0.000 {min}
        1    3.809    3.809    4.074    4.074 ocsvm.py:45(rho)
        1    0.000    0.000    1.888    1.888 linalg.py:404(inv)
        1    0.000    0.000    1.883    1.883 linalg.py:244(solve)
        1    1.740    1.740    1.740    1.740 {numpy.linalg.lapack_lite.dgesv}
        5    1.316    0.263    1.316    0.263 {cvxopt.base.gemm}
       10    1.191    0.119    1.191    0.119 {cvxopt.lapack.potrf}
        8    0.000    0.000    0.438    0.055 validation.py:268(check_array)
        8    0.000    0.000    0.429    0.054 validation.py:43(_assert_all_finite)
        8    0.428    0.054    0.428    0.054 {method 'sum' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.394    0.099 extmath.py:87(_impose_f_order)
    18950    0.115    0.000    0.369    0.000 numeric.py:1791(ones)
       56    0.255    0.005    0.255    0.005 {cvxopt.base.gemv}
        9    0.000    0.000    0.222    0.025 misc.py:1489(solve)
    12630    0.210    0.000    0.210    0.000 {numpy.core.multiarray.where}
        8    0.000    0.000    0.199    0.025 coneprog.py:2333(f4)
        8    0.000    0.000    0.198    0.025 coneprog.py:2291(f4_no_ir)
        2    0.000    0.000    0.160    0.080 shape_base.py:177(vstack)
        2    0.158    0.079    0.158    0.079 {numpy.core.multiarray.concatenate}
        1    0.157    0.157    0.158    0.158 data.py:29(Xs)



*** PROFILER RESULTS ***
cvxopt_ocsvm (../evaluation_tmp.py:181)
function called 1 times

         1399 function calls in 851.843 seconds

   Ordered by: cumulative time, internal time, call count
   List reduced from 123 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000  851.843  851.843 evaluation_tmp.py:181(cvxopt_ocsvm)
        1    0.215    0.215  851.843  851.843 ocsvm.py:35(fit)
        1   13.610   13.610  836.090  836.090 ocsvm.py:62(alpha)
        1    0.085    0.085  805.091  805.091 coneprog.py:4159(qp)
        1    0.009    0.009  805.006  805.006 coneprog.py:1441(coneqp)
        5    0.000    0.000  797.632  159.526 coneprog.py:1984(kktsolver)
        5    2.340    0.468  797.632  159.526 misc.py:1389(factor)
        5  630.443  126.089  630.443  126.089 {cvxopt.base.syrk}
       10  110.158   11.016  110.158   11.016 {cvxopt.lapack.potrf}
        5   53.899   10.780   53.899   10.780 {cvxopt.base.gemm}
        2    0.000    0.000   25.810   12.905 ocsvm.py:58(gram)
        2    0.000    0.000   25.810   12.905 pairwise.py:1164(pairwise_kernels)
        2    0.012    0.006   25.810   12.905 pairwise.py:949(_parallel_pairwise)
        2    3.824    1.912   25.798   12.899 pairwise.py:740(rbf_kernel)
        2    1.760    0.880   21.800   10.900 pairwise.py:136(euclidean_distances)
        2    0.000    0.000   19.970    9.985 extmath.py:171(safe_sparse_dot)
        2    0.000    0.000   19.970    9.985 extmath.py:129(fast_dot)
        2   19.296    9.648   19.970    9.985 extmath.py:97(_fast_dot)
        1    0.000    0.000   15.538   15.538 ocsvm.py:45(rho)
        2    0.000    0.000    5.915    2.957 shape_base.py:177(vstack)
        2    5.914    2.957    5.914    2.957 {numpy.core.multiarray.concatenate}
       56    5.881    0.105    5.881    0.105 {cvxopt.base.gemv}
        9    0.001    0.000    4.780    0.531 misc.py:1489(solve)
        8    0.000    0.000    4.241    0.530 coneprog.py:2333(f4)
        8    0.000    0.000    4.241    0.530 coneprog.py:2291(f4_no_ir)
       10    0.000    0.000    2.122    0.212 coneprog.py:1900(fG)
       10    0.000    0.000    2.122    0.212 misc.py:801(sgemv)
       18    1.019    0.057    1.019    0.057 {cvxopt.blas.trsv}
       10    0.000    0.000    0.893    0.089 validation.py:268(check_array)
        2    0.001    0.001    0.841    0.420 twodim_base.py:220(diag)
        4    0.840    0.210    0.840    0.210 {numpy.core.multiarray.zeros}
        5    0.780    0.156    0.780    0.156 {cvxopt.blas.trsm}
       10    0.000    0.000    0.763    0.076 validation.py:43(_assert_all_finite)
       10    0.762    0.076    0.762    0.076 {method 'sum' of 'numpy.ndarray' objects}
        4    0.000    0.000    0.674    0.168 extmath.py:87(_impose_f_order)
        5    0.000    0.000    0.432    0.086 coneprog.py:1847(fP)
        5    0.432    0.086    0.432    0.086 {cvxopt.base.symv}
        2    0.256    0.128    0.257    0.129 data.py:29(Xs)
        4    0.000    0.000    0.219    0.055 pairwise.py:57(check_pairwise_arrays)
       39    0.130    0.003    0.130    0.003 {numpy.core.multiarray.array}



*** PROFILER RESULTS ***
sklearn_ocsvm (../evaluation_tmp.py:177)
function called 1 times

         61 function calls in 437.500 seconds

   Ordered by: cumulative time, internal time, call count

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000  437.500  437.500 evaluation_tmp.py:177(sklearn_ocsvm)
        1    0.004    0.004  437.500  437.500 classes.py:941(fit)
        1    0.000    0.000  437.496  437.496 base.py:99(fit)
        1    0.000    0.000  437.436  437.436 base.py:211(_dense_fit)
        1  437.436  437.436  437.436  437.436 {sklearn.svm.libsvm.fit}
        1    0.000    0.000    0.059    0.059 validation.py:268(check_array)
        5    0.044    0.009    0.044    0.009 {numpy.core.multiarray.array}
        1    0.000    0.000    0.015    0.015 validation.py:43(_assert_all_finite)
        1    0.015    0.015    0.015    0.015 {method 'sum' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 base.py:193(_validate_targets)
        1    0.000    0.000    0.000    0.000 validation.py:126(_shape_repr)
        2    0.000    0.000    0.000    0.000 numeric.py:167(asarray)
        1    0.000    0.000    0.000    0.000 numeric.py:1791(ones)
        1    0.000    0.000    0.000    0.000 {method 'join' of 'str' objects}
        2    0.000    0.000    0.000    0.000 base.py:553(isspmatrix)
        1    0.000    0.000    0.000    0.000 {method 'fill' of 'numpy.ndarray' objects}
        2    0.000    0.000    0.000    0.000 sputils.py:116(_isinstance)
        2    0.000    0.000    0.000    0.000 numeric.py:237(asanyarray)
        3    0.000    0.000    0.000    0.000 validation.py:153(<genexpr>)
        1    0.000    0.000    0.000    0.000 getlimits.py:234(__init__)
        2    0.000    0.000    0.000    0.000 {numpy.core.multiarray.empty}
        1    0.000    0.000    0.000    0.000 validation.py:105(_num_samples)
        1    0.000    0.000    0.000    0.000 {sklearn.svm.libsvm.set_verbosity_wrap}
        1    0.000    0.000    0.000    0.000 shape_base.py:58(atleast_2d)
        1    0.000    0.000    0.000    0.000 {method 'copy' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 validation.py:503(check_random_state)
        3    0.000    0.000    0.000    0.000 {hasattr}
        1    0.000    0.000    0.000    0.000 getlimits.py:259(max)
        1    0.000    0.000    0.000    0.000 base.py:203(_warn_from_fit_status)
        1    0.000    0.000    0.000    0.000 {method 'randint' of 'mtrand.RandomState' objects}
        1    0.000    0.000    0.000    0.000 {method 'index' of 'list' objects}
        6    0.000    0.000    0.000    0.000 {len}
        4    0.000    0.000    0.000    0.000 {method 'split' of 'str' objects}
        3    0.000    0.000    0.000    0.000 {isinstance}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        2    0.000    0.000    0.000    0.000 {callable}
        1    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
        0    0.000             0.000          profile:0(profiler)


