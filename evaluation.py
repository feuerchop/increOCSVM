__author__ = 'LT'
import numpy as np
import ocsvm
import sys
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from operator import add, div
from math import ceil
from scipy.optimize import check_grad
import os
from sklearn import svm
import cProfile as profile
import pandas


from sklearn import decomposition


def get5FoldCV(filePath):
    X5fold = []
    for i in range(1,6):
        tra = filePath + "%stra.dat" % i
        lst = filePath + "%stst.dat" % i
        X_tra, label_tra = loadData(tra)
        X_lst, label_lst = loadData(lst)
        X5fold.append((X_tra, label_tra, X_lst,label_lst))
    return X5fold

def loadData(filePath):
    f = open(filePath)
    X = []
    label = []

    for l in f:
        if l.startswith("@"): continue
        p = l.split(", ")
        X.append(map(float,p[:-1]))
        l = -1 if p[-1:][0].strip() == "positive" else 1
        label.append(l)
    return X, label

def testScilearnOCSVM(nu, gamma, X5fold):
    precision = []
    recall = []
    f1 = []
    error = []
    for X_tra, label_tra, X_lst,label_lst in X5fold:
        clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
        clf.fit(X_tra)
        y_pred_train = clf.predict(X_tra)
        y_pred_test = clf.predict(X_lst)
        prec, rec, f1score, err = score(label_tra, y_pred_train, label_lst, y_pred_test)
        precision.append(prec)
        recall.append(rec)
        f1.append(f1score)
        error.append(err)
    avgprec = float(sum(precision))/len(precision)
    avgrec = float(sum(recall))/len(recall)
    avgf1 = float(sum(f1))/len(f1)
    avgerror = float(sum(error)/len(error))
    print "scikit-learn -> precision: %s, recall: %s, f1: %s, error: %s" % (avgprec, avgrec, avgf1, avgerror)

    for X_tra, label_tra, X_lst,label_lst in X5fold:
        clf = ocsvm.OCSVM("rbf", nu=nu, gamma=gamma)
        clf.train(np.asarray(X_tra))
        y_pred_train = clf.predict(np.asarray(X_tra))
        y_pred_test = clf.predict(np.asarray(X_lst))
        prec, rec, f1score, err = score(label_tra, y_pred_train, label_lst, y_pred_test)
        precision.append(prec)
        recall.append(rec)
        f1.append(f1score)
        error.append(err)
    avgprec = float(sum(precision))/len(precision)
    avgrec = float(sum(recall))/len(recall)
    avgf1 = float(sum(f1))/len(f1)
    avgerror = float(sum(error)/len(error))
    print "own impl -> precision: %s, recall: %s, f1: %s, error: %s" % (avgprec, avgrec, avgf1, avgerror)

    for X_tra, label_tra, X_lst,label_lst in X5fold:
        clf = ocsvm.OCSVM("rbf", nu=nu, gamma=gamma)
        clf.train(np.asarray(X_tra), scale=nu*len(X_tra))
        y_pred_train = clf.predict(np.asarray(X_tra))
        y_pred_test = clf.predict(np.asarray(X_lst))
        prec, rec, f1score, err = score(label_tra, y_pred_train, label_lst, y_pred_test)
        precision.append(prec)
        recall.append(rec)
        f1.append(f1score)
        error.append(err)
    avgprec = float(sum(precision))/len(precision)
    avgrec = float(sum(recall))/len(recall)
    avgf1 = float(sum(f1))/len(f1)
    avgerror = float(sum(error)/len(error))
    print "scaled impl -> precision: %s, recall: %s, f1: %s, error: %s" % (avgprec, avgrec, avgf1, avgerror)

def getBestParas(X5fold):
    nu = np.arange(0.2,1,0.2)
    gamma = np.arange(0.2, 5, 0.2)
    best_paras = []
    for n in nu:

        for g in gamma:
            #print "nu: %s; gamma: %s" % (n,g)
            precision = []
            recall = []
            f1 = []
            error = []
            i = 1
            for X_tra, label_tra, X_lst,label_lst in X5fold:
                #print i
                i += 1
                ### own implementation
                #print "predicting ..."
                clf = ocsvm.OCSVM("rbf", nu=n, gamma=g)
                #print "training ..."
                clf.train(np.asarray(X_tra), scale=n*len(X_tra))
                #print "evaluating ..."
                train_predict = clf.predict(np.asarray(X_tra))
                test_predict = clf.predict(np.asarray(X_lst))
                prec, rec, f1score, err = score(label_tra, train_predict, label_lst, test_predict)
                precision.append(prec)
                recall.append(rec)
                f1.append(f1score)
                error.append(err)
            avgprec = float(sum(precision))/len(precision)
            avgrec = float(sum(recall))/len(recall)
            avgf1 = float(sum(f1))/len(f1)
            avgerror = float(sum(error)/len(error))
            #print "precision: %s, recall: %s, f1: %s, error: %s" % (avgprec, avgrec, avgf1, avgerror)
            if len(best_paras) == 0:
                best_paras = [avgf1, avgprec, avgrec, n, g, avgerror]
            else:
                if avgf1 > best_paras[0]:
                    best_paras = [avgf1, avgprec, avgrec, n, g, avgerror]

def getBestParasIncr(X5fold):
    nu = np.arange(0.2,1,0.2)
    gamma = np.arange(0.2, 5, 0.2)
    best_paras = []
    for n in nu:

        for g in gamma:
            precision = []
            recall = []
            f1 = []
            error = []
            i = 1
            eval = True
            for X_tra, label_tra, X_lst,label_lst in X5fold:
                #print i
                i += 1
                ### own implementation
                #print "predicting ..."
                clf = ocsvm.OCSVM("rbf", nu=n, gamma=g)
                #print "training ..."
                try:
                    clf = train(np.asarray(X_tra), n, g)
                    #print "evaluating ..."
                    train_predict = clf.predict(np.asarray(X_tra))
                    test_predict = clf.predict(np.asarray(X_lst))
                    prec, rec, f1score, err = score(label_tra, train_predict, label_lst, test_predict)
                    precision.append(prec)
                    recall.append(rec)
                    f1.append(f1score)
                    error.append(err)
                except:
                    eval = False
            if eval:
                avgprec = float(sum(precision))/len(precision)
                avgrec = float(sum(recall))/len(recall)
                avgf1 = float(sum(f1))/len(f1)
                avgerror = float(sum(error)/len(error))
                #print "precision: %s, recall: %s, f1: %s, error: %s" % (avgprec, avgrec, avgf1, avgerror)
                if len(best_paras) == 0:
                    best_paras = [avgf1, avgprec, avgrec, n, g, avgerror]
                else:
                    if avgf1 > best_paras[0]:
                        best_paras = [avgf1, avgprec, avgrec, n, g, avgerror]

    print "f1: %s, precision: %s, recall: %s, nu: %s, gamma: %s, error: %s" % (best_paras[0], best_paras[1],
                                                                               best_paras[2], best_paras[3], best_paras[4], best_paras[5])
    return best_paras

def plot(X, label=None, D3 = False):
    X = X
    y = np.asarray(label)

    fig = plt.figure()
    plt.clf()
    if D3:
        ax = fig.add_subplot(111, projection='3d')

    plt.cla()
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)
    # Reorder the labels to have colors matching the cluster results
    if label == None:
        plt.scatter(X[:, 0], X[:, 1], c='white')
    else:
        X_plus = X[y == 1]
        X_minus = X[y == -1]
        if D3:
            a = ax.plot(X_plus[:, 0], X_plus[:, 1], X_plus[:, 2], 'o', c='white', label="positive")
            b = ax.plot(X_minus[:, 0], X_minus[:, 1], X_minus[:, 2], 'o', c='red', label="negative")
            ax.legend()
        else:
            a = plt.scatter(X_plus[:, 0], X_plus[:, 1], c='white')
            b = plt.scatter(X_minus[:, 0], X_minus[:, 1], c='red')
            plt.legend([a,b], ["positive", "negative"],loc="upper left",
                       prop=matplotlib.font_manager.FontProperties(size=11))
    plt.show()

def score(labelTrain, predictTrain, labelTest, predictTest):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i, label in enumerate(labelTrain):

        if label == predictTrain[i]:
            if label == 1:
                tn += 1
            else:
                tp += 1
        elif label < predictTrain[i]:
            fn += 1
        else:
            fp += 1
    for i, label in enumerate(labelTest):
        if label == predictTest[i]:
            if label == 1:
                tn += 1
            else:
                tp += 1
        elif label < predictTest[i]:
            fn += 1
        else:
            fp += 1
    #print "tn: %s, tp: %s, fn: %s, fp: %s" % (tn, tp, fn, fp)
    #print "train + test: %s" % (len(labelTest) + len(labelTrain))
    if tp + fp > 0: prec = float(tp)/(tp + fp)
    else: prec = 0
    if tp + fn > 0: rec = float(tp)/(tp + fn)
    else: rec = 0
    if prec + rec > 0:
        f1 = 2*prec*rec/(prec+rec)
    else: f1 = 0
    err = float(fp + fn) / (len(labelTrain) + len(labelTest))
    #print "err: %s" % err
    return prec, rec, f1, err

def incrementEval(X5fold, nu, gamma):
    batch_incr = {'precision': [0,0], 'recall': [0,0], 'f1-score': [0,0], 'error': [0,0]}
    i = 0

    for X_tra, label_tra, X_lst,label_lst in X5fold:
        train_size = ceil(len(X_tra) * nu / 0.9)
        nu_new = (len(X_tra) * nu) / train_size
        #print "Train data size: %s" % len(X_tra)
        #print "beginning train size: %s" % train_size
        #print "nu new: %s" % nu_new
        clf_gold = ocsvm.OCSVM("rbf", nu=nu, gamma=gamma)
        clf_gold.train(np.asarray(X_tra), scale=len(X_tra) * nu)
        train_predict = clf_gold.predict(X_tra)
        test_predict = clf_gold.predict(X_lst)
        pb, rb, fb, eb = score(label_tra, train_predict, label_lst, test_predict)

        #print "batch -> prec: %s, rec: %s, f1score: %s, err: %s" % (p, r, f, e)
        #print "gold alpha_s: %s" % clf_gold._data.alpha_s()
        clf = train(np.asarray(X_tra), nu_new, gamma)
        train_predict = clf.predict(X_tra)
        test_predict = clf.predict(X_lst)
        p, r, f, e = score(label_tra, train_predict, label_lst, test_predict)
        #print "incremental -> prec: %s, rec: %s, f1score: %s, err: %s" % (p, r, f, e)

        i += 1
        batch_incr['precision'][0] += p
        batch_incr['recall'][0] += r
        batch_incr['f1-score'][0] += f
        batch_incr['error'][0] += e
        batch_incr['precision'][1] += pb
        batch_incr['recall'][1] += rb
        batch_incr['f1-score'][1] += fb
        batch_incr['error'][1] += eb
    for k in batch_incr.keys():
        batch_incr[k][0] /= i
        batch_incr[k][1] /= i
    print pandas.DataFrame(batch_incr, index=['incremental', 'batch'])

    return clf, X_tra, X_lst, train_predict, test_predict

def train(X_tra, n, g, size = 5):

    clf = ocsvm.OCSVM("rbf", nu=n, gamma=g)
    clf.train(X_tra[0:size], scale=n*len(X_tra[0:size]))
    X_tra = X_tra[size:]
    for i,x in enumerate(X_tra):
        #print "========================== INCREMENTAL %s" %i
        clf.increment(x, init_ac=n)
    return clf

def findBestParasIncr(path):

    txtFiles = [root + "/" + file for root, dirs, files in os.walk(path) for file in files if file.endswith("-names.txt")]
    for f in txtFiles:
        #print f
        path = f.replace(f.split("/")[-1:][0], "")
        print path
        datName = path.split("/")[-2:][0]

        datAll = path + datName + ".dat"
        if os.path.isfile(datAll):
            X, label = loadData(datAll)
        else:
            datFile = [root + "/" + file for root, dirs, files in os.walk(path) for file in files if file.endswith(".dat") if root + "/" == path]
            X, label = loadData(datFile)
        #plot(X, label, D3=False)
        #plot(np.asarray(X), np.asarray(label))
        fiveFoldPath = path + datName + "-5-fold/" + datName + "-5-"
        X5fold = get5FoldCV(fiveFoldPath)
        print "Find best parameters for %s" % datName
        bestParas = getBestParas(X5fold)

def findBestParasBatch(path):
    dats = ["vehicle2", "vehicle3", "glass0", "newthyroid2", "yeast3", "vehicle1", "wisconsin" \
        , "page-blocks0", "new-thyroid1", "haberman", "pima", "glass-0-1-2-3_vs_4-5-6", "yeast1", "ecoli-0_vs_1", \
            "glass6"]
    txtFiles = [root + "/" + file for root, dirs, files in os.walk(path) for file in files if file.endswith("-names.txt")]
    for f in txtFiles:
        #print f
        path = f.replace(f.split("/")[-1:][0], "")
        print path
        datName = path.split("/")[-2:][0]
        if datName not in dats:

            datAll = path + datName + ".dat"
            print datAll
            if os.path.isfile(datAll):
                X, label = loadData(datAll)
            else:
                datFile = [root + "/" + file for root, dirs, files in os.walk(path) for file in files if file.endswith(".dat") if root + "/" == path]
                X, label = loadData(datFile)
            #plot(X, label, D3=False)
            #plot(np.asarray(X), np.asarray(label))
            fiveFoldPath = path + datName + "-5-fold/" + datName + "-5-"
            X5fold = get5FoldCV(fiveFoldPath)
            print "Find best parameters for %s" % datName
            bestParas = getBestParas(X5fold)


if __name__ == "__main__":
    #findBestParasIncr("/mnt/project/predictppi/data/MA/increOCSVM/imbalanced_data/")
    #sys.exit()
    txtFiles = [root + "/" + file for root, dirs, files in os.walk("/Users/LT/Documents/Uni/MA/increOCSVM/imbalanced_data")
                for file in files if file.endswith("-names.txt")]
    for f in txtFiles:
        path = f.replace(f.split("/")[-1:][0], "")
        datName = path.split("/")[-2:][0]
        if "haberman" in datName:
            print path
            datAll = path + datName + ".dat"
            if os.path.isfile(datAll):
                X, label = loadData(datAll)
            else:
                datFile = [root + "/" + file for root, dirs, files in os.walk(path) for file in files if file.endswith(".dat") if root + "/" == path]
                X, label = loadData(datFile)
            fiveFoldPath = path + datName + "-5-fold/" + datName + "-5-"
            X5fold = get5FoldCV(fiveFoldPath)
            nu = 0.8
            gamma = 1.6
            print "increment evaluation"
            clf, X_tra, X_lst, train_predict, test_predict = incrementEval(X5fold, nu, gamma)

    sys.exit()


