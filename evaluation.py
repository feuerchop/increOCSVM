__author__ = 'LT'
import numpy as np
import ocsvm
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from operator import add, div


from sklearn import decomposition
from sklearn import datasets

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
        l = 1 if p[-1:][0].strip() == "positive" else -1
        label.append(l)
    return X, label

def getBestParas(X5fold):
    nu = np.arange(0.2,1,0.2)
    gamma = np.arange(0.2, 5, 0.2)
    best_paras = []
    for n in nu:
        for g in gamma:
            precision = []
            recall = []
            f1 = []
            error = []
            for X_tra, label_tra, X_lst,label_lst in X5fold:
                clf = ocsvm.OCSVM("rbf", nu=n, gamma=g)
                clf.train(np.asarray(X_tra))
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
            if len(best_paras) == 0:
                best_paras = [avgf1, avgprec, avgrec, n, g, avgerror]
            else:
                if avgf1 > best_paras[0]:
                    best_paras = [avgf1, avgprec, avgrec, n, g, avgerror]
    clf = ocsvm.OCSVM("rbf", nu=best_paras[3], gamma=best_paras[4])
    clf.train(np.asarray(X_tra))
    train_predict = clf.predict(np.asarray(X_tra))
    test_predict = clf.predict(np.asarray(X_lst))
    prec, rec, f1score, err = score(label_tra, train_predict, label_lst, test_predict)
    print prec, rec, f1score, err
    print len(clf._data.alpha_s()), len(clf._data.alpha())
    #print clf._data.alpha()
    print "gold standard alpha s: %s, sum(as): %s" % (clf._data.alpha_s(), sum(clf._data.alpha_s()))
    print "f1: %s, precision: %s, recall: %s, nu: %s, gamma: %s, error: %s" % (best_paras[0], best_paras[1],
                                                                               best_paras[2], best_paras[3], best_paras[4], best_paras[5])
    return best_paras

def plot(X, label):
    X = X
    y = label

    fig = plt.figure()
    plt.clf()
    #ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)
    # Reorder the labels to have colors matching the cluster results
    X_plus = X[y == 1]
    X_minus = X[y == -1]
    plt.scatter(X_plus[:, 0], X_plus[:, 1], c='white')
    plt.scatter(X_minus[:, 0], X_minus[:, 1], c='red')
    plt.show()

def score(labelTrain, predictTrain, labelTest, predictTest):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i, label in enumerate(labelTrain):

        if label == predictTrain[i]:
            if label == 1:
                tp += 1
            else:
                tn += 1
        elif label < predictTrain[i]:
            fp += 1
        else:
            fn += 1
        for i, label in enumerate(labelTest):
            if label == predictTest[i]:
                if label == 1:
                    tp += 1
                else:
                    tn += 1
            elif label < predictTest[i]:
                fp += 1
            else:
                fn +=1
    if tn + fn > 0: prec = float(tn)/(tn + fn)
    else: prec = 0
    if tn + fp > 0: rec = float(tn)/(tn + fp)
    else: rec = 0
    if prec + rec > 0:
        f1 = 2*prec*rec/(prec+rec)
    else: f1 = 0
    err = float(fp + fn) / (len(labelTrain) + len(labelTest))
    return prec, rec, f1, err

def incrementEval(X5fold, nu, gamma):
    prec = 0
    rec = 0
    f1score = 0
    err = 0
    #i = 0

    for X_tra, label_tra, X_lst,label_lst in X5fold:
        clf = train(np.asarray(X_tra), nu, gamma)
        print len(clf._data.alpha_s())
        train_predict = clf.predict(X_tra)
        test_predict = clf.predict(X_lst)
        p, r, f, e = score(label_tra, train_predict, label_lst, test_predict)
        print "prec: %s, rec: %s, f1score: %s, err: %s" % (p, r, f, e)

        #i += 1
        #if i == 3: break
        prec += p
        rec += r
        f1score += f
        err += e
    print "prec: %s, rec: %s, f1score: %s, err: %s" % (float(prec)/5, float(rec)/5, float(f1score)/5, float(err)/5)

    return clf, X_tra, X_lst, train_predict, test_predict


def train(X_tra, n, g):
    clf = ocsvm.OCSVM("rbf", nu=n, gamma=g)
    clf.train(X_tra[0:3])
    print clf._data.alpha_s()
    X_tra = X_tra[3:]
    for i,x in enumerate(X_tra):
        print "========================== INCREMENTAL %s" %i
        clf.increment(x)

    return clf



if __name__ == "__main__":
    ecoli0 = "/Users/LT/Documents/Uni/MA/increOCSVM/imbalanced_data/ecoli-0_vs_1/"
    ecoli0All = ecoli0 + "ecoli-0_vs_1.dat"
    X, label = loadData(ecoli0All)

    #plot(np.asarray(X), np.asarray(label))
    ecoli05fold = "/Users/LT/Documents/Uni/MA/increOCSVM/imbalanced_data/ecoli-0_vs_1/ecoli-0_vs_1-5-fold/ecoli-0_vs_1-5-"
    X5fold = get5FoldCV(ecoli05fold)
    bestParas = getBestParas(X5fold)

    clf, X_tra, X_lst, train_predict, test_predict = incrementEval(X5fold, bestParas[3], bestParas[4])
    #plot(np.vstack((X_tra, X_lst)), np.hstack((train_predict, test_predict)))

