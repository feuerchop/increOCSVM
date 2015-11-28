__author__ = 'LT'
import numpy as np
import ocsvm
import sys
import matplotlib
import matplotlib.pyplot as plt
from math import ceil
import os
from sklearn import svm, datasets
import pandas
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_recall_fscore_support
from sklearn.datasets import load_digits
from sklearn import decomposition
# get 5-fold cross validation from imbalanced dataset
def get_cv_data(filePath):
    cv_5_data = {}
    for i in range(1, 6):
        tra = filePath + "%stra.dat" % i
        lst = filePath + "%stst.dat" % i
        X_tra, label_tra = load_data(tra)
        X_lst, label_lst = load_data(lst)
        cv_5_data[i] = (X_tra, label_tra, X_lst,label_lst)
    return cv_5_data

# get data from imbalanced dataset
def load_data(filePath):
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

# fit one-class svm classifier
def train_clf(nu, gamma, train_data, scale = 1):
    clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
    clf.fit(train_data, scale=scale)
    return clf

def get_best_parameters(cv_data, nu=None, gamma=None):
    nu = np.arange(0.2,1,0.2)
    gamma = np.arange(0.2, 5, 0.2)
    best_parameter = {'precision': 0, 'recall': 0, 'f1': 0, 'error': 0}
    for n in nu:
        for g in gamma:
            parameters_result = {'precision': 0, 'recall': 0, 'f1': 0, 'error': 0}
            n_cv = len(cv_data)
            for X_train, label_train, X_test, label_test in cv_data:
                clf = train_clf(n, g, np.asarray(X_train), scale=n*len(X_train))
                train_predict = clf.predict(np.asarray(X_train))
                test_predict = clf.predict(np.asarray(X_test))
                precision, recall, f1, error = score(label_train, train_predict, label_test, test_predict)
                parameters_result['precision'] += precision
                parameters_result['recall'] += recall
                parameters_result['f1'] += f1
                parameters_result['error'] += error

            if parameters_result['f1'] / float(n_cv) > best_parameter['f1']:
                parameters_result.update({n: parameters_result[n] / float(n_cv) for n in parameters_result.keys()})
                best_parameter = parameters_result
                best_parameter['nu'] = n
                best_parameter['gamma'] = g

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

def compare_labels(target, prediction):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i, label in enumerate(target):

        if label == prediction[i]:
            if label == 1:
                tn += 1
            else:
                tp += 1
        elif label < prediction[i]:
            fn += 1
        else:
            fp += 1
    return tp, fp, tn, fn

def score(labelTrain, predictTrain, labelTest, predictTest):

    tp, fp, tn, fn = compare_labels(labelTrain, predictTrain)

    if labelTest != None:
        tp1, fp1, tn1, fn1 = compare_labels(labelTest, predictTest)
        tp += tp1
        fp += fp1
        tn += tn1
        fn += fn1
    #print "tn: %s, tp: %s, fn: %s, fp: %s" % (tn, tp, fn, fp)
    #print "fit + test: %s" % (len(labelTest) + len(labelTrain))
    if tp + fp > 0: prec = float(tp)/(tp + fp)
    else: prec = 0
    if tp + fn > 0: rec = float(tp)/(tp + fn)
    else: rec = 0
    if prec + rec > 0:
        f1 = 2*prec*rec/(prec+rec)
    else: f1 = 0
    if labelTest != None:
        err = float(fp + fn) / (len(labelTrain) + len(labelTest))
    else: err=float(fp + fn) / len(labelTrain)
    #print "err: %s" % err
    return prec, rec, f1, err

def show_confusion_matrix(true_label, pred_label):
    y_actu = pandas.Series([2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2], name='Actual')
    y_pred = pandas.Series([0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2], name='Predicted')
    df_confusion = pandas.crosstab(y_actu, y_pred)

def draw_confusion_matrix(target, prediction):
    # Compute confusion matrix
    cm = confusion_matrix(target, prediction)
    # Show confusion matrix in a separate window
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def evaluate_incr_ocsvm(cv_data, nu, gamma):
    batch_incr = {'precision': [0,0], 'recall': [0,0], 'f1-score': [0,0], 'error': [0,0]}
    cv_data_keys = cv_data.keys()
    n_data = len(cv_data_keys)
    for i in cv_data_keys:
        if i != 4: continue
        X_train, label_train, X_test, label_test = cv_data[i]
        train_size = ceil(len(X_train) * nu / 0.9)
        nu_new = nu
        nu_new = (len(X_train) * nu) / train_size
        #print nu

        #print "training batch version"
        clf_gold = ocsvm.OCSVM("rbf", nu=nu, gamma=gamma)
        clf_gold.fit(np.asarray(X_train), scale=len(X_train) * nu)
        train_predict = clf_gold.predict(X_train)
        test_predict = clf_gold.predict(X_test)
        pb, rb, fb, eb = score(label_train, train_predict, label_test, test_predict)

        #print "specific training batch version (+69)"

        clf = train(np.asarray(X_train), nu_new, gamma, size=train_size, init_ac=0)
        #clf = fit(np.asarray(X_train), nu, gamma, init_ac=nu)
        train_predict = clf.predict(X_train)
        test_predict = clf.predict(X_test)
        p, r, f, e = score(label_train, train_predict, label_test, test_predict)

        batch_incr['precision'][0] += p
        batch_incr['recall'][0] += r
        batch_incr['f1-score'][0] += f
        batch_incr['error'][0] += e
        batch_incr['precision'][1] += pb
        batch_incr['recall'][1] += rb
        batch_incr['f1-score'][1] += fb
        batch_incr['error'][1] += eb
    for k in batch_incr.keys():
        batch_incr[k][0] /= n_data
        batch_incr[k][1] /= n_data
    print pandas.DataFrame(batch_incr, index=['incremental', 'batch'])

    distance_pred_train = clf.decision_function(X_train)
    distance_pred_test = clf.decision_function(X_test)
    show_precision_recall(np.hstack((label_train, label_test)),
                          np.hstack((distance_pred_train, distance_pred_test)))
    draw_confusion_matrix(np.hstack((label_train, label_test)),
                          np.hstack((train_predict, test_predict)))

    return clf, X_train, X_test, train_predict, test_predict

def plot_coef_trajectory(alpha_traject):
    return True

def show_precision_recall(target, predicted):
    precision, recall, _ = precision_recall_curve(target, predicted)
    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall, precision,
                 label='Precision-recall curve')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()

def train(X_tra, nu, gamma, size = 20, init_ac=0):

    clf = ocsvm.OCSVM("rbf", nu=nu, gamma=gamma)
    #print "fit size: %s,%s" % (X_tra[0:size].shape[0], X_tra[0:size].shape[1])
    print "test batch version: "
    #
    print "nu: %s" % nu
    nu_test = nu*size / (size + 1)
    print "nu_test: %s" % nu_test
    print "test batch version alpha"
    print "size: %s" % size
    clf1 = ocsvm.OCSVM("rbf", nu=nu_test, gamma=gamma)
    clf1.fit(np.vstack((X_tra[-1:], X_tra[:size])), scale=nu_test*(size+1))
    #clf1.fit(X_tra[:size], scale=nu*size)


    a = clf1._data.alpha()
    print sum(a)
    e = clf._data._e
    C = 1
    inds = [i for i, bool in enumerate(np.all([a > e, a < C - e], axis=0)) if bool]
    inde = [i for i, bool in enumerate(a >= 1- e) if bool]
    indo = [i for i, bool in enumerate(a <= e) if bool]
    print "inds: %s" % inds
    print "KKT: %s" % clf1.KKT(clf1._data.X(), a)

    #print "X: %s" % clf1._data.X()

    print "rho: %s" % clf1._rho
    #print "a[180]: %s" % a[180]

    clf.fit(X_tra[:size], scale=nu*size)
    a = clf._data.alpha()
    inds = [i for i, bool in enumerate(np.all([a > 1e-5, a < 1 - 1e-5], axis=0)) if bool]

    #print "before incremental alphas"
    inde = [i for i, bool in enumerate(a >= 1- 1e-5) if bool]
    indo = [i for i, bool in enumerate(a <= 1e-5) if bool]
    #print "sum(a): %s" % sum(a)
    #print "a[inds]: %s" % a[inds]
    #print "inds: %s" % inds

    X_tra = X_tra[size:]
    clf.increment(X_tra)
    '''
    if not clf.KKT():
        print "KKT not satisfied"
        #sys.exit()
    for i,x in enumerate(X_tra):
        print "========================== INCREMENTAL %s" %i
        clf.increment(x, init_ac=init_ac)
        if not clf.KKT():
            print "KKT not satisfied"

            sys.exit()
        #print clf._data.alpha_s()
        #if i == 1: break
    '''
    clf.rho()
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
            X, label = load_data(datAll)
        else:
            datFile = [root + "/" + file for root, dirs, files in os.walk(path) for file in files if file.endswith(".dat") if root + "/" == path]
            X, label = load_data(datFile)
        #plot(X, label, D3=False)
        #plot(np.asarray(X), np.asarray(label))
        fiveFoldPath = path + datName + "-5-fold/" + datName + "-5-"
        X5fold = get_cv_data(fiveFoldPath)
        print "Find best parameters for %s" % datName
        bestParas = get_best_parameters(X5fold)

# find best parameters for all imbalanced datasets
# find it with batch ocsvm (faster)
def find_best_parameters_batch(path):
    imbalanced_sets = ["vehicle2", "vehicle3", "glass0",
                       "newthyroid2", "yeast3", "vehicle1",
                       "wisconsin", "page-blocks0", "new-thyroid1",
                       "haberman", "pima", "glass-0-1-2-3_vs_4-5-6",
                       "yeast1", "ecoli-0_vs_1", "glass6"]
    txtFiles = [root + "/" + file for root, dirs, files in os.walk(path)
                for file in files if file.endswith("-names.txt")]
    for f in txtFiles:
        path = os.path.dirname(f)
        file_name = os.path.basename(f)
        if file_name not in imbalanced_sets:

            datAll = path + file_name + ".dat"
            print datAll
            if os.path.isfile(datAll):
                X, label = load_data(datAll)
            else:
                datFile = [root + "/" + file for root, dirs, files in os.walk(path)
                           for file in files if file.endswith(".dat") if root + "/" == path]
                X, label = load_data(datFile)
            #plot(X, label, D3=False)
            #plot(np.asarray(X), np.asarray(label))
            fiveFoldPath = path + file_name + "-5-fold/" + file_name + "-5-"
            X5fold = get_cv_data(fiveFoldPath)
            print "Find best parameters for %s" % file_name
            bestParas = get_best_parameters(X5fold)

def find_best_paras_handwriting():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.data
    label = digits.target
    one_class = label == 2
    label_binary = np.ones(len(label)) * -1
    label_binary[one_class] = 1

def train_handwriting_digits():
    nu = 0.19
    gamma = 0.004
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.data
    label = digits.target
    one_class = label == 2
    label_binary = np.ones(len(label)) * -1
    label_binary[one_class] = 1

    clf_gold = ocsvm.OCSVM("rbf", nu=nu, gamma=gamma)
    clf_gold.fit(data, scale=data.shape[0] * nu)
    train_predict = clf_gold.predict(data)
    p, r, f, e = score(train_predict, label_binary, None, None)
    print "prec: %s, rec: %s, f1score: %s, err: %s" % (p, r, f, e)

def handwriting_digits_small():
    # The digits dataset
    digits = load_digits()

    # The data that we are interested in is made of 8x8 images of digits, let's
    # have a look at the first 3 images, stored in the `images` attribute of the
    # dataset.  If we were working from image files, we could load them using
    # pylab.imread.  Note that each image must have the same size. For these
    # images, we know which digit they represent: it is given in the 'target' of
    # the dataset.
    images_and_labels = list(zip(digits.images, digits.target))
    for index, (image, label) in enumerate(images_and_labels[:4]):
        plt.subplot(2, 4, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Training: %i' % label)

    # To apply a classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    target = digits.target



    # only take some classes of digits
    c1 = 1
    c2 = [2]
    target_bin_index = [i for i, t in enumerate(target) if t == c1 or t in c2]
    bin_samples = len(target_bin_index)
    binary_data = data[target_bin_index]
    binary_target = target[target_bin_index]
    binary_target[binary_target == c1] = 1
    binary_target[binary_target != c1] = -1

    nu_gamma_f1 = [0, 0, 0]
    nu_range = [0.1*i for i in range(1,10)]
    gamma_range = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    for nu in nu_range:
        for gamma in gamma_range:
            clf = ocsvm.OCSVM("rbf", nu=nu, gamma=gamma)
            clf.fit(binary_data[:bin_samples/2])
            expected = binary_target[:bin_samples/ 2:]
            predicted = clf.predict(binary_data[:bin_samples/2:])

            precision, recall, f1score, support = precision_recall_fscore_support(expected, predicted, average='binary')
            if f1score > nu_gamma_f1[2]:
                nu_gamma_f1 = [nu, gamma, f1score]

    print "nu_gamma_f1: %s" % nu_gamma_f1
    clf = ocsvm.OCSVM("rbf", nu=nu_gamma_f1[0], gamma=nu_gamma_f1[1])
    clf.fit(binary_data[:bin_samples/2])
    expected = binary_target[:bin_samples/ 2:]
    predicted = clf.predict(binary_data[:bin_samples/2:])
    print("Confusion matrix:\n%s" % confusion_matrix(expected, predicted))
    precision, recall, f1score, support = precision_recall_fscore_support(expected, predicted, average='binary')
    print "precision: %s, recall: %s, f1-score: %s" % (precision, recall, f1score)

    clf = ocsvm.OCSVM("rbf", nu=0.5, gamma=0.01)
    clf.fit(binary_data[:bin_samples/2])
    expected = binary_target[:bin_samples/ 2:]
    predicted = clf.predict(binary_data[:bin_samples/2:])
    print("Confusion matrix:\n%s" % confusion_matrix(expected, predicted))
    precision, recall, f1score, support = precision_recall_fscore_support(expected, predicted, average='binary')
    print "precision: %s, recall: %s, f1-score: %s" % (precision, recall, f1score)

if __name__ == "__main__":
    handwriting_digits_small()
    sys.exit()
    # evaluation for imbalanced datasets
    parameters_imbalanced_datasets = {
        'segment0': {'nu': 0.2, 'gamma': 1.4},
        'ecoli1': {'nu': 0.6, 'gamma': 0.2},
        'haberman': {'nu': 0.6, 'gamma': 1.6},
        'page-blocks0': {'nu': 0.4, 'gamma': 0.6},
        'pima': {'nu': 0.6, 'gamma': 0.6},
    }
    imbalanced_datasets_file_path = '/Users/LT/Documents/Uni/MA/increOCSVM/imbalanced_data/'
    txtFiles = [root + "/" + file for root, dirs, files in os.walk(imbalanced_datasets_file_path)
                for file in files if file.endswith("-names.txt")]
    for f in txtFiles:
        path = os.path.dirname(f)
        file_name = os.path.basename(f).replace('-names.txt', '').strip()
        if file_name in parameters_imbalanced_datasets and 'page-blocks0' in file_name:
            cv_5_path = os.path.join(path, file_name + '-5-fold/', file_name + '-5-')
            print file_name
            cv_5_data = get_cv_data(cv_5_path)
            nu = parameters_imbalanced_datasets[file_name]['nu']
            gamma = parameters_imbalanced_datasets[file_name]['gamma']
            print "increment evaluation"
            evaluate_incr_ocsvm(cv_5_data, nu, gamma)


    #findBestParasIncr("/mnt/project/predictppi/data/MA/increOCSVM/imbalanced_data/")
    #sys.exit()
    ##train_handwriting_digits()




