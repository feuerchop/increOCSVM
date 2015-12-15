__author__ = 'LT'
import sys
import matplotlib
matplotlib.use('Agg')
from sklearn import preprocessing, cross_validation, svm
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_recall_fscore_support, average_precision_score
import pandas as pd
from pandas import Series, crosstab
import numpy as np
from numpy import array, int8, uint8, zeros
import matplotlib.pyplot as plt
from profilehooks import profile
from sklearn.feature_selection import SelectKBest, chi2
from math import floor, ceil
from array import array as pyarray
import matplotlib
import os, struct, ocsvm, sys, plot_data
import cPickle as pickle
import random
novelty = -1

# load raw data from
# keel dataset
def load_raw_data(file_path):
    f = open(file_path)
    X_raw = []
    label_raw = []
    for l in f:
        if l.startswith("@"): continue
        p = l.split(",")
        X_raw.append(map(float,p[:-1]))
        l = p[-1:][0].strip()
        label_raw.append(l)
    return X_raw, label_raw

# normalize data
def norm_data(X_raw):
    # normalization
    X = np.asarray(X_raw)
    X = preprocessing.normalize(X)
    return X

# convert output to y in {-1,+1}
def convert_output(label_raw, class_labels):
    # we assume there are only two classes
    n_labels = [len(np.asarray(label_raw)[np.asarray(label_raw) == c]) for c in class_labels]
    # get labels with highest count
    max_label_count = max(n_labels)
    index_max_label = n_labels.index(max_label_count)
    pos_label = class_labels[index_max_label]
    return [1 if l == pos_label else -1 for l in label_raw]

# load data with given filepath
def load_data(file_path):
    X_raw, label_raw = load_raw_data(file_path)
    X = norm_data(X_raw)
    label = np.asarray(convert_output(label_raw, list(set(label_raw))))
    return X, label

#generates k-fold indeces with given number of samples
def get_kfold_cv(n_samples, kfold):
    return cross_validation.KFold(n_samples, n_folds=kfold)

# gridsearch for sklearn OCSVM
def grid_search_sklearn(X, label,
                split=0.8,
                nu_range=[0.05*i for i in range(1,20)],
                gamma_range=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30],
                novelty=novelty,
                verbose=True):
    all_data = []
    train_split = int(floor(X.shape[0]*split))
    nu_gamma_f1 = [0, 0, 0]
    for nu in nu_range:
        for gamma in gamma_range:
            clf = ocsvm.OCSVM("rbf", nu=nu, gamma=gamma)
            try:
                clf.fit(X[:train_split], scale=nu*len(X[:train_split]))
                expected = np.asarray(label) * novelty
                predicted = clf.predict(X) * novelty
                precision, recall, f1score, support = precision_recall_fscore_support(expected, predicted, average='binary')
                #print "nu: %s, gamma: %s -> precision: %s, recall: %s, f1: %s" % (nu, gamma, precision, recall, f1score)
                all_data.append([nu, gamma, precision, recall, f1score])
                if f1score > nu_gamma_f1[2] and recall != 1.0:
                    nu_gamma_f1 = [nu, gamma, f1score]
            except Exception,e:
                print "train error"
                continue
    all_data = sorted(all_data, key=lambda x: -x[4])
    if verbose:
        pd.set_option('display.max_rows', None)
        df = pd.DataFrame(all_data, columns=['nu', 'gamma', 'precision', 'recall', 'f1'])
        print df
    return nu_gamma_f1

# gridsearch for cvxopt OCSVM
def grid_search_cvxopt(X, label,
                split=0.8,
                nu_range=[0.05*i for i in range(1,20)],
                gamma_range=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30],
                novelty=novelty,
                verbose=True):
    all_data = []
    train_split = int(floor(X.shape[0]*split))
    nu_gamma_f1 = [0, 0, 0]
    for nu in nu_range:
        for gamma in gamma_range:
            clf = ocsvm.OCSVM("rbf", nu=nu, gamma=gamma)
            try:
                clf.fit(X[:train_split], scale=nu*len(X[:train_split]))
                expected = np.asarray(label) * novelty
                predicted = clf.predict(X) * novelty
                precision, recall, f1score, support = precision_recall_fscore_support(expected, predicted, average='binary')
                #print "nu: %s, gamma: %s -> precision: %s, recall: %s, f1: %s" % (nu, gamma, precision, recall, f1score)
                all_data.append([nu, gamma, precision, recall, f1score])
                if f1score > nu_gamma_f1[2] and recall != 1.0:
                    nu_gamma_f1 = [nu, gamma, f1score]
            except Exception,e:
                print "train error"
                continue
            print nu, gamma
    all_data = sorted(all_data, key=lambda x: -x[4])
    if verbose:
        pd.set_option('display.max_rows', None)
        df = pd.DataFrame(all_data, columns=['nu', 'gamma', 'precision', 'recall', 'f1'])
        print df
    return nu_gamma_f1

def evaluate_kdd99(X, label, nu, gamma):
    split = 0.8
    print "grid search incremental"
    all_data = []
    train_split = int(floor(X.shape[0]*split))
    nu_start = 0.975
    start_train_size = ceil(len(X) * split * nu / nu_start)
    clf = ocsvm.OCSVM("rbf", nu=nu, gamma=gamma)

    clf.fit(X[:train_split][:start_train_size], scale=nu*len(X[:train_split][:start_train_size]))
    clf.increment(X[:train_split][start_train_size:])
    expected = np.asarray(label) * novelty
    predicted = clf.predict(X) * novelty
    precision, recall, f1score, support = precision_recall_fscore_support(expected, predicted, average='binary')
    print "nu: %s, gamma: %s -> precision: %s, recall: %s, f1: %s" % (nu, gamma, precision, recall, f1score)

def grid_search_incre(X, label,
                split=0.8,
                nu_range=[0.01*i for i in range(1,100)],
                gamma_range=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30],
                novelty=novelty,
                verbose=True):
    print "grid search incremental"
    all_data = []
    train_split = int(floor(X.shape[0]*split))
    nu_gamma_f1 = [0, 0, 0]
    nu_start = 0.975
    kfold = get_kfold_cv(X.shape[0], 5)
    #kfold = pickle.load(open('datasets/kddcup99/best_kfold_pima.p'))
    stop = False
    for nu in nu_range:

        start_train_size = int(len(X) * split * nu / nu_start)
        for gamma in gamma_range:
            #if nu == 0.03 and gamma == 0.3:
            print nu, gamma
            precision_5fold = {'min': 1, 'max': 0, 'avg': 0}
            recall_5fold = {'min': 1, 'max': 0, 'avg': 0}
            i = 0
            for train_index, test_index in kfold:
                #try:

                X_train = X[train_index]
                label_train = label[train_index]
                clf = ocsvm.OCSVM("rbf", nu=nu_start, gamma=gamma)
                clf.fit(X_train[:start_train_size], scale=nu_start*len(X_train[:start_train_size]))
                success = clf.increment(X_train[start_train_size:])
                if not success: continue
                #split_1 = int(X_train[start_train_size:].shape[0] * 0.95)
                #clf.increment(X_train[start_train_size:][:split_1], init_ac=0)
                #success = clf.increment_supervised(X_train[start_train_size:][split_1:],
                #                         label_train[start_train_size:][split_1:],init_ac=0)
                #if not success: continue
                expected = np.asarray(label) * novelty
                predicted = clf.predict(X) * novelty
                precision, recall, f1score, _ = precision_recall_fscore_support(expected, predicted, average='binary')

                #print "precision: %s, recall: %s, f1score: %s" % (precision, recall, f1score)
                precision_5fold['avg'] += precision
                if precision < precision_5fold['min']:
                    precision_5fold['min'] = precision
                if precision > precision_5fold['max']:
                    precision_5fold['max'] = precision

                if recall < recall_5fold['min']:
                    recall_5fold['min'] = recall
                if recall > recall_5fold['max']:
                    recall_5fold['max'] = recall

                recall_5fold['avg'] += recall
                #confusion = output_cf(expected, predicted)
                #print("Confusion matrix:\n%s" % confusion)
                i += 1
                #if f1score > 0.86:
                    #stop = True
                    #kfold_result = [[train_index, test_index]]
                    #kfold_result = kfold
                    #break
                #except:
                #    print "train error"
            if i == 0: continue
            precision_5fold['avg'] /= float(i)
            recall_5fold['avg'] /= float(i)
            if precision_5fold['avg'] + recall_5fold['avg'] > 0:
                f1 = 2 * (precision_5fold['avg'] * recall_5fold['avg']) \
                     / (precision_5fold['avg'] + recall_5fold['avg'])
            else: f1 = 0
            #print "averages: nu: %s, gamma: %s -> precision: %s, recall: %s, f1: %s" \
                  #% (nu, gamma, precision_5fold['avg'], recall_5fold['avg'], f1)
            all_data.append([nu, gamma, precision_5fold['avg'], recall_5fold['avg'], f1])
            if f1 > nu_gamma_f1[2] and recall != 1.0:
                nu_gamma_f1 = [nu, gamma, f1]
                print "averages: nu: %s, gamma: %s -> precision: %s, recall: %s, f1: %s" \
                  % (nu, gamma, precision_5fold['avg'], recall_5fold['avg'], f1)
                if f1 > 0.85:
                    stop = True
            print "--------------------------------------------------------------------"
            if stop: break
        if stop: break



    all_data = sorted(all_data, key=lambda x: -x[4])
    if verbose:
        pd.set_option('display.max_rows', None)
        df = pd.DataFrame(all_data, columns=['nu', 'gamma', 'precision', 'recall', 'f1'])
        print df
    return nu_gamma_f1, kfold

def evaluate_dataset(X, label, mnist=False, dataset=None):
    split = 0.8
    train_split = int(floor(X.shape[0]*split))
    nu_gamma_f1, kfold = grid_search_incre(X, label, split=split,
                                           nu_range=[0.01*i for i in range(80,95)],
                                           gamma_range=[0.01, 0.03, 0.1, 0.3, 1, 3])#,
                              #nu_range=[0.05*i for i in range(1,11)],
                              #gamma_range=[0.01, 0.03, 0.1, 0.3, 1, 3])
    pickle.dump(kfold, open('datasets/kddcup99/best_kfold_%s.p' % dataset, 'w+'))
    # train with best
    print "nu_gamma_f1: %s" % nu_gamma_f1
    clf = ocsvm.OCSVM("rbf", nu=nu_gamma_f1[0], gamma=nu_gamma_f1[1])
    clf.fit(X[:train_split], scale=nu_gamma_f1[0]*len(X[:train_split]))

    expected = np.asarray(label)*novelty
    if mnist:
        expected = expected.tolist()
        expected = [e[0] for e in expected]
    predicted = clf.predict(X)*novelty


    confusion = output_cf(expected, predicted)
    print("Confusion matrix:\n%s" % confusion)
    #print("Confusion matrix:\n%s" % confusion_matrix(expected, predicted))
    precision, recall, f1score, support = precision_recall_fscore_support(expected, predicted, average='binary')
    print "precision: %s, recall: %s, f1-score: %s" % (precision, recall, f1score)

def precision(confusion_matrix):
    tp = confusion_matrix[1][1]
    fp = confusion_matrix

@profile
def sklearn_ocsvm(clf, X_train):
    clf.fit(X_train)

@profile
def cvxopt_ocsvm(clf, X_train, scale=1, nu=None):
    clf.fit(X_train, scale=scale, v_target=nu)

@profile
def incremental_ocsvm(clf, X_train, train_size, init_ac, break_count):
    clf.increment(X_train[train_size:], init_ac=init_ac)

def get_precision_recall_data(expected, predicted_prob):
    precision, recall, _ = precision_recall_curve(expected, predicted_prob)
    return precision, recall,\
           average_precision_score(expected, predicted_prob)

def evaluate_incremental(X, label, nu, gamma, train_size=20, zero_init=True, dataset=None):
    nu_start = 0.975
    dataset = dataset if dataset is not None else "mnist"
    print "data size: %s, nu: %s, gamma: %s" % (len(X), nu, gamma)
    i = 1
    kfold = get_kfold_cv(X.shape[0], 5)
    #kfold = pickle.load(open('datasets/kddcup99/best_kfold_None.p'))
    expected = label*novelty
    expected = expected.ravel()


    precision_recall_f1 = {'IncreOCSVM': {'Precision': 0,
                                          'Recall': 0,
                                          'F1': 0},
                           'cvxopt-OCSVM': {'Precision': 0,
                                          'Recall': 0,
                                          'F1': 0},
                           'sklearn-OCSVM': {'Precision': 0,
                                          'Recall': 0,
                                          'F1': 0}}
    precision_recall_avg = []
    for train_index, test_index in kfold:

        print "============ %s. Fold of CV ============" % i
        print "1) Incremental OCSVM"
        break_count = len(X) - train_size
        X_train, X_test = X[train_index], X[test_index]
        if zero_init:
            clf_inc = ocsvm.OCSVM("rbf", nu=nu_start, gamma=gamma)
            if nu < nu_start:
                train_size = ceil(len(X_train) * nu / nu_start)

            clf_inc.fit(X_train[:train_size], scale=nu_start*train_size, rho=False)
            incremental_ocsvm(clf_inc, X_train, train_size, 0, break_count)
        else:
            clf_inc = ocsvm.OCSVM("rbf", nu=nu, gamma=gamma)
            clf_inc.fit(X_train[:train_size], scale=nu_start*train_size, rho=False)
            incremental_ocsvm(clf_inc, X_train, train_size, nu, break_count)
        predicted = clf_inc.predict(X)*novelty
        ### confusion matrix and precision recall only for the first fold
        if i == 1:
            confusion1 = confusion_matrix(expected, predicted)
            predicted_prob = clf_inc.decision_function(X)*novelty
            both_labels = np.concatenate((expected.reshape((len(expected),1)), predicted_prob.reshape((len(predicted_prob),1))), axis=1)
            both_labels = both_labels[both_labels[:, 1].argsort()]
            tmp_dict = {'label': 'Incremental OCSVM'}
            tmp_dict['precision'], tmp_dict['recall'], tmp_dict['avg_precision'] = \
            get_precision_recall_data(both_labels[:,0][::-1], both_labels[:,1][::-1])
            precision_recall_avg.append(tmp_dict)
        confusion = output_cf(expected, predicted)

        print("Confusion matrix:\n%s" % confusion)
        precision, recall, f1score, support = precision_recall_fscore_support(expected, predicted, average='binary')
        precision_recall_f1['IncreOCSVM']['Precision'] += precision
        precision_recall_f1['IncreOCSVM']['Recall'] += recall
        precision_recall_f1['IncreOCSVM']['F1'] += f1score
        print "precision: %s, recall: %s, f1-score: %s" % (precision, recall, f1score)
        print "Number of support vectors: %s" % len(clf_inc._data.alpha_s())
        print "-----------"
        if X_train.shape[0] < 5000:
            print "2) cvxopt-OCSVM"
            clf = ocsvm.OCSVM("rbf", nu=nu, gamma=gamma)
            cvxopt_ocsvm(clf, X_train, nu*X_train.shape[0], nu)
            predicted = clf.predict(X)*novelty
            ### confusion matrix and precision recall only for the first fold
            if i == 1:
                confusion2 = confusion_matrix(expected, predicted)
                predicted_prob = clf.decision_function(X)*novelty
                both_labels = np.concatenate((expected.reshape((len(expected),1)), predicted_prob.reshape((len(predicted_prob),1))), axis=1)
                both_labels = both_labels[both_labels[:, 1].argsort()]
                tmp_dict = {'label': 'cvxopt-OCSVM'}
                tmp_dict['precision'], tmp_dict['recall'], tmp_dict['avg_precision'] = \
                get_precision_recall_data(both_labels[:,0][::-1], both_labels[:,1][::-1])
                precision_recall_avg.append(tmp_dict)
            confusion = output_cf(expected, predicted)
            print("Confusion matrix:\n%s" % confusion)
            precision, recall, f1score, support = precision_recall_fscore_support(expected, predicted, average='binary')
            precision_recall_f1['cvxopt-OCSVM']['Precision'] += precision
            precision_recall_f1['cvxopt-OCSVM']['Recall'] += recall
            precision_recall_f1['cvxopt-OCSVM']['F1'] += f1score
            print "precision: %s, recall: %s, f1-score: %s" % (precision, recall, f1score)
            print "Number of support vectors: %s" % len(clf._data.alpha_s())
            print "---------"
        else:
            print "2) Datasize too big for cvxopt-OCSVM. Not enough memory."
        print "3) sklearn-OCSVM"
        clf = svm.OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
        sklearn_ocsvm(clf, X_train)
        predicted = clf.predict(X)*novelty
        ### confusion matrix and precision recall only for the first fold
        if i == 1:

            confusion3 = confusion_matrix(expected, predicted)
            predicted_prob = clf.decision_function(X)*novelty
            both_labels = np.concatenate((expected.reshape((len(expected),1)), predicted_prob.reshape((len(predicted_prob),1))), axis=1)
            both_labels = both_labels[both_labels[:, 1].argsort()]
            tmp_dict = {'label': 'sklearn-OCSVM'}
            tmp_dict['precision'], tmp_dict['recall'], tmp_dict['avg_precision'] = \
            get_precision_recall_data(both_labels[:,0][::-1], both_labels[:,1][::-1])
            precision_recall_avg.append(tmp_dict)
        confusion = output_cf(expected, predicted)
        print("Confusion matrix:\n%s" % confusion)
        precision, recall, f1score, support = precision_recall_fscore_support(expected, predicted, average='binary')
        precision_recall_f1['sklearn-OCSVM']['Precision'] += precision
        precision_recall_f1['sklearn-OCSVM']['Recall'] += recall
        precision_recall_f1['sklearn-OCSVM']['F1'] += f1score
        print "Number of support vectors: %s" % len(clf.support_vectors_)
        print "precision: %s, recall: %s, f1-score: %s" % (precision, recall, f1score)
        #plot_data.plot_multiple_cf(cm1_normalized, ['negative', 'positive'], cm2_normalized, cm3_normalized, colorbar=True)
        if i == 1 and X_train.shape[0] < 5000:
            if zero_init:
                zi = "zero"
            else:
                zi = "nonzero"
            plot_data.plot_multiple_cf(confusion1, ['negative', 'positive'], ['Incremental OCSVM', 'cvxopt-OCSVM', 'sklearn-OCSVM'],
                                   confusion2, confusion3, colorbar=True,
                                   filename_prefix="results_performance/%s_%s_%s_%s-%s" % (dataset, nu, gamma, zi, i))
            plot_data.plot_multiple_precision_recall_curves(precision_recall_avg,
                                                            filename_prefix="results_performance/%s_%s_%s_%s-%s" % (dataset, nu, gamma, zi, i))
        i += 1
        #break

    print "========================================"
    print "Average Incremental OCSVM results:"
    precision = precision_recall_f1['IncreOCSVM']['Precision'] / (i-1)
    recall = precision_recall_f1['IncreOCSVM']['Recall'] / (i-1)
    f1 = 2 * precision * recall / (precision + recall)
    print "precision: %s, recall: %s, f1-score: %s" % (precision,
                                                       recall,
                                                       f1)
    print "Average cvxopt-OCSVM results:"
    precision = precision_recall_f1['cvxopt-OCSVM']['Precision'] / (i-1)
    recall = precision_recall_f1['cvxopt-OCSVM']['Recall'] / (i-1)
    f1 = 2 * precision * recall / (precision + recall)
    print "precision: %s, recall: %s, f1-score: %s" % (precision,
                                                       recall,
                                                       f1)
    print "Average sklearn-OCSVM results:"
    precision = precision_recall_f1['sklearn-OCSVM']['Precision'] / (i-1)
    recall = precision_recall_f1['sklearn-OCSVM']['Recall'] / (i-1)
    f1 = 2 * precision * recall / (precision + recall)
    print "precision: %s, recall: %s, f1-score: %s" % (precision,
                                                       recall,
                                                       f1)

def evaluate_semi_supervised(X, label, nu, gamma, train_size=20,
                             zero_init=True, dataset=None, ratio=0.99):
    nu_start = 0.975
    dataset = dataset if dataset is not None else "mnist"
    print "data size: %s, nu: %s, gamma: %s" % (len(X), nu, gamma)
    i = 1
    #kfold = get_kfold_cv(X.shape[0], 5)
    kfold = pickle.load(open('datasets/kddcup99/best_kfold_None.p'))
    expected = label*novelty
    expected = expected.ravel()


    precision_recall_f1 = {'IncreOCSVM': {'Precision': 0,
                                          'Recall': 0,
                                          'F1': 0},
                           'cvxopt-OCSVM': {'Precision': 0,
                                          'Recall': 0,
                                          'F1': 0},
                           'sklearn-OCSVM': {'Precision': 0,
                                          'Recall': 0,
                                          'F1': 0}}
    precision_recall_avg = []
    for train_index, test_index in kfold:

        print "============ %s. Fold of CV ============" % i
        print "1) Incremental OCSVM"
        break_count = len(X) - train_size
        X_train, X_test = X[train_index], X[test_index]
        label_train, label_test = label[train_index], label[test_index]

        if zero_init:
            clf_inc = ocsvm.OCSVM("rbf", nu=nu_start, gamma=gamma)
            if nu < nu_start:
                train_size = ceil(len(X_train) * nu / nu_start)

                clf_inc.fit(X_train[:train_size], scale=nu_start*train_size, rho=False)
                split = X_train[train_size:].shape[0] * ratio
                clf_inc.increment(X_train[train_size:][:split], init_ac=0)
                clf_inc.increment_supervised(X_train[train_size:][split:],
                                             label_train[train_size:][split:],init_ac=0)

        else:
            clf_inc = ocsvm.OCSVM("rbf", nu=nu, gamma=gamma)
            clf_inc.fit(X_train[:train_size], scale=nu_start*train_size, rho=False)
            incremental_ocsvm(clf_inc, X_train, train_size, nu, break_count)
        predicted = clf_inc.predict(X)*novelty
        ### confusion matrix and precision recall only for the first fold
        if i == 5:
            confusion1 = confusion_matrix(expected, predicted)
            predicted_prob = clf_inc.decision_function(X)*novelty
            both_labels = np.concatenate((expected.reshape((len(expected),1)), predicted_prob.reshape((len(predicted_prob),1))), axis=1)
            both_labels = both_labels[both_labels[:, 1].argsort()]
            tmp_dict = {'label': 'Incremental Semi-Supervised OCSVM'}
            tmp_dict['precision'], tmp_dict['recall'], tmp_dict['avg_precision'] = \
            get_precision_recall_data(both_labels[:,0][::-1], both_labels[:,1][::-1])
            precision_recall_avg.append(tmp_dict)
        confusion = output_cf(expected, predicted)

        print("Confusion matrix:\n%s" % confusion)
        precision, recall, f1score, support = precision_recall_fscore_support(expected, predicted, average='binary')
        precision_recall_f1['IncreOCSVM']['Precision'] += precision
        precision_recall_f1['IncreOCSVM']['Recall'] += recall
        precision_recall_f1['IncreOCSVM']['F1'] += f1score
        print "precision: %s, recall: %s, f1-score: %s" % (precision, recall, f1score)
        print "Number of support vectors: %s" % len(clf_inc._data.alpha_s())
        print "-----------"
        if X_train.shape[0] < 1000:
            print "2) cvxopt-OCSVM"
            clf = ocsvm.OCSVM("rbf", nu=nu, gamma=gamma)
            cvxopt_ocsvm(clf, X_train, nu*X_train.shape[0], nu)
            predicted = clf.predict(X)*novelty
            ### confusion matrix and precision recall only for the first fold
            if i == 5:
                confusion2 = confusion_matrix(expected, predicted)
                predicted_prob = clf.decision_function(X)*novelty
                both_labels = np.concatenate((expected.reshape((len(expected),1)), predicted_prob.reshape((len(predicted_prob),1))), axis=1)
                both_labels = both_labels[both_labels[:, 1].argsort()]
                tmp_dict = {'label': 'cvxopt-OCSVM'}
                tmp_dict['precision'], tmp_dict['recall'], tmp_dict['avg_precision'] = \
                get_precision_recall_data(both_labels[:,0][::-1], both_labels[:,1][::-1])
                precision_recall_avg.append(tmp_dict)
            confusion = output_cf(expected, predicted)
            print("Confusion matrix:\n%s" % confusion)
            precision, recall, f1score, support = precision_recall_fscore_support(expected, predicted, average='binary')
            precision_recall_f1['cvxopt-OCSVM']['Precision'] += precision
            precision_recall_f1['cvxopt-OCSVM']['Recall'] += recall
            precision_recall_f1['cvxopt-OCSVM']['F1'] += f1score
            print "precision: %s, recall: %s, f1-score: %s" % (precision, recall, f1score)
            print "Number of support vectors: %s" % len(clf._data.alpha_s())
            print "---------"
        else:
            print "2) Datasize too big for cvxopt-OCSVM. Not enough memory."
        print "3) sklearn-OCSVM"
        clf = svm.OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
        sklearn_ocsvm(clf, X_train)
        predicted = clf.predict(X)*novelty
        ### confusion matrix and precision recall only for the first fold
        if i == 5:

            confusion3 = confusion_matrix(expected, predicted)
            predicted_prob = clf.decision_function(X)*novelty
            both_labels = np.concatenate((expected.reshape((len(expected),1)), predicted_prob.reshape((len(predicted_prob),1))), axis=1)
            both_labels = both_labels[both_labels[:, 1].argsort()]
            tmp_dict = {'label': 'sklearn-OCSVM'}
            tmp_dict['precision'], tmp_dict['recall'], tmp_dict['avg_precision'] = \
            get_precision_recall_data(both_labels[:,0][::-1], both_labels[:,1][::-1])
            precision_recall_avg.append(tmp_dict)
        confusion = output_cf(expected, predicted)
        print("Confusion matrix:\n%s" % confusion)
        precision, recall, f1score, support = precision_recall_fscore_support(expected, predicted, average='binary')
        precision_recall_f1['sklearn-OCSVM']['Precision'] += precision
        precision_recall_f1['sklearn-OCSVM']['Recall'] += recall
        precision_recall_f1['sklearn-OCSVM']['F1'] += f1score
        print "Number of support vectors: %s" % len(clf.support_vectors_)
        print "precision: %s, recall: %s, f1-score: %s" % (precision, recall, f1score)
        #plot_data.plot_multiple_cf(cm1_normalized, ['negative', 'positive'], cm2_normalized, cm3_normalized, colorbar=True)
        if i == 5 and X_train.shape[0] < 1000:
            if zero_init:
                zi = "zero"
            else:
                zi = "nonzero"
            plot_data.plot_multiple_cf(confusion1, ['negative', 'positive'], ['Incremental OCSVM', 'cvxopt-OCSVM', 'sklearn-OCSVM'],
                                   confusion2, confusion3, colorbar=True,
                                   filename_prefix="results_performance/%s_%s_%s_%s-%s" % (dataset, nu, gamma, zi, i))
            plot_data.plot_multiple_precision_recall_curves(precision_recall_avg,
                                                            filename_prefix="results_performance/%s_%s_%s_%s-%s" % (dataset, nu, gamma, zi, i))
        i += 1
        #break

    print "========================================"
    print "Average Incremental OCSVM results:"
    precision = precision_recall_f1['IncreOCSVM']['Precision'] / (i-1)
    recall = precision_recall_f1['IncreOCSVM']['Recall'] / (i-1)
    f1 = 2 * precision * recall / (precision + recall)
    print "precision: %s, recall: %s, f1-score: %s" % (precision,
                                                       recall,
                                                       f1)
    print "Average cvxopt-OCSVM results:"
    precision = precision_recall_f1['cvxopt-OCSVM']['Precision'] / (i-1)
    recall = precision_recall_f1['cvxopt-OCSVM']['Recall'] / (i-1)
    f1 = 2 * precision * recall / (precision + recall)
    print "precision: %s, recall: %s, f1-score: %s" % (precision,
                                                       recall,
                                                       f1)
    print "Average sklearn-OCSVM results:"
    precision = precision_recall_f1['sklearn-OCSVM']['Precision'] / (i-1)
    recall = precision_recall_f1['sklearn-OCSVM']['Recall'] / (i-1)
    f1 = 2 * precision * recall / (precision + recall)
    print "precision: %s, recall: %s, f1-score: %s" % (precision,
                                                       recall,
                                                       f1)

def output_cf(expected, predictions):
    expected = Series(expected, name="Target")
    predictions = Series(predictions, name="Prediction")
    return crosstab(expected, predictions)

def load_wisconsin():
    wisconsin = "/Users/LT/Documents/Uni/MA/increOCSVM/datasets/wisconsin/wisconsin.dat"
    return load_data(wisconsin)

def load_ecoli1():
    ecoli1 = "datasets/ecoli1/ecoli1.dat"
    return load_data(ecoli1)

def load_digits_small():
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

    nu_gamma_precision = [0, 0, 0]
    nu_range = [0.1*i for i in range(1,10)]
    gamma_range = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    for nu in nu_range:
        for gamma in gamma_range:
            clf = ocsvm.OCSVM("rbf", nu=nu, gamma=gamma)
            clf.fit(binary_data[:bin_samples/2])
            expected = binary_target[:bin_samples/ 2:]
            predicted = clf.predict(binary_data[:bin_samples/2:])

            precision, recall, f1score, _ = precision_recall_fscore_support(expected, predicted, average='binary')
            if precision > nu_gamma_precision[2]:
                nu_gamma_precision = [nu, gamma, precision]

    print "nu_gamma_precision: %s" % nu_gamma_precision
    clf = ocsvm.OCSVM("rbf", nu=nu_gamma_precision[0], gamma=nu_gamma_precision[1])
    clf.fit(binary_data[:bin_samples/2])
    expected = binary_target[:bin_samples/ 2:]
    predicted = clf.predict(binary_data[:bin_samples/2:])
    print("Confusion matrix:\n%s" % confusion_matrix(expected, predicted))
    precision, recall, f1score, support = precision_recall_fscore_support(expected, predicted, average='binary')
    print "precision: %s, recall: %s, f1-score: %s" % (precision, recall, f1score)

def load_haberman():
    haberman = "datasets/haberman/haberman.dat"
    return load_data(haberman)

def load_pima():
    pima = "datasets/pima/pima.dat"
    return load_data(pima)

def load_pageblocks0():
    pb = "datasets/page-blocks0/page-blocks0.dat"
    return load_data(pb)

def load_yeast1():
    yeast = "datasets/yeast1/yeast1.dat"
    return load_data(yeast)

def load_segment0():
    segment = 'datasets/segment0/segment0.dat'
    return load_data(segment)

# Loads MNIST files into 3D numpy arrays
def load_mnist(dataset="training",
               digits=np.arange(10),
               path="raw_data/digits_handwriting/"):

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols :
        (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

def reshape_data(data):
    n_data = data.shape[0]
    return data.reshape((n_data, -1))

def load_digits(classes=1, max_size=None):
    images_labels = None

    for c in range(classes):
        images_train, labels_train = load_mnist('training', digits=[c + 1])
        images_train = reshape_data(images_train)
        images_test, labels_test = load_mnist('testing', digits=[c + 1])
        images_test = reshape_data(images_test)
        if c > 0:
            labels_train[:] = -1
            labels_test[:] = -1
        train = np.concatenate((images_train, labels_train), axis=1)
        test = np.concatenate((images_test, labels_test), axis=1)
        if images_labels is None:
            images_labels = np.concatenate((train, test), axis=0)
        else:
            new_images_labels = np.concatenate((train, test), axis=0)
            images_labels = np.concatenate((images_labels, new_images_labels), axis=0)
    if max_size is not None:
        if max_size > images_labels.shape[0]:
            images_labels_all = None
            for c in range(classes,9):
                images_train, labels_train = load_mnist('training', digits=[c + 1])
                images_train = reshape_data(images_train)
                images_test, labels_test = load_mnist('testing', digits=[c + 1])
                images_test = reshape_data(images_test)
                labels_train[:] = -1
                labels_test[:] = -1
                train = np.concatenate((images_train, labels_train), axis=1)
                test = np.concatenate((images_test, labels_test), axis=1)
                if images_labels_all is None:
                    images_labels_all = np.concatenate((train, test), axis=0)
                else:
                    new_images_labels = np.concatenate((train, test), axis=0)
                    images_labels_all = np.concatenate((images_labels_all, new_images_labels), axis=0)
            np.random.shuffle(images_labels_all)
            images_labels = np.concatenate((images_labels,
                                            images_labels_all[0:max_size - images_labels.shape[0], :]), axis=0)


    np.random.shuffle(images_labels)
    return images_labels[:, :-1], images_labels[:, -1:]


def load_random_data(size=20, variance=2, density=0.3):
    half_size = size/2
    # Generate train data
    X = density * np.random.randn(half_size, 2)
    X_train = np.r_[X + variance, X - variance]
    # Generate some regular novel observations
    X = density * np.random.randn(half_size * 0.2, 2)
    X_test = np.r_[X + variance, X - variance]
    # Generate some abnormal novel observations
    X_outliers = np.random.uniform(low=-variance*2, high=variance*2, size=(half_size * 0.2, 2))

    return X_train, X_test, X_outliers

def plot_random_data(X_train, X_test, X_outliers):
    plt.scatter(X_train[:, 0], X_train[:, 1], c='white')
    plt.scatter(X_test[:, 0], X_test[:, 1], c='green')
    plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red')
    plt.show()

def classify_random_data(X_train, X_test, X_outliers, variance=20):
    # fit the model
    xx, yy = np.meshgrid(np.linspace(-variance, variance, 500), np.linspace(-variance, variance, 500))
    #clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf = ocsvm.OCSVM(nu=0.1, gamma=0.1)
    #clf.fit(X_train)
    clf.fit(X_train, scale=0.1*X_train.shape[0])
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(X_outliers)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

    # plot the line, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title("Novelty Detection")
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')

    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white')
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green')
    c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red')
    plt.axis('tight')
    plt.xlim((-variance, variance))
    plt.ylim((-variance, variance))
    plt.legend([a.collections[0], b1, b2, c],
               ["learned frontier", "training observations",
                "new regular observations", "new abnormal observations"],
               loc="upper left",
               prop=matplotlib.font_manager.FontProperties(size=11))
    plt.xlabel(
        "error train: %d/200 ; errors novel regular: %d/40 ; "
        "errors novel abnormal: %d/40"
        % (n_error_train, n_error_test, n_error_outliers))
    plt.show()
def profile_incremental(X_train, labels=None):
    nu_start = 0.95
    nu = 0.3
    clf = ocsvm.OCSVM("rbf", nu=nu_start, gamma=10)
    if nu < nu_start:
        train_size = ceil(len(X_train) * nu / nu_start)
    break_count = X_train.shape[0]
    incremental_ocsvm(clf, X_train, train_size, nu_start*train_size, 0, break_count)
    #profile.runctx('incremental_ocsvm(clf, X_train, train_size, scale, ac, break_count)',
    #               globals(), {'clf':clf, 'X_train': X_train, 'train_size': train_size,
    #                           'scale':  nu_start*train_size, 'ac': 0, 'break_count': break_count},
    #               filename='stats')
    #p = pstats.Stats('stats')
    #p.strip_dirs().sort_stats('cumulative').print_stats()
    expected = labels*novelty
    predicted = clf.predict(X_train)*novelty
    predicted[predicted == 0] = novelty
    confusion = confusion_matrix(expected, predicted)
    print("Confusion matrix:\n%s" % confusion)
    precision, recall, f1score, support = precision_recall_fscore_support(expected, predicted, average='binary')
    print "precision: %s, recall: %s, f1-score: %s" % (precision, recall, f1score)

def reduce_mnist_features(X):
    change = True
    i = 0
    while change:
        if X[:,i][X[:,i] == 0].shape[0] == X.shape[0]:
            X = X[:,i+1:]
            i += 1
        else:
            change = False
    change = True
    while change:
        if X[:,-1][X[:,-1] == 0].shape[0] == X.shape[0]:
            X = X[:,:-1]
        else:
            change = False
    return X

def train_cvxopt_ocsvm(X_train):
    nu = 0.2
    gamma = 1
    clf = ocsvm.OCSVM("rbf", nu=nu, gamma=gamma)
    cvxopt_ocsvm(clf, X_train, nu*X_train.shape[0], nu)

def evaluate_mnist(method, size, nu=None, gamma=None, zero_init=False):

    #### grid search or evaluation for mnist
    if len(sys.argv) > 1:
        print sys.argv
        size = int(sys.argv[1])
    else:
        size = 200
    classes = 2
    print "mnist classes = %s" % classes
    X, label = load_digits(classes=classes, max_size=30000)
    X = reduce_mnist_features(X[:size])
    label = label[:size]

    print "size: %s" % X.shape[0]
    print label[label == 1].shape
    print label[label == -1].shape
    #evaluate_dataset(X,label, mnist=True)
    #profile_incremental(X,label)
    #evaluate_incremental(X, label, 0.2, 1)
    print train_cvxopt_ocsvm(X)
    #evaluate_incremental(X, label, 0.5, 10, zero_init=False)
    sys.exit()

def load_shuttle():
    shuttle_train = "datasets/shuttle/shuttle.trn"
    shuttle_test = "datasets/shuttle/shuttle.tst"
    data = []
    with open(shuttle_train) as f:
        for l in f:
            if len(l.strip()) > 0:
                data.append(map(int, l.strip().split(" ")))
    with open(shuttle_test) as f:
        for l in f:
            if len(l.strip()) > 0:
                data.append(map(int, l.strip().split(" ")))
    random.shuffle(data)
    data = np.asarray(data)
    X = data[:, :-1]
    label = data[:, -1:].T[0]
    label[label != 1] = -1
    return X, label

def evaluate_shuttle(size, nu, gamma):
    #shuttle
    X, label = load_shuttle()


    X = X[:size]
    label = label[:size]
    print "Data distribution (-1, 1): %s, %s" % \
          (label[label==-1].shape[0], label[label==1].shape[0])
    #evaluate_dataset(X, label)
    evaluate_incremental(X[:size], label[:size], nu, gamma)
    evaluate_semi_supervised(X[:size], label[:size], nu, gamma)

def load_kdd99(maxsize=None):
    data = []
    data_file = "datasets/kddcup99/kddcup.data"
    with open(data_file) as f:
        for line in f:
            if len(line.strip()) > 0:
                row = line.split(",")
                row[-1] = row[-1].strip(".\n")
                data.append(row)
            if maxsize is not None:
                if len(data) % maxsize == 0:
                    break
    n_row = len(data)
    n_col = len(data[0])
    arr_data = np.asarray(data)
    normal_size = arr_data[arr_data[:,-1] == "normal"].shape[0]
    print "normal data: %s" % normal_size
    print "outlier data: %s" % (n_row - normal_size)
    print "number of columns: %s" % n_col
    outliers = np.unique(arr_data[:,-1]).tolist()
    print "outlier types: %s" % ",".join(outliers)
    for outlier in outliers:
        print "number of data for %s: %s" % (outlier, arr_data[arr_data[:,-1] == outlier].shape[0])
    print "dump to file"
    pickle.dump(arr_data, open("datasets/kddcup99/kddcup_data_processed.p", 'w+'))
def load_kdd99_from_dump(outlier=None, maxsize=None, features=None, outlier_ratio=None):
    mapping = {1: ['icmp', 'tcp', 'udp'],
               2: ['auth', 'domain_u', 'eco_i', 'ecr_i', 'finger', 'ftp', 'http',
                   'ntp_u', 'smtp', 'telnet', 'other', 'urp_i'],
               3: ['S1', 'S2', 'S3', 'SF', 'REJ', 'OTH', 'RSTO']}
    map_to_int = [0] + range(4,24) + range(31,33)
    map_to_float = range(24,31) + range(33,41)
    print "load from dump"
    data = pickle.load(open("datasets/kddcup99/kddcup_data_processed.p", "r"))
    # feature selection
    n_row, n_col = data.shape
    print "number of features: %s" % n_col
    print "feature selection"
    if outlier is not None:
        if outlier_ratio is None:
            data_filtered = data[np.any((data[:,-1] == outlier, data[:,-1] == "normal"), axis=0),:]
        else:
            n_normal = data[data[:,-1] == "normal",:].shape[0]
            n_outlier = int(n_normal*outlier_ratio)
            data_filtered = np.vstack((
                data[data[:,-1] == "normal"],
                data[data[:,-1] == outlier][:n_outlier,:]))
    else:
        data_filtered = data


    #get unique features and indices
    #for i in range(n_col):
    #    print i
    #    print np.unique(data_filtered[:,i])
    #sys.exit()
    for i,l in mapping.iteritems():
        for j,item in enumerate(l):
            filter = data_filtered[:,i] == item
            data_filtered[filter, i] = j
    np.random.shuffle(data_filtered)
    if maxsize is None:
        X = data_filtered[:, :-1].astype(float)
        label = data_filtered[:, -1]
    else:
        X = data_filtered[:maxsize, :-1].astype(float)
        label = data_filtered[:maxsize, -1]
    label[label == "normal"] = 1
    label[label == outlier] = -1
    label = label.astype(int)
    if features is not None:
        X = SelectKBest(chi2, k=features).fit_transform(X, label)
    print np.unique(label)
    # return X, label
    print "number of features after feature selection: %s" % X.shape[0]

def evaluate_imbalance(dataset, nu=None, gamma=None, zero_init=True):

    if dataset == "ecoli1":
        X, label = load_ecoli1()
    elif dataset == "pima":
        X, label = load_pima()
    elif dataset == "yeast1":
        X, label = load_yeast1()
    elif dataset == "haberman":
        X, label = load_haberman()
    elif dataset == "segment0":
        X, label = load_segment0()
    elif dataset == "page-blocks0":
        X, label = load_pageblocks0()
    elif dataset == "yeast1":
        X, label = load_yeast1()
    else:
        X, label = load_ecoli1()
        dataset = "ecoli1"
    print "distribution"
    print label[label==1].shape[0], label[label==-1].shape[0]
    #evaluate_dataset(X, label, dataset=dataset)
    evaluate_incremental(X, label, nu, gamma, zero_init=True, dataset=dataset)

def evaluate_mnist(size, classes, nu=None, gamma=None):
    print "mnist classes = %s" % classes
    #X, label = load_digits(classes=classes, max_size=30000)
    #pickle.dump(X, open('datasets/mnist/X.p', 'w+'))
    #pickle.dump(label, open('datasets/mnist/label.p', 'w+'))
    X = pickle.load(open('datasets/mnist/X.p'))
    label = pickle.load(open('datasets/mnist/label.p'))

    X = reduce_mnist_features(X[:size])
    label = label[:size]
    print "size: %s" % X.shape[0]
    print label[label == 1].shape
    print label[label == -1].shape
    #evaluate_dataset(X,label, mnist=True)
    #profile_incremental(X,label)
    evaluate_incremental(X, label, nu, gamma)
    #print train_cvxopt_ocsvm(X)

def evaluate_kdd99(size, nu=None, gamma=None):
    #load_kdd99(maxsize=100000)
    #X, label = load_kdd99_from_dump("smurf", outlier_ratio=0.05, maxsize=5000, features=6)
    #pickle.dump(X, open('datasets/kddcup99/X.p', 'w+'))
    #pickle.dump(label, open('datasets/kddcup99/label.p', 'w+'))
    X = pickle.load(open('datasets/kddcup99/X.p'))
    label = pickle.load(open('datasets/kddcup99/label.p'))
    norm_X = norm_data(X[:size])
    label = label[:size]
    print label[label == 1].shape[0], label[label == -1].shape[0]
    #evaluate_dataset(norm_X, label[:size])
    #evaluate_incremental(norm_X, label[:size], nu, gamma, dataset='kdd99-incr')
    evaluate_semi_supervised(norm_X, label[:size], nu, gamma, ratio=0.90, dataset='kdd99-semiincr')


if __name__ == "__main__":
    option = 2

    if option == 1:

        # evaluation for keel dataset
        if len(sys.argv) > 1:
            dataset = sys.argv[1]
        else:
            dataset = "segment0"
        evaluate_imbalance(dataset, nu=0.75, gamma=0.003)
    elif option == 2:
        if len(sys.argv) > 1:
            print sys.argv
            size = int(sys.argv[1])
        else:
            size = 1000
        print size
        evaluate_kdd99(size, nu=0.3, gamma=0.3)
    elif option == 3:
        #### grid search or evaluation for mnist
        if len(sys.argv) > 1:
            print sys.argv
            size = int(sys.argv[1])
        else:
            size = 1000
        classes = 2

        evaluate_mnist(size, classes, nu=0.21, gamma=0.001)
    elif option == 3:
        #shuttle
        evaluate_shuttle(1000, 0.1, 0.3)


