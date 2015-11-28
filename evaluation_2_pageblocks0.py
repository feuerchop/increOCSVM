__author__ = 'LT'
import sys
from sklearn import preprocessing, cross_validation, svm
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_recall_fscore_support
import pandas as pd
from pandas import Series, crosstab
import numpy as np
from numpy import array, int8, uint8, zeros
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from profilehooks import profile
from math import floor, ceil
from array import array as pyarray
import matplotlib
import os, struct, ocsvm, sys
novelty = -1
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

def norm_data(X_raw):
    # normalization
    X = np.asarray(X_raw)
    X = preprocessing.normalize(X)
    return X

def convert_output(label_raw, class_labels):
    # we assume there are only two classes
    n_labels = [len(label_raw[label_raw == c]) for c in class_labels]
    # get labels with highest count
    max_label_count = max(n_labels)
    index_max_label = n_labels.index(max_label_count)
    pos_label = class_labels[index_max_label]
    return [1 if l == pos_label else -1 for l in label_raw]

def load_data(file_path):
    X_raw, label_raw = load_raw_data(file_path)
    X = norm_data(X_raw)
    label = np.asarray(convert_output(label_raw, list(set(label_raw))))
    return X, label

def get_kfold_cv(n_samples, kfold):
    return cross_validation.KFold(n_samples, n_folds=kfold)

def evaluate_yeast3():
    yeast3 = "/Users/LT/Documents/Uni/MA/increOCSVM/datasets/yeast3/yeast3.dat"
    X_raw, label_raw = load_raw_data(yeast3)
    print X_raw
    X = np.asarray(norm_data(X_raw))
    print X
    n_samples = X.shape[0]
    label = convert_output(label_raw, list(set(label_raw)))
    nu_gamma_f1 = [0, 0, 0]
    nu_range = [0.1*i for i in range(1,10)]
    gamma_range = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    train_test_split = int(floor(n_samples*0.8))
    for nu in nu_range:
        for gamma in gamma_range:
            clf = ocsvm.OCSVM("rbf", nu=nu, gamma=gamma)
            clf.fit(X[:train_test_split])
            expected = label[train_test_split:]
            predicted = clf.predict(X[train_test_split:])
            precision, recall, f1score, support = precision_recall_fscore_support(expected, predicted, average='binary')
            if f1score > nu_gamma_f1[2]:
                nu_gamma_f1 = [nu, gamma, f1score]
            print f1score
    print "nu_gamma_f1: %s" % nu_gamma_f1
    clf = ocsvm.OCSVM("rbf", nu=nu_gamma_f1[0], gamma=nu_gamma_f1[1])
    clf.fit(X[:train_test_split])
    predicted = clf.predict(X[train_test_split:])
    precision, recall, f1score, support = precision_recall_fscore_support(expected, predicted, average='binary')
    print("Confusion matrix:\n%s" % confusion_matrix(expected, predicted))
    precision, recall, f1score, support = precision_recall_fscore_support(expected, predicted, average='binary')
    print "precision: %s, recall: %s, f1-score: %s" % (precision, recall, f1score)
    cv_indices = get_kfold_cv(len(X), 5)

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
    all_data = sorted(all_data, key=lambda x: -x[4])
    if verbose:
        pd.set_option('display.max_rows', None)
        df = pd.DataFrame(all_data, columns=['nu', 'gamma', 'precision', 'recall', 'f1'])
        print df
    return nu_gamma_f1

def evaluate_dataset(X, label):
    split = 0.8
    train_split = int(floor(X.shape[0]*split))
    nu_gamma_f1 = grid_search_sklearn(X, label, split=split)#,
                              #nu_range=[0.05*i for i in range(1,11)],
                              #gamma_range=[0.01, 0.03, 0.1, 0.3, 1, 3, 5])
    # train with best
    print "nu_gamma_f1: %s" % nu_gamma_f1
    clf = ocsvm.OCSVM("rbf", nu=nu_gamma_f1[0], gamma=nu_gamma_f1[1])
    clf.fit(X[:train_split], scale=nu_gamma_f1[0]*len(X[:train_split]))
    clf._data.alpha_s()
    expected = np.asarray(label)*novelty
    predicted = clf.predict(X)*novelty
    confusion = output_cf(expected, predicted)
    print("Confusion matrix:\n%s" % confusion)
    print("Confusion matrix:\n%s" % confusion_matrix(expected, predicted))
    precision, recall, f1score, support = precision_recall_fscore_support(expected, predicted, average='binary')
    print "precision: %s, recall: %s, f1-score: %s" % (precision, recall, f1score)

def precision(confusion_matrix):
    tp = confusion_matrix[1][1]
    fp = confusion_matrix

#@profile
def sklearn_ocsvm(clf, X_train):
    clf.fit(X_train)

#@profile
def cvxopt_ocsvm(clf, X_train, scale=1, nu=None):
    clf.fit(X_train, scale=scale, v_target=nu)

#@profile
def incremental_ocsvm(clf, X_train, train_size, scale, init_ac, break_count):
    clf.fit(X_train[:train_size], scale=scale, rho=False)
    clf.increment(X_train[train_size:], init_ac=init_ac, break_count=break_count)
    print len(clf._data.alpha_s())
    print clf._data.alpha_s()

def evaluate_incremental(X, label, nu, gamma, train_size=20):
    nu_start = 0.95
    print "data size: %s" % len(X)

    kfold = get_kfold_cv(X.shape[0], 5)
    for train_index, test_index in kfold:
        break_count = len(X) - train_size
        #break_count = 30

        print "break_count: %s" % break_count
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = label[train_index], label[test_index]
        #clf = ocsvm.OCSVM("rbf", nu=nu, gamma=gamma)
        clf = ocsvm.OCSVM("rbf", nu=nu_start, gamma=gamma)
        if nu < nu_start:
            train_size = ceil(len(X_train) * nu / nu_start)
        print "train_size: %s" % train_size

        #clf.fit(X_train[:train_size], scale=nu_start*train_size)
        #clf.increment(X_train[train_size:], init_ac=0, break_count=break_count)
        incremental_ocsvm(clf, X_train, train_size, nu_start*train_size, 0, break_count)
        #clf.increment_work(X_train[train_size:], init_ac=0, break_count=break_count)
        #clf.KKT(clf._data.X(), clf._data.alpha())
        expected = label*novelty
        predicted = clf.predict(X)*novelty
        confusion = confusion_matrix(expected, predicted)
        print("Confusion matrix:\n%s" % confusion)
        precision, recall, f1score, support = precision_recall_fscore_support(expected, predicted, average='binary')
        print "precision: %s, recall: %s, f1-score: %s" % (precision, recall, f1score)
        print "--------"
        X_train_batch = np.vstack((X_train[-break_count:], X_train[:train_size]))
        tmp = float(len(X_train_batch))
        clf = ocsvm.OCSVM("rbf", nu=nu_start*train_size / tmp, gamma=gamma)
        cvxopt_ocsvm(clf, X_train_batch, nu_start*train_size, nu)
        #a = clf._data.alpha()
        #clf.KKT(clf._data.X(), a)
        expected = label*novelty
        #print "expected = 1: %s" % (expected[expected==1].shape[0])
        predicted = clf.predict(X)*novelty
        confusion = output_cf(expected, predicted)
        print("Confusion matrix:\n%s" % confusion)
        precision, recall, f1score, support = precision_recall_fscore_support(expected, predicted, average='binary')
        print "precision: %s, recall: %s, f1-score: %s" % (precision, recall, f1score)
        print "---------"
        clf = svm.OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
        sklearn_ocsvm(clf, X_train)
        expected = label*novelty
        predicted = clf.predict(X)*novelty
        confusion = output_cf(expected, predicted)
        print("Confusion matrix:\n%s" % confusion)
        precision, recall, f1score, support = precision_recall_fscore_support(expected, predicted, average='binary')
        print "precision: %s, recall: %s, f1-score: %s" % (precision, recall, f1score)

def output_cf(expected, predictions):
    expected = Series(expected, name="Target")
    predictions = Series(predictions, name="Prediction")
    return crosstab(expected, predictions)

def load_wisconsin():
    wisconsin = "/Users/LT/Documents/Uni/MA/increOCSVM/datasets/wisconsin/wisconsin.dat"
    return load_data(wisconsin)

def load_ecoli1():
    ecoli1 = "/Users/LT/Documents/Uni/MA/increOCSVM/datasets/ecoli1/ecoli1.dat"
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

            precision, recall, f1score, support = precision_recall_fscore_support(expected, predicted, average='binary')
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

    clf = ocsvm.OCSVM("rbf", nu=nu_start, gamma=0.1)
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
    print expected
    predicted = clf.predict(X_train)*novelty
    predicted[predicted == 0] = novelty
    print predicted
    confusion = confusion_matrix(expected, predicted)
    print("Confusion matrix:\n%s" % confusion)
    precision, recall, f1score, support = precision_recall_fscore_support(expected, predicted, average='binary')
    print "precision: %s, recall: %s, f1-score: %s" % (precision, recall, f1score)

if __name__ == "__main__":
    #size = 10000
    #novelty = 1
    #X_train, X_test, X_outliers = load_random_data(size=size, variance=5, density=1)
    #print "size: %s" % X_train.shape[0]
    #profile_incremental(X_train, X_test, X_outliers)
    #classify_random_data(X_train, X_test, X_outliers, variance=20)
    #plot_random_data(X_train, X_test, X_outliers)
    #evaluate_incremental(X_train, np.ones(X_train.shape[0]) * 1, 0.1, 0.1)
    #sys.exit()
    #nu = 0.9
    #gamma = 30
    #print "mnist digits: nu=%s, gamma=%s" % (nu, gamma)
    #print "yeast1"
    '''
    if len(sys.argv) > 1:
        print sys.argv
        size = int(sys.argv[1])
    else:
        size = 500
    print "mnist classes = 2"
    X, label = load_digits(classes=1, max_size=10000)
    X = X[:size]
    label = label[:size]
    print "size: %s" % X.shape[0]
    print label[label == 1].shape
    print label[label == -1].shape
    profile_incremental(X,label)
    '''
    if len(sys.argv) > 1:
        print sys.argv
        size = int(sys.argv[1])
    else:
        size = 1000

    X, label = load_pageblocks0()
    X = X[:size]
    label = label[:size]
    print "size: %s" % X.shape[0]
    profile_incremental(X,label)

    #evaluate_dataset(X, label)

    #evaluate_incremental(X, label, nu, gamma)
    #load_digits_small()



