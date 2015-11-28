print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, colorbar=True):
    f, ax1 = plt.subplots(1,3, figsize=(30,8))
    im = ax1[0].imshow(cm, interpolation='nearest', cmap=cmap)

    ax1[0].set_title(title)
    if colorbar:
        f.colorbar(im)
    tick_marks = np.arange(len(iris.target_names))
    ax1[0].set_xticks(tick_marks)
    ax1[0].set_xticklabels(iris.target_names, rotation=45)
    print iris.target_names
    #ax1.set_yticks(tick_marks, iris.target_names)
    ax1[0].set_yticks(tick_marks)
    ax1[0].set_yticklabels(iris.target_names)
    #ax1.tight_layout()
    ax1[0].set_ylabel('True label')
    ax1[0].set_xlabel('Predicted label')

    im = ax1[1].imshow(cm, interpolation='nearest', cmap=cmap)

    ax1[1].set_title(title)
    if colorbar:
        f.colorbar(im)
    tick_marks = np.arange(len(iris.target_names))
    ax1[1].set_xticks(tick_marks)
    ax1[1].set_xticklabels(iris.target_names, rotation=45)
    #ax1.set_yticks(tick_marks, iris.target_names)
    ax1[1].set_yticks(tick_marks)
    #ax1[1].set_yticklabels(iris.target_names)
    ax1[1].yaxis.set_visible(False)
    f.tight_layout()
    #ax1.tight_layout()
    #ax1[1].set_ylabel('True label')
    #ax1[1].set_xlabel('Predicted label')


# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
print cm.shape
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plot_confusion_matrix(cm)
'''
# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
'''
plt.show()