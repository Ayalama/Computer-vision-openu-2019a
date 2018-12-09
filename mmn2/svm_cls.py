import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix

from mmn2 import utils


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h).astype(np.float16),
                         np.arange(y_min, y_max, h).astype(np.float16), sparse=True)
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def visualize_classes_2d(x_train, y_train):
    X_train_pca, _ = utils.PrincipalComponents(2, x_train)
    C = 1.0

    models = (svm.SVC(kernel='linear', C=C),
              svm.LinearSVC(C=C),
              svm.SVC(kernel='rbf', gamma=0.7, C=C),
              svm.SVC(kernel='poly', degree=3, C=C))

    models = (clf.fit(X_train_pca, y_train) for clf in models)
    # title for the plots
    titles = ('SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel')

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X_train_pca[:, 0], X_train_pca[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('PCA component 1')
        ax.set_ylabel('PCA component 2')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
    plt.show()


def SVM(trainData, valData, trainLabels, valLabels):
    # initialize the values of c for our SVM classifier
    CVals = [10 ** (-20), 10 ** (-10), 0.001, 0.1, 1]

    # list of accuracies for each value of c
    accuracies = []
    best_c = 10 ** (-20)
    best_kernel = 'poly'
    best_accuracy = 0
    valPred_best=None
    valPred_prob_best=None

    # for fig_num, kernel in enumerate(('poly', 'linear')):
    for fig_num, kernel in enumerate(('poly', 'linear', 'rbf')):
        print("kernal = " + kernel + " begin ")
        for c in CVals:
            print("C = " + str(c) + " begin ")
            start = time.time()
            # train the SVM classifier with the current value of `c`
            model = svm.SVC(kernel=kernel, C=c, degree=3, probability=True, gamma=10)
            model.fit(trainData, trainLabels)
            valPred = model.predict(valData)
            valPred_prob = model.predict_proba(valData)

            # evaluate the model and update the accuracies list
            score = model.score(valData, valLabels)
            accuracies.append(score)
            end = time.time()

            # output performance
            print("classification report:")
            print(classification_report(valLabels, valPred))
            print("confusion matrix:")
            print(confusion_matrix(valLabels, valPred))
            print("c=%d, accuracy=%.2f%%" % (c, score * 100))
            print("Complete time: " + str(end - start) + " Secs.")
        # plot accuracy by c values
        plt.figure(fig_num)
        plt.clf()
        plt.plot(CVals, accuracies)
        plt.xlabel('c value (penalty)')
        plt.ylabel('accuracy of validation set (25%)')
        plt.title(kernel)

        i = int(np.argmax(accuracies))
        print("c=%.2f%% achieved highest accuracy of %.2f%% on validation data" % (CVals[i], accuracies[i] * 100))
        if accuracies[i] > best_accuracy:
            best_c = CVals[i]
            best_kernel = kernel
            best_accuracy = accuracies[i]
            valPred_best=valPred
            valPred_prob_best=valPred_prob
        accuracies = []
    plt.show()

    return valPred_best, valPred_prob_best
