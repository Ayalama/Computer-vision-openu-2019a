import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from mmn2 import load_img
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA


def plot_confusion(confusion_matrix):
    plt.figure()
    plt.matshow(confusion_matrix)
    plt.title('Confusion Matrix for Validation Data')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


##Take n principal components
def PrincipalComponents(n, datatrain, datatest=None):
    X_test_pca = None
    pca = PCA(n_components=n)
    X_train_pca = pca.fit_transform(datatrain)
    if datatest is not None:
        X_test_pca = pca.transform(datatest)
    return X_train_pca, X_test_pca


def plot_roc(test_y, tesy_y_prob, title=''):
    n_classes = np.unique(test_y).shape[0]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(np.where(test_y == i, 1, 0), tesy_y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area

    test_y_multi = tesy_y_prob
    for i in range(n_classes):
        test_y_multi[:, i] = np.where(test_y == i, 1, 0)
    fpr["micro"], tpr["micro"], _ = roc_curve(test_y_multi.ravel(), tesy_y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class %d (area = %.4f%%)' % (i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC multi-class')
    plt.legend(loc="lower right")
    plt.title(title)
    plt.show()


def load_data_set(dataset="MNIST"):
    images = None

    ##############load data set of images and prepare training, test and validation sets###############
    if dataset == "MNIST":
        images = load_img.load_train_from_dir()
    if dataset == "iris":
        images = load_img.load_iris()

    # take the MNIST data and construct the training and testing split, using 75% of the
    # data for training and 25% for testing
    # visualize_images_set(images.data)
    (trainData, valData, trainLabels, valLabels) = train_test_split(np.array(images.data),
                                                                    images.target, test_size=0.25, random_state=42)

    # # now, let's take 10% of the training data and use that for validation
    # (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
    #                                                                 test_size=0.1, random_state=84)

    # show the sizes of each data split
    print("training data points: {}".format(len(trainLabels)))
    print("validation data points: {}".format(len(valLabels)))
    return (trainData, valData, trainLabels, valLabels)


def visualize_images_set(img_data):
    plt.figure(figsize=(5, 5))
    for digit_num in range(0, 64):
        plt.subplot(8, 8, digit_num + 1)
        grid_data = img_data.iloc[digit_num].as_matrix().reshape(28, 28)  # reshape from 1d to 2d pixel array
        plt.imshow(grid_data, interpolation="none", cmap="bone_r")
        plt.xticks([])
        plt.yticks([])

    plt.show()
