import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mmn2 import features
from mmn2 import knn_cls
from mmn2 import svm_cls
from mmn2 import utils

mnist_dir = "MNIST\\"
test = True
test_size = 20000

# 1. load data set
train_data = pd.read_csv(mnist_dir + "train.csv")
train_data.reset_index()

np.random.seed(10)
n_sample = len(train_data)
order = np.random.permutation(n_sample)
train_data = train_data.iloc[order]

if test:
    train_data = train_data.head(test_size)

X = train_data.drop("label", 1)
y = train_data['label']

## construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="name of classification model- one of knn or svm")
ap.add_argument("-f", "--feature", required=True,
                help="name of used feature- one of baseline, hog or sift")
args = vars(ap.parse_args())

# 2. get features
## split train ans test sets according to selected feature
if args["feature"] == 'baseline':
    (trainData, valData, trainLabels, valLabels) = train_test_split(X,
                                                                    y, test_size=0.3, random_state=42)
elif args["feature"] == 'hog':
    digits = np.asarray(X).reshape((X.shape[0], 28, 28))
    hog_rep = features.hog_batch_representation(digits, orientations=3, pixelsPerCell=(2, 2),
                                                cellsPerBlock=(4, 4), block_norm='L2-Hys')
    (trainData, valData, trainLabels, valLabels) = train_test_split(np.array(hog_rep),
                                                                 y, test_size=0.3, random_state=42)
elif args["feature"] == 'sift':
    images = X
    X_sift = features.sift_batch_representation(images)
    (trainData, valData, trainLabels, valLabels) = train_test_split(X_sift,
                                                                    y, test_size=0.3, random_state=42)
else:
    digits = np.asarray(X).reshape((X.shape[0], 28, 28))
    hog_rep = features.hog_batch_representation(digits, orientations=3, pixelsPerCell=(2, 2),
                                                cellsPerBlock=(4, 4), block_norm='L2-Hys')
    images = X
    X_sift = features.sift_batch_representation(images)
    comb = np.concatenate((hog_rep, X_sift), axis=1)
    (trainData, valData, trainLabels, valLabels) = train_test_split(comb,
                                                                    y, test_size=0.3, random_state=42)

# 3. classify according to selected model
print('execute classification using {model}'.format(model=args["model"]))
if args["model"] == 'knn':
    valPred, valPred_prob = knn_cls.knn(trainData, valData, trainLabels, valLabels)
else:
    valPred, valPred_prob = svm_cls.SVM(trainData, valData, trainLabels, valLabels)
# 4. ROC,AUC
utils.plot_roc(valLabels, valPred_prob, title="MNIST {model} ROC".format(model=args["model"]))
