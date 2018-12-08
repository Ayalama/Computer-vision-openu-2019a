import argparse

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

from mmn2 import knn_cls
from mmn2 import svm_cls
from mmn2 import utils

# 1. load data set
iris = load_iris()
X, y = iris.data, iris.target
np.random.seed(10)
n_sample = len(X)
order = np.random.permutation(n_sample)
X = X[order]
y = y[order]

# split train ans test sets
(trainData, valData, trainLabels, valLabels) = train_test_split(X,
                                                                y, test_size=0.3, random_state=42)

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="name of classification model- one of knn or svm")
args = vars(ap.parse_args())

# classify according to selected model
print('execute classification using {model}'.format(model=args["model"]))
if args["model"] == 'knn':
    valPred, valPred_prob = knn_cls.knn(trainData, valData, trainLabels, valLabels)
else:
    valPred, valPred_prob = svm_cls.SVM(trainData, valData, trainLabels, valLabels)
###ROC,AUC
utils.plot_roc(valLabels, valPred_prob, title="Iris {model} ROC".format(model=args["model"]))

pd.DataFrame(trainData).to_csv("Iris\\trainData.csv")
pd.DataFrame(valData).to_csv("Iris\\valData.csv")
pd.DataFrame(trainLabels).to_csv("Iris\\trainLabels.csv")
pd.DataFrame(valLabels).to_csv("Iris\\valLabels.csv")
pd.DataFrame(valPred).to_csv("Iris\\valPred_{model}.csv".format(model=args["model"]))
pd.DataFrame(valPred_prob).to_csv("Iris\\valPred_prob_{model}.csv".format(model=args["model"]))
