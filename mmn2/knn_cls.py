import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

def knn(trainData, valData, trainLabels, valLabels):
    # initialize the values of k for our k-Nearest Neighbor classifier along with the
    kVals = range(1, 20, 2)
    # list of accuracies for each value of k
    accuracies = []

    for k in kVals:
        print("k = " + str(k) + " begin ")
        start = time.time()
        # train the k-Nearest Neighbor classifier with the current value of `k`
        # model = KNeighborsClassifier(n_neighbors=k, weights='distance')
        model = KNeighborsClassifier(n_neighbors=k)
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
        print("k=%d, accuracy=%.2f%%" % (k, score * 100))
        print("Complete time: " + str(end - start) + " Secs.")

    plt.xlabel('K value (number of neighbors)')
    plt.ylabel('accuracy of validation set (25%)')
    plt.show()

    # find the value of k that has the largest accuracy
    i = int(np.argmax(accuracies))
    print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i], accuracies[i] * 100))
    # plot accuracy by K values
    plt.plot(kVals, accuracies)
    model = KNeighborsClassifier(n_neighbors=kVals[i])
    model.fit(trainData, trainLabels)
    valPred = model.predict(valData)
    valPred_prob = model.predict_proba(valData)
    return valPred, valPred_prob