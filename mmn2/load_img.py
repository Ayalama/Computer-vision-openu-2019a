from sklearn import datasets
import numpy as np
import pandas as pd
mnist_dit = "MNIST\\"
test = True
test_size=100


def load_iris():
    iris = datasets.load_iris()
    return iris


# patial digits data set from sklearn
def load_mnist():
    mnist = datasets.load_digits()
    return mnist


def load_train_from_dir(data_dir=mnist_dit):
    # train_data = open(data_dir + "train.csv").read()
    # train_data = train_data.split("\n")[1:-1]
    # if test:
    #     train_data = train_data[0:5000]
    # train_data = [i.split(",") for i in train_data]
    #
    # X_train = np.array([[int(i[j]) for j in range(1, len(i))] for i in train_data])
    # Y_train = np.array([int(i[0]) for i in train_data])
    # train_data = data_set(X_train, Y_train)


    train_data = pd.read_csv(data_dir + "train.csv")
    if test:
        train_data = train_data.head(test_size)

    X_train = train_data.drop("label",1)
    Y_train = train_data['label']
    train_data = data_set(X_train, Y_train)


    # test_data = open(data_dir + "test.csv").read()
    # test_data = test_data.split("\n")[1:-1]
    # test_data = [i.split(",") for i in test_data]
    # # print(len(test_data))
    # X_test = np.array([[int(i[j]) for j in range(0, len(i))] for i in test_data])
    # test_data=data_set(X_test,None)
    return train_data


class data_set():
    "images data set splited into data and target"

    def __init__(self, data, target):
        self.data = data
        self.target = target
