import matplotlib.pyplot as plt
import numpy as np

# given input points, use leastSquares methods to find parabula which fits this points
if __name__ == '__main__':
    input_points = np.asarray([[1, 3.96], [4, 27.96], [3, 15.15], [5, 45.8], [2, 7.07], [6, 69.4]])
    X = input_points[:, 0]
    X_matx = np.transpose([np.square(X), X, np.zeros(X.shape) + 1])
    Y = input_points[:, 1]
    Y_matx = Y.reshape(-1, 1)

    # define Y,X,B
    print("Equestion params: ")
    print("Y=")
    print(Y_matx)
    print()
    print("X=")
    print(X_matx)
    print()
    print("B=")
    B_matx = np.asarray(["a", "b", "c"])
    B_matx = B_matx.reshape(-1, 1)
    print(B_matx)
    print()

    print("Solving equation XtXB=XtY ==> B=((XtX)^(-1))*XtY")

    # calc (XtX)
    xtx= np.matmul(np.transpose(X_matx), X_matx)
    print("XtX=")
    print(xtx)
    print()
    # calc (XtX)^(-1)
    xtx_inv = np.linalg.inv(xtx)
    print("XtX^(-1)=")
    print(xtx_inv)
    print()
    # calc (XtY)
    xty = np.matmul(np.transpose(X_matx), Y_matx)
    print("XtY=")
    print(xty)
    print()
    # calc ((XtX)^(-1))*(XtY)
    B_matx = np.matmul(xtx_inv, xty)
    print("((XtX)^(-1))*(XtY)=B=")
    print(B_matx)
    print()

    z = np.polyfit(X, Y, 2)
    p = np.poly1d(z)
    print("fitted parabula:")
    print("a={}".format(z[0]))
    print("b={}".format(z[1]))
    print("c={}".format(z[2]))

    xp = np.linspace(-6, 10, 100)
    _ = plt.plot(X, Y, '.', xp, p(xp), '-')

    plt.show()
