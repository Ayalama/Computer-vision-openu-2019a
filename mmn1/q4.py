from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt
import numpy as np


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->',
                      linewidth=2,
                      shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


# given input points, use PCA methods on this points
# restore points after information loss and draw
if __name__ == '__main__':
    input_points = np.asarray(
        [[2.5, 2.9], [0.5, 1.2], [2.2, 3.4], [1.9, 2.7], [3.1, 3.5], [2.3, 3.2], [2, 2.1], [1, 1.6], [1.5, 2.1],
         [1.1, 1.4]])
    print("input points:")
    print(input_points)
    print()
    X = input_points[:, 0]
    Y = input_points[:, 1]

    #get PCA points and componenets
    pca = sklearnPCA(n_components=1)  # 1-dimensional PCA
    points_pca = pca.fit_transform(input_points)
    print("sample covariance = (XtX)/(n-1): ")
    cov_matrix = pca.get_covariance()
    print(cov_matrix)
    print()

    print("PCA componenets (W) : ")
    print(pca.components_)
    print()

    print("PCA eigenvalues = (v_i)t*(cov_matrix*v_i) where v_i is an eigenvector: ")
    for eigenvector in pca.components_:
        print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
    print()
    # eigenvalues can be produces by solving det(covmatrix-lambada*I)=0
    # eigenvectors can be produces by solving covmatrix*v = lambada*v

    print("each point converted using (x_new)=Wt((x,y)-(x_avg,y_avg)):")
    print(points_pca)
    print()

    # plot input data
    fig, ax = plt.subplots(1, 3, figsize=(16, 6))
    fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
    ax[0].scatter(input_points[:, 0],input_points[:, 1], alpha=0.2)
    # draw PCA vectors on original scatter
    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * np.sqrt(length)
        draw_vector(pca.mean_, pca.mean_ + v, ax=ax[0])
        ax[0].axes.autoscale(enable=True, axis='both', tight=True)
    ax[0].axis('equal')
    ax[0].set(xlabel='x', ylabel='y', title='input points')

    # # plot principal components- 2d
    # ax[1].scatter(points_pca[:, 0], points_pca[:, 1], alpha=0.2)
    # draw_vector([0, 0], [0, np.sqrt(pca.explained_variance_[1])], ax=ax[1])
    # draw_vector([0, 0], [np.sqrt(pca.explained_variance_[0]), 0], ax=ax[1])
    # ax[1].axis('equal')
    # ax[1].set(xlabel='component 1', ylabel='component 2',
    #           title='principal components',
    #           xlim=(-5, 5), ylim=(-3, 3.1))

    # plot principal components- 1d
    ax[1].scatter(points_pca[:, 0],np.zeros(len(points_pca[:, 0])), alpha=0.2)
    draw_vector([0, 0], [np.sqrt(pca.explained_variance_[0]), 0], ax=ax[1])
    ax[1].axis('equal')
    ax[1].set(xlabel='component 1', ylabel='component 2',
              title='principal components',
              xlim=(-5, 5), ylim=(-3, 3.1))

    # plot restored data
    print("each point restored using (x_restored,y_restored)=x_pca*W+(x_avg,y_avg)):")
    points_restoration = pca.inverse_transform(points_pca)
    print(points_restoration)

    ax[2].scatter(points_restoration[:, 0], points_restoration[:, 1], alpha=0.2)
    ax[2].axis('equal')
    ax[2].set(xlabel='restored x', ylabel='restored y', title='restored data points')
    plt.tight_layout()
    plt.show()
