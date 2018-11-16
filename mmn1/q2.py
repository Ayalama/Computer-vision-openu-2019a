import random
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin


def get_rand_points(N=8, max_idx=12, max_idy=12):
    rangeX = (1, max_idx)
    rangeY = (1, max_idy)
    # Generate a set of all points within 200 of the origin, to be used as offsets later
    # There's probably a more efficient way to do this.
    deltas = set()
    for x in range(1,12):
        for y in range(1,12):
            deltas.add((x, y))

    randPoints = []
    excluded = set()
    i = 0
    while i < N:
        x = random.randrange(*rangeX)
        y = random.randrange(*rangeY)
        if (x, y) in excluded: continue
        randPoints.append((x, y))
        i += 1
        excluded.update((x + dx, y + dy) for (dx, dy) in deltas)
    print(randPoints)
    return randPoints


def plot_clusters(arrpoints,labels, centers, iteration_idx):
    plt.figure()
    plt.scatter(arrpoints[:, 0], arrpoints[:, 1], c=labels,
                s=50, cmap='viridis');
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.2);
    plt.title("kmeans created clusters, {iteration_idx}".format(iteration_idx=iteration_idx))
    plt.show()

# returned y_kmeans are cluster labels assigned for each point
# returned kmeans is the sklearn object holding kmeans paramater according to our input
def compute_knn_clusters(arrpoints, k, verbose=True, rseed=2):
    # 1. Randomly choose clusters
    arrpoints = np.asarray(arrpoints)

    if verbose:
        print("input points:")
        print(arrpoints)

    rng = np.random.RandomState(rseed)
    i = rng.permutation(arrpoints.shape[0])[:k]
    centers = arrpoints[i]

    itr = 0
    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(arrpoints, centers)

        if verbose:
            print("iteration number: {itr}".format(itr=itr))
            print("cluster centers:")
            print(centers)
            print("labels for each point:")
            print(labels)
            plot_clusters(arrpoints,labels,centers,itr)

        # 2b. Find new centers from means of points
        new_centers = np.array([arrpoints[labels == i].mean(0)
                                for i in range(k)])

        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
        itr = itr + 1

    if verbose:
        print("previous iteration centers equals to current iteration centers => process finished at itr {itr}".format(itr=itr))
        print("final clusters centers:")
        print(centers)
        print("final label for each point:")
        print(labels)
        plot_clusters(arrpoints, labels, centers, iteration_idx="final")

    return centers, labels

# randomly select 8 points in 2d space of size 12X12
# run kmeans with k=3
if __name__ == '__main__':
    N = 8
    max_idx = 12
    max_idy = 12
    k = 3

    # get 2d points
    points = get_rand_points(N, max_idx, max_idy)
    arrpoints = np.asarray(points)
    plt.scatter(arrpoints[:, 0], arrpoints[:, 1], s=50);
    plt.title("randomly selected points")
    plt.show()

    # kmeans manual
    centers, labels = compute_knn_clusters(points, k)