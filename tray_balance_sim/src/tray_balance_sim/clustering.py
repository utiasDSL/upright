import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy, vq
import IPython


def unit(x):
    return x / np.linalg.norm(x)


def ritter(ps, eps=1e-8):
    """Ritter's bounding circle algorithm."""
    # this can be any point in the dataset
    x = ps[0, :]

    dx = np.sum(np.square(ps - x), axis=1)
    y = ps[np.argmax(dx), :]

    dy = np.sum(np.square(ps - y), axis=1)
    z = ps[np.argmax(dy), :]

    # initial proposal for center and radius
    r = 0.5 * np.linalg.norm(z - y)
    c = 0.5 * (y + z)

    # while some points are not contained in the bounding sphere, expand it to
    # contain that point + previous sphere
    # eps is needed to avoid small floating point errors causing the loop not
    # to break
    d = np.sum(np.square(ps - c), axis=1)
    idx = np.argmax(d)
    while d[idx] > (r + eps)**2:
        r = 0.5 * (r + np.sqrt(d[idx]))
        v = unit(c - ps[idx, :])
        c = ps[idx, :] + r * v

        d = np.sum(np.square(ps - c), axis=1)
        idx = np.argmax(d)

    return c, r


def iterative_ritter(assignments, points, k, n=10):
    """Iterative Ritter's algorithm for multiple spheres.

    Given initial assignments for each point, we iterate:
    * finding a bound sphere for each cluster using Ritter's algorithm
    * re-assigning points to the nearest cluster (in terms of Euclidean distance)

    Returns: (centers, radii) of the bounding spheres
    """
    centers = np.zeros((k, points.shape[1]))
    radii = np.zeros(k)
    dists = np.zeros((points.shape[0], k))

    for _ in range(n):
        # compute new center based on Ritter's bounding sphere
        for i in range(k):
            idx, = np.nonzero(assignments == i)
            centers[i, :], radii[i] = ritter(points[idx, :])

        # compute new assigments based on closest center for each point
        for i in range(k):
            dists[:, i] = np.sum(np.square(points - centers[i, :]), axis=1)
        assigments = np.argmin(dists, axis=1)
    return centers, radii


def cluster_greedy_kcenter(points, k=3):
    # k-centers, initialize the first one randomly
    C = np.zeros((k, points.shape[1]))
    C[0, :] = points[0, :]

    # compute distances to the current center points
    D = np.zeros((points.shape[0], k))
    D[:, 0] = np.sum(np.square(points - C[0, :]), axis=1)

    for i in range(1, k):
        Dmin = np.min(D[:, :i], axis=1)

        # new center point is farthest from all current centers
        idx = np.argmax(Dmin)
        print(f"new center index = {idx}")
        C[i, :] = points[idx, :]
        D[:, i] = np.sum(np.square(points - C[i, :]), axis=1)

    assignments = np.argmin(D, axis=1)
    return C, assignments


def cluster_kmeans(points, k=3):
    """Cluster points using k-means."""
    whitened_points = vq.whiten(points)
    whitened_centers, _ = vq.kmeans2(whitened_points, k=k, minit="random")
    assignments, dists = vq.vq(whitened_points, whitened_centers)

    centers = np.zeros_like(whitened_centers)
    for i in range(k):
        centers[i, :] = np.mean(points[assignments == i], axis=0)
    return centers, assignments


def cluster_hierarchy(points, k=3):
    """Cluster points using hierarchical clustering (complete linkage)."""
    dists = pdist(points)
    clusters = hierarchy.complete(dists)
    assignments = hierarchy.fcluster(clusters, t=k, criterion="maxclust") - 1
    return clusters, assignments


def cluster_and_bound(points, k, cluster_type="kmeans", bound_type="ritter", n=10):
    # cluster
    if cluster_type == "kmeans":
        _, assignments = cluster_kmeans(points, k=k)
    elif cluster_type == "hierarchy":
        _, assignments = cluster_hierarchy(points, k=k)
    elif cluster_type == "greedy":
        _, assignments = cluster_greedy_kcenter(points, k=k)
    else:
        raise Exception(f"Unknown cluster type: {cluster_type}")

    # bound
    # only ritter algorithm is available now
    centers, radii = iterative_ritter(assignments, points, k, n=n)
    return centers, radii
