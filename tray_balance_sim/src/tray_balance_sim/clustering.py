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


def cluster_kmeans(points, k=3):
    """Cluster points using k-means."""
    whitened_points = vq.whiten(points)
    whitened_centers, _ = vq.kmeans(whitened_points, k_or_guess=k)
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


def cluster_and_bound(points, k, cluster_type="kmeans", bound_type="ritter"):
    # cluster
    if cluster_type == "kmeans":
        _, assignments = cluster_kmeans(points, k=k)
    elif cluster_type == "hierarchy":
        _, assignments = cluster_hierarchy(points, k=k)
    else:
        raise Exception(f"Unknown cluster type: {cluster_type}")

    # bound
    # only ritter algorithm is available now
    centers = np.zeros((k, points.shape[1]))
    radii = np.zeros(k)
    for i in range(k):
        idx, = np.nonzero(assignments == i)
        centers[i, :], radii[i] = ritter(points[idx, :])
    return centers, radii
