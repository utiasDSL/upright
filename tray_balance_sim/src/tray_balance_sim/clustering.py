import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy, vq
import IPython


def ritter(ps):
    """Ritter's bounding circle algorithm."""
    # x = ps[np.random.randint(n), :]
    x = ps[0, :]

    dx = np.sum(np.square(ps - x), axis=1)
    y = ps[np.argmax(dx), :]

    dy = np.sum(np.square(ps - y), axis=1)
    z = ps[np.argmax(dy), :]

    r = 0.5 * np.linalg.norm(z - y)
    c = 0.5 * (y + z)
    d = np.sum(np.square(ps - c), axis=1)

    # TODO this should really be iterative and recenter the ball too, but here
    # I'm just growing the ball to ensure if covers all points
    r = np.sqrt(np.max(d))

    return c, r


def cluster_kmeans(points, k=3):
    whitened_points = vq.whiten(points)
    whitened_centers, _ = vq.kmeans(whitened_points, k_or_guess=k)
    assignments, dists = vq.vq(whitened_points, whitened_centers)

    centers = np.zeros_like(whitened_centers)
    for i in range(k):
        centers[i, :] = np.mean(points[assignments == i], axis=0)
    return centers, assignments


def cluster_hierarchy(points, k=3):
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
