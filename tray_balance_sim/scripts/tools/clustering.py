#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from tray_balance_sim import clustering

import IPython


np.random.seed(0)


def main():
    n = 100
    ps = np.random.random((n, 2))

    # z, f = cluster_kmeans(ps)
    clusters, assigments = clustering.cluster_hierarchy(ps)

    plt.figure()
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.grid()

    colors = ["r", "g", "b"]
    for i in range(3):
        idx, = np.nonzero(assigments == i)
        # ax.plot(clusters[i, 0], clusters[i, 1], "x", color=colors[i])
        ax.plot(ps[idx, 0], ps[idx, 1], "o", color=colors[i])
        c, r = clustering.ritter(ps[idx, :])
        ax.add_patch(plt.Circle(c, radius=r, color=colors[i], fill=False))

    plt.show()


if __name__ == "__main__":
    main()
