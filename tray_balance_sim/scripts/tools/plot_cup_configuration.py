"""Visualize the configuration of cups balanced on the tray."""
import matplotlib.pyplot as plt
import numpy as np

import IPython


def dist_to_edge(v1, v2, p):
    S = np.array([[0, 1], [-1, 0]])

    normal = S @ (v2 - v1)  # inward-facing normal vector

    normal = normal / np.linalg.norm(normal)

    return -(p - v1).dot(normal)


def main():
    s = 0.2
    h = s / (2 * np.sqrt(3))
    r = 2 * h

    # triangle support area vertices
    vertices = np.array([[r, 0], [-h, 0.5 * s], [-h, -0.5 * s]])

    # unit normals to each of the vertices
    n0 = vertices[0, :] / np.linalg.norm(vertices[0, :])
    n1 = vertices[1, :] / np.linalg.norm(vertices[1, :])
    n2 = vertices[2, :] / np.linalg.norm(vertices[2, :])

    # point to place each cup is some distance along the normal
    L = 0.08
    c0 = L * n0
    c1 = L * n1
    c2 = L * n2

    print(dist_to_edge(vertices[0, :], vertices[1, :], np.zeros(2)))
    # IPython.embed()

    plt.figure()
    ax = plt.gca()
    ax.add_patch(plt.Polygon(vertices, fill=False))
    ax.add_patch(plt.Circle(c0, 0.05, fill=False))
    ax.add_patch(plt.Circle(c1, 0.05, fill=False))
    ax.add_patch(plt.Circle(c2, 0.05, fill=False))
    plt.plot([0, c0[0], c1[0], c2[0]], [0, c0[1], c1[1], c2[1]], "o", color="k")
    plt.grid()
    plt.xlim([-0.1, 0.15])
    plt.ylim([-0.15, 0.15])
    plt.show()


if __name__ == "__main__":
    main()
