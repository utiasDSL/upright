import numpy as np
import matplotlib.pyplot as plt

from util import Body, cuboid_inertia_matrix, compose_bodies, polygon_zmp_constraints

import IPython


def test_compose_bodies():
    mass1 = 1
    side_lengths1 = [1, 1, 1]
    inertia1 = cuboid_inertia_matrix(mass1, side_lengths1)
    body1 = Body(mass=mass1, inertia=inertia1, com=[0.5, 0.5, 0.5])

    mass2 = 1
    body2 = Body(mass=mass2, inertia=inertia1, com=[0.5, 0.5, 1.5])

    inertia3 = cuboid_inertia_matrix(mass1 + mass2, [1, 1, 2])
    body3 = compose_bodies([body1, body2])
    IPython.embed()


def test_zmp_constraints():
    vertices = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    point = np.array([-1.5, 0.5])
    g = polygon_zmp_constraints(point, vertices)
    print(np.all(g >= 0))

    patch = plt.Polygon(vertices, closed=True)
    plt.figure()
    ax = plt.gca()
    ax.add_patch(patch)
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    plt.plot([point[0]], [point[1]], "o", color="r")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    test_zmp_constraints()
