import numpy as np

from util import Body, cuboid_inertia_matrix, cylinder_inertia_matrix, compose_bodies

import IPython


def main():
    mass1 = 1
    side_lengths1 = [1, 1, 1]
    inertia1 = cuboid_inertia_matrix(mass1, side_lengths1)
    body1 = Body(mass=mass1, inertia=inertia1, com=[0.5, 0.5, 0.5])

    mass2 = 1
    body2 = Body(mass=mass2, inertia=inertia1, com=[0.5, 0.5, 1.5])

    inertia3 = cuboid_inertia_matrix(mass1 + mass2, [1, 1, 2])
    body3 = compose_bodies([body1, body2])
    IPython.embed()


if __name__ == "__main__":
    main()
