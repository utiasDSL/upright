import numpy as np

import tray_balance_ocs2.MobileManipulatorPythonInterface as ocs2


def equilateral_triangle_inscribed_radius(side_length):
    """Compute radius of the inscribed circle of equilateral triangle."""
    return side_length / (2 * np.sqrt(3))


def circle_r_tau(radius):
    return 2.0 * radius / 3


def cuboid_support_vertices(side_lengths):
    """Generate vertices of support area for cuboid with given side_lengths.

    side_lengths are arranged (x, y, z). Only x and y are used.
    """
    hx, hy = 0.5 * np.array(side_lengths)[:2]
    vertices = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
    return vertices


class PolygonSupportArea:
    """Polygonal support area

    vertices: N*2 array of vertices arranged in order, counter-clockwise.
    offset: the 2D vector pointing from the projection of the CoM on the
    support plane to the center of the support area
    """

    def __init__(self, vertices, offset=(0, 0), margin=0):
        self.vertices = np.array(vertices)
        self.offset = np.array(offset)
        self.margin = margin
        self.num_constraints = self.vertices.shape[0]

    def convert_to_ocs2(self):
        vertices = [self.vertices[i, :] for i in range(self.vertices.shape[0])]
        return ocs2.PolygonSupportArea(vertices, self.offset, self.margin)

    @classmethod
    def rectangle(cls, w, h, offset=(0, 0), margin=0):
        # TODO
        pass

    @classmethod
    def circle(cls, radius, offset=(0, 0), margin=0):
        s = np.sqrt(2.0) * radius
        vertices = cuboid_support_vertices([s, s])
        return cls(vertices, offset, margin)

    # TODO this is not a fully general implementation, since it doesn't account
    # for different orientations
    @classmethod
    def equilateral_triangle(cls, side_length, offset=(0, 0), margin=0):
        r = equilateral_triangle_inscribed_radius(side_length)
        vertices = np.array(
            [[2 * r, 0], [-r, 0.5 * side_length], [-r, -0.5 * side_length]]
        )
        return cls(vertices, offset, margin)


class CircleSupportArea:
    """Circular support area

    offset: the 2D vector pointing from the projection of the CoM on the
    support plane to the center of the support area
    """

    num_constraints = 1

    def __init__(self, radius, offset=(0, 0), margin=0):
        self.radius = radius
        self.offset = np.array(offset)
        self.margin = margin

    def convert_to_ocs2(self):
        return ocs2.CircleSupportArea(self.radius, self.offset, self.margin)
