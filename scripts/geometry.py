import numpy as np
import jax


def equilateral_triangle_inscribed_radius(side_length):
    """Compute radius of the inscribed circle of equilateral triangle."""
    return side_length / (2 * np.sqrt(3))


def circle_r_tau(radius):
    return 2.0 * radius / 3


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

    # TODO margin is not tested yet
    @staticmethod
    def edge_zmp_constraint(zmp, v1, v2, margin):
        """ZMP constraint for a single edge of a polygon.

        zmp is the zero-moment point
        v1 and v2 are the endpoints of the segment.
        """
        S = np.array([[0, 1], [-1, 0]])
        normal = S @ (v2 - v1)  # inward-facing
        normal = normal / np.linalg.norm(normal)
        return -(zmp - v1) @ normal - margin  # negative because g >= 0

    def zmp_constraints(self, zmp):
        def scan_func(v0, v1):
            return v1, PolygonSupportArea.edge_zmp_constraint(zmp, v0, v1, self.margin)

        _, g = jax.lax.scan(scan_func, self.vertices[-1, :], self.vertices)
        return g

    def zmp_constraints_numpy(self, zmp):
        N = self.vertices.shape[0]

        g = np.zeros(N)
        for i in range(N - 1):
            v1 = self.vertices[i, :]
            v2 = self.vertices[i + 1, :]
            g[i] = PolygonSupportArea.edge_zmp_constraint(zmp, v1, v2, self.margin)
        g[-1] = PolygonSupportArea.edge_zmp_constraint(
            zmp, self.vertices[-1, :], self.vertices[0, :]
        )
        return g


class CircleSupportArea:
    """Circular support area

    offset: the 2D vector pointing from the projection of the CoM on the
    support plane to the center of the support area
    """

    def __init__(self, radius, offset=(0, 0), margin=0):
        self.radius = radius
        self.offset = np.array(offset)
        self.margin = margin

    def zmp_constraints(self, zmp):
        """Generate ZMP stability constraint.


        zmp: 2D point to check for inclusion in the support area
        margin: minimum distance from edge of support area to be considered inside

        Returns a value g, where g >= 0 satisfies the ZMP constraint
        """
        e = zmp - self.offset
        return (self.radius - self.margin) ** 2 - e @ e

    def zmp_constraints_scaled(self, αz_zmp, αz):
        e = αz_zmp - αz * self.offset
        return (αz * (self.radius - self.margin)) ** 2 - e @ e
