import numpy as np
import jax
import jax.numpy as jnp


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

        # TODO could pre-process normals

    # TODO margin is not tested yet
    @staticmethod
    def edge_zmp_constraint(zmp, v1, v2, margin, np=np):
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
            return v1, PolygonSupportArea.edge_zmp_constraint(
                zmp, v0, v1, self.margin, np=jnp
            )

        _, g = jax.lax.scan(scan_func, self.vertices[-1, :], self.vertices)
        return g

    # def zmp_constraints_scaled(self, αz_zmp, αz):
    #     pass

    def zmp_constraints_numpy(self, zmp):
        N = self.vertices.shape[0]

        # TODO offset is not handled
        g = np.zeros(N)
        for i in range(N - 1):
            v1 = self.vertices[i, :]
            v2 = self.vertices[i + 1, :]
            g[i] = PolygonSupportArea.edge_zmp_constraint(zmp, v1, v2, self.margin)
        g[-1] = PolygonSupportArea.edge_zmp_constraint(
            zmp, self.vertices[-1, :], self.vertices[0, :], self.margin
        )
        return g


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

    def zmp_constraints(self, zmp):
        """Generate ZMP stability constraint.


        zmp: 2D point to check for inclusion in the support area
        margin: minimum distance from edge of support area to be considered inside

        Returns a value g, where g >= 0 satisfies the ZMP constraint
        """
        e = zmp - self.offset
        return jnp.array([(self.radius - self.margin) ** 2 - e @ e])

    def zmp_constraints_scaled(self, αz_zmp, αz):
        e = αz_zmp - αz * self.offset
        return jnp.array([(αz * (self.radius - self.margin)) ** 2 - e @ e])


if __name__ == "__main__":
    import IPython
    import matplotlib.pyplot as plt

    s = 0.2
    r = equilateral_triangle_inscribed_radius(s)
    αz = 5
    vertices = αz * np.array([[2 * r, 0], [-r, 0.5 * s], [-r, -0.5 * s]])
    support = PolygonSupportArea(vertices)
    points = αz * np.array([[0.05, 0.05], [-0.06, 0], [0.05, -0.05]])

    for i, point in enumerate(points):
        d = support.zmp_constraints_numpy(point)
        print(f"Point {i} = {d}")

    plt.plot(vertices[:, 0], vertices[:, 1], "o", color="k")
    plt.plot(
        np.append(vertices[:, 0], vertices[0, 0]),
        np.append(vertices[:, 1], vertices[0, 1]),
        color="k",
    )
    plt.plot(points[:, 0], points[:, 1], "o", color="r")
    plt.grid()
    plt.show()
