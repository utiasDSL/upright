import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog

from upright_core.math import plane_span

import IPython

# TODO rename this file to polyhedron


class ConvexPolyhedron:
    def __init__(self, vertices, normals, position=None, rotation=None, incidence=None):
        self.nv = vertices.shape[0]  # number of vertices
        self.nf = normals.shape[0]  # number of faces

        self.local_vertices = vertices
        self.local_normals = normals
        self.update_pose(position, rotation)

        # if position is None:
        #     position = np.zeros(3)
        # if rotation is None:
        #     rotation = np.eye(3)
        #
        # self.position = position
        # self.rotation = rotation
        # self.vertices = position + (rotation @ self.local_vertices.T).T
        # self.normals = (rotation @ self.local_normals.T).T

        # TODO possibly more convenient to compute wound indices of vertices
        # for each face: then clipping can be done face-wise
        if incidence is None:
            self.incidence = self._compute_incidence_matrix()
            self.ne = np.count_nonzero(np.tril(self.incidence))
        else:
            assert incidence.dtype == bool, "Incidence matrix must have dtype=bool"
            assert incidence.shape == (self.nv, self.nv), "Incidence matrix of wrong size"
            assert incidence == edges.T, "Incidence matrix is not symmetric"
            self.incidence = incidence
            self.ne = np.count_nonzero(np.tril(incidence))  # number of edges

    @classmethod
    def box(cls, half_extents, position=None, rotation=None):
        half_extents = np.array(half_extents)
        assert (half_extents > 0).all(), "Half extents must be positive."
        x, y, z = half_extents

        # fmt: off
        local_normals = np.vstack((np.eye(3), -np.eye(3)))
        local_vertices = np.array([
            [ x,  y,  z],
            [ x,  y, -z],
            [ x, -y, -z],
            [ x, -y,  z],
            [-x,  y,  z],
            [-x,  y, -z],
            [-x, -y, -z],
            [-x, -y,  z]])
        # fmt: on

        return cls(local_vertices, local_normals, position, rotation)

    # TODO might be nice to have this return a new object
    def update_pose(self, position=None, rotation=None):
        if position is None:
            position = np.zeros(3)
        if rotation is None:
            rotation = np.eye(3)

        self.position = position
        self.rotation = rotation
        self.vertices = position + (rotation @ self.local_vertices.T).T
        self.normals = (rotation @ self.local_normals.T).T

    def transform(self, position=None, rotation=None):
        if position is None:
            position = np.zeros(3)
        if rotation is None:
            rotation = np.eye(3)

        # TODO: update existing position and rotation and create a new object
        new_position = rotation @ self.position + position
        new_rotation = rotation @ self.rotation

        return ConvexPolyhedron()  # TODO


    def _compute_incidence_matrix(self, tol=1e-8):
        C = np.zeros((self.nv, self.nv), dtype=bool)
        for i in range(self.nf):
            # plane defining this face
            normal = self.normals[i, :]
            span = plane_span(normal)

            # TODO this can be streamlined considerably
            point = self.max_vertex_along_axis(normal)

            # get the vertices on this face
            projection = project_vertices_on_axes(self.vertices, point, normal)
            (idx,) = np.nonzero(np.abs(projection) < tol)
            assert len(idx) >= 3
            V_3d = self.vertices[idx]

            # project into 2D to wind them
            V_2d = project_vertices_on_axes(V_3d, point, span)
            _, idx2 = wind_polygon_vertices(V_2d)
            idx_wound = idx[idx2]

            # record connections for the face given the winding order
            for j in range(len(idx_wound)):
                a = idx_wound[j - 1]
                b = idx_wound[j]
                C[a, b] = True
                C[b, a] = True
        return C

    def limits_along_axis(self, axis):
        """Minimum and maximum values of the shape projected onto an axis."""
        axis = axis / np.linalg.norm(axis)
        projection = axis @ self.vertices.T
        return np.array([np.min(projection), np.max(projection)])

    def length_along_axis(self, axis):
        """Length of the shape along a given axis.

        i.e., this is the length of the projection of the shape onto the given axis.
        """
        axis = axis / np.linalg.norm(axis)
        limits = self.limits_along_axis(axis)
        return limits[1] - limits[0]

    def height(self):
        """Height of the polyhedron is the length in the z-direction."""
        return self.length_along_axis(np.array([0, 0, 1]))

    def max_vertex_along_axis(self, axis):
        """Get the vertex that is farthest along the given axis."""
        axis = axis / np.linalg.norm(axis)
        projection = axis @ self.vertices.T
        return self.vertices[np.argmax(projection), :]

    def get_vertices_in_plane(self, point, normal, tol=1e-8):
        """Get the vertices that lie in the plane defined by the point and normal."""
        projection = project_vertices_on_axes(self.vertices, point, normal)
        return self.vertices[np.nonzero(np.abs(projection) < tol)]

    def get_polygon_in_plane(self, point, plane_normal, plane_span, tol=1e-8):
        """Get the interection of this shape with the plane defined by the
        point and normal.

        The resultant polygon is projected onto the span basis.
        """
        V_3d = self.get_vertices_in_plane(point, plane_normal, tol=tol)
        V_2d = project_vertices_on_axes(V_3d, point, plane_span)
        return wind_polygon_vertices(V_2d)[0]

    def distance_from_centroid_to_boundary(self, axis, offset=None, tol=1e-8):
        """Get the distance from the shape's position to its boundary in
        direction given by axis.

        An offset can be provided, which is relative to the position of the
        shape.
        """
        axis = axis / np.linalg.norm(axis)

        # this is a linear programming problem
        n = self.vertices.shape[0]
        c = np.zeros(n + 1)
        c[0] = -1

        if offset is None:
            offset = np.zeros(3)

        # optimal point needs to be a convex combination of vertices (to still
        # be inside the shape)
        A_eq = np.zeros((4, n + 1))
        A_eq[:3, 0] = axis
        A_eq[:3, 1:] = -self.vertices.T
        A_eq[3, 1:] = np.ones(n)

        b_eq = np.ones(4)
        b_eq[:3] = -(self.position + offset)

        # TODO probably don't need the bounds: distance should be non-negative
        # bounds = [(None, None)]
        # bounds.extend([(0, None) for _ in range(n)])

        # solve the LP
        res = linprog(c, A_eq=A_eq, b_eq=b_eq)  # , bounds=bounds)
        d = res.x[0]
        assert d >= -tol, "Distance to boundary is negative!"
        return d

    def clip_with_half_space(V, point, normal, tol=1e-8):
        clipped_V = []
        clipped_indices = []

        # TODO we also need to handle the normals: these are changed based on
        # clipping

        # iterate over all edges (pairs of vertices) and clip each with the
        # halfspace, possibly discarding and generating new vertices
        nv = self.vertices.shape[0]
        for i in range(nv):
            for j in range(i, nv):
                if not self.incidence[i, j]:
                    continue
                v1 = self.vertices[i, :]
                v2 = self.vertices[j, :]
                indices, new_v = clip_line_segment_with_half_space(
                    v1, v2, point, normal, tol=tol
                )

            if 0 in indices:
                clipped_V.append(V[i, :])
                clipped_indices.append(i)
            if new_v is not None:
                clipped_V.append(new_v)
                clipped_indices.append(None)
            if 1 in indices:
                clipped_V.append(V[j, :])
                clipped_indices.append(j)

        new_V = []
        for i in range(len(clipped_V)):
            idx = clipped_indices[i]
            if idx is not None and clipped_indices[(i + 1) % len(clipped_V)] == idx:
                continue
            new_V.append(clipped_V[i])
        assert len(new_V) > 0

        return np.vstack(new_V)


# TODO deprecate in favour of polygon
# def Box3d(half_extents, position=None, rotation=None):
#     return ConvexPolyhedron.box(half_extents, position, rotation)


def orth2d(a):
    """Return vector `a` rotated by 90 degrees counter-clockwise."""
    # equivalent to np.array([[0, -1], [1, 0]]) @ a
    return np.array([-a[1], a[0]])


# TODO it would be nice to generalize this to a half space
def edge_line_intersection(v1, v2, p, a, tol=1e-8):
    """Find the intersection point of an edge with end points v1 and v2 and
       line going through point p in direction a.

    Returns:
        The intersection point if it exists, otherwise None.
    """
    b = v2 - v1
    c = orth2d(a)

    if np.abs(b @ c) < tol:
        return None

    bd = (p - v1) @ c / (b @ c)
    if bd < 0 or bd > 1.0:
        return None
    return v1 + bd * b


def line_segment_half_space_intersection(v1, v2, point, normal, tol=1e-8):
    assert v1.shape == v2.shape == point.shape == normal.shape

    normal = normal / np.linalg.norm(normal)
    d1 = normal @ (v1 - point)
    d2 = normal @ (v2 - point)

    # if either vertex is on the plane defining the halfspace, just return that
    # vertex. If both are, then the line lies on the plane and we can return
    # any point on the segment.
    if np.abs(d1) < tol:
        return v1
    if np.abs(d2) < tol:
        return v2

    # if both vertices are on one side of the half space, there is no
    # intersection
    if (d1 < tol and d2 < tol) or (d1 > -tol and d2 > -tol):
        return None

    t = normal @ (point - v1) / (normal @ (v2 - v1))
    assert 0 <= t <= 1
    return v1 + t * (v2 - v1)


def clip_line_segment_with_half_space(v1, v2, point, normal, tol=1e-8):
    assert v1.shape == v2.shape == point.shape == normal.shape

    normal = normal / np.linalg.norm(normal)
    d1 = normal @ (v1 - point)
    d2 = normal @ (v2 - point)

    # if both vertices are on one side of the line, we either keep both or
    # discard both (discard if the edge is a subset of the line)
    if d1 >= -tol and d2 >= -tol:
        return (0, 1), None
    if d1 <= tol and d2 <= tol:
        return (), None

    intersection = line_segment_half_space_intersection(v1, v2, point, normal, tol=tol)
    assert intersection is not None

    # keep the vertex on the left of the line as well as the intersection
    # point, while maintaining vertex order
    if d1 > 0:
        return (0,), intersection
    else:
        return (1,), intersection


# TODO it would be nice to generalize this to a half space
def clip_line_segment_with_line(v1, v2, p, a, tol=1e-8):
    """Clip a line segment with a line.

    The line segment is defined by end points v1 and v2. The half space is
    defined by point p and axis a. All arguments are 2D.

    Returns a tuple of indices for the vertices kept (i.e., (0, 1) when both
    vertices kept, (0,) only the first, (1,) only the second, () for neither)
    and the intersection point of the edge and line (can be None).
    """
    assert v1.shape == v2.shape == p.shape == a.shape == (2,)
    a = a / np.linalg(a)

    c = orth2d(a)
    d1 = c @ (v1 - p)
    d2 = c @ (v2 - p)

    # if both vertices are on one side of the line, we either keep both or
    # discard both (discard if the edge is a subset of the line)
    if d1 >= -tol and d2 >= -tol:
        return (0, 1), None
    elif d1 <= tol and d2 <= tol:
        return (), None

    intersection = edge_line_intersection(v1, v2, p, a, tol=tol)
    assert intersection is not None

    # keep the vertex on the left of the line as well as the intersection
    # point, while maintaining vertex order
    if d1 > 0:
        return (0,), intersection
    else:
        return (1,), intersection


def clip_polyhedron_with_half_space(V, point, normal, tol=1e-8):
    clipped_V = []
    clipped_indices = []

    # iterate over all edges (pairs of vertices)
    # TODO this is more complicated because there isn't an obvious canonical
    # way to get edges from just the vertices
    for i in range(V.shape[0]):
        j = (i + 1) % V.shape[0]
        v1 = V[i, :]
        v2 = V[j, :]
        indices, new_v = clip_line_segment_with_half_space(
            v1, v2, point, normal, tol=tol
        )
        if 0 in indices:
            clipped_V.append(V[i, :])
            clipped_indices.append(i)
        if new_v is not None:
            clipped_V.append(new_v)
            clipped_indices.append(None)
        if 1 in indices:
            clipped_V.append(V[j, :])
            clipped_indices.append(j)

    new_V = []
    for i in range(len(clipped_V)):
        idx = clipped_indices[i]
        if idx is not None and clipped_indices[(i + 1) % len(clipped_V)] == idx:
            continue
        new_V.append(clipped_V[i])
    assert len(new_V) > 0

    return np.vstack(new_V)


def clip_polygon_with_line(V, p, a, tol=1e-8):
    assert V.shape[1] == 2

    clipped_V = []
    clipped_indices = []

    for i in range(V.shape[0]):
        j = (i + 1) % V.shape[0]
        v1 = V[i, :]
        v2 = V[j, :]
        indices, new_v = clip_line_segment_with_line(v1, v2, p, a, tol=tol)
        if 0 in indices:
            clipped_V.append(V[i, :])
            clipped_indices.append(i)
        if new_v is not None:
            clipped_V.append(new_v)
            clipped_indices.append(None)
        if 1 in indices:
            clipped_V.append(V[j, :])
            clipped_indices.append(j)

    new_V = []
    for i in range(len(clipped_V)):
        idx = clipped_indices[i]
        if idx is not None and clipped_indices[(i + 1) % len(clipped_V)] == idx:
            continue
        new_V.append(clipped_V[i])
    assert len(new_V) > 0

    return np.vstack(new_V)


def clip_polygon_with_polygon(V1, V2, tol=1e-8):
    """Get the polygonal overlap between convex polygons V1 and V2.

    Parameters:
        V1 and V2 are vertex arrays of shape (m, 2) and (n, 2)

    Returns:
        Array of vertices defining the overlap region.
    """
    assert V1.shape[1] == 2
    assert V2.shape[1] == 2

    n = V2.shape[0]

    V = V1
    for i in range(n):
        p = V2[i, :]
        a = V2[(i + 1) % n] - p
        if np.linalg.norm(a) < tol:
            raise ValueError("Clipping polygon has repeated vertices.")
        a = a / np.linalg.norm(a)
        V = clip_polygon_with_line(V, p, a, tol=tol)

    return V


def project_vertices_on_axes(vertices, point, axes):
    return (axes @ (vertices - point).T).T


def wind_polygon_vertices(V):
    """Order vertices counter-clockwise.

    Parameters:
        V: shape (n, 2) array

    Returns:
        An array of shape (n, 2) containing the sorted vertices.
    """
    # order by increasing angle around a central point
    assert V.shape[1] == 2
    c = np.mean(V, axis=0)
    angles = np.arctan2(V[:, 1] - c[1], V[:, 0] - c[0])
    idx = np.argsort(angles)
    return V[idx, :], idx


# TODO rename now that we can handle arbitrary polyhedra
def box_box_axis_aligned_contact(box1, box2, tol=1e-8):

    # in general, we need to test all face normals and all axes that are cross
    # products between pairs of face normals, one from each shape
    # <https://www.geometrictools.com/Documentation/DynamicCollisionDetection.pdf>
    cross_normals = []
    for i in range(box1.normals.shape[0]):
        for j in range(box2.normals.shape[0]):
            a = np.cross(box1.normals[i, :], box2.normals[j, :])
            mag = np.linalg.norm(a)
            if mag > tol:
                cross_normals.append(a / mag)
    axes = np.vstack((box1.normals, box2.normals, cross_normals))
    axis_idx = None

    for i in range(axes.shape[0]):
        axis = axes[i, :]

        limits1 = box1.limits_along_axis(axis)
        limits2 = box2.limits_along_axis(axis)

        upper = np.min([limits1[1], limits2[1]])
        lower = np.max([limits1[0], limits2[0]])

        # find the plane that is just touching, or determine that shapes do not
        # intersect
        if np.abs(upper - lower) < tol:
            if limits1[0] < limits2[0]:
                # projection of first box is smaller on this axis
                # this means we also want to return the normal pointing in the
                # reverse direction, such that it points into the first box
                point = box1.max_vertex_along_axis(axis)
                normal_multiplier = -1
            else:
                point = box2.max_vertex_along_axis(axis)
                normal_multiplier = 1
            axis_idx = i
        elif upper < lower:
            # shapes are not intersecting, nothing more to do
            print("Shapes not intersecting.")
            import IPython

            IPython.embed()
            return None, None

    # shapes are penetrating: this is more complicated, and we don't deal with
    # it here
    if axis_idx is None:
        print("Shapes are penetrating.")
        return None, None

    plane_normal = axes[axis_idx, :]
    span = plane_span(plane_normal)

    # get the polygonal "slice" of each box in the contact plane
    V1 = box1.get_polygon_in_plane(point, plane_normal, span, tol=tol)
    V2 = box2.get_polygon_in_plane(point, plane_normal, span, tol=tol)

    # find the overlapping region
    Vp = clip_polygon_with_polygon(V1, V2, tol=tol)

    # unproject back into world coordinates
    V = point + Vp @ span

    # normal points into the first shape
    return V, normal_multiplier * plane_normal
