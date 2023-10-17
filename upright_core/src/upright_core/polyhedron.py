import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog

from upright_core.math import plane_span


DEFAULT_TOLERANCE = 1e-8


class ConvexPolyhedron:
    def __init__(self, vertices, normals, position=None, rotation=None, incidence=None):
        self.nv = vertices.shape[0]  # number of vertices
        self.nf = normals.shape[0]  # number of faces

        self.vertices = vertices
        self.normals = normals

        # nominal position and rotation are stored alongside the vertices
        # when transformed, these values along with the vertices and normals
        # are updated
        if position is None:
            position = np.zeros(3)
        if rotation is None:
            rotation = np.eye(3)
        self.position = position
        self.rotation = rotation

        # TODO possibly more convenient to compute wound indices of vertices
        # for each face: then clipping can be done face-wise
        if incidence is None:
            self.incidence = self._compute_incidence_matrix()
        else:
            assert incidence.dtype == bool, "Incidence matrix must have dtype=bool"
            assert incidence.shape == (
                self.nv,
                self.nv,
            ), "Incidence matrix of wrong size"
            assert np.all(incidence == incidence.T), "Incidence matrix is not symmetric"
            self.incidence = incidence
        self.ne = np.count_nonzero(np.tril(self.incidence))  # number of edges

    @classmethod
    def box(cls, half_extents, position=None, rotation=None):
        """Create a box: three pairs of parellel sides."""
        half_extents = np.array(half_extents)
        assert (half_extents > 0).all(), "Half extents must be positive."
        x, y, z = half_extents

        # fmt: off
        normals = np.vstack((np.eye(3), -np.eye(3)))
        vertices = np.array([
            [ x,  y,  z],
            [ x,  y, -z],
            [ x, -y, -z],
            [ x, -y,  z],
            [-x,  y,  z],
            [-x,  y, -z],
            [-x, -y, -z],
            [-x, -y,  z]])
        # fmt: on

        return cls(vertices, normals, position, rotation)

    @classmethod
    def wedge(cls, half_extents, position=None, rotation=None):
        """Create a wedge: two right-angle triangular faces and three rectangular faces.

        The slope faces the positive x-direction.
        """
        half_extents = np.array(half_extents)
        assert (half_extents > 0).all(), "Half extents must be positive."
        hx, hy, hz = half_extents

        # fmt: off
        vertices = np.array([
            [-hx, -hy, -hz], [hx, -hy, -hz], [-hx, -hy, hz],
            [-hx,  hy, -hz], [hx,  hy, -hz], [-hx,  hy, hz]])
        # fmt: on

        # compute normal of the non-axis-aligned face
        e12 = vertices[2, :] - vertices[1, :]
        e14 = vertices[4, :] - vertices[1, :]
        n = np.cross(e14, e12)
        n = n / np.linalg.norm(n)

        # other normals point backward (-x), down (-z), and to each side (+- y)
        normals = np.vstack((-np.eye(3), np.array([0, 1, 0]), n))

        return cls(vertices, normals, position, rotation)

    def transform(self, translation=None, rotation=None):
        """Apply a rigid body transform to the polyhedron.

        A new transformed polyhedron is returned; the original is unchanged.
        """
        if translation is None:
            translation = np.zeros(3)
        if rotation is None:
            rotation = np.eye(3)

        # apply transform to existing pose
        new_position = rotation @ self.position + translation
        new_rotation = rotation @ self.rotation

        # apply transform to vertices and normals
        vertices = translation + (rotation @ self.vertices.T).T
        normals = (rotation @ self.normals.T).T

        # build the new polyhedron
        # the incidence matrix is invariant to tranforms
        return ConvexPolyhedron(
            vertices=vertices,
            normals=normals,
            position=new_position,
            rotation=new_rotation,
            incidence=self.incidence.copy(),
        )

    def _compute_incidence_matrix(self, tol=DEFAULT_TOLERANCE):
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
            # each face must have at least three vertices
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

    def _compute_faces(self):
        # TODO compute 2d list of lists: one list of vertex indices per face
        pass

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

    def get_vertices_in_plane(self, point, normal, tol=DEFAULT_TOLERANCE):
        """Get the vertices that lie in the plane defined by the point and normal."""
        projection = project_vertices_on_axes(self.vertices, point, normal)
        return self.vertices[np.nonzero(np.abs(projection) < tol)]

    def get_polygon_in_plane(
        self, point, plane_normal, plane_span, tol=DEFAULT_TOLERANCE
    ):
        """Get the interection of this shape with the plane defined by the
        point and normal.

        The resultant polygon is projected onto the span basis.
        """
        V_3d = self.get_vertices_in_plane(point, plane_normal, tol=tol)
        V_2d = project_vertices_on_axes(V_3d, point, plane_span)
        return wind_polygon_vertices(V_2d)[0]

    def distance_from_centroid_to_boundary(
        self, axis, offset=None, tol=DEFAULT_TOLERANCE
    ):
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

        # solve the LP
        res = linprog(c, A_eq=A_eq, b_eq=b_eq)  # , bounds=bounds)
        d = res.x[0]
        assert d >= -tol, "Distance to boundary is negative!"
        return d

    def clip_with_half_space(V, point, normal, tol=DEFAULT_TOLERANCE):
        raise NotImplementedError()

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


def orth2d(a):
    """Return vector `a` rotated by 90 degrees counter-clockwise.

    For a polygon with counter-clockwise winding, this gives the inward-facing
    normal for  agiven edge.
    """
    # equivalent to np.array([[0, -1], [1, 0]]) @ a
    return np.array([-a[1], a[0]])


def line_segment_half_space_intersection(v1, v2, point, normal, tol=DEFAULT_TOLERANCE):
    """Intersection of a line segment and a half space.

    The line segment is defined by vertices `v1` and `v2`. The half-space
    passes through `point` and is defined by `normal`.

    Returns the intersection point or None if there is no intersection.
    """
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


def clip_line_segment_with_half_space(v1, v2, point, normal, tol=DEFAULT_TOLERANCE):
    """Clip a line segment with a half space.

    The line segment is defined by vertices `v1` and `v2`. The half-space
    passes through `point` and is defined by `normal`.

    Returns a tuple consisting the new vertices, which be either:
    * the first vertex and the intersection point;
    * the intersection point and the second vertex;
    * both existing vertices (whole segment is kept);
    * empty (whole segment is discarded).
    """
    assert v1.shape == v2.shape == point.shape == normal.shape

    normal = normal / np.linalg.norm(normal)
    d1 = normal @ (v1 - point)
    d2 = normal @ (v2 - point)

    # if both vertices are on one side of the line, we either keep both or
    # discard both
    if d1 >= -tol and d2 >= -tol:
        return v1, v2
    if d1 <= tol and d2 <= tol:
        return ()

    intersection = line_segment_half_space_intersection(v1, v2, point, normal, tol=tol)
    assert intersection is not None

    if d1 > 0:
        return v1, intersection
    else:
        return intersection, v2


def clip_polygon_with_half_space(V, point, normal, tol=DEFAULT_TOLERANCE):
    """Clip a polygon consisting of vertices `V` by a line passing through `p`
    in direction `a`.

    Returns a new set of vertices defining the clipped polygon.
    """
    assert V.shape[1] == 2

    # clip each edge of the polygon with the half space
    clipped_vertices = []
    for i in range(V.shape[0]):
        j = (i + 1) % V.shape[0]
        v1 = V[i, :]
        v2 = V[j, :]
        new_vs = clip_line_segment_with_half_space(v1, v2, point, normal, tol=tol)
        clipped_vertices.extend(new_vs)

    # early return if the whole polygon is removed
    if len(clipped_vertices) == 0:
        return None

    # filter out duplicate vertices
    new_vertices = []
    for candidate_vertex in clipped_vertices:
        exists = False
        for existing_vertex in new_vertices:
            if np.linalg.norm(candidate_vertex - existing_vertex) < tol:
                exists = True
                break
        if exists:
            continue
        new_vertices.append(candidate_vertex)

    # NOTE: no need to re-wind vertices because ordering is preserved by above
    # routines
    return np.array(new_vertices)


def clip_polygon_with_polygon(V1, V2, tol=DEFAULT_TOLERANCE):
    """Get the polygonal overlap between convex polygons V1 and V2.

    Parameters:
        V1 and V2 are vertex arrays of shape (m, 2) and (n, 2), assumed to be
        wound in counter-clockwise order.

    Returns:
        Array of vertices defining the overlap region.
    """
    assert V1.shape[1] == 2
    assert V2.shape[1] == 2

    n = V2.shape[0]

    V = V1
    for i in range(n):
        point = V2[i, :]
        a = V2[(i + 1) % n] - point
        if np.linalg.norm(a) < tol:
            raise ValueError("Clipping polygon has repeated vertices.")
        a = a / np.linalg.norm(a)
        normal = orth2d(a)  # inward-facing normal
        V = clip_polygon_with_half_space(V, point, normal, tol=tol)

        # if the clip ever excluded the whole polygon (i.e. there is no
        # overlap), then we are done
        if V is None:
            return None
    return V


def project_vertices_on_axes(vertices, point, axes):
    """Project vertices on a plane spanned by `axes` and passing through `point`."""
    # NOTE: there is of course a loss of information in that vertices that
    # differ only in the nullspace of `axes` map to the same point in the
    # projection
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


# TODO this name is somewhat misleading
def axis_aligned_contact(poly1, poly2, tol=DEFAULT_TOLERANCE):
    """Compute contact points and normal between two polyhedra.

    Returns: a tuple (V, n), where V is the array of contact points (one per
    row) and n is the (shared) contact normal.
    """

    # in general, we need to test all face normals and all axes that are cross
    # products between pairs of face normals, one from each shape
    # <https://www.geometrictools.com/Documentation/DynamicCollisionDetection.pdf>
    cross_normals = []
    for i in range(poly1.normals.shape[0]):
        for j in range(poly2.normals.shape[0]):
            a = np.cross(poly1.normals[i, :], poly2.normals[j, :])
            mag = np.linalg.norm(a)
            if mag > tol:
                cross_normals.append(a / mag)
    axes = np.vstack((poly1.normals, poly2.normals, cross_normals))
    axis_idx = None

    for i in range(axes.shape[0]):
        axis = axes[i, :]

        limits1 = poly1.limits_along_axis(axis)
        limits2 = poly2.limits_along_axis(axis)

        upper = np.min([limits1[1], limits2[1]])
        lower = np.max([limits1[0], limits2[0]])

        # find the plane that is just touching, or determine that shapes do not
        # intersect
        if np.abs(upper - lower) < tol:
            if limits1[0] < limits2[0]:
                # projection of first polyhedron is smaller on this axis
                # this means we also want to return the normal pointing in the
                # reverse direction, such that it points into the first polyhedron
                point = poly1.max_vertex_along_axis(axis)
                normal_multiplier = -1
            else:
                point = poly2.max_vertex_along_axis(axis)
                normal_multiplier = 1
            axis_idx = i
        elif upper < lower:
            # shapes are not intersecting with distance of at least `lower -
            # upper` between them: nothing more to do
            print("Shapes not intersecting.")
            return None, None

    # shapes are penetrating: this is more complicated, and we don't deal with
    # it here
    if axis_idx is None:
        print("Shapes are penetrating.")
        return None, None

    plane_normal = axes[axis_idx, :]
    span = plane_span(plane_normal)

    # get the polygonal "slice" of each polyhedron in the contact plane
    V1 = poly1.get_polygon_in_plane(point, plane_normal, span, tol=tol)
    V2 = poly2.get_polygon_in_plane(point, plane_normal, span, tol=tol)

    # find the overlapping region
    Vp = clip_polygon_with_polygon(V1, V2, tol=tol)

    # unproject back into world coordinates
    V = point + Vp @ span

    # normal points into the first shape
    return V, normal_multiplier * plane_normal
