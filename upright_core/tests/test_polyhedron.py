import numpy as np
import pytest
import upright_core as core
from upright_core.util import sort_canonical


def test_polyhedron_transform():
    box = core.polyhedron.ConvexPolyhedron.box([1, 1, 1])

    # default pose
    assert np.allclose(box.position, np.zeros(3))
    assert np.allclose(box.rotation, np.eye(3))

    # translation
    translation = np.array([1, 0, 0])
    box2 = box.transform(translation=translation)
    assert np.allclose(box2.position, translation)
    assert np.allclose(box2.normals, box.normals)
    assert np.allclose(box.vertices + translation, box2.vertices)

    # rotation
    # since we have a cube, rotations of 90-degrees about principal axes
    # shouldn't change the values of the vertices or normals, but it will
    # change their order
    rotation = core.math.rotx(np.pi / 2) @ core.math.roty(np.pi / 2)
    box3 = box.transform(rotation=rotation)
    assert np.allclose(box3.rotation, rotation)
    assert np.allclose(sort_canonical(box.vertices), sort_canonical(box3.vertices))
    assert np.allclose(sort_canonical(box.normals), sort_canonical(box3.normals))

    # rotation and translation
    box4 = box.transform(translation=translation, rotation=rotation)
    assert np.allclose(
        sort_canonical(box.vertices + translation), sort_canonical(box4.vertices)
    )
    assert np.allclose(sort_canonical(box.normals), sort_canonical(box4.normals))


def test_polyhedron_limits():
    box = core.polyhedron.ConvexPolyhedron.box([1, 1, 1])

    # principal axes
    assert np.allclose(box.limits_along_axis([1, 0, 0]), [-1, 1])
    assert np.allclose(box.limits_along_axis([0, 1, 0]), [-1, 1])
    assert np.allclose(box.limits_along_axis([0, 0, 1]), [-1, 1])

    # diagonal
    axis = np.array([1, 1, 1])
    d = np.linalg.norm(axis)
    axis = axis / d
    assert np.allclose(box.limits_along_axis(axis), [-d, d])

    # rotate 45 degrees about z-axis and check limits along that axis
    rotation = core.math.rotz(np.pi / 4)
    box2 = box.transform(rotation=rotation)
    axis = np.array([1, 1, 0])
    axis = axis / np.linalg.norm(axis)
    assert np.allclose(box2.limits_along_axis(axis), [-1, 1])


def test_polyhedron_lengths():
    box = core.polyhedron.ConvexPolyhedron.box([1, 1, 1])

    # principal axes
    assert np.isclose(box.length_along_axis([1, 0, 0]), 2)
    assert np.isclose(box.length_along_axis([0, 1, 0]), 2)
    assert np.isclose(box.length_along_axis([0, 0, 1]), 2)

    # height is just the z-axis again
    assert np.isclose(box.height(), 2)

    # diagonal
    axis = np.array([1, 1, 1])
    d = np.linalg.norm(axis)
    axis = axis / d
    assert np.isclose(box.length_along_axis(axis), 2 * d)


def test_max_vertex_along_axis():
    box = core.polyhedron.ConvexPolyhedron.box([1, 1, 1])
    axis = np.array([1, 1, 1])
    axis = axis / np.linalg.norm(axis)

    assert np.allclose(box.max_vertex_along_axis(axis), [1, 1, 1])
    assert np.allclose(box.max_vertex_along_axis(-axis), [-1, -1, -1])


def test_get_vertices_in_plane():
    box = core.polyhedron.ConvexPolyhedron.box([1, 1, 1])
    n = np.array([1, 0, 0])
    p = np.array([1, 0, 0])

    # vertices are in no particular order
    V_actual = box.get_vertices_in_plane(p, n)
    V_expected = np.array([[1, 1, 1], [1, -1, 1], [1, -1, -1], [1, 1, -1]])
    assert np.allclose(sort_canonical(V_actual), sort_canonical(V_expected))


def test_get_polygon_in_plane():
    box = core.polyhedron.ConvexPolyhedron.box([1, 1, 1])
    n = np.array([1, 0, 0])
    S = np.array([[0, 1, 0], [0, 0, 1]])
    p = np.array([1, 0, 0])

    # vertices are wound counter-clockwise but can start at an arbitrary index
    V_actual = box.get_polygon_in_plane(p, n, S)
    V_expected = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    assert np.allclose(sort_canonical(V_actual), sort_canonical(V_expected))


def test_distance_from_centroid_to_boundary():
    box = core.polyhedron.ConvexPolyhedron.box([1, 1, 1])
    n = np.array([1, 0, 0])

    d = box.distance_from_centroid_to_boundary([1, 0, 0])
    assert np.isclose(d, 1)

    axis = np.array([1, 1, 1])
    a = np.linalg.norm(axis)
    axis = axis / a
    d = box.distance_from_centroid_to_boundary(axis)
    assert np.isclose(d, a)


def test_orth2d():
    a = [1, 2]
    assert np.allclose(core.polyhedron.orth2d(a), [-2, 1])


def test_line_segment_half_space_intersection_2d():
    point = np.array([1, 0])
    normal = np.array([0, 1])

    # intersection
    v1 = np.array([1, 1])
    v2 = np.array([-1, -1])
    r = core.polyhedron.line_segment_half_space_intersection(v1, v2, point, normal)
    assert np.allclose(r, [0, 0])

    # intersection with one of the vertices
    v1 = np.array([0, 0])
    r = core.polyhedron.line_segment_half_space_intersection(v1, v2, point, normal)
    assert np.allclose(r, v1)

    # no intersection
    v1 = np.array([-0.5, -0.5])
    r = core.polyhedron.line_segment_half_space_intersection(v1, v2, point, normal)
    assert r is None


def test_line_segment_half_space_intersection_3d():
    point = np.array([1, 0, 0])
    normal = np.array([0, 1, 0])

    # intersection
    v1 = np.array([1, 1, 1])
    v2 = np.array([-1, -1, -1])
    r = core.polyhedron.line_segment_half_space_intersection(v1, v2, point, normal)
    assert np.allclose(r, [0, 0, 0])

    # intersection with one of the vertices
    v1 = np.array([0, 0, 0])
    r = core.polyhedron.line_segment_half_space_intersection(v1, v2, point, normal)
    assert np.allclose(r, v1)

    # no intersection
    v1 = np.array([-0.5, -0.5, -0.5])
    r = core.polyhedron.line_segment_half_space_intersection(v1, v2, point, normal)
    assert r is None


def test_wedge():
    """Test that wedge shape is build correctly."""
    wedge = core.polyhedron.ConvexPolyhedron.wedge([1, 1, 1])
    V_expected = np.array(
        [[-1, 1, 1], [-1, 1, -1], [-1, -1, -1], [-1, -1, 1], [1, -1, -1], [1, 1, -1]]
    )
    n = np.array([1, 0, 1])  # non-axis-aligned normal
    N_expected = np.vstack((-np.eye(3), [0, 1, 0], n / np.linalg.norm(n)))
    assert np.allclose(sort_canonical(wedge.vertices), sort_canonical(V_expected))
    assert np.allclose(sort_canonical(wedge.normals), sort_canonical(N_expected))


def test_box_box_contact():
    box1 = core.polyhedron.ConvexPolyhedron.box([1, 1, 1])
    box2 = core.polyhedron.ConvexPolyhedron.box([0.5, 0.5, 0.5])

    # place box2 atop box1, and offset into a corner
    box2 = box2.transform(translation=[0.5, 0.5, 1.5])

    points, normal = core.polyhedron.axis_aligned_contact(box1, box2)
    points_expected = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]])

    assert np.allclose(normal, [0, 0, -1])
    assert np.allclose(sort_canonical(points_expected), sort_canonical(points))

    # shapes are penetrating: nothing should be returned
    box3 = box2.transform(translation=[0, 0, -0.1])
    ret = core.polyhedron.axis_aligned_contact(box1, box3)
    assert ret == (None, None)

    # shapes are not intersecting: nothing should be returned
    box4 = box2.transform(translation=[0, 0, 0.1])
    ret = core.polyhedron.axis_aligned_contact(box1, box4)
    assert ret == (None, None)


def test_wedge_box_contact():
    wedge = core.polyhedron.ConvexPolyhedron.wedge([1, 1, 1])
    box = core.polyhedron.ConvexPolyhedron.box([1, 1, 1])

    n = np.array([1, 0, 1])
    n = n / np.linalg.norm(n)

    # rotate the box so its bottom face is aligned with the slope of the wedge
    rotation = core.math.roty(-np.pi / 4)
    box = box.transform(rotation=rotation, translation=n)

    points, normal = core.polyhedron.axis_aligned_contact(wedge, box)

    # normal should point into the wedge (the first shape)
    assert np.allclose(normal, -n)

    a = np.sqrt(2) / 2
    points_expected = np.array([[-a, 1, a], [-a, -1, a], [a, -1, -a], [a, 1, -a]])
    assert np.allclose(sort_canonical(points_expected), sort_canonical(points))


def test_line_contact():
    """Test when contact area is a line (i.e. contact polygon is degenerate)."""
    box1 = core.polyhedron.ConvexPolyhedron.box([0.03, 0.03, 0.3])
    rotation = core.math.rotz(np.pi / 4)
    box1 = box1.transform(rotation=rotation)

    box2 = core.polyhedron.ConvexPolyhedron.box([0.1, 0.1, 0.1])
    dx = box1.distance_from_centroid_to_boundary([1, 0, 0])
    box2 = box2.transform(translation=[dx + 0.1, 0, 0])

    points, normal = core.polyhedron.axis_aligned_contact(box1, box2)

    # print(points)
    # print(normal)
    #
    # import IPython
    # IPython.embed()

    # only two contact points
    points_expected = np.array([[dx, 0, -0.1], [dx, 0, 0.1]])

    assert np.allclose(normal, [-1, 0, 0])
    assert np.allclose(sort_canonical(points), sort_canonical(points_expected))


def test_vertices_projection_on_axes():
    axes = np.array([[1, 0, 0], [0, 1, 0]])
    point = np.array([0, 0, 1])

    vertices = core.polyhedron.ConvexPolyhedron.box([1, 1, 1]).vertices
    projection = core.polyhedron.project_vertices_on_axes(vertices, point, axes)

    assert np.allclose(projection, vertices[:, :2])


def test_wind_polygon_vertices():
    # same sets of vertices, different orders
    V1 = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
    V2 = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]])

    V1_wound, _ = core.polyhedron.wind_polygon_vertices(V1)
    V2_wound, _ = core.polyhedron.wind_polygon_vertices(V2)

    # wound order should be the same
    assert np.allclose(V1_wound, V2_wound)


def test_clip_line_segment_with_half_space():
    v1 = np.array([0, 0])
    v2 = np.array([2, 0])

    # line along x=1
    point = np.array([1, 0])
    normal = np.array([-1, 0])

    # clip the line
    new_vs = core.polyhedron.clip_line_segment_with_half_space(
        v1, v2, point, normal
    )
    assert np.allclose(new_vs, [[0, 0], [1, 0]])

    # no intersection and the segment is kept
    point = np.array([3, 0])
    new_vs = core.polyhedron.clip_line_segment_with_half_space(
        v1, v2, point, normal
    )
    assert np.allclose(new_vs, [v1, v2])

    # no intersection and the segment is discarded
    point = np.array([-1, 0])
    new_vs = core.polyhedron.clip_line_segment_with_half_space(
        v1, v2, point, normal
    )
    assert new_vs == ()


def test_clip_polygon_with_half_space():
    # square
    V = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
    V, _ = core.polyhedron.wind_polygon_vertices(V)

    point = np.array([0, 0])
    normal = np.array([-1, 1]) / np.sqrt(2)

    V_clipped = core.polyhedron.clip_polygon_with_half_space(V, point, normal)
    V_clipped_expected = np.array([[-1, -1], [1, 1], [-1, 1]])
    assert np.allclose(sort_canonical(V_clipped), sort_canonical(V_clipped_expected))

    # no intersection but shape is kept
    point = np.array([1, -1])
    V_clipped = core.polyhedron.clip_polygon_with_half_space(V, point, normal)
    assert np.allclose(sort_canonical(V_clipped), sort_canonical(V))

    # no intersection and shape is discarded
    point = np.array([-1, 1])
    V_clipped = core.polyhedron.clip_polygon_with_half_space(V, point, normal)
    assert V_clipped is None


def test_clip_polygon_with_polygon():
    # square
    V1 = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
    V1, _ = core.polyhedron.wind_polygon_vertices(V1)
    V2 = V1 + [1, 1]

    V = core.polyhedron.clip_polygon_with_polygon(V1, V2)
    V_expected = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    assert np.allclose(sort_canonical(V), sort_canonical(V_expected))

    # no overlap
    V2 = V1 + [-2, 2]
    V = core.polyhedron.clip_polygon_with_polygon(V1, V2)
    assert V is None


# NOTE: experimental
def test_incidence():
    # TODO this relies on internal implementation details of the box (i.e. the
    # order of vertices): should be revised (i.e. specify the vertices manually)
    box = core.polyhedron.ConvexPolyhedron.box([1, 1, 1])

    C_expected = np.zeros((box.nv, box.nv), dtype=bool)
    C_expected[0, 1] = True
    C_expected[1, 2] = True
    C_expected[2, 3] = True
    C_expected[3, 0] = True

    C_expected[4, 5] = True
    C_expected[5, 6] = True
    C_expected[6, 7] = True
    C_expected[7, 4] = True

    C_expected[0, 4] = True
    C_expected[1, 5] = True
    C_expected[2, 6] = True
    C_expected[3, 7] = True

    # make both connections are represented in the matrix
    C_expected = np.logical_or(C_expected, C_expected.T)

    assert np.all(box.incidence == C_expected)
