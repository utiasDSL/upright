import numpy as np
import pytest
import upright_core as core


def sort_canonical(A):
    V = np.copy(A)
    for i in range(len(V.shape)):
        V.sort(axis=-i-1)
    return V


def test_polyhedron_limits():
    box = core.geometry.ConvexPolyhedron.box([1, 1, 1])

    # principal axes
    assert np.allclose(box.limits_along_axis([1, 0, 0]), [-1, 1])
    assert np.allclose(box.limits_along_axis([0, 1, 0]), [-1, 1])
    assert np.allclose(box.limits_along_axis([0, 0, 1]), [-1, 1])

    # diagonal
    axis = np.array([1, 1, 1])
    d = np.linalg.norm(axis)
    axis = axis / d
    assert np.allclose(box.limits_along_axis(axis), [-d, d])

    # TODO test with a rotated polyhedron as well


def test_polyhedron_lengths():
    box = core.geometry.ConvexPolyhedron.box([1, 1, 1])

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
    box = core.geometry.ConvexPolyhedron.box([1, 1, 1])
    axis = np.array([1, 1, 1])
    axis = axis / np.linalg.norm(axis)

    assert np.allclose(box.max_vertex_along_axis(axis), [1, 1, 1])
    assert np.allclose(box.max_vertex_along_axis(-axis), [-1, -1, -1])


def test_get_vertices_in_plane():
    box = core.geometry.ConvexPolyhedron.box([1, 1, 1])
    n = np.array([1, 0, 0])
    p = np.array([1, 0, 0])

    # vertices are in no particular order
    V_actual = box.get_vertices_in_plane(p, n)
    V_expected = np.array([[1, 1, 1], [1, -1, 1], [1, -1, -1], [1, 1, -1]])
    assert np.allclose(sort_canonical(V_actual), sort_canonical(V_expected))


def test_get_polygon_in_plane():
    box = core.geometry.ConvexPolyhedron.box([1, 1, 1])
    n = np.array([1, 0, 0])
    S = np.array([[0, 1, 0], [0, 0, 1]])
    p = np.array([1, 0, 0])

    # vertices are wound counter-clockwise but can start at an arbitrary index
    V_actual = box.get_polygon_in_plane(p, n, S)
    V_expected = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    assert np.allclose(sort_canonical(V_actual), sort_canonical(V_expected))


def test_distance_from_centroid_to_boundary():
    box = core.geometry.ConvexPolyhedron.box([1, 1, 1])
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
    assert np.allclose(core.geometry.orth2d(a), [-2, 1])


def test_line_segment_half_space_intersection_2d():
    point = np.array([1, 0])
    normal = np.array([0, 1])

    # intersection
    v1 = np.array([1, 1])
    v2 = np.array([-1, -1])
    r = core.geometry.line_segment_half_space_intersection(v1, v2, point, normal)
    assert np.allclose(r, [0, 0])

    # intersection with one of the vertices
    v1 = np.array([0, 0])
    r = core.geometry.line_segment_half_space_intersection(v1, v2, point, normal)
    assert np.allclose(r, v1)

    # no intersection
    v1 = np.array([-0.5, -0.5])
    r = core.geometry.line_segment_half_space_intersection(v1, v2, point, normal)
    assert r is None


def test_line_segment_half_space_intersection_3d():
    point = np.array([1, 0, 0])
    normal = np.array([0, 1, 0])

    # intersection
    v1 = np.array([1, 1, 1])
    v2 = np.array([-1, -1, -1])
    r = core.geometry.line_segment_half_space_intersection(v1, v2, point, normal)
    assert np.allclose(r, [0, 0, 0])

    # intersection with one of the vertices
    v1 = np.array([0, 0, 0])
    r = core.geometry.line_segment_half_space_intersection(v1, v2, point, normal)
    assert np.allclose(r, v1)

    # no intersection
    v1 = np.array([-0.5, -0.5, -0.5])
    r = core.geometry.line_segment_half_space_intersection(v1, v2, point, normal)
    assert r is None

def test_incidence():
    box = core.geometry.ConvexPolyhedron.box([1, 1, 1])

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
