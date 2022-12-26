"""Tests for upright_core.math module."""
import numpy as np
import pytest
import upright_core as core


def test_skew3():
    """Test construction of skew-symmetric matrix."""
    v = [1, 2, 3]
    V_actual = core.math.skew3(v)
    V_expected = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
    assert np.allclose(V_actual, V_expected)


def test_quat_rot_identity():
    """Test conversions between identity rotation matrices and quaternions."""
    # identity roations
    q0 = np.array([0, 0, 0, 1])
    C0 = np.eye(3)

    assert np.allclose(core.math.quat_to_rot(q0), C0)
    assert np.allclose(core.math.rot_to_quat(C0), q0)


def test_quat_rot_arbitrary():
    """Test conversion of arbitrary quaternion to rotation matrix and back."""
    q = np.array([1, 2, 3, 4])
    q = q / np.linalg.norm(q)

    C = core.math.quat_to_rot(q)
    q2 = core.math.rot_to_quat(C)

    assert np.allclose(q, q2)


def test_quat_angle():
    """Test computation of angle from a quaternion."""
    # no rotation
    q0 = np.array([0, 0, 0, 1])
    assert np.isclose(core.math.quat_angle(q0), 0)

    # 90-degree rotation about x-axis
    q1 = np.array([1, 0, 0, 1]) / np.sqrt(2)
    assert np.isclose(core.math.quat_angle(q1), 0.5 * np.pi)


def test_quat_rotate():
    """Test rotating a point by a quaternion."""
    q = np.array([1, 2, 3, 4])
    q = q / np.linalg.norm(q)
    C = core.math.quat_to_rot(q)

    v = np.array([1, 2, 3])

    assert np.allclose(core.math.quat_rotate(q, v), C @ v)


def test_quat_transform():
    """Test rotating and translating a point."""
    q = np.array([1, 2, 3, 4])
    q = q / np.linalg.norm(q)
    C = core.math.quat_to_rot(q)
    r = np.array([-1, -2, -3])

    v = np.array([1, 2, 3])

    assert np.allclose(core.math.quat_transform(r, q, v), C @ v + r)


def test_inset_vertex():
    """Test insetting a vertex toward the origin."""
    v = np.array([1, 1])
    inset = np.sqrt(2) / 2
    v_inset = core.math.inset_vertex(v, inset)
    assert np.allclose(v_inset, [0.5, 0.5])

    # now with a negative number
    v = np.array([-1, 1])
    v_inset = core.math.inset_vertex(v, inset)
    assert np.allclose(v_inset, [-0.5, 0.5])

    # an inset that is too big should raise an exception
    inset = 2
    with pytest.raises(ValueError):
        core.math.inset_vertex(v, inset)


def test_inset_vertex_abs():
    """Test inset of vertex using absolute/taxicab method."""
    v = np.array([1, 1])
    inset = 0.5
    v_inset = core.math.inset_vertex_abs(v, inset)
    assert np.allclose(v_inset, [0.5, 0.5])

    v = np.array([-1, 1])
    v_inset = core.math.inset_vertex_abs(v, inset)
    assert np.allclose(v_inset, [-0.5, 0.5])

    inset = 1.1
    with pytest.raises(ValueError):
        core.math.inset_vertex_abs(v, inset)


def test_plane_span():
    """Test computation of span of a plane."""
    n = np.array([0, 0, 1])
    S = core.math.plane_span(n)
    x = np.array([1, 2, 3])
    assert np.allclose(S @ n, 0)
    assert np.allclose(np.linalg.norm(S @ x), np.linalg.norm(x[:2]))
