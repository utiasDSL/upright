import numpy as np


def right_triangular_prism_vertices_normals(half_extents):
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

    normals = np.vstack((np.eye(3), n))

    return vertices, normals


def right_triangular_prism_inertia_normalized(half_extents):
    hx, hy, hz = half_extents

    # computed using sympy script right_triangular_prism_inertia.py
    # fmt: off
    J = np.array([
        [hy**2/3 + 2*hz**2/9,                     0,             hx*hz/9],
        [                  0, 2*hx**2/9 + 2*hz**2/9,                   0],
        [            hx*hz/9,                     0, 2*hx**2/9 + hy**2/3]])
    # fmt: on

    d, C = np.linalg.eig(J)
    D = np.diag(d)

    # J = C @ D @ C.T
    # D is the (diagonal) inertia tensor in the local inertial frame
    # C is the orientation of the inertial frame w.r.t. the object frame
    return D, C
