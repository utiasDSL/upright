import numpy as np


def right_triangular_prism_vertices_normals(half_extents):
    hx, hy, hz = half_extents

    # fmt: off
    vertices = [[-hx, -hy, -hz], [hx, -hy, -hz], [hx, -hy, hz],
                [-hx,  hy, -hz], [hx,  hy, -hz], [hx,  hy, hz]]
    # fmt: on

    # compute normal of the non-axis-aligned face
    e12 = vertices[2, :] - vertices[1, :]
    e14 = vertices[4, :] - vertices[1, :]
    n = np.cross(e12, e14)
    n = n / np.linalg.norm(n)

    normals = np.vstack((np.eye(3), n))

    return vertices, normals


def right_triangular_prism_inertia_normalized(half_extents):
    hx, hy, hz = half_extents

    # fmt: off
    J = np.array([
        [4*hx*hy*hz*(3*hy**2 + 2*hz**2)/9,                            0,               4*hx**2*hy*hz**2/9],
        [                               0, 8*hx*hy*hz*(hx**2 + hz**2)/9,                                0],
        [              4*hx**2*hy*hz**2/9,                            0, 4*hx*hy*hz*(2*hx**2 + 3*hy**2)/9]])
    # fmt: on

    d, C = np.linalg.eig(J)
    D = np.diag(d)

    # J = C @ D @ C.T
    # D is the (diagonal) inertia tensor in the local inertial frame
    # C is the orientation of the inertial frame w.r.t. the object frame
    return D, C
