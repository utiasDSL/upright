import numpy as np
from scipy.linalg import block_diag

import upright_core as core

import rigeo as rg


def skew6_matrices():
    """Compute list of matrices S such that skew6(V) == sum(S[i] * V[i])"""
    S = []
    for i in range(6):
        E = np.zeros(6)
        E[i] = 1
        S.append(rg.skew6(E))
    return S


def lift6_matrices():
    """Compute list of matrices L such that lift6(V) == sum(L[i] * V[i])"""
    # TODO make a global constant?
    L = []
    for i in range(6):
        E = np.zeros(6)
        E[i] = 1
        L.append(rg.lift6(E))
    return L


# TODO: unused, remove eventually
# def pim_trace_vec_matrices():
#     """Generate the matrices A_i such that tr(A_i @ J) == θ_i"""
#     As = [np.zeros((4, 4)) for _ in range(10)]
#     As[0][3, 3] = 1  # mass
#
#     As[1][0, 3] = 1  # hx
#     As[2][1, 3] = 1  # hy
#     As[3][2, 3] = 1  # hz
#
#     # Ixx
#     As[4][1, 1] = 1
#     As[4][2, 2] = 1
#
#     As[5][0, 1] = -1  # Ixy
#     As[6][0, 2] = -1  # Ixz
#
#     # Iyy
#     As[7][0, 0] = 1
#     As[7][2, 2] = 1
#
#     As[8][1, 2] = -1  # Iyz
#
#     # Izz
#     As[9][0, 0] = 1
#     As[9][1, 1] = 1
#
#     return As


def body_gravito_inertial_wrench(C, V, A, obj):
    """Gravito-inertial wrench in the body frame.

    The supplied velocity twist V and acceleration A must also be in the body
    frame.
    """
    # G = np.array([0, 0, -9.81])  # TODO
    # Ag = np.concatenate((C @ G, np.zeros(3)))
    G = body_gravity6(C)
    return obj.M @ (A - G) + rg.skew6(V) @ obj.M @ V


def body_contact_wrench(forces, contacts):
    """Contact wrench in the body frame.

    forces is an (n, 3) array of contact forces
    contacts is the list of contact points
    """
    return np.sum([c.W @ f for c, f in zip(contacts, forces)], axis=0)


def friction_cone_constraints(forces, contacts):
    """Constraints are non-negative if all contact forces are inside their friction cones."""
    return np.concatenate([c.F @ f for c, f in zip(contacts, forces)])


# def body_regressor(V, A):
#     """Compute regressor matrix Y given body frame velocity V and acceleration A.
#
#     The regressor maps the inertial parameters to the body inertial wrench: w = Yθ.
#     """
#     return rg.RigidBody.regressor(V, A)


def body_regressor_A_by_vector(f):
    """Compute a matrix D such that d + D @ A == Y.T @ f for some vector f.

    A: body acceleration twist
    Y: body regressor
    """
    n = f.shape[0] // 6
    I = np.eye(n)
    return np.vstack([np.kron(I, Y).T @ f for Y in lift6_matrices()]).T


def body_regressor_VG_by_vector(V, G, f):
    """Compute a vector d such that d + D @ A == Y.T @ f for some vector f.

    V: body velocity twist
    A: body acceleration twist
    G: body gravity twist

    The vector d contains velocity and gravity components.
    """
    n = f.shape[0] // 6
    I = np.eye(n)
    Y0 = rg.RigidBody.regressor(V, -G)
    return np.kron(I, Y0).T @ f


def body_regressor_VG_by_vector_vectorized(V, G, F):
    """A vectorized version of the above function, computed for each row f of F.

    V: body velocity twist
    G: body gravity twist
    F: arbitrary matrix with n * 6 columns

    Returns a matrix M with each column corresponding to the vector d from the
    above function for each row of F.
    """
    n = F.shape[1] // 6  # number of wrenches
    # I = np.eye(n)
    Y0 = rg.RigidBody.regressor(V, -G)

    # some alternative approaches here:
    # M = np.kron(I, Y0).T @ F.T
    M = block_diag(*[Y0.T] * n) @ F.T
    return M


def body_regressor_VG_by_vector_tilde_vectorized(V, G, F):
    """Same as the above function but the "tilde" means that each vector d ends
    with an extra zero (for the robust formulation)."""
    M = body_regressor_VG_by_vector_vectorized(V, G, F)
    # TODO this copy is likely expensive!
    return np.vstack((M, np.zeros((1, M.shape[1]))))


def body_gravity3(C_ew, g=9.81):
    """Compute body acceleration vector."""
    return C_ew @ [0, 0, -g]


def body_gravity6(C_ew, g=9.81):
    """Compute body acceleration twist."""
    return np.concatenate((body_gravity3(C_ew, g), np.zeros(3)))


def cone_span_to_face_form(S):
    """Convert the span form of a polyhedral convex cone to face form.

    Span form is { Sz | z  >= 0 }
    Face form is { x  | Ax <= 0 }

    Return A of the face form.
    """
    ff = rg.SpanForm(rays=S.T).to_face_form()
    assert np.allclose(ff.b, 0)
    return ff.A


def cwc(contacts):
    """Build the (face form of the) contact wrench cone from contact points of an object."""
    # combine span form of each contact wrench cone to get the overall CWC in
    # span form
    S = np.hstack([c.W @ c.S for c in contacts])

    # convert to face form
    # Aw <= 0 implies there exist feasible contact forces to support wrench w
    A = cone_span_to_face_form(S)
    return A


# def body_regressor_by_vector_matrix(C, V, z):
#     """Compute a matrix D such that d0 + D @ A == Y.T @ z for some vector z."""
#     Y0, Ys = body_regressor_components(C, V)
#     d0 = Y0.T @ z
#     D = np.vstack([Y.T @ z for Y in Ys]).T
#     return d0, D


def body_regressor_by_vector_acceleration_matrix(x):
    """Compute a matrix D such that x.T @ Y(0, A) == A.T @ D for some vector x."""
    Ls = lift6_matrices()
    return np.array([x.T @ L for L in Ls])


def body_regressor_by_vector_velocity_matrix(x):
    """Compute a matrix D such that x.T @ Y(V, 0) == z.T @ D for some vector x,
    where z = vec(V @ V.T)."""
    S = skew6_matrices()
    L = lift6_matrices()
    As = []

    for i in range(6):
        for j in range(6):
            As.append(S[i] @ L[j])

    # matrix with rows of f.T * A[i]
    # this is the linear representation required for the optimization problem
    return np.array([x.T @ A for A in As])


def body_regressor_by_vector_velocity_matrix_half(x):
    """Compute a matrix D such that x.T @ Y(V, 0) == z.T @ D for some vector x,
    where z = vec(V @ V.T)."""
    S = skew6_matrices()
    L = lift6_matrices()
    As = []

    for i in range(6):
        for j in range(i, 6):
            A = S[i] @ L[j]
            if i != j:
                A += S[j] @ L[i]
            As.append(A)

    idx = [4, 5, 8, 10, 12, 13, 15, 16, 17, 18, 19, 20]

    # matrix with rows of f.T * A[i]
    # this is the linear representation required for the optimization problem
    return np.array([x.T @ As[i] for i in idx])
