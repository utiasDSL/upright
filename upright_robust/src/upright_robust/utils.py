import numpy as np
import cdd
from scipy.spatial import ConvexHull
from scipy.linalg import block_diag

import upright_core as core

import rigeo as rg


def lift3(x):
    """Lift a 3-vector x such that A @ x = lift(x) @ vech(A) for symmetric A."""
    # fmt: off
    return np.array([
        [x[0], x[1], x[2], 0, 0, 0],
        [0, x[0], 0, x[1], x[2], 0],
        [0, 0, x[0], 0, x[1], x[2]]
    ])
    # fmt: on


def skew6(V):
    """6D cross product matrix"""
    v, ω = V[:3], V[3:]
    Sv = core.math.skew3(v)
    Sω = core.math.skew3(ω)
    return np.block([[Sω, np.zeros((3, 3))], [Sv, Sω]])


def skew6_matrices():
    """Compute list of matrices S such that skew6(V) = Σ S[i] * V[i]"""
    S = []
    for i in range(6):
        E = np.zeros(6)
        E[i] = 1
        S.append(skew6(E))
    return S


def lift6(x):
    """Lift a 6-vector V such that A @ V = lift(V) @ vech(A) for symmetric A."""
    a = x[:3]
    b = x[3:]
    # fmt: off
    return np.block([
        [a[:, None], core.math.skew3(b), np.zeros((3, 6))],
        [np.zeros((3, 1)), -core.math.skew3(a), lift3(b)]])
    # fmt: on


def lift6_matrices():
    """Compute list of matrices L such that lift6(V) = Σ L[i] * V[i]"""
    # TODO make a global constant?
    L = []
    for i in range(6):
        E = np.zeros(6)
        E[i] = 1
        L.append(lift6(E))
    return L


def vech(J):
    """Half-vectorize the inertia matrix"""
    return np.array([J[0, 0], J[0, 1], J[0, 2], J[1, 1], J[1, 2], J[2, 2]])


# def J_vec_constraint(J, θ, eps=1e-4):
#     """Constraint to enforce consistency between J and θ representations."""
#     H = J[:3, :3]
#     I = cp.trace(H) * np.eye(3) - H
#     return [
#         J >> eps * np.eye(4),
#         J[3, 3] == θ[0],
#         J[:3, 3] == θ[1:4],
#         I[0, :3] == θ[4:7],
#         I[1, 1:3] == θ[7:9],
#         I[2, 2] == θ[9],
#     ]


def pim_trace_vec_matrices():
    """Generate the matrices A_i such that tr(A_i @ J) == θ_i"""
    As = [np.zeros((4, 4)) for _ in range(10)]
    As[0][3, 3] = 1  # mass

    As[1][0, 3] = 1  # hx
    As[2][1, 3] = 1  # hy
    As[3][2, 3] = 1  # hz

    # Ixx
    As[4][1, 1] = 1
    As[4][2, 2] = 1

    As[5][0, 1] = -1  # Ixy
    As[6][0, 2] = -1  # Ixz

    # Iyy
    As[7][0, 0] = 1
    As[7][2, 2] = 1

    As[8][1, 2] = -1  # Iyz

    # Izz
    As[9][0, 0] = 1
    As[9][1, 1] = 1

    return As


def pim_sum_vec_matrices():
    """Generate the matrices A_i such that J == sum(A_i * θ_i)"""
    As = [np.zeros((4, 4)) for _ in range(10)]
    As[0][3, 3] = 1  # mass

    # hx
    As[1][0, 3] = 1
    As[1][3, 0] = 1

    # hy
    As[2][1, 3] = 1
    As[2][3, 1] = 1

    # hz
    As[3][2, 3] = 1
    As[3][3, 2] = 1

    # Ixx
    As[4][0, 0] = -0.5
    As[4][1, 1] = 0.5
    As[4][2, 2] = 0.5

    # Ixy
    As[5][0, 1] = -1
    As[5][1, 0] = -1

    # Ixz
    As[6][0, 2] = -1
    As[6][2, 0] = -1

    # Iyy
    As[7][0, 0] = 0.5
    As[7][1, 1] = -0.5
    As[7][2, 2] = 0.5

    # Iyz
    As[8][1, 2] = -1
    As[8][2, 1] = -1

    # Izz
    As[9][0, 0] = 0.5
    As[9][1, 1] = 0.5
    As[9][2, 2] = -0.5

    return As


def body_gravito_inertial_wrench(C, V, A, obj):
    """Gravito-inertial wrench in the body frame.

    The supplied velocity twist V and acceleration A must also be in the body
    frame.
    """
    G = np.array([0, 0, -9.81])  # TODO
    Ag = np.concatenate((C @ G, np.zeros(3)))
    return obj.M @ (A - Ag) + skew6(V) @ obj.M @ V


def body_contact_wrench(forces, contacts):
    """Contact wrench in the body frame.

    forces is an (n, 3) array of contact forces
    contacts is the list of contact points
    """
    return np.sum([c.W @ f for c, f in zip(contacts, forces)], axis=0)


def friction_cone_constraints(forces, contacts):
    """Constraints are non-negative if all contact forces are inside their friction cones."""
    return np.concatenate([c.F @ f for c, f in zip(contacts, forces)])


def body_regressor(V, A):
    """Compute regressor matrix Y given body frame velocity V and acceleration A.

    The regressor maps the inertial parameters to the body inertial wrench: w = Yθ.
    """
    return lift6(A) + skew6(V) @ lift6(V)


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
    Y0 = body_regressor(V, -G)
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
    Y0 = body_regressor(V, -G)

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


def _span_to_face_form_qhull(S):
    points = np.vstack((np.zeros((1, S.shape[0])), S.T))
    hull = ConvexHull(points)
    mask = np.any(hull.simplices == 0, axis=1)
    F = hull.equations[mask, :]
    G = [F[0, :]]
    for i in range(1, F.shape[0]):
        if np.allclose(F[i, :], G[-1]):
            continue
        G.append(F[i, :])
    G = np.array(G)

    A = G[:, :-1]
    b = G[:, -1]
    return A, b


def _span_to_face_form_cdd(S):
    # span form
    # we have generators as columns but cdd wants it as rows, hence the transpose
    Smat = cdd.Matrix(np.hstack((np.zeros((S.shape[1], 1)), S.T)))
    Smat.rep_type = cdd.RepType.GENERATOR

    # polyhedron
    poly = cdd.Polyhedron(Smat)

    # general face form is Ax <= b, which cdd stores as one matrix [b -A]
    Fmat = poly.get_inequalities()

    face_form = rg.FaceForm.from_cdd_matrix(Fmat)
    return face_form.A, face_form.b

    # # ensure no equality constraints: we are only setup to handle inequalities
    # # at the moment
    # assert len(Fmat.lin_set) == 0
    #
    # F = np.array([Fmat[i] for i in range(Fmat.row_size)])
    # b = F[:, 0]
    # A = -F[:, 1:]
    # return A, b


def span_to_face_form(S, library="cdd"):
    """Convert the span form of a polyhedral convex cone to face form.

    Span form is { Sz | z  >= 0 }
    Face form is { x  | Ax <= 0 }

    Return A of the face form.
    """
    if library == "cdd":
        A, b = _span_to_face_form_cdd(S)
    elif library == "qhull":
        A, b = _span_to_face_form_qhull(S)
    else:
        raise ValueError(f"library much be either 'cdd' or 'qhull', not {library}")

    # for cones b should be zero
    assert np.allclose(b, 0)

    return A


def cwc(contacts):
    """Build the (face form of the) contact wrench cone from contact points of an object."""
    # combine span form of each contact wrench cone to get the overall CWC in
    # span form
    S = np.hstack([c.W @ c.S for c in contacts])

    # convert to face form
    # Aw <= 0 implies there exist feasible contact forces to support wrench w
    A = span_to_face_form(S)
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


class RunningAverage:
    def __init__(self, size=None):
        self.max = -np.inf
        self.count = 0
        if size is None:
            self.average = 0
        else:
            self.average = np.zeros(size)

    def update(self, value):
        self.average = (self.count * self.average + value) / (1 + self.count)
        self.count += 1
        self.max = max(self.max, value)
