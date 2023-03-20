import numpy as np
import cdd

import upright_core as core


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
    L = []
    for i in range(6):
        E = np.zeros(6)
        E[i] = 1
        L.append(lift6(E))
    return L


def vech(J):
    """Half-vectorize the inertia matrix"""
    return np.array([J[0, 0], J[0, 1], J[0, 2], J[1, 1], J[1, 2], J[2, 2]])


class ContactPoint:
    def __init__(self, position, normal, μ):
        self.position = np.array(position)
        self.normal = np.array(normal)
        self.span = core.math.plane_span(self.normal)
        self.μ = μ

        # matrix to convert contact force into body contact wrench
        self.W = np.vstack((np.eye(3), core.math.skew3(position)))

        # matrix to enforce friction cone constraint F @ f >= 0 ==> inside FC
        # (this is the negative of the face form)
        # fmt: off
        P = np.vstack((self.normal, self.span))
        self.F = np.array([
            [1,  0,  0],
            [μ, -1, -1],
            [μ,  1, -1],
            [μ, -1,  1],
            [μ,  1,  1],
        ]) @ P
        # fmt: on

        # span (generator) form matrix FC = {Sz | z >= 0}
        # fmt: off
        self.S = P @ np.array([
            [1,  1, 1,  1],
            [μ, -μ, 0,  0],
            [0,  0, μ, -μ]
        ])
        # fmt: on


class BalancedObject:
    def __init__(self, m, h, δ, μ, h0=0, x0=0):
        self.m = m
        self.h = h  # height of CoM above base of object
        self.δ = δ
        self.μ = μ

        side_lengths = 2 * np.array([δ, δ, h])
        self.J = core.math.cuboid_inertia_matrix(m, side_lengths)

        self.h0 = h0  # height of base above tray
        self.x0 = x0

        self.origin = np.array([self.x0, 0, self.h + self.h0])
        S = core.math.skew3(self.origin)

        # mass matrix
        self.M = np.block([[m * np.eye(3), -m * S], [m * S, self.J]])

        # polytopic constraints on the inertial parameters
        # Pθ >= p
        Jvec = vech(self.J)
        self.θ = np.concatenate(([m], m * self.origin, Jvec))
        Δθ = np.concatenate(([0.1, 0.01 * δ, 0.01 * δ, 0.01 * h], 0 * Jvec))
        θ_min = self.θ - Δθ
        θ_max = self.θ + Δθ
        self.P = np.vstack((np.eye(self.θ.shape[0]), -np.eye(self.θ.shape[0])))
        self.p = np.concatenate((θ_min, -θ_max))

    def contacts(self):
        # contacts are in the body frame w.r.t. to the origin
        C1 = ContactPoint(
            position=[-self.δ, -self.δ, -self.h], normal=[0, 0, 1], μ=self.μ
        )
        C2 = ContactPoint(
            position=[self.δ, -self.δ, -self.h], normal=[0, 0, 1], μ=self.μ
        )
        C3 = ContactPoint(
            position=[self.δ, self.δ, -self.h], normal=[0, 0, 1], μ=self.μ
        )
        C4 = ContactPoint(
            position=[-self.δ, self.δ, -self.h], normal=[0, 0, 1], μ=self.μ
        )
        return [C1, C2, C3, C4]


def body_gravito_inertial_wrench(C, V, A, obj):
    """Gravito-inertial wrench in the body frame.

    The supplied velocity twist V and acceleration A must also be in the body
    frame.
    """
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
    v, ω = V[:3], V[3:]
    a, α = A[:3], A[3:]

    # account for gravity
    # a = a - C @ G

    Sω = core.math.skew3(ω)
    Sv = core.math.skew3(v)
    Sa = core.math.skew3(a)
    Sα = core.math.skew3(α)
    Lω = lift3(ω)
    Lα = lift3(α)

    # fmt: off
    return np.block([
        [(a + Sω @ v)[:, None], Sα + Sω @ Sω, np.zeros((3, 6))],
        [np.zeros((3, 1)), -Sa - core.math.skew3(Sω @ v), Lα + Sω @ Lω]
    ])
    # fmt: on


def body_regressor_acceleration_matrices():
    # TODO I think these are just the lift6_matrices
    Ys = []
    V = np.zeros(6)
    for i in range(6):
        A = np.zeros(6)
        A[i] = 1
        Ys.append(body_regressor(V, A))
    return Ys


def body_regressor_components(C, V):
    """Compute components {Yi} of the regressor matrix Y such that
    Y = sum(Yi * Ai forall i)
    """
    # velocity + gravity component
    Ag = np.concatenate((C @ G, np.zeros(3)))
    Y0 = body_regressor(V, -Ag)

    # acceleration component
    Ys = body_regressor_acceleration_matrices()
    return Y0, Ys


def body_regressor_by_vector_matrix(C, V, z):
    """Compute a matrix D such that d0 + D @ A == Y.T @ z for some vector z."""
    Y0, Ys = body_regressor_components(C, V)
    d0 = Y0.T @ z
    D = np.vstack([Y.T @ z for Y in Ys]).T
    return d0, D


def span_to_face_form(S):
    """Convert the span form of a polyhedral cone to face form.

    Span form is { Sz | z  >= 0 }
    Face form is { x  | Fx >= 0 }
    """
    # span form
    # we have generators as columns but cdd wants it as rows, hence the transpose
    Smat = cdd.Matrix(np.hstack((np.zeros((S.shape[1], 1)), S.T)))
    Smat.rep_type = cdd.RepType.GENERATOR

    # polyhedron
    poly = cdd.Polyhedron(Smat)

    # face form: Ax <= b
    Fmat = poly.get_inequalities()
    F = np.array([Fmat[i] for i in range(Fmat.row_size)])
    b = F[:, 0]
    A = -F[:, 1:]
    return A, b


def cwc(contacts):
    """Build the (face form of the) contact wrench cone from contact points of an object."""
    # combine span form of each contact wrench cone to get the overall CWC in
    # span form
    S = np.hstack([c.W @ c.S for c in contacts])

    # convert to face form
    A, b = span_to_face_form(S)
    assert np.allclose(b, 0)

    # Fw >= 0 ==> there exist feasible contact forces to support wrench w
    return A



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
