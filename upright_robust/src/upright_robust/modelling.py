import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag

import rigeo as rg

import upright_core as core
import upright_robust.utils as utils


def unit_H_min_max(bounding_box, com_box):
    """Compute the element-wise extreme values of the unit H matrix.

    The unit H matrix the second moment matrix H normalized to correspond to
    unit mass.

    Parameters
    ----------
    bounding_box : rg.Box
        The bounding box of the entire object.
    com_box : rg.Box
        The bounding box for the object's center of mass. This box should be
        fully contained within the object's bounding box.

    Returns
    -------
    :
        A tuple ``(H_min, H_max)``, where ``H_min`` is a matrix containing the
        minimum values of the elements of H, and ``H_max`` contains the maximum
        values.
    """
    J = cp.Variable((4, 4), PSD=True)
    H = J[:3, :3]
    h = J[:3, 3]
    m = J[3, 3]

    constraints = (
        [m == 1.0]  # normalized mass
        + com_box.must_contain(points=h, scale=m)
        + bounding_box.moment_sdp_constraints(J)
    )

    H_min = np.zeros((3, 3))
    H_max = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            objective = cp.Maximize(H[i, j])
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.MOSEK)
            H_max[i, j] = objective.value
            H_max[j, i] = objective.value

            objective = cp.Minimize(J[i, j])
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.MOSEK)
            H_min[i, j] = objective.value
            H_min[j, i] = objective.value

    # we know the diagonal must be non-negative, but small numerical errors in
    # the solvers can cause that to fail, which CDD doesn't like, so we
    # manually clamp to zero here
    np.fill_diagonal(H_min, np.maximum(np.diag(H_min), 0))

    return H_min, H_max


def unit_I_min_max(bounding_box, com_box):
    """Compute the element-wise extreme values of the unit inertia (gyration) matrix.

    Parameters
    ----------
    bounding_box : rg.Box
        The bounding box of the entire object.
    com_box : rg.Box
        The bounding box for the object's center of mass. This box should be
        fully contained within the object's bounding box.

    Returns
    -------
    :
        A tuple ``(I_min, I_max)``, where ``I_min`` is a matrix containing the
        minimum values of the elements of I, and ``I_max`` contains the maximum
        values.
    """
    J = cp.Variable((4, 4), PSD=True)
    H = J[:3, :3]
    h = J[:3, 3]
    m = J[3, 3]
    I = cp.trace(H) * np.eye(3) - H

    constraints = (
        [m == 1.0]  # normalized mass
        + com_box.must_contain(points=h, scale=m)
        + bounding_box.moment_sdp_constraints(J)
    )

    I_min = np.zeros((3, 3))
    I_max = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            objective = cp.Maximize(I[i, j])
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.MOSEK)
            I_max[i, j] = objective.value
            I_max[j, i] = objective.value

    # we know the diagonal must be non-negative, but small numerical errors in
    # the solvers can cause that to fail, which CDD doesn't like, so we
    # manually clamp to zero here
    np.fill_diagonal(I_min, np.maximum(np.diag(I_min), 0))

    return I_min, I_max


def unit_vec2_max(bounding_box, com_box):
    """Compute the maximum squared values of the mass-normalized inertial
    parameter vector.

    Note that the resulting values are always positive since they are squares.
    """
    J = cp.Variable((4, 4), PSD=True)
    θ = cp.Variable(10)

    H = J[:3, :3]
    h = J[:3, 3]
    m = J[3, 3]

    I = cp.trace(H) * np.eye(3) - H

    constraints = (
        [m == 1.0, J == rg.pim_must_equal_vec(θ)]
        + com_box.must_contain(points=h, scale=m)
        + bounding_box.moment_sdp_constraints(J)
    )

    θ2_max = np.zeros(10)
    θ2_max[0] = 1.0
    for i in range(1, 10):
        if i == 9:
            objective = cp.Maximize(I[2, 2])
        else:
            objective = cp.Maximize(θ[i])
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        θi_max = objective.value

        objective = cp.Minimize(θ[i])
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        θi_min = objective.value

        θ2_max[i] = max(θi_min**2, θi_max**2)

    return θ2_max


class RobustContactPoint:
    def __init__(self, contact):
        # TODO eventually we can merge this functionality into the base
        # ContactPoint class from upright directly
        self.contact = contact
        self.normal = contact.normal
        self.span = contact.span
        μ = contact.mu

        # grasp matrix: convert contact force into body contact wrench
        self.W1 = np.vstack((np.eye(3), core.math.skew3(contact.r_co_o1)))
        self.W2 = np.vstack((np.eye(3), core.math.skew3(contact.r_co_o2)))

        # matrix to enforce friction cone constraint F @ f >= 0 ==> inside FC
        # (this is the negative of the face form)
        # TODO make negative to be consistent with face form
        # fmt: off
        self.F = np.array([
            [1,  0,  0],
            [μ, -1, -1],
            [μ,  1, -1],
            [μ, -1,  1],
            [μ,  1,  1],
        ]) @ np.vstack((self.normal, self.span))
        # fmt: on

        # span (generator) form matrix FC = {Sz | z >= 0}
        # this is w.r.t. the first object (since the normal points into the
        # first object)
        # fmt: off
        self.S = np.vstack([
            self.normal + μ * self.span[0, :],
            self.normal + μ * self.span[1, :],
            self.normal - μ * self.span[0, :],
            self.normal - μ * self.span[1, :]]).T
        # fmt: on


class ObjectBounds:
    """Bounds on the inertial parameters of a rigid body.

    Parameters
    ----------
    bounding_box : rg.Box
        The bounding box for the entire object.
    com_box : rg.Box
        The box which the CoM is known to lie within.
    approx_inertia : bool
        True to approximate inertia as a fixed value, False to use any
        physically realizable value.
    """

    def __init__(
        self,
        bounding_box,
        com_box=None,
        approx_inertia=False,
    ):
        self.bounding_box = bounding_box
        self.bounding_ellipsoid = bounding_box.mbe()
        self.com_box = com_box
        self.approx_inertia = approx_inertia

    def polytope(self, m, c, I):
        """Build the polytope containing the inertial parameters given the
        nominal values.

        Parameters
        ----------
        m : float
            The nominal mass.
        c : np.ndarray, shape (3,)
            The nominal center of mass.
        I : np.ndarray, shape (3, 3)
            The nominal inertia matrix.

        Returns
        -------
        : tuple
            A tuple (P, p) with matrix P and vector p such that Pθ <= p.
        """
        Ivec = rg.vech(I)

        # mass is always assumed to be exact
        P_m = np.hstack(([[-1], [1]], np.zeros((2, 9))))
        p_m = np.array([-m, m])

        # CoM bounds
        E3 = np.eye(3)
        com_lower = self.com_box.center - self.com_box.half_extents
        com_upper = self.com_box.center + self.com_box.half_extents
        P_h = np.block(
            [
                [com_lower[:, None], -E3, np.zeros((3, 6))],
                [-com_upper[:, None], E3, np.zeros((3, 6))],
            ]
        )
        p_h = np.zeros(6)

        # inertia bounds
        if self.approx_inertia:
            # assume gyration matrix is fixed
            Gvec = Ivec / m  # vectorized gyration matrix

            P_I = np.zeros((12, 10))
            p_I = np.zeros(12)

            # vech(I) == m * Gvec
            P_I[:6, 4:] = -np.eye(6)
            P_I[:6, 0] = Gvec
            P_I[6:, 4:] = np.eye(6)
            P_I[6:, 0] = -Gvec
        else:
            # otherwise we assume no prior knowledge on inertia except object
            # bounding shapes; only constraints required for realizability
            P_I = np.zeros((13, 10))
            p_I = np.zeros(13)

            # using constraints on I is not as tight!
            USE_I_CONSTRAINTS = False
            if USE_I_CONSTRAINTS:
                I_min, I_max = unit_I_min_max(self.bounding_box, self.com_box)

                # H diagonal bounds
                # diag(I) >= diag(I)_min
                P_I[0, :] = [I_min[0, 0], 0, 0, 0, -1, 0, 0, 0, 0, 0]  # -Ixx
                P_I[1, :] = [I_min[1, 1], 0, 0, 0, 0, 0, 0, -1, 0, 0]  # -Iyy
                P_I[2, :] = [I_min[2, 2], 0, 0, 0, 0, 0, 0, 0, 0, -1]  # -Izz

                # diag(I) <= diag(I)_max
                P_I[3, :] = [-I_max[0, 0], 0, 0, 0, 1, 0, 0, 0, 0, 0]  # Ixx
                P_I[4, :] = [-I_max[1, 1], 0, 0, 0, 0, 0, 0, 1, 0, 0]  # Iyy
                P_I[5, :] = [-I_max[2, 2], 0, 0, 0, 0, 0, 0, 0, 0, 1]  # Izz

                # off-diagonal I values
                P_I[6, :] = [I_min[0, 1], 0, 0, 0, 0, -1, 0, 0, 0, 0]  # -Ixy
                P_I[7, :] = [I_min[0, 2], 0, 0, 0, 0, 0, -1, 0, 0, 0]  # -Ixz
                P_I[8, :] = [I_min[1, 2], 0, 0, 0, 0, 0, 0, 0, -1, 0]  # -Iyz

                P_I[9, :] = [-I_max[0, 1], 0, 0, 0, 0, 1, 0, 0, 0, 0]  # Ixy
                P_I[10, :] = [-I_max[0, 2], 0, 0, 0, 0, 0, 1, 0, 0, 0]  # Ixz
                P_I[11, :] = [-I_max[1, 2], 0, 0, 0, 0, 0, 0, 0, 1, 0]  # Iyz
            else:
                H_min, H_max = unit_H_min_max(self.bounding_box, self.com_box)

                # H diagonal bounds
                # diag(H) >= diag(H)_min
                P_I[0, :] = [H_min[0, 0], 0, 0, 0, 0.5, 0, 0, -0.5, 0, -0.5]  # -Hxx
                P_I[1, :] = [H_min[1, 1], 0, 0, 0, -0.5, 0, 0, 0.5, 0, -0.5]  # -Hyy
                P_I[2, :] = [H_min[2, 2], 0, 0, 0, -0.5, 0, 0, -0.5, 0, 0.5]  # -Hzz

                # diag(H) <= diag(H)_max
                P_I[3, :] = [-H_max[0, 0], 0, 0, 0, -0.5, 0, 0, 0.5, 0, 0.5]  # Hxx
                P_I[4, :] = [-H_max[1, 1], 0, 0, 0, 0.5, 0, 0, -0.5, 0, 0.5]  # Hyy
                P_I[5, :] = [-H_max[2, 2], 0, 0, 0, 0.5, 0, 0, 0.5, 0, -0.5]  # Hzz

                # off-diagonal H values
                P_I[6, :] = [H_min[0, 1], 0, 0, 0, 0, 1, 0, 0, 0, 0]  # -Hxy
                P_I[7, :] = [H_min[0, 2], 0, 0, 0, 0, 0, 1, 0, 0, 0]  # -Hxz
                P_I[8, :] = [H_min[1, 2], 0, 0, 0, 0, 0, 0, 0, 1, 0]  # -Hyz

                P_I[9, :] = [-H_max[0, 1], 0, 0, 0, 0, -1, 0, 0, 0, 0]  # Hxy
                P_I[10, :] = [-H_max[0, 2], 0, 0, 0, 0, 0, -1, 0, 0, 0]  # Hxz
                P_I[11, :] = [-H_max[1, 2], 0, 0, 0, 0, 0, 0, 0, -1, 0]  # Hyz

            # ellipsoid density realizability
            # tr(ΠQ) >= 0
            ell = self.bounding_ellipsoid
            P_I[12, :] = [-np.trace(A @ ell.Q) for A in rg.pim_sum_vec_matrices()]

        # combined bounds
        P = np.vstack((P_m, P_h, P_I))
        p = np.concatenate((p_m, p_h, p_I))
        return P, p

    def unit_vec2_max(self, m, c, I):
        return unit_vec2_max(self.bounding_box, self.com_box)


class UncertainObject:
    def __init__(self, body, bounds=None):
        self.body = body

        # inertial quantities are taken w.r.t./about the EE origin (i.e.,
        # self.body.com is the vector from the EE origin to the CoM of this
        # object)
        m = self.body.mass
        c = self.body.com

        Sc = core.math.skew3(c)
        I = self.body.inertia - m * Sc @ Sc

        # spatial mass matrix
        # fmt: off
        self.M = np.block([
            [m * np.eye(3), -m * Sc],
            [m * Sc, I]
        ])
        # fmt: on

        # polytopic parameter uncertainty: Pθ <= p
        if bounds is not None:
            self.P, self.p = bounds.polytope(m, c, I)
            self.bounding_box = bounds.bounding_box
            self.com_box = bounds.com_box

            # use a fixed inertia value for approx_inertia case
            self.unit_vec2_max = bounds.unit_vec2_max(m, c, I)
            if bounds.approx_inertia:
                self.unit_vec2_max[-6:] = rg.vech(I)

    def bias(self, V):
        """Compute Coriolis and centrifugal terms."""
        return rg.skew6(V) @ self.M @ V

    def wrench(self, A, V):
        """Compute the body-frame inertial wrench."""
        return self.M @ A + self.bias(V)


def compute_object_name_index(names):
    """Compute mapping from object names to indices."""
    return {name: idx for idx, name in enumerate(names)}


# TODO rename to compute_grasp_matrix
def compute_contact_force_to_wrench_map(name_index, contacts):
    """Compute mapping W from contact forces (f_1, ..., f_nc) to object wrenches
    (w_1, ..., w_no)"""
    no = len(name_index)
    nc = len(contacts)
    W = np.zeros((no * 6, nc * 3))
    for i, c in enumerate(contacts):
        # ignore EE object
        if c.contact.object1_name != "ee":
            r = name_index[c.contact.object1_name]
            W[r * 6 : (r + 1) * 6, i * 3 : (i + 1) * 3] = c.W1

        # second object is negative because a positive force on the first
        # object = negative force on the second object
        r = name_index[c.contact.object2_name]
        W[r * 6 : (r + 1) * 6, i * 3 : (i + 1) * 3] = -c.W2
    return W


def compute_cwc_span_form(name_index, contacts):
    no = len(name_index)
    nc = len(contacts)
    H = np.zeros((no * 6, nc * 4))
    for i, c in enumerate(contacts):
        # ignore EE object
        if c.contact.object1_name != "ee":
            r = name_index[c.contact.object1_name]
            H[r * 6 : (r + 1) * 6, i * 4 : (i + 1) * 4] = c.W1 @ c.S

        # second object is negative because a positive force on the first
        # object = negative force on the second object
        r = name_index[c.contact.object2_name]
        H[r * 6 : (r + 1) * 6, i * 4 : (i + 1) * 4] = -c.W2 @ c.S
    return H


def compute_cwc_face_form(name_index, contacts):
    """Build the (face form of the) contact wrench cone from contact points of an object."""

    # # compute mapping W from contact forces (f_1, ..., f_nc) to object wrenches
    # # (w_1, ..., w_no)
    # W = compute_contact_force_to_wrench_map(name_index, contacts)
    #
    # # computing mapping from face form of contact forces to span form
    # # TODO this should use the name index
    # S = block_diag(*[c.S for c in contacts])
    #
    # # span form of the CWC
    # H = W @ S

    H = compute_cwc_span_form(name_index, contacts)

    # convert the whole contact wrench cone to face form
    A = utils.cone_span_to_face_form(H)

    # Aw <= 0 implies there exist feasible contact forces to support wrench w
    return A


def compute_P_tilde_matrix(P, p):
    return np.block([[P, p[:, None]], [np.zeros((1, P.shape[1])), np.array([[1]])]])
