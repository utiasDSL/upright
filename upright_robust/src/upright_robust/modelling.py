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
        + bounding_box.must_realize(J)
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
        + bounding_box.must_realize(J)
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

        # matrix to convert contact force into body contact wrench
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
    """Bounds on the inertial parameters of a rigid body."""

    def __init__(
        self,
        mass_lower=None,
        mass_upper=None,
        com_lower=None,
        com_upper=None,
        ellipsoid_half_extents=None,
        box_half_extents=None,
        centroid=None,
        approx_inertia=False,
    ):
        """All quantities are *relative to the nominal value*.

        For example, we want:
            mass_actual >= mass_nominal + mass_lower
            mass_actual <= mass_nominal + mass_upper
        """
        if mass_lower is None:
            mass_lower = 0
        self.mass_lower = mass_lower
        assert self.mass_lower <= 0

        if mass_upper is None:
            mass_upper = 0
        self.mass_upper = mass_upper
        assert self.mass_upper >= 0

        if com_lower is None:
            com_lower = np.zeros(3)
        self.com_lower = np.array(com_lower)

        if com_upper is None:
            com_upper = np.zeros(3)
        self.com_upper = np.array(com_upper)

        # ellipsoid_half_extents can be None
        self.ellipsoid_half_extents = ellipsoid_half_extents
        self.box_half_extents = box_half_extents
        self.centroid = centroid

        self.approx_inertia = approx_inertia

    @classmethod
    def from_config(cls, config, approx_inertia=False):
        mass_lower = config.get("mass_lower", None)
        mass_upper = config.get("mass_upper", None)
        com_lower = config.get("com_lower", None)
        com_upper = config.get("com_upper", None)

        centroid = np.array(config["centroid"])  # required
        if "ellipsoid_half_extents" in config:
            ell_half_extents = np.array(config["ellipsoid_half_extents"])
            assert ell_half_extents.shape == (3,)
        else:
            ell_half_extents = None
        if "box_half_extents" in config:
            box_half_extents = np.array(config["box_half_extents"])
            assert box_half_extents.shape == (3,)
        else:
            box_half_extents = None

        return cls(
            mass_lower=mass_lower,
            mass_upper=mass_upper,
            com_lower=com_lower,
            com_upper=com_upper,
            ellipsoid_half_extents=ell_half_extents,
            box_half_extents=box_half_extents,
            centroid=centroid,
            approx_inertia=approx_inertia,
        )

    def bounding_box(self, c):
        return rg.Box(half_extents=self.box_half_extents, center=self.centroid)

    def bounding_ellipsoid(self):
        return rg.Ellipsoid(
            half_extents=self.ellipsoid_half_extents, center=self.centroid
        )

    def com_box(self, c):
        return rg.Box.from_two_vertices(c + self.com_lower, c + self.com_upper)

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

        # mass bounds
        # TODO can we get rid of these?
        P_m = np.hstack(([[-1], [1]], np.zeros((2, 9))))
        p_m = np.array([-(m + self.mass_lower), m + self.mass_upper])

        # CoM bounds
        E3 = np.eye(3)
        P_h = np.block(
            [
                [(c + self.com_lower)[:, None], -E3, np.zeros((3, 6))],
                [-(c + self.com_upper)[:, None], E3, np.zeros((3, 6))],
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

            H_min, H_max = unit_H_min_max(self.bounding_box(c), self.com_box(c))

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
            ell = self.bounding_ellipsoid()
            P_I[12, :] = [-np.trace(A @ ell.Q) for A in rg.pim_sum_vec_matrices()]

        # combined bounds
        P = np.vstack((P_m, P_h, P_I))
        p = np.concatenate((p_m, p_h, p_I))
        return P, p

    def unit_vec2_max(self, m, c, I):
        return unit_vec2_max(self.bounding_box(c), self.com_box(c))


# TODO in principle this could be a composition of an rg.RigidBody and the
# inertial parameter bounds?
class UncertainObject:
    def __init__(self, obj, bounds=None, compute_bounds=True):
        self.object = obj
        self.body = obj.body

        # inertial quantities are taken w.r.t./about the EE origin (i.e.,
        # self.body.com is the vector from the EE origin to the CoM of this
        # object)
        m = self.body.mass
        c = self.body.com

        Sc = core.math.skew3(c)
        I = obj.body.inertia - m * Sc @ Sc

        # spatial mass matrix
        # fmt: off
        self.M = np.block([
            [m * np.eye(3), -m * Sc],
            [m * Sc, I]
        ])
        # fmt: on

        # polytopic parameter uncertainty: Pθ <= p
        if compute_bounds:
            self.P, self.p = bounds.polytope(m, c, I)

            self.mass_min = m + bounds.mass_lower
            self.mass_max = m + bounds.mass_upper
            self.bounding_box = bounds.bounding_box(c)
            self.com_box = bounds.com_box(c)

            # use a fixed inertia value for approx_inertia case
            self.unit_vec2_max = bounds.unit_vec2_max(m, c, I)
            if bounds.approx_inertia:
                self.unit_vec2_max[-6:] = rg.vech(I)

            # nominal inertial parameter vector
            # NOTE: unused, previously part of primal relaxation approach
            # if bounds.approx_inertia:
            #     self.θ_nom = np.concatenate(([m], m * c, rg.vech(I)))
            # else:
            #     self.θ_nom = np.concatenate(([m], m * c, rg.vech(-m * Sc @ Sc)))

    def bias(self, V):
        """Compute Coriolis and centrifugal terms."""
        return rg.skew6(V) @ self.M @ V

    def wrench(self, A, V):
        """Compute the body-frame inertial wrench."""
        return self.M @ A + self.bias(V)

    # def inertial_com_wrench(self, C_we, A_ew_w, V_ew_w):
    #     # NOTE gravity needs to be included in A_ew_w
    #     m = self.body.mass
    #     c = self.body.com
    #     I = self.body.inertia
    #
    #     α_ew_e = C_we.T @ A_ew_w[3:]
    #     ω_ew_e = C_we.T @ V_ew_w[3:]
    #     Sα = core.math.skew3(A_ew_w[3:])
    #     Sω = core.math.skew3(V_ew_w[3:])
    #     ddC_we = (Sα + Sω @ Sω) @ C_we
    #     f = m * C_we.T @ (A_ew_w[:3] + ddC_we @ c)
    #     τ = I @ α_ew_e + np.cross(ω_ew_e, I @ ω_ew_e)
    #     return np.concatenate((f, τ))


def compute_object_name_index(names):
    """Compute mapping from object names to indices."""
    return {name: idx for idx, name in enumerate(names)}


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

        # second object is negative
        r = name_index[c.contact.object2_name]
        W[r * 6 : (r + 1) * 6, i * 3 : (i + 1) * 3] = -c.W2
    return W


def compute_cwc_face_form(name_index, contacts):
    """Build the (face form of the) contact wrench cone from contact points of an object."""

    # compute mapping W from contact forces (f_1, ..., f_nc) to object wrenches
    # (w_1, ..., w_no)
    W = compute_contact_force_to_wrench_map(name_index, contacts)

    # computing mapping from face form of contact forces to span form
    S = block_diag(*[c.S for c in contacts])

    # span form of the CWC
    H = W @ S

    # convert the whole contact wrench cone to face form
    A = utils.cone_span_to_face_form(H)

    # Aw <= 0 implies there exist feasible contact forces to support wrench w
    return A


def compute_P_tilde_matrix(P, p):
    return np.block([[P, p[:, None]], [np.zeros((1, P.shape[1])), np.array([[1]])]])
