import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag

# TODO probably don't want this dependency at the moment
import inertial_params as ip

import upright_core as core
import upright_robust.utils as utils


def H_min_max(bounding_box, com_box, mass_min, mass_max):
    Es = bounding_box.as_ellipsoidal_intersection()

    J = cp.Variable((4, 4), PSD=True)
    h = J[:3, 3]
    m = J[3, 3]
    constraints = (
        [
            m >= mass_min,
            m <= mass_max,
        ]
        + com_box.must_contain(points=h, scale=m)
        + bounding_box.must_realize(J)
        # + [cp.trace(E.Q @ J) >= 0 for E in Es]
    )

    H_min = np.zeros((3, 3))
    H_max = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            objective = cp.Maximize(J[i, j])
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
    # the solvers can cause that to fail, which CDD doesn't like
    np.fill_diagonal(H_min, np.maximum(np.diag(H_min), 0))

    return H_min, H_max


class RobustContactPoint:
    def __init__(self, contact):
        # TODO eventually we can merge this functionality into the base
        # ContactPoint class directly
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

        self.approx_inertia = approx_inertia

    @classmethod
    def from_config(cls, config, approx_inertia=False):
        mass_lower = config.get("mass_lower", None)
        mass_upper = config.get("mass_upper", None)
        com_lower = config.get("com_lower", None)
        com_upper = config.get("com_upper", None)

        # the center of the bounding shapes is the nominal CoM
        if "ellipsoid" in config:
            ell_half_extents = np.array(config["ellipsoid"]["half_extents"])
        else:
            ell_half_extents = None
        if "box" in config:
            box_half_extents = np.array(config["box"]["half_extents"])
        else:
            box_half_extents = None

        return cls(
            mass_lower=mass_lower,
            mass_upper=mass_upper,
            com_lower=com_lower,
            com_upper=com_upper,
            ellipsoid_half_extents=ell_half_extents,
            box_half_extents=box_half_extents,
            approx_inertia=approx_inertia,
        )

    def polytope(self, m, c, J):
        """Build the polytope containing the inertial parameters given the
        nominal values.

        Parameters
        ----------
        m : float
            The nominal mass.
        c : np.ndarray, shape (3,)
            The nominal center of mass.
        J : np.ndarray, shape (3, 3)
            The nominal inertia matrix.

        Returns
        -------
        : tuple
            A tuple (P, p) with matrix P and vector p such that Pθ <= p.
        """
        Jvec = utils.vech(J)

        # mass bounds
        P_m = np.hstack(([[-1], [1]], np.zeros((2, 9))))
        p_m = np.array([-(m + self.mass_lower), m + self.mass_upper])

        # CoM bounds
        I3 = np.eye(3)
        P_h = np.block(
            [
                [(c + self.com_lower)[:, None], -I3, np.zeros((3, 6))],
                [-(c + self.com_upper)[:, None], I3, np.zeros((3, 6))],
            ]
        )
        p_h = np.zeros(6)

        # inertia bounds
        if self.approx_inertia:
            # assume inertia J is exact
            P_J = np.zeros((12, 10))
            p_J = np.zeros(12)

            P_J[:6, 4:] = -np.eye(6)
            P_J[6:, 4:] = np.eye(6)

            p_J[:6] = -Jvec
            p_J[6:] = Jvec
        else:
            # otherwise we assume no prior knowledge on inertia except object
            # bounding shapes; only constraints required for realizability
            P_J = np.zeros((13, 10))
            p_J = np.zeros(13)

            bounding_box = ip.Box(half_extents=self.box_half_extents, center=c)
            com_box = ip.Box.from_two_vertices(c + self.com_lower, c + self.com_upper)
            mass_min = m + self.mass_lower
            mass_max = m + self.mass_upper
            H_min, H_max = H_min_max(bounding_box, com_box, mass_min, mass_max)

            # H diagonal bounds
            # TODO we actually want to find Hxx_min as a function of mass!
            # diag(H) >= diag(H)_min
            P_J[0, :] = -0.5 * np.array([0, 0, 0, 0, -1, 0, 0, 1, 0, 1])  # -Hxx
            P_J[1, :] = -0.5 * np.array([0, 0, 0, 0, 1, 0, 0, -1, 0, 1])  # -Hyy
            P_J[2, :] = -0.5 * np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, -1])  # -Hzz
            p_J[:3] = -np.diag(H_min)

            # diag(H) <= diag(H)_max
            P_J[3, :] = -0.5 * np.array([0, 0, 0, 0, 1, 0, 0, -1, 0, -1])  # Hxx
            P_J[4, :] = -0.5 * np.array([0, 0, 0, 0, -1, 0, 0, 1, 0, -1])  # Hyy
            P_J[5, :] = -0.5 * np.array([0, 0, 0, 0, -1, 0, 0, -1, 0, 1])  # Hzz
            p_J[3:6] = np.diag(H_max)

            # off-diagonal H values
            P_J[6, :] = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  # -Hxy
            P_J[7, :] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  # -Hxz
            P_J[8, :] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  # -Hyz
            p_J[6:9] = -np.array([H_min[0, 1], H_min[0, 2], H_min[1, 2]])

            P_J[9, :] = [0, 0, 0, 0, 0, -1, 0, 0, 0, 0]  # Hxy
            P_J[10, :] = [0, 0, 0, 0, 0, 0, -1, 0, 0, 0]  # Hxz
            P_J[11, :] = [0, 0, 0, 0, 0, 0, 0, 0, -1, 0]  # Hyz
            p_J[9:12] = np.array([H_max[0, 1], H_max[0, 2], H_max[1, 2]])

            # ellipsoid density realizability
            # tr(ΠQ) >= 0
            ell = ip.Ellipsoid(half_extents=self.ellipsoid_half_extents, center=c)
            P_J[12, :] = [-np.trace(A @ ell.Q) for A in utils.pim_sum_vec_matrices()]

            # TODO why does this make less constraints negative than the
            # ellipsoidal constraint?
            # box density realizability
            # As = utils.pim_sum_vec_matrices()
            # Es = bounding_box.as_ellipsoidal_intersection()
            # P_J[12, :] = [-np.trace(A @ Es[0].Q) for A in As]
            # P_J[13, :] = [-np.trace(A @ Es[1].Q) for A in As]
            # P_J[14, :] = [-np.trace(A @ Es[2].Q) for A in As]

        # combined bounds
        P = np.vstack((P_m, P_h, P_J))
        p = np.concatenate((p_m, p_h, p_J))
        return P, p


class UncertainObject:
    def __init__(self, obj, bounds=None):
        self.object = obj
        self.body = obj.body

        # inertial quantities are taken w.r.t./about the EE origin (i.e.,
        # self.body.com is the vector from the EE origin to the CoM of this
        # object)
        m = self.body.mass
        c = self.body.com
        h = m * c
        H = core.math.skew3(h)
        J = obj.body.inertia - H @ core.math.skew3(self.body.com)

        # spatial mass matrix
        # fmt: off
        self.M = np.block([
            [m * np.eye(3), -H],
            [H, J]
        ])
        # fmt: on

        # polytopic parameter uncertainty: Pθ <= p
        # TODO still may be bugs in the controller since I switched from Pθ >= p
        if bounds is None:
            bounds = ObjectBounds()
        self.P, self.p = bounds.polytope(m, c, J)

    def bias(self, V):
        """Compute Coriolis and centrifugal terms."""
        return utils.skew6(V) @ self.M @ V


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

    H = W @ S

    # convert the whole contact wrench cone to face form
    A = utils.span_to_face_form(H)

    # Aw <= 0 implies there exist feasible contact forces to support wrench w
    return A


def compute_P_tilde_matrix(P, p):
    return np.block([[P, p[:, None]], [np.zeros((1, P.shape[1])), np.array([[1]])]])
