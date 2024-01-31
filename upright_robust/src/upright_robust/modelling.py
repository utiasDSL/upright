import numpy as np
from scipy.linalg import block_diag

# TODO probably don't want this dependency at the moment
import inertial_params as ip

import upright_core as core
import upright_robust.utils as utils


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

    @classmethod
    def from_config(cls, config):
        mass_lower = config.get("mass_lower", None)
        mass_upper = config.get("mass_upper", None)
        com_lower = config.get("com_lower", None)
        com_upper = config.get("com_upper", None)
        if "ellipsoid" in config:
            # the center of the ellipsoid will be the nominal CoM
            half_extents = np.array(config["ellipsoid"]["half_extents"])
        else:
            half_extents = None
        return cls(
            mass_lower=mass_lower,
            mass_upper=mass_upper,
            com_lower=com_lower,
            com_upper=com_upper,
            ellipsoid_half_extents=half_extents,
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
        if self.ellipsoid_half_extents is None:
            # assume inertia J is exact
            P_J = np.zeros((12, 10))
            p_J = np.zeros(12)

            P_J[:6, 4:] = -np.eye(6)
            P_J[6:, 4:] = np.eye(6)

            p_J[:6] = -Jvec
            p_J[6:] = Jvec
        else:
            # no prior knowledge on inertia; only constraints required for
            # realizability
            P_J = np.zeros((13, 10))
            p_J = np.zeros(13)

            # H diagonal bounds
            # TODO we actually want to find Hxx_min as a function of mass!
            # diag(H) >= diag(H)_min
            P_J[0, :] = -0.5 * np.array([0, 0, 0, 0, -1, 0, 0, 1, 0, 1])  # -Hxx
            P_J[1, :] = -0.5 * np.array([0, 0, 0, 0, 1, 0, 0, -1, 0, 1])  # -Hyy
            P_J[2, :] = -0.5 * np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, -1])  # -Hzz
            p_J[:3] = -np.array([Hxx_min, Hyy_min, Hzz_min])

            # diag(H) <= diag(H)_max
            P_J[3, :] = -0.5 * np.array([0, 0, 0, 0, 1, 0, 0, -1, 0, -1])  # Hxx
            P_J[4, :] = -0.5 * np.array([0, 0, 0, 0, -1, 0, 0, 1, 0, -1])  # Hyy
            P_J[5, :] = -0.5 * np.array([0, 0, 0, 0, -1, 0, 0, -1, 0, 1])  # Hzz
            p_J[3:6] = np.array([Hxx_max, Hyy_max, Hzz_max])

            # off-diagonal H values
            P_J[6, :] = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  # -Hxy
            P_J[7, :] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  # -Hxz
            P_J[8, :] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  # -Hyz
            p_J[6:9] = -np.array([Hxy_min, Hxz_min, Hyz_min])

            P_J[9, :] = [0, 0, 0, 0, 0, -1, 0, 0, 0, 0]  # Hxy
            P_J[10, :] = [0, 0, 0, 0, 0, 0, -1, 0, 0, 0]  # Hxz
            P_J[11, :] = [0, 0, 0, 0, 0, 0, 0, 0, -1, 0]  # Hyz
            p_J[9:12] = np.array([Hxy_max, Hxz_max, Hyz_max])

            # ellipsoid density realizability
            # tr(ΠQ) >= 0
            Q = ip.Ellipsoid.from_half_extents(
                half_extents=self.ellipsoid_half_extents, center=c
            ).Q
            P_J[12, :] = [-np.trace(A @ Q) for A in utils.pim_sum_vec_matrices()]

            # # TODO let's put upper bounds on as well
            # # diag(H) >= 0
            # # recall θ = [m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
            # P_J[1, :] = [0, 0, 0, 0, -1, 0, 0, 1, 0, 1]  # Hxx >= 0
            # P_J[2, :] = [0, 0, 0, 0, 1, 0, 0, -1, 0, 1]  # Hyy >= 0
            # P_J[3, :] = [0, 0, 0, 0, 1, 0, 0, 1, 0, -1]  # Hzz >= 0
            #
            # # TODO: depends on the mass, right now just using nominal
            # H_diag_upper = m * self.ellipsoid_half_extents ** 2
            #
            # # negative because <=
            # P_J[4:7, :] = -P_J[1:4, :]
            # p_J[4:7] = -2 * H_diag_upper
            #
            # # TODO force other elements to zero
            # P_J[7, :] = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  # Hxy
            # P_J[8, :] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  # Hxz
            # P_J[9, :] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  # Hyz
            # P_J[10:13, :] = -P_J[7:10, :]

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
