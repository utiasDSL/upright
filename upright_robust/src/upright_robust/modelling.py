import numpy as np
from scipy.linalg import block_diag

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
        # this is w.r.t. to the first object (since the normal points into the
        # first object)
        # fmt: off
        self.S = np.vstack([
            self.normal + μ * self.span[0, :],
            self.normal + μ * self.span[1, :],
            self.normal - μ * self.span[0, :],
            self.normal - μ * self.span[1, :]]).T
        # fmt: on


def parameter_bounds(θ, θ_min, θ_max):
    θ_min_actual = θ
    θ_max_actual = θ

    n = θ.shape[0]
    if θ_min is not None:
        θ_min_actual = np.array(
            [θ[i] if θ_min[i] is None else θ_min[i] for i in range(n)]
        )
    if θ_max is not None:
        θ_max_actual = np.array(
            [θ[i] if θ_max[i] is None else θ_max[i] for i in range(n)]
        )
    return θ_min_actual, θ_max_actual


class UncertainObject:
    def __init__(self, obj, θ_min=None, θ_max=None):
        self.object = obj
        self.body = obj.body

        # inertial quantities are taken w.r.t./about the EE origin (i.e.,
        # self.body.com is the vector from the EE origin to the CoM of this
        # object)
        m = self.body.mass
        h = m * self.body.com
        H = core.math.skew3(h)
        J = obj.body.inertia - H @ core.math.skew3(self.body.com)

        # fmt: off
        self.M = np.block([
            [m * np.eye(3), -H],
            [H, J]
        ])
        # fmt: on

        # polytopic parameter uncertainty: Pθ + p >= 0
        self.θ = np.concatenate(([m], h, utils.vech(J)))
        self.θ_min, self.θ_max = parameter_bounds(self.θ, θ_min, θ_max)
        I = np.eye(self.θ.shape[0])
        self.P = np.vstack((I, -I))
        self.p = np.concatenate((self.θ_min, -self.θ_max))

    def bias(self, V):
        """Compute Coriolis and centrifugal terms."""
        return utils.skew6(V) @ self.M @ V


def compute_object_name_index(names):
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

    # convert the whole contact wrench cone to face form
    A, b = utils.span_to_face_form(W @ S)
    assert np.allclose(b, 0)

    # Aw <= 0 implies there exist feasible contact forces to support wrench w
    return A
