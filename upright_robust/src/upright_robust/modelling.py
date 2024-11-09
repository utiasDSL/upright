import numpy as np

import rigeo as rg

import upright_core as core
import upright_robust.utils as utils


class RobustContactPoint:
    def __init__(self, contact):
        # TODO eventually we can merge this functionality into the base
        # ContactPoint class from upright directly
        self.contact = contact
        self.normal = contact.normal
        self.span = contact.span
        μ = contact.mu

        # grasp matrix: convert contact force into body contact wrench
        self.G1 = np.vstack((np.eye(3), core.math.skew3(contact.r_co_o1)))
        self.G2 = np.vstack((np.eye(3), core.math.skew3(contact.r_co_o2)))

        # matrix to enforce friction cone constraint F @ f >= 0 ==> inside FC
        # (this is the negative of the face form)
        # TODO make negative to be consistent with face form
        # fmt: off
        # self.F = np.array([
        #     [1,  0,  0],
        #     [μ, -1, -1],
        #     [μ,  1, -1],
        #     [μ, -1,  1],
        #     [μ,  1,  1],
        # ]) @ np.vstack((self.normal, self.span))
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


class UncertainObject:
    def __init__(self, body, bounding_box=None, com_box=None):
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

        self.bounding_box = bounding_box
        self.com_box = com_box

    def bias(self, V):
        """Compute Coriolis and centrifugal terms."""
        return rg.skew6(V) @ self.M @ V

    def wrench(self, A, V):
        """Compute the body-frame inertial wrench."""
        return self.M @ A + self.bias(V)


def compute_object_name_index(names):
    """Compute mapping from object names to indices."""
    return {name: idx for idx, name in enumerate(names)}


def compute_grasp_matrix(name_index, contacts):
    """Compute the grasp matrix.

    The grasp matrix G maps the contact forces (f_1, ..., f_nc) to object
    wrenches (w_1, ..., w_no)."""
    no = len(name_index)
    nc = len(contacts)
    G = np.zeros((no * 6, nc * 3))
    for i, c in enumerate(contacts):
        # ignore EE object
        if c.contact.object1_name != "ee":
            r = name_index[c.contact.object1_name]
            G[r * 6 : (r + 1) * 6, i * 3 : (i + 1) * 3] = c.G1

        # second object is negative because a positive force on the first
        # object = negative force on the second object
        r = name_index[c.contact.object2_name]
        G[r * 6 : (r + 1) * 6, i * 3 : (i + 1) * 3] = -c.G2
    return G


def compute_cwc_span_form(name_index, contacts):
    """Build the span form of the contact wrench cone from contact points of an object."""
    no = len(name_index)
    nc = len(contacts)
    H = np.zeros((no * 6, nc * 4))
    for i, c in enumerate(contacts):
        # ignore EE object
        if c.contact.object1_name != "ee":
            r = name_index[c.contact.object1_name]
            H[r * 6 : (r + 1) * 6, i * 4 : (i + 1) * 4] = c.G1 @ c.S

        # second object is negative because a positive force on the first
        # object = negative force on the second object
        r = name_index[c.contact.object2_name]
        H[r * 6 : (r + 1) * 6, i * 4 : (i + 1) * 4] = -c.G2 @ c.S
    return H


def compute_cwc_face_form(name_index, contacts):
    """Build the face form of the contact wrench cone from contact points of an object."""

    # convert the whole contact wrench cone to face form
    H = compute_cwc_span_form(name_index, contacts)
    A = utils.cone_span_to_face_form(H)

    # Aw <= 0 implies there exist feasible contact forces to support wrench w
    return A
