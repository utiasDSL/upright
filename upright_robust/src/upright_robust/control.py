from enum import Enum

import numpy as np
from scipy import sparse
from scipy.linalg import block_diag
from qpsolvers import solve_qp

import upright_core as core
import upright_robust.utils as utils
import upright_robust.modelling as mdl


class ReactiveBalancingController:
    """Base class for reactive balancing control."""

    def __init__(
        self,
        robot,
        objects,
        contacts,
        dt,
        a_cart_weight=1,
        α_cart_weight=5,
        a_joint_weight=0.01,
        v_joint_weight=0.1,
        a_cart_max=5,
        α_cart_max=1,
        a_joint_max=5,
        solver="proxqp",
    ):
        self.solver = solver
        self.robot = robot
        self.objects = objects
        self.contacts = contacts
        self.dt = dt

        self.nc = len(self.contacts)  # number of contacts

        # canonical map of object names to indices
        names = list(self.objects.keys())
        self.object_name_index = mdl.compute_object_name_index(names)

        # default optimization variables are (u, A)
        # shared optimization weight
        self.v_joint_weight = v_joint_weight
        self.a_cart_weight = a_cart_weight
        self.α_cart_weight = α_cart_weight
        self.P_joint = (a_joint_weight + dt**2 * v_joint_weight) * np.eye(9)
        self.P_cart = block_diag(a_cart_weight * np.eye(3), α_cart_weight * np.eye(3))

        # shared optimization bounds
        self.a_cart_max = a_cart_max
        self.α_cart_max = α_cart_max

        self.ub = np.zeros(15)
        self.ub[:9] = a_joint_max
        self.ub[9:12] = a_cart_max
        self.ub[12:] = α_cart_max
        self.lb = -self.ub

        self.W = mdl.compute_contact_force_to_wrench_map(
            self.object_name_index, self.contacts
        )
        self.M = np.vstack([obj.M for obj in self.objects.values()])

        # face form of the CWC
        self.face = mdl.compute_cwc_face_form(self.object_name_index, self.contacts)

    def update(self, q, v):
        self.robot.update(q, v)

    def _solve_qp(
        self, P, q, G=None, h=None, A=None, b=None, lb=None, ub=None, x0=None
    ):
        x = solve_qp(
            P=P,
            q=q,
            G=G,
            h=h,
            A=A,
            b=b,
            lb=lb,
            ub=ub,
            initvals=x0,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=10000,
            solver=self.solver,
        )
        u = x[:9]
        A = x[9:15]
        return u, A

    def _initial_guess(self, a_ew_e_cmd):
        x0 = np.zeros(self.nv)
        x0[9:12] = a_ew_e_cmd
        return x0

    def _compute_q(self, v, a_ew_e_cmd):
        q = np.zeros(self.nv)
        q[:9] = self.v_joint_weight * self.dt * v
        q[9:12] = -self.a_cart_weight * a_ew_e_cmd
        return q

    def solve(self, q, v, a_ew_w_cmd, **kwargs):
        """Solve for an updated joint acceleration command given current robot
        state (q, v) and desired EE (world-frame) acceleration A_ew_w_cmd.
        """
        self.robot.forward(q, v)
        J = self.robot.jacobian(q, frame="local")
        δ = np.concatenate(self.robot.link_classical_acceleration(frame="local"))

        C_we = self.robot.link_pose(rotation_matrix=True)[1]
        V_ew_e = np.concatenate(self.robot.link_velocity(frame="local"))

        # rotate command into the body frame
        a_ew_e_cmd = C_we.T @ a_ew_w_cmd

        # body-frame gravity
        G_e = utils.body_gravity6(C_we.T)

        return self._setup_and_solve_qp(G_e, V_ew_e, a_ew_e_cmd, δ, J, v, **kwargs)


class NominalReactiveBalancingControllerFlat(ReactiveBalancingController):
    """Reactive balancing controller that keeps the tray flat."""

    def __init__(
        self, robot, objects, contacts, dt, use_balancing_constraints=False, **kwargs
    ):
        super().__init__(robot, objects, contacts, dt, **kwargs)

        self.use_balancing_constraints = use_balancing_constraints

        if use_balancing_constraints:
            self.nf = 3 * self.nc

            # pre-compute part of equality constraints
            self.A_eq_bal = np.hstack((np.zeros((self.M.shape[0], 9)), self.M, self.W))

            # inequality constraints for balancing
            F = block_diag(*[c.F for c in self.contacts])
            self.A_ineq = np.hstack((np.zeros((F.shape[0], 9 + 6)), F))
            self.A_ineq_sparse = sparse.csc_matrix(self.A_ineq)
            self.b_ineq = np.zeros(self.A_ineq.shape[0])
        else:
            self.nf = 0
            self.A_ineq_sparse = None
            self.b_ineq = None

        # number of optimization variables: 9 joint accelerations, 6 EE
        # acceleration twist, nf force
        self.nv = 15 + self.nf

        # cost
        P_force = np.zeros((self.nf, self.nf))
        self.P = block_diag(self.P_joint, self.P_cart, P_force)
        self.P_sparse = sparse.csc_matrix(self.P)

        # optimization bounds
        self.lb = np.concatenate((self.lb, -np.inf * np.ones(self.nf)))
        self.ub = np.concatenate((self.ub, np.inf * np.ones(self.nf)))

    def _setup_and_solve_qp(self, G_e, V_ew_e, a_ew_e_cmd, δ, J, v):
        x0 = self._initial_guess(a_ew_e_cmd)

        # map joint acceleration to EE acceleration
        A_eq_track = np.hstack((-J, np.eye(6), np.zeros((6, self.W.shape[1]))))
        b_eq_track = δ

        # keep angular acceleration fixed to zero
        A_α_fixed = np.zeros((3, self.nv))
        A_α_fixed[:, 12:15] = np.eye(3)
        b_α_fixed = np.zeros(3)

        if self.use_balancing_constraints:
            # compute the equality constraints A_eq @ x == b
            # N-E equations for object balancing
            h = np.concatenate([obj.bias(V_ew_e) for obj in self.objects.values()])
            b_eq_bal = self.M @ G_e - h
            A_eq = np.vstack((A_eq_track, A_α_fixed, self.A_eq_bal))
            b_eq = np.concatenate((b_eq_track, b_α_fixed, b_eq_bal))
        else:
            A_eq = np.vstack((A_eq_track, A_α_fixed))
            b_eq = np.concatenate((b_eq_track, b_α_fixed))

        # compute the cost: 0.5 * x @ P @ x + q @ x
        q = self._compute_q(v, a_ew_e_cmd)

        return self._solve_qp(
            P=self.P_sparse,
            q=q,
            G=self.A_ineq_sparse,
            h=self.b_ineq,
            A=sparse.csc_matrix(A_eq),
            b=b_eq,
            lb=self.lb,
            ub=self.ub,
            x0=x0,
        )


class NominalReactiveBalancingControllerTrayTilting(ReactiveBalancingController):
    """Reactive balancing controller that tilts so that the tray is normal to total acceleration.

    Neglects the rotational dynamics of the tray in the tilting computation.
    Optionally also enforces the overall balancing constraints on the objects
    (these do account for rotational dynamics).
    """

    def __init__(
        self,
        robot,
        objects,
        contacts,
        dt,
        kθ=0,
        kω=0,
        use_balancing_constraints=False,
        **kwargs
    ):
        super().__init__(robot, objects, contacts, dt, **kwargs)

        self.use_balancing_constraints = use_balancing_constraints

        # angular control gains
        self.kθ = kθ
        self.kω = kω

        # inequality constraints
        if use_balancing_constraints:
            self.nf = 3 * self.nc

            # pre-compute part of equality constraints
            nM = self.M.shape[0]
            self.A_eq_bal = np.hstack(
                (np.zeros((nM, 9)), self.M, np.zeros((nM, 3)), self.W)
            )

            F = block_diag(*[c.F for c in self.contacts])
            self.A_ineq = np.hstack((np.zeros((F.shape[0], 18)), F))
            self.A_ineq_sparse = sparse.csc_matrix(self.A_ineq)
            self.b_ineq = np.zeros(self.A_ineq.shape[0])
        else:
            self.nf = 0
            self.A_ineq_sparse = None
            self.b_ineq = None

        # number of optimization variables: 9 joint accelerations, 6 EE
        # acceleration twist, desired angular acceleration, nf force
        self.nv = 9 + 6 + 3 + self.nf

        # cost
        I3 = np.eye(3)
        P_a_cart = self.a_cart_weight * I3
        P_α_cart = self.α_cart_weight * np.block([[I3, -I3], [-I3, I3]])
        P_force = np.zeros((self.nf, self.nf))
        P_cart = block_diag(P_a_cart, P_α_cart)
        self.P = block_diag(self.P_joint, P_cart, P_force)
        self.P_sparse = sparse.csc_matrix(self.P)

        # bounds
        self.ub = np.concatenate(
            (self.ub, self.α_cart_max * np.ones(3), np.inf * np.ones(self.nf))
        )
        self.lb = -self.ub

    def _setup_and_solve_qp(self, G_e, V_ew_e, a_ew_e_cmd, δ, J, v):
        x0 = self._initial_guess(a_ew_e_cmd)

        # map joint acceleration to EE acceleration
        A_eq_track = np.hstack((-J, np.eye(6), np.zeros((6, 3 + self.nf))))
        b_eq_track = δ

        # equality constraints for new consistency stuff
        I3 = np.eye(3)
        z = np.array([0, 0, 1])
        A_eq_tilt = np.hstack(
            (
                np.zeros((3, 9)),
                -self.kθ * core.math.skew3(z),
                np.zeros((3, 3)),
                np.eye(3),
                np.zeros((3, self.nf)),
            )
        )
        g = G_e[:3]
        ω_ew_e = V_ew_e[3:]
        b_eq_tilt = -self.kω * ω_ew_e - self.kθ * np.cross(z, g)

        if self.use_balancing_constraints:
            # compute the equality constraints A_eq @ x == b
            # N-E equations for object balancing
            h = np.concatenate([obj.bias(V_ew_e) for obj in self.objects.values()])
            b_eq_bal = self.M @ G_e - h
            A_eq = np.vstack((A_eq_track, self.A_eq_bal, A_eq_tilt))
            b_eq = np.concatenate((b_eq_track, b_eq_bal, b_eq_tilt))
        else:
            A_eq = np.vstack((A_eq_track, A_eq_tilt))
            b_eq = np.concatenate((b_eq_track, b_eq_tilt))

        # compute the cost: 0.5 * x @ P @ x + q @ x
        q = self._compute_q(v, a_ew_e_cmd)

        return self._solve_qp(
            P=self.P_sparse,
            q=q,
            G=self.A_ineq_sparse,
            h=self.b_ineq,
            A=sparse.csc_matrix(A_eq),
            b=b_eq,
            lb=self.lb,
            ub=self.ub,
            x0=x0,
        )


class ReactiveBalancingControllerFullTilting(ReactiveBalancingController):
    """Reactive balancing controller that tilts to account for all balanced objects.

    Constraints can be nominal with force variable or face form (if
    `use_face_form=True`), or robust constraints can be used
    (`use_robust_constraints=True`).

    Currently, non-face form robust constraints are not supported (so when
    `use_robust_constraints=True`, then the value of `use_face_form` does not
    matter).
    """

    def __init__(
        self,
        robot,
        objects,
        contacts,
        dt,
        kθ=0,
        kω=0,
        use_dvdt_scaling=False,
        use_face_form=False,
        use_robust_constraints=False,
        **kwargs
    ):
        super().__init__(robot, objects, contacts, dt, **kwargs)

        # angular control gains
        self.kθ = kθ
        self.kω = kω
        self.use_dvdt_scaling = use_dvdt_scaling
        self.use_face_form = use_face_form
        self.use_robust_constraints = use_robust_constraints

        self.no = len(self.objects)

        if use_robust_constraints:
            self.nf = 0

            # polytopic uncertainty Pθ + p >= 0
            P = block_diag(*[obj.P for obj in self.objects.values()])
            p = np.concatenate([obj.p for obj in self.objects.values()])

            # fmt: off
            self.P_tilde = np.block([
                [P, p[:, None]],
                [np.zeros((1, P.shape[1])), np.array([[-1]])]])
            # fmt: on
            self.R = utils.span_to_face_form(self.P_tilde.T)[0]

            # pre-compute inequality matrix
            n_face = self.face.shape[0]
            n_ineq = self.R.shape[0]
            self.A_ineq = np.zeros((n_ineq * n_face, 15 + 6 * self.no))
            for i in range(n_face):
                D = utils.body_regressor_A_by_vector(self.face[i, :])
                D_tilde = np.vstack((D, np.zeros((1, D.shape[1]))))
                self.A_ineq[i * n_ineq : (i + 1) * n_ineq, 9:15] = -self.R @ D_tilde

            # no balancing equality constraints
            self.A_eq_bal = np.zeros((0, 15 + 6 * self.no))

        elif use_face_form:
            self.nf = 0
            n_face = self.face.shape[0]
            self.A_ineq = np.hstack(
                (
                    np.zeros((n_face, 9)),
                    self.face @ self.M,
                    np.zeros((n_face, 6 * self.no)),
                )
            )

            # no balancing equality constraints
            self.A_eq_bal = np.zeros((0, 15 + 6 * self.no))

        else:
            self.nf = 3 * self.nc

            # pre-compute part of equality constraints
            nM = self.M.shape[0]
            self.A_eq_bal = np.hstack(
                (np.zeros((nM, 9)), self.M, np.zeros((nM, 6 * self.no)), self.W)
            )

            # inequality constraints
            F = block_diag(*[c.F for c in self.contacts])
            self.A_ineq = np.hstack((np.zeros((F.shape[0], 15 + 6 * self.no)), F))

        self.A_ineq_sparse = sparse.csc_matrix(self.A_ineq)

        # equality constraint for linear velocity at object CoMs
        I3 = np.eye(3)
        self.coms = np.array([o.body.com for o in self.objects.values()])
        self.A_eq_tilt1 = np.hstack(
            (
                np.zeros((3 * self.no, 9)),  # joints
                np.tile(-I3, (self.no, 1)),  # dvedt
                np.vstack([core.math.skew3(c) for c in self.coms]),  # dωedt
                np.zeros((3 * self.no, 3 * self.no)),  # dωddt
                np.eye(3 * self.no),  # dvodt
                np.zeros((3 * self.no, self.nf)),  # contact forces
            )
        )

        # number of optimization variables: 9 joint accelerations, 6 EE
        # acceleration twist, no * desired angular acceleration no * object
        # linear acceleration, nf force
        self.nv = 9 + 6 + 6 * self.no + self.nf

        # cost
        P_α_cart = np.eye(3 * (1 + self.no))
        P_α_cart[:3, :3] = self.no * I3
        P_α_cart[:3, 3:] = np.tile(-I3, self.no)
        P_α_cart[3:, :3] = np.tile(-I3, self.no).T
        P_α_cart = self.α_cart_weight * P_α_cart / self.no
        P_cart = block_diag(
            self.a_cart_weight * I3, P_α_cart, self.a_cart_weight * np.eye(3 * self.no)
        )

        P_force = np.zeros((self.nf, self.nf))
        self.P = block_diag(self.P_joint, P_cart, P_force)
        self.P_sparse = sparse.csc_matrix(self.P)

        # bounds
        self.ub = np.concatenate(
            (
                self.ub,
                self.α_cart_max * np.ones(3 * self.no),
                self.a_cart_max * np.ones(3 * self.no),
                np.inf * np.ones(self.nf),
            )
        )
        self.lb = -self.ub

    def _setup_and_solve_qp(self, G_e, V_ew_e, a_ew_e_cmd, δ, J, v):
        x0 = self._initial_guess(a_ew_e_cmd)

        # compute the equality constraints A_eq @ x == b
        # N-E equations for object balancing
        h = np.concatenate([obj.bias(V_ew_e) for obj in self.objects.values()])

        # map joint acceleration to EE acceleration
        A_eq_track = np.hstack((-J, np.eye(6), np.zeros((6, 6 * self.no + self.nf))))
        b_eq_track = δ

        # equality constraints for new consistency stuff
        ω_ew_e = V_ew_e[3:]
        b_eq_tilt1 = np.concatenate(
            [np.cross(ω_ew_e, np.cross(ω_ew_e, c)) for c in self.coms]
        )

        scale = self.kθ
        if self.use_dvdt_scaling:
            scale /= np.linalg.norm(a_ew_e_cmd - G_e[:3])

        z = np.array([0, 0, 1])
        A_eq_tilt2 = np.hstack(
            (
                np.zeros((3 * self.no, 9 + 6)),
                np.eye(3 * self.no),
                np.kron(np.eye(self.no), -scale * core.math.skew3(z)),
                np.zeros((3 * self.no, self.nf)),
            )
        )
        g = G_e[:3]
        b_eq_tilt2 = np.tile(-self.kω * ω_ew_e - scale * np.cross(z, g), self.no)

        if self.use_robust_constraints:
            B = utils.body_regressor_VG_by_vector_tilde_vectorized(
                V_ew_e, G_e, self.face
            )
            b_ineq = (self.R @ B).T.flatten()
            b_eq_bal = np.zeros(0)
        elif self.use_face_form:
            b_ineq = self.face @ (self.M @ G_e - h)
            b_eq_bal = np.zeros(0)
        else:
            b_ineq = np.zeros(self.A_ineq.shape[0])
            b_eq_bal = self.M @ G_e - h

        A_eq = np.vstack((A_eq_track, self.A_eq_bal, self.A_eq_tilt1, A_eq_tilt2))
        b_eq = np.concatenate((b_eq_track, b_eq_bal, b_eq_tilt1, b_eq_tilt2))

        # compute the cost: 0.5 * x @ P @ x + q @ x
        q = self._compute_q(v, a_ew_e_cmd)

        return self._solve_qp(
            P=self.P_sparse,
            q=q,
            G=self.A_ineq_sparse,
            h=b_ineq,
            A=sparse.csc_matrix(A_eq),
            b=b_eq,
            lb=self.lb,
            ub=self.ub,
            x0=x0,
        )


# TODO deprecated but currently still here for reference
# class RobustReactiveBalancingController(ReactiveBalancingController):
#     def __init__(self, model, dt, **kwargs):
#         super().__init__(model, dt, **kwargs)
#
#         self.P_sparse = sparse.csc_matrix(self.P)
#
#         # polytopic uncertainty Pθ + p >= 0
#         P = block_diag(*[obj.P for obj in self.objects.values()])
#         p = np.concatenate([obj.p for obj in self.objects.values()])
#
#         # fmt: off
#         self.P_tilde = np.block([
#             [P, p[:, None]],
#             [np.zeros((1, P.shape[1])), np.array([[-1]])]])
#         # fmt: on
#         self.R = utils.span_to_face_form(self.P_tilde.T)[0]
#
#         # pre-compute inequality matrix
#         self.nv = 15
#
#         nf = self.face.shape[0]
#         n_ineq = self.R.shape[0]
#         N_ineq = n_ineq * nf
#         self.A_ineq = np.zeros((N_ineq, nv))
#         for i in range(nf):
#             D = utils.body_regressor_A_by_vector(self.face[i, :])
#             D_tilde = np.vstack((D, np.zeros((1, D.shape[1]))))
#             self.A_ineq[i * n_ineq : (i + 1) * n_ineq, 9:] = -self.R @ D_tilde
#         self.A_ineq_sparse = sparse.csc_matrix(self.A_ineq)
#
#     def _setup_and_solve_qp(self, G_e, V_ew_e, a_ew_e_cmd, δ, J, v):
#         x0 = self._initial_guess(a_ew_e_cmd)
#
#         # map joint acceleration to EE acceleration
#         A_eq = np.hstack((-J, np.eye(6)))
#         b_eq = δ
#
#         # build robust constraints
#         B = utils.body_regressor_VG_by_vector_tilde_vectorized(V_ew_e, G_e, self.face)
#         b_ineq = (self.R @ B).T.flatten()
#
#         # compute the cost: 0.5 * x @ P @ x + q @ x
#         q = self._compute_q(v, a_ew_e_cmd)
#
#         return self._solve_qp(
#             P=self.P_sparse,
#             q=q,
#             G=self.A_ineq_sparse,
#             h=b_ineq,
#             A=sparse.csc_matrix(A_eq),
#             b=b_eq,
#             lb=self.lb,
#             ub=self.ub,
#             x0=x0,
#         )
