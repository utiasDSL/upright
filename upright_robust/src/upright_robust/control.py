import time

import numpy as np
from scipy import sparse
from scipy.linalg import block_diag, null_space
from qpsolvers import solve_qp
from lpsolvers import solve_lp

import upright_core as core
import upright_robust.utils as utils
import upright_robust.modelling as mdl

import IPython


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
        print("Computing CWC face form...")
        self.face = mdl.compute_cwc_face_form(self.object_name_index, self.contacts)
        print("...done.")

        self.x_last = None

    def update(self, q, v):
        self.robot.update(q, v)

    def _solve_qp(
        self, P, q, G=None, h=None, A=None, b=None, lb=None, ub=None, x0=None
    ):
        if x0 is None:
            x0 = self.x_last
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
        self.x_last = x
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


def check_redundant_linear_constraints(A, b):
    """{x | Ax <= b}"""

    n = A.shape[0]
    for i in range(n):
        a = A[i, :]
        h = np.copy(b)
        h[i] = 1
        x = solve_lp(c=-a, G=A, h=h)
        p = a @ x
        print(p - b[i])
    IPython.embed()


def approx_polytope_with_hypercuboid(R):
    """{x | Rx <= 0}"""
    # dimension of the space
    d = R.shape[1]

    z = np.zeros(R.shape[0])

    lb = np.zeros(d)
    ub = np.zeros(d)

    for i in range(d):
        c = np.zeros(d)
        c[i] = 1

        # TODO problems are infeasible: this is a cone containing zero
        try:
            # max
            x = solve_lp(c=-c, G=R, h=z, solver="cdd")
            ub[i] = c @ x

            # min
            x = solve_lp(c=c, G=R, h=z, solver="cdd")
            lb[i] = c @ x
        except ValueError as e:
            print(e)
            IPython.embed()

    normals = np.vstack((np.eye(d), -np.eye(d)))
    bounds = np.concatenate((ub, -lb))

    IPython.embed()


def solve_max_min_value(A):
    c = np.zeros(A.shape[1] + 1)
    c[0] = 1

    G = np.hstack((np.ones((A.shape[0], 1)), -A))
    x = solve_lp(c=-c, G=G, h=np.zeros(A.shape[0]))
    IPython.embed()


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

        self.nλ = 0

        # polytopic parameter uncertainty
        P = block_diag(*[obj.P for obj in self.objects.values()])
        p = np.concatenate([obj.p for obj in self.objects.values()])

        # fmt: off
        self.P_tilde = np.block([
            [P, p[:, None]],
            [np.zeros((1, P.shape[1])), np.array([[-1]])]])
        # fmt: on

        print("Computing inertial parameter face form...")
        self.R = utils.span_to_face_form(self.P_tilde.T)
        print("...done.")

        n_face = self.face.shape[0]
        n_ineq = self.R.shape[0]
        self.A_ineq = np.zeros((n_ineq * n_face, 15 + 6 * self.no))
        for i in range(n_face):
            D = utils.body_regressor_A_by_vector(self.face[i, :])
            D_tilde = np.vstack((D, np.zeros((1, D.shape[1]))))
            self.A_ineq[i * n_ineq : (i + 1) * n_ineq, 9:15] = -self.R @ D_tilde

        self.A_ineq_rob = self.A_ineq

        if use_robust_constraints:
            # robust constraints without using DD to remove the extra dual variables
            self.nf = 0

            # polytopic parameter uncertainty
            P = block_diag(*[obj.P for obj in self.objects.values()])
            p = np.concatenate([obj.p for obj in self.objects.values()])

            # fmt: off
            self.P_tilde = np.block([
                [P, p[:, None]],
                [np.zeros((1, P.shape[1])), np.array([[-1]])]])
            # fmt: on

            if use_face_form:
                print("Computing inertial parameter face form...")
                self.R = utils.span_to_face_form(self.P_tilde.T)
                print("...done.")

                # self.R /= np.max(np.abs(self.R))

                # pre-compute inequality matrix
                n_face = self.face.shape[0]
                n_ineq = self.R.shape[0]
                self.A_ineq = np.zeros((n_ineq * n_face, 15 + 6 * self.no))
                for i in range(n_face):
                    D = utils.body_regressor_A_by_vector(self.face[i, :])
                    D_tilde = np.vstack((D, np.zeros((1, D.shape[1]))))
                    self.A_ineq[i * n_ineq : (i + 1) * n_ineq, 9:15] = -self.R @ D_tilde

                # self.A_ineq_max = np.max(np.abs(self.A_ineq))
                # self.A_ineq /= self.A_ineq_max
                self.A_eq_bal = np.zeros((0, 15 + 6 * self.no))
            else:
                self.P_inv = np.linalg.pinv(self.P_tilde.T)
                P_null = null_space(self.P_tilde.T)

                # number of extra dual variables
                n_face = self.face.shape[0]
                n_dual = P_null.shape[1]  # self.P_tilde.shape[0]
                self.nλ = n_dual * n_face

                # equalities tying together the dual variables and object
                # accelerations
                n_ineq = P_null.shape[0]  #self.P_tilde.shape[1]
                self.A_eq_bal = np.zeros((n_ineq * n_face, 15 + 6 * self.no + self.nλ))
                self.A_ineq = np.zeros((n_ineq * n_face, 15 + 6 * self.no + self.nλ))
                for i in range(n_face):
                    D = utils.body_regressor_A_by_vector(self.face[i, :])
                    D_tilde = np.vstack((D, np.zeros((1, D.shape[1]))))
                    self.A_ineq[i * n_ineq : (i + 1) * n_ineq, 9:15] = self.P_inv @ D_tilde
                    self.A_ineq[
                        i * n_ineq : (i + 1) * n_ineq,
                        15 + 6 * self.no + i * n_dual : 15 + 6 * self.no + (i + 1) * n_dual,
                    ] = -P_null

                    # self.A_eq_bal[i * n_ineq : (i + 1) * n_ineq, 9:15] = D_tilde
                    # self.A_eq_bal[
                    #     i * n_ineq : (i + 1) * n_ineq,
                    #     15 + 6 * self.no + i * n_dual : 15 + 6 * self.no + (i + 1) * n_dual,
                    # ] = self.P_tilde.T
                    # self.A_eq_bal[
                    #     i * n_ineq : (i + 1) * n_ineq,
                    #     15 + 6 * self.no: 15 + 6 * self.no + n_dual,
                    # ] = self.P_tilde.T

                # self.A_ineq = np.zeros((0, 15 + 6 * self.no + self.nλ))
                self.A_eq_bal = np.zeros((0, 15 + 6 * self.no + self.nλ))

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
                np.zeros((3 * self.no, self.nf + self.nλ)),  # contact forces
            )
        )

        # number of optimization variables: 9 joint accelerations, 6 EE
        # acceleration twist, no * desired angular acceleration no * object
        # linear acceleration, nf force
        self.nv = 9 + 6 + 6 * self.no + self.nf + self.nλ

        # cost
        P_α_cart = np.eye(3 * (1 + self.no))
        P_α_cart[:3, :3] = self.no * I3
        P_α_cart[:3, 3:] = np.tile(-I3, self.no)
        P_α_cart[3:, :3] = np.tile(-I3, self.no).T
        P_α_cart = self.α_cart_weight * P_α_cart / self.no
        P_cart = block_diag(
            self.a_cart_weight * I3, P_α_cart, self.a_cart_weight * np.eye(3 * self.no)
        )

        P_force = 0 * np.eye(self.nf + self.nλ)
        self.P = block_diag(self.P_joint, P_cart, P_force)
        self.P_sparse = sparse.csc_matrix(self.P)

        # bounds
        self.ub = np.concatenate(
            (
                self.ub,
                self.α_cart_max * np.ones(3 * self.no),
                self.a_cart_max * np.ones(3 * self.no),
                np.inf * np.ones(self.nf),
                np.inf * np.ones(self.nλ),
            )
        )
        self.lb = -self.ub
        # self.lb[-self.nλ:] = 0

    def _setup_and_solve_qp(self, G_e, V_ew_e, a_ew_e_cmd, δ, J, v):
        x0 = self._initial_guess(a_ew_e_cmd)

        # compute the equality constraints A_eq @ x == b
        # N-E equations for object balancing
        h = np.concatenate([obj.bias(V_ew_e) for obj in self.objects.values()])

        # map joint acceleration to EE acceleration
        A_eq_track = np.hstack((-J, np.eye(6), np.zeros((6, 6 * self.no + self.nf + self.nλ))))
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
                np.zeros((3 * self.no, self.nf + self.nλ)),
            )
        )
        g = G_e[:3]
        b_eq_tilt2 = np.tile(-self.kω * ω_ew_e - scale * np.cross(z, g), self.no)

        if self.use_robust_constraints:
            B = utils.body_regressor_VG_by_vector_tilde_vectorized(
                V_ew_e, G_e, self.face
            )
            if self.use_face_form:
                b_ineq = (self.R @ B).T.flatten()  #/ self.A_ineq_max
                # check_redundant_linear_constraints(self.A_ineq, b_ineq)
                b_eq_bal = np.zeros(0)
            else:
                # b_ineq = np.zeros(0)
                # b_eq_bal = -B.T.flatten()
                b_ineq = -(self.P_inv @ B).T.flatten()
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

        u, A = self._solve_qp(
            P=self.P_sparse,
            q=q,
            G=self.A_ineq_sparse,
            h=b_ineq,
            A=sparse.csc_matrix(A_eq),
            b=b_eq,
            lb=self.lb,
            ub=self.ub,
            # x0=x0,
        )
        # return u, A

        n = self.face.shape[1] // 6  # number of wrenches
        RT = self.R[:, :-1].T

        # NOTE seems to work somewhat, but is still quite slow just to compute
        # these quantities
        # TODO is this just projection onto the polyhedral cone?
        t0 = time.time()

        # B = utils.body_regressor_VG_by_vector_vectorized(V_ew_e, G_e, self.face)
        #
        # t11 = time.time()
        # # equivalent to but faster than: b_ineq_rob = (self.R @ B).T.flatten()
        # b_ineq_rob = np.ravel(self.R[:, :-1] @ B, order="F")
        # t12 = time.time()

        Y0 = utils.body_regressor(V_ew_e, -G_e)
        b_ineq_rob = np.ravel(self.face @ block_diag(*[Y0] * n) @ RT)

        # t13 = time.time()

        x = self.A_ineq_rob[:, 9:15] @ A
        mask = x > b_ineq_rob
        s = b_ineq_rob[mask] / x[mask]

        # short-circuit if there are no constraint violations
        if s.shape == (0,):
            return u, A

        if np.any(s < 0):
            raise ValueError("Different signs in inequality constraints!")

        scale = np.min(s)
        sA = scale * A
        su = np.linalg.lstsq(J, sA - δ, rcond=None)[0]

        t2 = time.time()

        print(f"scale took {1000 * (t2 - t0)} ms")

        return su, sA


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
