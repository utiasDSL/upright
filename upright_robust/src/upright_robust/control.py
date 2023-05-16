import numpy as np
from scipy import sparse
from scipy.linalg import block_diag
from qpsolvers import solve_qp

import upright_core as core
import upright_robust.utils as utils
import upright_robust.modelling as mdl


class ReactiveBalancingController:
    def __init__(
        self,
        model,
        dt,
        θ_min=None,
        θ_max=None,
        a_cart_weight=1,
        α_cart_weight=1,
        a_joint_weight=0,
        v_joint_weight=0.1,
        a_cart_max=5,
        α_cart_max=1,
        a_joint_max=5,
        solver="proxqp",
    ):
        self.solver = solver
        self.robot = model.robot
        self.dt = dt

        self.objects = {
            k: mdl.UncertainObject(v, θ_min, θ_max)
            for k, v in model.settings.balancing_settings.objects.items()
        }
        self.contacts = [
            mdl.RobustContactPoint(c)
            for c in model.settings.balancing_settings.contacts
        ]

        # canonical map of object names to indices
        names = list(self.objects.keys())
        self.object_name_index = mdl.compute_object_name_index(names)

        # default optimization variables are (u, A)
        # shared optimization weight
        self.v_joint_weight = v_joint_weight
        self.P_joint = (a_joint_weight + dt**2 * v_joint_weight) * np.eye(9)
        self.P_cart = block_diag(a_cart_weight * np.eye(3), α_cart_weight * np.eye(3))
        self.P = block_diag(self.P_joint, self.P_cart)

        # shared optimization bounds
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
        self.F = mdl.compute_cwc_face_form(self.object_name_index, self.contacts)

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
        a = x[:9]
        A = x[9:15]
        return a, A

    def _initial_guess(self, nv, A_ew_e_cmd):
        x0 = np.zeros(nv)
        x0[9:15] = A_ew_e_cmd
        return x0

    def _compute_q(self, nv, v, A_ew_e_cmd):
        q = np.zeros(nv)
        q[:9] = self.v_joint_weight * self.dt * v
        q[9:15] = -self.P_cart @ A_ew_e_cmd
        return q

    def solve(self, q, v, A_ew_w_cmd, **kwargs):
        """Solve for an updated joint acceleration command given current robot
        state (q, v) and desired EE (world-frame) acceleration A_ew_w_cmd.
        """
        self.robot.forward(q, v)
        J = self.robot.jacobian(q, frame="local")
        δ = np.concatenate(self.robot.link_classical_acceleration(frame="local"))

        C_we = self.robot.link_pose(rotation_matrix=True)[1]
        V_ew_e = np.concatenate(self.robot.link_velocity(frame="local"))

        test_link_idx = self.robot.get_link_index("extra_link")
        V_ew_e_test = np.concatenate(
            self.robot.link_velocity(link_idx=test_link_idx, frame="local")
        )
        Δr = np.array([0, 0, 0.2])
        ω = V_ew_e[3:]

        # print(f"ΔV = {V_ew_e + np.concatenate((np.cross(ω, Δr), np.zeros(3))) - V_ew_e_test}")

        # rotate command into the body frame
        A_ew_e_cmd = block_diag(C_we, C_we).T @ A_ew_w_cmd

        # body-frame gravity
        G_e = utils.body_gravity6(C_we.T)

        W = core.math.skew3(ω)
        X = np.block([[np.eye(3), -core.math.skew3(Δr)], [np.zeros((3, 3)), np.eye(3)]])
        d = np.concatenate((W @ W @ Δr, np.zeros(3)))

        # A_ew_w_cmd = np.linalg.solve(X, A_ew_e_cmd - d)

        return self._setup_and_solve_qp(G_e, V_ew_e, A_ew_e_cmd, δ, J, v, **kwargs)


class NominalReactiveController(ReactiveBalancingController):
    """Reactive controller with no balancing constraints."""

    def __init__(self, model, dt):
        super().__init__(model, dt)

        # cost
        self.P_sparse = sparse.csc_matrix(self.P)

    def _setup_and_solve_qp(self, G_e, V_ew_e, A_ew_e_cmd, δ, J, v):
        nv = 15
        x0 = self._initial_guess(nv, A_ew_e_cmd)

        # map joint acceleration to EE acceleration
        A_eq = np.hstack((-J, np.eye(6)))
        b_eq = δ

        # compute the cost: 0.5 * x @ P @ x + q @ x
        q = self._compute_q(nv, v, A_ew_e_cmd)

        return self._solve_qp(
            P=self.P_sparse,
            q=q,
            A=sparse.csc_matrix(A_eq),
            b=b_eq,
            lb=self.lb,
            ub=self.ub,
            x0=x0,
        )


class NominalReactiveBalancingController(ReactiveBalancingController):
    """Reactive balancing controller that explicitly computes contact forces."""

    def __init__(self, model, dt, **kwargs):
        super().__init__(model, dt, **kwargs)

        nc = 3 * len(self.contacts)

        # add weight on the tangential forces
        ft_weight = 0
        Pf = np.zeros((len(self.contacts), 3))
        Pf[:, :2] = ft_weight
        Pf = np.diag(Pf.flatten())

        # cost
        self.P = block_diag(self.P, Pf)
        self.P_sparse = sparse.csc_matrix(self.P)

        # optimization bounds
        self.lb = np.concatenate((self.lb, -np.inf * np.ones(nc)))
        self.ub = np.concatenate((self.ub, np.inf * np.ones(nc)))

        # pre-compute part of equality constraints
        self.A_eq_bal = np.hstack((np.zeros((self.M.shape[0], 9)), self.M, self.W))

        # inequality constraints
        F = block_diag(*[c.F for c in self.contacts])
        self.A_ineq = np.hstack((np.zeros((F.shape[0], 9 + 6)), F))
        self.A_ineq_sparse = sparse.csc_matrix(self.A_ineq)

    def _setup_and_solve_qp(self, G_e, V_ew_e, A_ew_e_cmd, δ, J, v, fixed_α=False):
        # number of optimization variables
        nv = 15 + 3 * len(self.contacts)
        x0 = self._initial_guess(nv, A_ew_e_cmd)

        # compute the equality constraints A_eq @ x == b
        # N-E equations for object balancing
        h = np.concatenate([obj.bias(V_ew_e) for obj in self.objects.values()])
        b_eq_bal = self.M @ G_e - h

        # map joint acceleration to EE acceleration
        A_eq_track = np.hstack((-J, np.eye(6), np.zeros((6, self.W.shape[1]))))
        b_eq_track = δ

        if fixed_α:
            A_α_fixed = np.zeros((3, nv))
            A_α_fixed[:, 12:15] = np.eye(3)
            b_α_fixed = A_ew_e_cmd[3:]

            A_eq_track = np.vstack((A_eq_track, A_α_fixed))
            b_eq_track = np.concatenate((b_eq_track, b_α_fixed))

        A_eq = np.vstack((A_eq_track, self.A_eq_bal))
        b_eq = np.concatenate((b_eq_track, b_eq_bal))

        # compute the cost: 0.5 * x @ P @ x + q @ x
        q = self._compute_q(nv, v, A_ew_e_cmd)

        return self._solve_qp(
            P=self.P_sparse,
            q=q,
            G=self.A_ineq_sparse,
            h=np.zeros(self.A_ineq.shape[0]),
            A=sparse.csc_matrix(A_eq),
            b=b_eq,
            lb=self.lb,
            ub=self.ub,
            x0=x0,
        )


class NominalReactiveBalancingControllerTilting(ReactiveBalancingController):
    """Reactive balancing controller that explicitly computes contact forces."""

    def __init__(
        self, model, dt, kθ=1, kω=1, a_cart_max=5, α_cart_max=1, a_joint_max=5, **kwargs
    ):
        super().__init__(model, dt, **kwargs)

        self.no = len(self.objects)
        self.coms = np.array([o.body.com for o in self.objects.values()])

        # number of contact force variables
        self.nc = 3 * len(self.contacts)

        # number of optimization variables
        self.nv = 9 + 6 + 6 * self.no + self.nc

        # add weight on the tangential forces
        ft_weight = 0
        Pf = np.zeros((len(self.contacts), 3))
        Pf[:, :2] = ft_weight
        Pf = np.diag(Pf.flatten())

        # angular control gains
        self.kθ = kθ
        self.kω = kω

        # optimization variables are (u, ae, αe, {αd}_{i=1}^no, {ao}_{i=1}^no,
        # ξ), where ξ are the contact forces

        # cost
        I3 = np.eye(3)
        dωdt_weight = np.eye(3 * (1 + self.no))
        dωdt_weight[:3, 3:] = np.tile(-I3, self.no)
        dωdt_weight[3:, :3] = np.tile(-I3, self.no).T
        dvodt_weight = np.eye(3 * self.no)

        P_cart = block_diag(I3, dωdt_weight / self.no, dvodt_weight)
        self.P = block_diag(self.P_joint, P_cart, Pf)
        self.P_sparse = sparse.csc_matrix(self.P)

        # bounds
        self.ub = np.concatenate(
            (
                a_joint_max * np.ones(9),
                a_cart_max * np.ones(3),
                α_cart_max * np.ones(3 * (1 + self.no)),
                a_cart_max * np.ones(3 * self.no),
                np.inf * np.ones(self.nc),
            )
        )
        self.lb = -self.ub

        # pre-compute part of equality constraints
        nM = self.M.shape[0]
        self.A_eq_bal = np.hstack(
            (np.zeros((nM, 9)), self.M, np.zeros((nM, 6 * self.no)), self.W)
        )

        # inequality constraints
        F = block_diag(*[c.F for c in self.contacts])
        self.A_ineq = np.hstack((np.zeros((F.shape[0], self.nv - self.nc)), F))
        self.A_ineq_sparse = sparse.csc_matrix(self.A_ineq)

    def _setup_and_solve_qp(self, G_e, V_ew_e, A_ew_e_cmd, δ, J, v, fixed_α=None):
        assert fixed_α is None

        x0 = self._initial_guess(self.nv, A_ew_e_cmd)

        # compute the equality constraints A_eq @ x == b
        # N-E equations for object balancing
        h = np.concatenate([obj.bias(V_ew_e) for obj in self.objects.values()])
        b_eq_bal = self.M @ G_e - h

        # map joint acceleration to EE acceleration
        A_eq_track = np.hstack((-J, np.eye(6), np.zeros((6, 6 * self.no + self.nc))))
        b_eq_track = δ

        # equality constraints for new consistency stuff
        I3 = np.eye(3)
        z = np.array([0, 0, 1])
        A_eq_tilt1 = np.hstack(
            (
                np.zeros((3 * self.no, 9)),  # joints
                np.tile(-I3, (self.no, 1)),  # dvedt
                np.vstack([core.math.skew3(c) for c in self.coms]),  # dωedt
                np.zeros((3 * self.no, 3 * self.no)),  # dωddt
                np.eye(3 * self.no),  # dvodt
                np.zeros((3 * self.no, self.nc)),  # contact forces
            )
        )
        ω_ew_e = V_ew_e[3:]
        b_eq_tilt1 = np.concatenate(
            [np.cross(ω_ew_e, np.cross(ω_ew_e, c)) for c in self.coms]
        )

        A_eq_tilt2 = np.hstack(
            (
                np.zeros((3 * self.no, 9 + 6)),
                np.eye(3 * self.no),
                np.kron(np.eye(self.no), -self.kθ * core.math.skew3(z)),
                np.zeros((3 * self.no, self.nc)),
            )
        )
        g = G_e[:3]
        b_eq_tilt2 = np.tile(-self.kω * ω_ew_e - self.kθ * np.cross(z, g), self.no)

        A_eq = np.vstack((A_eq_track, self.A_eq_bal, A_eq_tilt1, A_eq_tilt2))
        b_eq = np.concatenate((b_eq_track, b_eq_bal, b_eq_tilt1, b_eq_tilt2))

        # compute the cost: 0.5 * x @ P @ x + q @ x
        q0 = self._compute_q(self.nv, v, A_ew_e_cmd)
        q = np.zeros(self.nv)
        q[:12] = q0[:12]

        return self._solve_qp(
            P=self.P_sparse,
            q=q,
            G=self.A_ineq_sparse,
            h=np.zeros(self.A_ineq.shape[0]),
            A=sparse.csc_matrix(A_eq),
            b=b_eq,
            lb=self.lb,
            ub=self.ub,
            x0=x0,
        )


class NominalReactiveBalancingControllerFaceForm(ReactiveBalancingController):
    """Reactive balancing controller without explicit contact force computation."""

    def __init__(self, model, dt):
        super().__init__(model, dt)

        self.P_sparse = sparse.csc_matrix(self.P)

        self.A_ineq = np.hstack((np.zeros((self.F.shape[0], 9)), self.F @ self.M))
        self.A_ineq_sparse = sparse.csc_matrix(self.A_ineq)

    def _setup_and_solve_qp(self, G_e, V_ew_e, A_ew_e_cmd, δ, J, v):
        nv = 15
        x0 = self._initial_guess(nv, A_ew_e_cmd)

        # map joint acceleration to EE acceleration
        A_eq = np.hstack((-J, np.eye(6)))
        b_eq = δ

        # compute the inequality constraints A_ineq @ x <= b_ineq
        h = np.concatenate([obj.bias(V_ew_e) for obj in self.objects.values()])
        b_ineq = self.F @ (self.M @ G_e - h)

        # compute the cost: 0.5 * x @ P @ x + q @ x
        q = self._compute_q(nv, v, A_ew_e_cmd)

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


class RobustReactiveBalancingController(ReactiveBalancingController):
    def __init__(self, model, dt, **kwargs):
        super().__init__(model, dt, **kwargs)

        self.P_sparse = sparse.csc_matrix(self.P)

        # polytopic uncertainty Pθ + p >= 0
        P = block_diag(*[obj.P for obj in self.objects.values()])
        p = np.concatenate([obj.p for obj in self.objects.values()])

        # fmt: off
        self.P_tilde = np.block([
            [P, p[:, None]],
            [np.zeros((1, P.shape[1])), np.array([[-1]])]])
        # fmt: on
        self.R = utils.span_to_face_form(self.P_tilde.T)[0]
        # self.R = self.R / np.max(np.abs(self.R))

        # pre-compute inequality matrix
        nv = 15
        nf = self.F.shape[0]
        n_ineq = self.R.shape[0]
        N_ineq = n_ineq * nf
        self.A_ineq = np.zeros((N_ineq, nv))
        for i in range(nf):
            D = utils.body_regressor_A_by_vector(self.F[i, :])
            D_tilde = np.vstack((D, np.zeros((1, D.shape[1]))))
            self.A_ineq[i * n_ineq : (i + 1) * n_ineq, 9:] = -self.R @ D_tilde
        # self.A_ineq_max = np.max(np.abs(self.A_ineq))
        # self.A_ineq = self.A_ineq / self.A_ineq_max
        self.A_ineq_sparse = sparse.csc_matrix(self.A_ineq)

    def _setup_and_solve_qp(self, G_e, V_ew_e, A_ew_e_cmd, δ, J, v, fixed_α=False):
        nv = 15
        x0 = self._initial_guess(nv, A_ew_e_cmd)

        # map joint acceleration to EE acceleration
        A_eq = np.hstack((-J, np.eye(6)))
        b_eq = δ

        if fixed_α:
            A_α_fixed = np.zeros((3, nv))
            A_α_fixed[:, 12:15] = np.eye(3)
            b_α_fixed = A_ew_e_cmd[3:]

            A_eq = np.vstack((A_eq, A_α_fixed))
            b_eq = np.concatenate((b_eq, b_α_fixed))

        # build robust constraints
        # t0 = time.time()
        B = utils.body_regressor_VG_by_vector_tilde_vectorized(V_ew_e, G_e, self.F)
        b_ineq = (self.R @ B).T.flatten()  # / self.A_ineq_max
        # t1 = time.time()
        # print(f"build time = {1000 * (t1 - t0)} ms")

        # compute the cost: 0.5 * x @ P @ x + q @ x
        q = self._compute_q(nv, v, A_ew_e_cmd)

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
