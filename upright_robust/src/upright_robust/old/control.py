import abc
import time

import numpy as np
from scipy import sparse
from scipy.linalg import block_diag, null_space
from qpsolvers import solve_qp, Problem, solve_problem
import proxsuite

import upright_core as core
import upright_robust.utils as utils
import upright_robust.modelling as mdl
from upright_robust.bindings import RobustBounds
from upright_robust.qp_utils import QPUpdateMask

import IPython


# TODO inconsistent use of dedicated proxqp interface
# TODO support different weights on different joints
# TODO clean up the recent messy implementation of the fact that only x and y
# components of angular acceleration should be constrained, and z component
# should be free but weighted

# use a dedicated interface to proxQP rather than going through qpsolvers
# this enables better warm-starting from timestep to timestep
USE_DEDICATED_PROXQP_INTERFACE = True

# constrain EE angular acceleration to be exactly equal to the desired value
# resulting from the adaptive tilting law
EQUALITY_CONSTRAIN_ANGULAR_ACC = True

# constrain the maximum tilt angle to be no more than the specified value
USE_MAX_TILT_ANGLE_CONSTRAINT = False

EXTRA_EE_ANG_ACC_WEIGHT = 1.0


def _make_matrix_weights(weight, dim):
    if np.isscalar(weight):
        return weight * np.eye(dim)
    if weight.ndim == 1:
        assert weight.shape == (dim,)
        return np.diag(weight)
    elif weight.ndim == 2:
        assert weight.shape == (dim, dim)
        return weight
    raise ValueError(f"Unexpected weight value {weight} with dim {dim}.")


class ReactiveBalancingController(abc.ABC):
    """Base class for reactive balancing control.

    Parameters
    ----------
    robot : RobotModel
        Robot model.
    objects : Iterable[UncertainObject]
        The balanced objects.
    contacts : Iterable[RobustContactPoint]
        The contact points between objects.
    dt: float, non-negative
        The controller timestep.

    a_cart_weight : float, non-negative
        The weight on the linear EE acceleration.
    α_cart_weight : float, non-negative
        The weight on the angular EE acceleration.
    v_joint_weight : float, non-negative
        The weight on the joint velocity.
    a_joint_weight : float, non-negative
        The weight on the joint acceleration.
    j_joint_weight : float, non-negative
        The weight on the joint jerk.

    a_cart_max : float, non-negative
        Maximum EE linear acceleration (in each dimension).
    α_cart_max : float, non-negative
        Maximum EE angular acceleration (in each dimension).
    v_cart_max : float, non-negative
        Maximum EE linear velocity (in each dimension).
    ω_cart_max : float, non-negative
        Maximum EE angular velocity (in each dimension).
    a_joint_max : float, non-negative
        Maximum joint acceleration.
    v_joint_max : float, non-negative
        Maximum joint velocity.

    use_slack : bool
        Set ``true`` to use slack variables to relax the equality between joint
        space and task space acceleration.
    slack_weight : float, non-negative
        The weight on the slack variables.
    """

    def __init__(
        self,
        robot,
        objects,
        contacts,
        dt,
        a_cart_weight=1,
        α_cart_weight=5,
        a_joint_weight=0.01,
        v_joint_weight=3,
        j_joint_weight=0,
        slack_weight=10,
        a_cart_max=5,
        α_cart_max=1,
        v_cart_max=1,
        ω_cart_max=1,
        tilt_angle_max=np.deg2rad(15),
        a_joint_max=5,
        v_joint_max=1,
        use_slack=True,
        solver="proxqp",
    ):
        self.solver = solver
        self.robot = robot
        self.objects = objects
        self.contacts = contacts
        self.dt = dt
        self.use_slack = use_slack

        self.no = len(self.objects)  # number of objects
        self.nc = len(self.contacts)  # number of contacts
        self.ns = use_slack * 6  # number of slack variables
        self.nu = 9  # number of inputs
        self.nv0 = self.nu + 6 + self.ns  # default number of decision variables

        # canonical map of object names to indices
        names = list(self.objects.keys())
        self.object_name_index = mdl.compute_object_name_index(names)

        # default optimization variables are (u, A, s), where s is the optimal
        # slack variables on the EE acceleration kinematics constraint

        # here we are penalizing ||u_k||^2 directly but also ||v_{k+1}||^2 =
        # ||v_k + dt * u_k||^2
        # the linear term from the velocity penalty is computed in the
        # _compute_linear_cost_term method
        a_joint_weight = _make_matrix_weights(a_joint_weight, dim=self.nu)
        v_joint_weight = _make_matrix_weights(v_joint_weight, dim=self.nu)
        j_joint_weight = _make_matrix_weights(j_joint_weight, dim=self.nu)

        self.P_joint = (
            a_joint_weight + dt**2 * v_joint_weight + j_joint_weight / dt**2
        )
        self.P_cart = block_diag(a_cart_weight * np.eye(3), α_cart_weight * np.eye(3))
        self.P_slack = slack_weight * np.eye(self.ns)

        # shared optimization weight
        self.v_joint_weight = v_joint_weight
        self.j_joint_weight = j_joint_weight
        self.a_cart_weight = a_cart_weight
        self.α_cart_weight = α_cart_weight
        self.slack_weight = slack_weight

        # slices for decision variables
        self.su = np.s_[: self.nu]
        self.ss = np.s_[self.nu : self.nu + self.ns]
        self.sa = np.s_[self.nu + self.ns : self.nu + self.ns + 3]
        self.sα = np.s_[self.nu + self.ns + 3 : self.nu + self.ns + 6]
        self.sA = np.s_[self.nu + self.ns : self.nu + self.ns + 6]

        # shared optimization bounds
        # task space
        self.a_cart_max = a_cart_max
        self.α_cart_max = α_cart_max
        self.v_cart_max = v_cart_max
        self.ω_cart_max = ω_cart_max
        self.tilt_angle_max = tilt_angle_max

        # joint space
        self.v_joint_max = v_joint_max
        self.a_joint_max = a_joint_max

        self.ub = np.zeros(self.nv0)
        self.ub[self.su] = a_joint_max  # inputs
        self.ub[self.ss] = np.inf  # slacks
        self.ub[self.sa] = a_cart_max  # linear EE acceleration
        self.ub[self.sα] = α_cart_max  # angular EE acceleration
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

        self.qp = None
        self.qp_prof = utils.RunningAverage()

    def update(self, q, v):
        self.robot.update(q, v)

    def _solve_qp_prox(self, P, q, G=None, h=None, A=None, b=None, lb=None, ub=None):
        """Solve QP using proxqp's dense solver."""
        if self.qp is None:
            nv = P.shape[0]
            n_eq = A.shape[0] if A is not None else 0
            n_ineq = G.shape[0] if G is not None else 0
            use_box = lb is not None
            self.qp = proxsuite.proxqp.dense.QP(nv, n_eq, n_ineq, use_box)
            # self.qp.settings.initial_guess = proxsuite.proxqp.WARM_START_WITH_PREVIOUS_RESULT
            l = -np.infty * np.ones_like(h)
            self.qp.init(P, q, A, b, G, l, h, lb, ub)
        else:
            l = None  # doesn't change
            P, q, A, b, G, h, lb, ub = self.update_mask.update(P, q, A, b, G, h, lb, ub)
            self.qp.update(P, q, A, b, G, l, h, lb, ub)

        # by default proxqp uses the previous solution as a warm-start
        self.qp.solve()
        x = self.qp.results.x
        u = x[self.su]
        A = x[self.sA]
        return u, A

    def _solve_qp(
        self, P, q, G=None, h=None, A=None, b=None, lb=None, ub=None, x0=None
    ):
        if USE_DEDICATED_PROXQP_INTERFACE:
            # be consistent! only use _solve_qp_prox when the flag is set
            raise ValueError(
                "Called _solve_qp but USE_DEDICATED_PROXQP_INTERFACE is true."
            )

        if x0 is None:
            x0 = self.x_last

        t0 = time.perf_counter_ns()
        problem = Problem(P=P, q=q, G=G, h=h, A=A, b=b, lb=lb, ub=ub)
        solution = solve_problem(
            problem,
            initvals=x0,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=10,  # TODO play with this
            solver=self.solver,
        )
        t1 = time.perf_counter_ns()
        x = solution.x
        self.qp_prof.update(t1 - t0)

        self.x_last = x
        u = x[self.su]
        A = x[self.sA]
        return u, A

    def _initial_guess(self, a_ew_e_cmd):
        """Compute the initial guess for the QP."""
        x0 = np.zeros(self.nv)
        x0[self.sa] = a_ew_e_cmd
        return x0

    def _compute_linear_cost_term(self, v, a_ew_e_cmd, u_last):
        """Compute the linear term of the QP's cost function.

        Parameters
        ----------
        v : np.ndarray, shape (nu,)
            The current joint velocity.
        a_ew_w_cmd : np.ndarray, shape (3,)
            The commanded EE linear acceleration.
        u_last : np.ndarray, shape (9,)
            The current joint acceleration (i.e., from the last timestep).

        Returns
        -------
        : np.ndarray shape (nv,)
            The linear term ``q`` of the QP's cost function.
        """
        q = np.zeros(self.nv)

        # joint acceleration
        q[self.su] = (
            self.dt * self.v_joint_weight @ v
            + self.j_joint_weight @ u_last / self.dt**2
        )

        # EE acceleration
        # note there is no linear term on the EE angular because α and α_d are
        # both decision variables
        q[self.sa] = -self.a_cart_weight * a_ew_e_cmd
        return q

    def _compute_bounds(self, v, V):
        """Compute bounds for the problem.

        We assume the bounds are fixed except for the joint acceleration, which
        depends on the current joint velocity.

        Parameters
        ----------
        v : np.ndarray, shape (nu,)
            The current joint velocity.
        V : np.ndarray, shape (6,)
            The current EE velocity.

        Returns
        -------
        : tuple
            The pair ``(lb, ub)``, representing the lower and upper bounds (box
            constraints) on the QP optimization variables.
        """
        lb = self.lb.copy()
        ub = self.ub.copy()

        # joint acceleration bounds
        # these combine the true acceleration bounds and the velocity joints
        # obtained using a first-order approximation
        lb[self.su] = np.maximum(-self.a_joint_max, (-self.v_joint_max - v) / self.dt)
        ub[self.su] = np.minimum(self.a_joint_max, (self.v_joint_max - v) / self.dt)

        # EE acceleration bounds
        lb[self.sa] = np.maximum(-self.a_cart_max, (-self.v_cart_max - V[:3]) / self.dt)
        ub[self.sa] = np.minimum(self.a_cart_max, (self.v_cart_max - V[:3]) / self.dt)
        lb[self.sα] = np.maximum(-self.α_cart_max, (-self.ω_cart_max - V[3:]) / self.dt)
        ub[self.sα] = np.minimum(self.α_cart_max, (self.ω_cart_max - V[3:]) / self.dt)

        return lb, ub

    def _compute_tilt_angle_constraint(self, C_we, V_ew_e):
        """Compute inequality constraint on the tray's maximum tilt angle.

        Parameters
        ----------
        C_we : np.ndarray, shape (3, 3)
            Rotation matrix representing the tray's orientation w.r.t. the world.
        V_ew_e : np.ndarray, shape (6,)
            The body-frame velocity twist of the tray.

        Returns
        -------
        : tuple
            The pair ``(A, b)`` representing the tilt angle constraint as ``Ax <= b``.
        """
        z_e = np.array([0, 0, 1])
        z_w = C_we @ z_e

        ω_ew_e = V_ew_e[3:]
        ω_ew_w = C_we @ ω_ew_e
        W = core.math.skew3(ω_ew_w)

        b = z_e @ (np.eye(3) + self.dt * W + 0.5 * self.dt**2 * W @ W) @ z_w - np.cos(
            self.tilt_angle_max
        )
        A = np.zeros(self.nv)
        A[self.sα] = 0.5 * self.dt**2 * z_e @ core.math.skew3(z_w)
        return A[None, :], np.array([b])

    def _compute_tracking_equalities(self, J, δ):
        """Compute matrix (A, b) such that Ax == b enforces EE acceleration tracking,
        with x the decision variables."""
        if self.use_slack:
            S = np.eye(self.ns)
        else:
            S = np.zeros((6, self.ns))
        A = np.hstack((-J, S, np.eye(6), np.zeros((6, self.nv - self.nv0))))
        b = δ

        # NOTE: constraint q[-2] to not accelerate
        # A2 = np.zeros((1, self.nv))
        # A2[0, self.nu - 2] = 1
        # b2 = np.array([0])
        # A = np.vstack((A, A2))
        # b = np.concatenate((b, b2))

        return A, b

    @abc.abstractmethod
    def _setup_and_solve_qp(
        self, C_we, G_e, V_ew_e, a_ew_e_cmd, δ, J, v, u_last, **kwargs
    ):
        pass

    def solve(self, q, v, a_ew_w_cmd, u_last, **kwargs):
        """Solve for an updated joint acceleration command given current robot
        state (q, v) and desired EE (world-frame) acceleration A_ew_w_cmd.
        """
        self.robot.forward(q, v)
        J = self.robot.jacobian(q, frame="local")

        # δ = \dot{J}v, where J is the robot Jacobian and v are the joint
        # velocities, since we are computing the EE acceleration with zero
        # joint acceleration
        δ = np.concatenate(self.robot.link_classical_acceleration(frame="local"))

        C_we = self.robot.link_pose(rotation_matrix=True)[1]
        V_ew_e = np.concatenate(self.robot.link_velocity(frame="local"))

        # rotate command into the body frame
        a_ew_e_cmd = C_we.T @ a_ew_w_cmd

        # body-frame gravity
        G_e = utils.body_gravity6(C_we.T)

        return self._setup_and_solve_qp(
            C_we, G_e, V_ew_e, a_ew_e_cmd, δ, J, v, u_last, **kwargs
        )


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
            self.A_eq_bal = np.hstack(
                (np.zeros((self.M.shape[0], self.nu + self.ns)), self.M, self.W)
            )

            # inequality constraints for balancing
            F = block_diag(*[c.F for c in self.contacts])
            self.A_ineq = np.hstack((np.zeros((F.shape[0], self.nu + self.ns + 6)), F))
            self.A_ineq_sparse = sparse.csc_matrix(self.A_ineq)
            self.b_ineq = np.zeros(self.A_ineq.shape[0])
        else:
            self.nf = 0
            self.A_ineq_sparse = None
            self.b_ineq = None

        # number of optimization variables: 9 joint accelerations, 6 slack, 6
        # EE acceleration twist, nf force
        self.nv = self.nv0 + self.nf

        # cost
        P_force = np.zeros((self.nf, self.nf))
        self.P = block_diag(self.P_joint, self.P_slack, self.P_cart, P_force)
        self.P_sparse = sparse.csc_matrix(self.P)

        # optimization bounds
        self.lb = np.concatenate((self.lb, -np.inf * np.ones(self.nf)))
        self.ub = np.concatenate((self.ub, np.inf * np.ones(self.nf)))

    def _setup_and_solve_qp(self, C_we, G_e, V_ew_e, a_ew_e_cmd, δ, J, v, u_last):
        x0 = self._initial_guess(a_ew_e_cmd)

        # map joint acceleration to EE acceleration
        A_eq_track, b_eq_track = self._compute_tracking_equalities(J, δ)

        # keep angular acceleration fixed to zero
        A_α_fixed = np.zeros((3, self.nv))
        A_α_fixed[:, self.sα] = np.eye(3)
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
        q = self._compute_linear_cost_term(v, a_ew_e_cmd, u_last)
        lb, ub = self._compute_bounds(v, V_ew_e)

        return self._solve_qp(
            P=self.P_sparse,
            q=q,
            G=self.A_ineq_sparse,
            h=self.b_ineq,
            A=sparse.csc_matrix(A_eq),
            b=b_eq,
            lb=lb,
            ub=ub,
            x0=x0,
        )


class NominalReactiveBalancingControllerTrayTilting(ReactiveBalancingController):
    """Reactive balancing controller that tilts so that the tray is normal to
    total acceleration.

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
        **kwargs,
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
                (np.zeros((nM, self.nu + self.ns)), self.M, np.zeros((nM, 3)), self.W)
            )

            F = block_diag(*[c.F for c in self.contacts])
            self.A_ineq = np.hstack((np.zeros((F.shape[0], self.nu + self.ns + 9)), F))
            self.A_ineq_sparse = sparse.csc_matrix(self.A_ineq)
            self.b_ineq = np.zeros(self.A_ineq.shape[0])
        else:
            self.nf = 0
            self.A_ineq_sparse = None
            self.b_ineq = None

        # number of optimization variables: 9 joint accelerations, 6 slack, 6
        # EE acceleration twist, 3 desired angular acceleration, nf force
        self.nv = self.nv0 + 3 + self.nf

        # cost
        I3 = np.eye(3)
        P_a_cart = self.a_cart_weight * I3
        P_α_cart = self.α_cart_weight * np.block([[I3, -I3], [-I3, I3]])
        P_force = np.zeros((self.nf, self.nf))
        P_cart = block_diag(P_a_cart, P_α_cart)
        self.P = block_diag(self.P_joint, self.P_slack, P_cart, P_force)
        self.P_sparse = sparse.csc_matrix(self.P)

        # bounds
        self.ub = np.concatenate(
            (self.ub, self.α_cart_max * np.ones(3), np.inf * np.ones(self.nf))
        )
        self.lb = -self.ub

    def _setup_and_solve_qp(self, C_we, G_e, V_ew_e, a_ew_e_cmd, δ, J, v, u_last):
        x0 = self._initial_guess(a_ew_e_cmd)

        # map joint acceleration to EE acceleration
        A_eq_track, b_eq_track = self._compute_tracking_equalities(J, δ)

        # equality constraints for new consistency stuff
        I3 = np.eye(3)
        z = np.array([0, 0, 1])
        A_eq_tilt = np.hstack(
            (
                np.zeros((3, self.nu + self.ns)),
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
        q = self._compute_linear_cost_term(v, a_ew_e_cmd, u_last)
        lb, ub = self._compute_bounds(v, V_ew_e)

        return self._solve_qp(
            P=self.P_sparse,
            q=q,
            G=self.A_ineq_sparse,
            h=self.b_ineq,
            A=sparse.csc_matrix(A_eq),
            b=b_eq,
            lb=lb,
            ub=ub,
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
        use_approx_robust_constraints=False,
        **kwargs,
    ):
        super().__init__(robot, objects, contacts, dt, **kwargs)

        # angular control gains
        self.kθ = kθ
        self.kω = kω
        self.use_dvdt_scaling = use_dvdt_scaling
        self.use_face_form = use_face_form
        self.use_robust_constraints = use_robust_constraints
        self.use_approx_robust_constraints = use_approx_robust_constraints

        self.nλ = 0

        # not the most elegantly implemented at the moment, but we can use
        # approximate robust constraints to compute the acceleration using the
        # nominal controller and then scale it to satisfy all robust
        # constraints
        assert not (use_approx_robust_constraints and use_robust_constraints)

        # TODO reduce duplication with subsequent code
        if use_approx_robust_constraints:
            # polytopic parameter uncertainty
            P = block_diag(*[obj.P for obj in self.objects.values()])
            p = np.concatenate([obj.p for obj in self.objects.values()])
            self.P_tilde = mdl.compute_P_tilde_matrix(P, p)

            print("Computing inertial parameter face form...")
            R = utils.span_to_face_form(self.P_tilde.T)
            print("...done.")

            n_face = self.face.shape[0]
            n_ineq = R.shape[0]
            self.A_ineq_rob = np.zeros((n_ineq * n_face, 6))
            for i in range(n_face):
                D = utils.body_regressor_A_by_vector(self.face[i, :])
                D_tilde = np.vstack((D, np.zeros((1, D.shape[1]))))
                self.A_ineq_rob[i * n_ineq : (i + 1) * n_ineq, :] = R @ D_tilde

            self.RT = R[:, :-1].T

            raise NotImplementedError(
                "robust bounds probably needs adjustment since I switched P and R signs"
            )
            self.robust_bounds = RobustBounds(self.RT, self.face, self.A_ineq_rob)

        if use_robust_constraints:
            # robust constraints without using DD to remove the extra dual variables
            self.nf = 0

            # polytopic parameter uncertainty
            P = block_diag(*[obj.P for obj in self.objects.values()])
            p = np.concatenate([obj.p for obj in self.objects.values()])
            self.P_tilde = mdl.compute_P_tilde_matrix(P, p)

            if use_face_form:
                print("Computing inertial parameter face form...")
                self.R = utils.span_to_face_form(self.P_tilde.T)
                print("...done.")

                self.RT = self.R[:, :-1].T

                # self.R /= np.max(np.abs(self.R))

                # pre-compute inequality matrix
                n_face = self.face.shape[0]
                n_ineq = self.R.shape[0]
                self.A_ineq = np.zeros((n_ineq * n_face, self.nv0 + 6 * self.no))
                for i in range(n_face):
                    D = utils.body_regressor_A_by_vector(self.face[i, :])
                    D_tilde = np.vstack((D, np.zeros((1, D.shape[1]))))
                    self.A_ineq[i * n_ineq : (i + 1) * n_ineq, self.sA] = (
                        self.R @ D_tilde
                    )

                # self.A_ineq_max = np.max(np.abs(self.A_ineq))
                # self.A_ineq /= self.A_ineq_max
                self.A_eq_bal = np.zeros((0, self.nv0 + 6 * self.no))
            else:
                self.P_inv = np.linalg.pinv(self.P_tilde.T)
                P_null = null_space(self.P_tilde.T)

                # number of extra dual variables
                n_face = self.face.shape[0]
                n_dual = P_null.shape[1]  # self.P_tilde.shape[0]
                self.nλ = n_dual * n_face

                # equalities tying together the dual variables and object
                # accelerations
                n_ineq = P_null.shape[0]  # self.P_tilde.shape[1]
                self.A_eq_bal = np.zeros(
                    (n_ineq * n_face, self.nv0 + 6 * self.no + self.nλ)
                )
                self.A_ineq = np.zeros(
                    (n_ineq * n_face, self.nv0 + 6 * self.no + self.nλ)
                )
                for i in range(n_face):
                    D = utils.body_regressor_A_by_vector(self.face[i, :])
                    D_tilde = np.vstack((D, np.zeros((1, D.shape[1]))))
                    self.A_ineq[i * n_ineq : (i + 1) * n_ineq, self.sA] = (
                        self.P_inv @ D_tilde
                    )
                    self.A_ineq[
                        i * n_ineq : (i + 1) * n_ineq,
                        self.nv0
                        + 6 * self.no
                        + i * n_dual : 15
                        + 6 * self.no
                        + (i + 1) * n_dual,
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
                self.A_eq_bal = np.zeros((0, self.nv0 + 6 * self.no + self.nλ))

        elif use_face_form:
            self.nf = 0
            n_face = self.face.shape[0]
            self.A_ineq = np.hstack(
                (
                    np.zeros((n_face, self.nu + self.ns)),
                    self.face @ self.M,
                    np.zeros((n_face, 6 * self.no)),
                )
            )

            # no balancing equality constraints
            self.A_eq_bal = np.zeros((0, self.nv0 + 6 * self.no))

        else:
            self.nf = 3 * self.nc

            # pre-compute part of equality constraints
            nM = self.M.shape[0]
            self.A_eq_bal = np.hstack(
                (
                    np.zeros((nM, self.nu + self.ns)),
                    self.M,
                    np.zeros((nM, 6 * self.no)),
                    self.W,
                )
            )

            # inequality constraints
            F = block_diag(*[c.F for c in self.contacts])
            self.A_ineq = np.hstack((np.zeros((F.shape[0], self.nv0 + 6 * self.no)), F))

        self.A_ineq_sparse = sparse.csc_matrix(self.A_ineq)

        # equality constraint for linear acceleration at object CoMs
        I3 = np.eye(3)
        self.coms = np.array([o.body.com for o in self.objects.values()])
        self.A_eq_tilt1 = np.hstack(
            (
                np.zeros((3 * self.no, self.nu + self.ns)),  # joints
                np.tile(-I3, (self.no, 1)),  # dvedt
                np.vstack([core.math.skew3(c) for c in self.coms]),  # dωedt
                np.zeros((3 * self.no, 3 * self.no)),  # dωddt
                np.eye(3 * self.no),  # dvodt
                np.zeros((3 * self.no, self.nf + self.nλ)),  # contact forces
            )
        )

        # number of optimization variables: 9 joint accelerations, 6 EE
        # acceleration twist, no * desired angular acceleration, no * object
        # linear acceleration, nf force
        self.nv = self.nv0 + 6 * self.no + self.nf + self.nλ

        # cost
        P_α_cart = np.eye(3 * (1 + self.no))
        P_α_cart[:3, :3] = self.no * I3
        P_α_cart[:3, 3:] = np.tile(-I3, self.no)
        P_α_cart[3:, :3] = np.tile(-I3, self.no).T
        P_α_cart = self.α_cart_weight * P_α_cart / self.no
        P_α_cart[2, 2] += EXTRA_EE_ANG_ACC_WEIGHT
        P_cart = block_diag(
            self.a_cart_weight * I3, P_α_cart, self.a_cart_weight * np.eye(3 * self.no)
        )

        P_force = 0 * np.eye(self.nf + self.nλ)
        self.P = block_diag(self.P_joint, self.P_slack, P_cart, P_force)
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

        self.update_mask = QPUpdateMask(P=False, G=False)
        self.approx_scale_prof = utils.RunningAverage()

    def _setup_and_solve_qp(self, C_we, G_e, V_ew_e, a_ew_e_cmd, δ, J, v, u_last):
        x0 = self._initial_guess(a_ew_e_cmd)

        # compute the equality constraints A_eq @ x == b
        # N-E equations for object balancing
        h = np.concatenate([obj.bias(V_ew_e) for obj in self.objects.values()])

        # map joint acceleration to EE acceleration
        A_eq_track, b_eq_track = self._compute_tracking_equalities(J, δ)

        # equality constraints for new consistency stuff
        # eq_tilt1 is for the linear acceleration at the object CoM
        ω_ew_e = V_ew_e[3:]
        b_eq_tilt1 = np.concatenate(
            [np.cross(ω_ew_e, np.cross(ω_ew_e, c)) for c in self.coms]
        )

        g = G_e[:3]
        scale = self.kθ
        if self.use_dvdt_scaling:
            scale /= np.linalg.norm(a_ew_e_cmd - g)

        # eq_tilt2 is for angular acceleration
        z = np.array([0, 0, 1])
        A_eq_tilt2 = np.hstack(
            (
                # np.zeros((3 * self.no, self.nv0)),
                # np.eye(3 * self.no),
                # np.kron(np.eye(self.no), -scale * core.math.skew3(z)),
                # np.zeros((3 * self.no, self.nf + self.nλ)),
                # TODO assumes one object
                np.zeros((2, self.nv0)),
                np.eye(2),
                np.zeros((2, 1)),
                -scale * core.math.skew3(z)[:2, :],  # \dot{v}_o
                np.zeros((2, self.nf + self.nλ)),
            )
        )
        # b_eq_tilt2 = np.tile(-self.kω * ω_ew_e - scale * np.cross(z, g), self.no)
        b_eq_tilt2 = -self.kω * ω_ew_e[:2] - scale * np.cross(z, g)[:2]
        # TODO assumes one object
        # b_eq_tilt2 = -self.kω * ω_ew_e - scale * np.cross(z, g)
        # b_eq_tilt2[2] = -self.kω * ω_ew_e[2]

        if self.use_robust_constraints:
            if self.use_face_form:
                Y0 = utils.body_regressor(V_ew_e, -G_e)
                # TODO added the negative to self.RT with the sign switch---is
                # this correct?
                b_ineq = np.ravel(self.face @ block_diag(*[Y0] * self.no) @ -self.RT)
                # b_ineq = (self.R @ B).T.flatten()  # / self.A_ineq_max
                b_eq_bal = np.zeros(0)
            else:
                # b_ineq = np.zeros(0)
                # b_eq_bal = -B.T.flatten()
                B = utils.body_regressor_VG_by_vector_tilde_vectorized(
                    V_ew_e, G_e, self.face
                )
                b_ineq = -(self.P_inv @ B).T.flatten()
                b_eq_bal = np.zeros(0)
        elif self.use_face_form:
            b_ineq = self.face @ (self.M @ G_e - h)
            b_eq_bal = np.zeros(0)
        else:
            b_ineq = np.zeros(self.A_ineq.shape[0])
            b_eq_bal = self.M @ G_e - h

        if EQUALITY_CONSTRAIN_ANGULAR_ACC:
            # dωdt == dωdt_desired
            # only works for one object
            A_eq_ω = np.zeros((3, self.nv))
            A_eq_ω[:, self.sα] = -np.eye(3)
            A_eq_ω[:, self.nv0 : self.nv0 + 3] = np.eye(3)
            b_eq_ω = np.zeros(3)

            A_eq = np.vstack(
                (A_eq_track, self.A_eq_bal, self.A_eq_tilt1, A_eq_tilt2, A_eq_ω)
            )
            b_eq = np.concatenate(
                (b_eq_track, b_eq_bal, b_eq_tilt1, b_eq_tilt2, b_eq_ω)
            )
        else:
            A_eq = np.vstack((A_eq_track, self.A_eq_bal, self.A_eq_tilt1, A_eq_tilt2))
            b_eq = np.concatenate((b_eq_track, b_eq_bal, b_eq_tilt1, b_eq_tilt2))

        # tilt angle constraint
        # TODO perhaps could be cleaned up
        if USE_MAX_TILT_ANGLE_CONSTRAINT:
            A_ineq_tilt, b_ineq_tilt = self._compute_tilt_angle_constraint(C_we, V_ew_e)
            A_ineq = np.vstack((self.A_ineq, A_ineq_tilt))
            b_ineq = np.concatenate((b_ineq, b_ineq_tilt))
        else:
            A_ineq = self.A_ineq

        # compute the cost: 0.5 * x @ P @ x + q @ x
        q = self._compute_linear_cost_term(v, a_ew_e_cmd, u_last)
        lb, ub = self._compute_bounds(v, V_ew_e)

        if USE_DEDICATED_PROXQP_INTERFACE:
            u, A = self._solve_qp_prox(
                P=self.P,
                q=q,
                G=A_ineq,
                h=b_ineq,
                A=A_eq,
                b=b_eq,
                lb=lb,
                ub=ub,
            )
        else:
            u, A = self._solve_qp(
                P=self.P_sparse,
                q=q,
                G=sparse.csc_matrix(A_ineq),
                h=b_ineq,
                A=sparse.csc_matrix(A_eq),
                b=b_eq,
                lb=lb,
                ub=ub,
                x0=x0,
            )
        if self.use_approx_robust_constraints:
            t0 = time.perf_counter_ns()
            scale = self.robust_bounds.compute_scale(V_ew_e, -G_e, A)
            assert scale >= 0, "Scale must be non-negative."

            # # the implementation below is equivalent to but faster than:
            # # B = utils.body_regressor_VG_by_vector_vectorized(V_ew_e, G_e, self.face)
            # # b_ineq_rob = (self.R @ B).T.flatten()
            #
            # Y0 = utils.body_regressor(V_ew_e, -G_e)
            # b_ineq_rob = np.ravel(self.face @ block_diag(*[Y0] * self.no) @ self.RT)
            #
            # x = self.A_ineq_rob @ A
            # mask = x > b_ineq_rob
            # s = b_ineq_rob[mask] / x[mask]
            #
            # # short-circuit if there are no constraint violations
            # if s.shape == (0,):
            #     return u, A
            #
            # if np.any(s < 0):
            #     raise ValueError("Different signs in inequality constraints!")
            #
            # scale = np.min(s)
            sA = scale * A
            su = np.linalg.lstsq(J, sA - δ, rcond=None)[0]

            t1 = time.perf_counter_ns()
            self.approx_scale_prof.update(t1 - t0)
            print(f"scale avg = {self.approx_scale_prof.average / 1e6} ms")

            return su, sA
        return u, A