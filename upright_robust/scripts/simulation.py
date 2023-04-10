#!/usr/bin/env python3
"""Closed-loop upright reactive simulation using Pybullet."""
import datetime
import time

import numpy as np
import pybullet as pyb
from pyb_utils.frame import debug_frame_world
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy import sparse
from qpsolvers import solve_qp

from upright_core.logging import DataLogger, DataPlotter
import mobile_manipulation_central as mm
import upright_sim as sim
import upright_core as core
import upright_control as ctrl
import upright_cmd as cmd
import upright_robust as rob

import IPython


g = 9.81
G = np.array([0, 0, -g])


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


class UncertainObject:
    def __init__(self, obj, θ_min=None, θ_max=None):
        self.object = obj
        self.body = obj.body

        m = self.body.mass
        h = m * self.body.com
        Jvec = rob.vech(self.body.inertia)

        # polytopic parameter uncertainty
        self.θ = np.concatenate(([m], h, Jvec))
        # Δθ = np.concatenate(([0.1, 0.01 * δ, 0.01 * δ, 0.01 * h], 0 * Jvec))
        self.θ_min = θ_min if θ_min is not None else self.θ
        self.θ_max = θ_max if θ_max is not None else self.θ
        self.P = np.vstack((np.eye(self.θ.shape[0]), -np.eye(self.θ.shape[0])))
        self.p = np.concatenate((self.θ_min, -self.θ_max))


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
    A, b = rob.span_to_face_form(W @ S)
    assert np.allclose(b, 0)

    # Aw <= 0 implies there exist feasible contact forces to support wrench w
    return A


def object_mass_matrix(obj):
    """Compute the object's mass matrix in the body frame."""
    h = obj.body.mass * obj.body.com
    # fmt: off
    return np.block([
        [obj.body.mass * np.eye(3), -core.math.skew3(h)],
        [core.math.skew3(h), obj.body.inertia]])
    # fmt: on


def object_velocity_terms(obj, V):
    """Compute the velocity terms of the GIW in the body frame."""
    M = object_mass_matrix(obj)
    return rob.skew6(V) @ M @ V


class ReactiveBalancingController:
    def __init__(
        self,
        model,
        dt,
        θ_min=None,
        θ_max=None,
        a_cart_weight=1,
        α_cart_weight=0.01,
        a_joint_weight=1e-6,
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
            k: UncertainObject(v)
            for k, v in model.settings.balancing_settings.objects.items()
        }
        self.contacts = [
            RobustContactPoint(c) for c in model.settings.balancing_settings.contacts
        ]

        # canonical map of object names to indices
        names = list(self.objects.keys())
        self.object_name_index = compute_object_name_index(names)

        # shared optimization weight
        self.v_joint_weight = v_joint_weight
        self.P = block_diag(
            (a_joint_weight + dt**2 * v_joint_weight) * np.eye(9),
            a_cart_weight * np.eye(3),
            α_cart_weight * np.eye(3),
        )

        # shared optimization bounds
        self.ub = np.zeros(15)
        self.ub[:9] = a_joint_max
        self.ub[9:12] = a_cart_max
        self.ub[12:] = α_cart_max
        self.lb = -self.ub

        self.W = compute_contact_force_to_wrench_map(
            self.object_name_index, self.contacts
        )
        self.M = np.vstack([object_mass_matrix(obj) for obj in self.objects.values()])

        # face form of the CWC
        self.F = compute_cwc_face_form(self.object_name_index, self.contacts)

    def update(self, q, v):
        self.robot.update(q, v)

    def solve(self, q, v, ad):
        """Solve for an updated joint acceleration command given current robot
        state (q, v) and desired EE acceleration ad."""
        self.robot.forward(q, v)  # TODO could be removed for speed
        J = self.robot.jacobian(q)[:3, :]
        dJdt = self.robot.jacobian_time_derivative(q, v)
        δ = dJdt[:3, :] @ v

        C_we = core.math.quat_to_rot(self.robot.link_pose()[1])
        V_ew_w = np.concatenate(self.robot.link_velocity())

        return self._solve_qp(C_we, V_ew_w, ad, δ, J, v)


class NominalReactiveBalancingController(ReactiveBalancingController):
    def __init__(self, model, dt):
        super().__init__(model, dt)

        nc = 3 * len(self.contacts)

        # cost
        self.P = block_diag(self.P, np.zeros((nc, nc)))
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

    def _solve_qp(self, C_we, V, ad, δ, J, v):
        # number of optimization variables
        nv = 9 + 6 + 3 * len(self.contacts)

        # initial guess
        x0 = np.zeros(nv)
        x0[9:12] = C_we.T @ ad

        Ag = np.concatenate((C_we.T @ G, np.zeros(3)))  # body-frame gravity

        # compute the equality constraints A_eq @ x == b
        # N-E equations for object balancing
        h = np.concatenate(
            [object_velocity_terms(obj, V) for obj in self.objects.values()]
        )
        b_eq_bal = self.M @ Ag - h

        # map joint acceleration to EE acceleration
        A_eq_track = np.hstack((-J, C_we, np.zeros((3, 3 + self.W.shape[1]))))
        b_eq_track = δ

        A_eq = np.vstack((A_eq_track, self.A_eq_bal))
        b_eq = np.concatenate((b_eq_track, b_eq_bal))

        # compute the cost: 0.5 * x @ P @ x + q @ x
        q = np.zeros(nv)
        q[:9] = self.v_joint_weight * self.dt * v
        q[9:12] = -C_we.T @ ad

        x = solve_qp(
            P=self.P_sparse,
            q=q,
            G=self.A_ineq_sparse,
            h=np.zeros(self.A_ineq.shape[0]),
            A=sparse.csc_matrix(A_eq),
            b=b_eq,
            lb=self.lb,
            ub=self.ub,
            initvals=x0,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=10000,
            solver=self.solver,
        )
        a = x[:9]
        A = x[9:15]
        return a, A


class NominalReactiveBalancingControllerFaceForm(ReactiveBalancingController):
    def __init__(self, model, dt):
        super().__init__(model, dt)

        self.P_sparse = sparse.csc_matrix(self.P)

        self.A_ineq = np.hstack((np.zeros((self.F.shape[0], 9)), self.F @ self.M))
        self.A_ineq_sparse = sparse.csc_matrix(self.A_ineq)

    def _solve_qp(self, C_we, V, ad, δ, J, v):
        nv = 15

        # initial guess
        x0 = np.zeros(nv)
        x0[9:12] = ad

        Ag = np.concatenate((C_we.T @ G, np.zeros(3)))  # body-frame gravity
        h = np.concatenate(
            [object_velocity_terms(obj, V) for obj in self.objects.values()]
        )

        # map joint acceleration to EE acceleration
        A_eq = np.hstack((-J, C_we, np.zeros((3, 3))))
        b_eq = δ

        # compute the inequality constraints A_ineq @ x <= b_ineq
        b_ineq = self.F @ (self.M @ Ag - h)

        # compute the cost: 0.5 * x @ P @ x + q @ x
        q = np.zeros(nv)
        q[:9] = self.v_joint_weight * self.dt * v
        q[9:12] = -C_we.T @ ad

        x = solve_qp(
            P=self.P_sparse,
            q=q,
            G=self.A_ineq_sparse,
            h=b_ineq,
            A=sparse.csc_matrix(A_eq),
            b=b_eq,
            lb=self.lb,
            ub=self.ub,
            initvals=x0,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=10000,
            solver=self.solver,
        )
        a = x[:9]
        A = x[9:15]
        return a, A


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
        self.R = rob.span_to_face_form(self.P_tilde.T)[0]

        # pre-compute inequality matrix
        nv = 15
        nf = self.F.shape[0]
        n_ineq = self.R.shape[0]
        N_ineq = n_ineq * nf
        self.A_ineq = np.zeros((N_ineq, nv))
        for i in range(nf):
            D = rob.body_regressor_A_by_vector(self.F[i, :])
            D_tilde = np.vstack((D, np.zeros((1, D.shape[1]))))
            self.A_ineq[i * n_ineq : (i + 1) * n_ineq, 9:] = -self.R @ D_tilde
        self.A_ineq_sparse = sparse.csc_matrix(self.A_ineq)

    def _solve_qp(self, C_we, V, ad, δ, J, v):
        nv = 15
        no = len(self.objects)
        nf = self.F.shape[0]

        # initial guess
        x0 = np.zeros(nv)
        x0[9:12] = ad

        # map joint acceleration to EE acceleration
        A_eq = np.hstack((-J, C_we, np.zeros((3, 3))))
        b_eq = δ

        # build robust constraints
        t0 = time.time()
        G = rob.body_gravity6(C_we.T)
        B = rob.body_regressor_VG_by_vector_tilde_vectorized(V, G, self.F)
        b_ineq = (self.R @ B).T.flatten()
        t1 = time.time()
        print(f"build time = {1000 * (t1 - t0)} ms")

        # compute the cost: 0.5 * x @ P @ x + q @ x
        q = np.zeros(nv)
        q[:9] = self.v_joint_weight * self.dt * v
        q[9:12] = -C_we.T @ ad

        x = solve_qp(
            P=self.P_sparse,
            q=q,
            G=self.A_ineq_sparse,
            h=b_ineq,
            A=sparse.csc_matrix(A_eq),
            b=b_eq,
            lb=self.lb,
            ub=self.ub,
            initvals=x0,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=10000,
            solver=self.solver,
        )
        a = x[:9]
        A = x[9:15]
        return a, A


def main():
    np.set_printoptions(precision=6, suppress=True)

    cli_args = cmd.cli.sim_arg_parser().parse_args()

    # load configuration
    config = core.parsing.load_config(cli_args.config)
    sim_config = config["simulation"]
    ctrl_config = config["controller"]
    log_config = config["logging"]

    timestamp = datetime.datetime.now()
    env = sim.simulation.UprightSimulation(
        config=sim_config,
        timestamp=timestamp,
        video_name=cli_args.video,
        extra_gui=sim_config.get("extra_gui", False),
    )
    env.settle(5.0)

    # controller
    model = ctrl.manager.ControllerModel.from_config(ctrl_config)
    robot = model.robot
    nominal_controller = NominalReactiveBalancingControllerFaceForm(model, env.timestep)
    robust_controller = RobustReactiveBalancingController(model, env.timestep)

    # tracking controller gains
    kp = 2
    kv = 1

    t = 0.0
    q, v = env.robot.joint_states()
    a = np.zeros(env.robot.nv)

    v_cmd = np.zeros(env.robot.nv)

    robot.forward(q, v)
    r_ew_w_0, Q_we_0 = robot.link_pose()
    r_ew_w_d = r_ew_w_0 + [0, 2, 0]

    # desired trajectory
    trajectory = mm.PointToPointTrajectory.quintic(
        r_ew_w_0, r_ew_w_d, max_vel=2, max_acc=5
    )

    rd = r_ew_w_d
    vd = np.zeros(3)
    ad = np.zeros(3)

    # goal position
    debug_frame_world(0.2, list(r_ew_w_d), orientation=Q_we_0, line_width=3)

    # simulation loop
    while t <= env.duration:
        # current joint state
        q, v = env.robot.joint_states(add_noise=False)

        # current EE state
        robot.forward(q, v)
        r_ew_w, Q_we = robot.link_pose()
        v_ew_w, ω_ew_w = robot.link_velocity()

        # desired EE state
        # rd, vd, ad = trajectory.sample(t)
        # print(vd)

        # commanded EE linear acceleration
        a_ew_w_cmd = kp * (rd - r_ew_w) + kv * (vd - v_ew_w) + ad

        # compute command
        t0 = time.time()
        u_n, A_n = nominal_controller.solve(q, v, a_ew_w_cmd)
        u_r, A_r = robust_controller.solve(q, v, a_ew_w_cmd)
        t1 = time.time()
        # print(f"solve took {1000 * (t1 - t0)} ms")
        print(f"A_n = {A_n}")
        print(f"A_r = {A_r}")

        print(f"u_n = {u_n}, norm = {np.linalg.norm(u_n)}")
        print(f"u_r = {u_r}, norm = {np.linalg.norm(u_r)}")
        IPython.embed()
        v_cmd = v_cmd + env.timestep * u_n

        env.robot.command_velocity(v_cmd, bodyframe=False)

        t = env.step(t, step_robot=False)[0]

    IPython.embed()


if __name__ == "__main__":
    main()
