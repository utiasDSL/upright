#!/usr/bin/env python
"""Baseline tray balancing formulation."""
from functools import partial
import time

import jax.numpy as jnp
import jax
from jaxlie import SO3
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import pybullet as pyb
import pybullet_data

import sqp
from util import (
    pose_error,
    state_error,
    pose_to_pos_quat,
    pose_from_pos_quat,
)
from bodies import Cylinder
import geometry
import balancing
from end_effector import EndEffector, EndEffectorModel

import IPython


# EE geometry parameters
EE_SIDE_LENGTH = 0.3
EE_INSCRIBED_RADIUS = geometry.equilateral_triangle_inscribed_radius(EE_SIDE_LENGTH)

# EE motion parameters
VEL_LIM = 4
ACC_LIM = 8

GRAVITY = 9.81

# tray parameters
TRAY_RADIUS = 0.25
TRAY_MASS = 0.5
TRAY_MU = 0.5
TRAY_COM_HEIGHT = 0.1

# simulation parameters
SIM_DT = 0.001  # simulation timestep (s)
MPC_DT = 0.1  # lookahead timestep of the controller MPC_STEPS = 10  # number of timesteps to lookahead
MPC_STEPS = 20
SQP_ITER = 3  # number of iterations for the SQP solved by the controller
PLOT_PERIOD = 100  # update plot every PLOT_PERIOD timesteps
CTRL_PERIOD = 100  # generate new control signal every CTRL_PERIOD timesteps
RECORD_PERIOD = 10
DURATION = 10.0  # duration of trajectory (s)


class TrayBalanceOptimizationEE:
    def __init__(self, model, tray):
        self.model = model
        self.obj_to_constrain = [tray]

        self.nv = model.ni + model.ns  # number of optimization variables per MPC step

        # NOTE: ns - 1 because the state has a constraint, meaning the error
        # can be represented as a 6-dim vector
        # something something minimal representation
        self.nc_eq = model.ns - 1  # number of equality constraints

        # NOTE: 2 per timestep
        self.nc_ineq = 2 * (2 + 1)  # number of inequality constraints
        self.nc = self.nc_eq + self.nc_ineq

        # MPC weights
        W = np.diag(
            np.concatenate((np.ones(6), np.zeros(6)))
        )  # Cartesian state error weights

        Q = np.diag(
            np.concatenate((0.01 * np.zeros(7), 0.01 * np.ones(model.ni)))
        )  # Cartesian state weights
        R = 0.01 * np.eye(model.ni)  # Accleration input weights

        # nominal target value of each state, for regularization. We can't just
        # leave this as zero because [0, 0, 0, 0] is not a valid quaternion.
        # Thus we regularize the quaternions to [0, 0, 0, 1], and all else to
        # zero.
        self.xt0 = np.zeros(model.ns)
        self.xt0[6] = 1  # w of quaternion
        self.xtbar = np.tile(self.xt0, MPC_STEPS)

        # lifted weight matrices
        Ibar = np.eye(MPC_STEPS)
        self.Qbar = np.kron(Ibar, Q)
        self.Wbar = np.kron(Ibar, W)
        self.Rbar = np.kron(Ibar, R)

        # H is identity (i.e., the system behaves like a decoupled double
        # integrator) except the quaternion term, which is fully coupled (that
        # is the 4x4 block of ones).
        H_mask0 = block_diag(np.eye(model.ni + 3), np.ones((4, 4)), np.eye(model.ni))
        self.H_mask = np.kron(Ibar, H_mask0)
        self.H_mask = np.triu(self.H_mask)  # only take upper-triangular part (for OSQP)
        self.H_nz_idx = np.nonzero(self.H_mask)

        A_mask0 = np.ones((self.nc, 2 * model.ns + model.ni))
        self.A_mask = np.zeros((self.nc * MPC_STEPS, self.nv * MPC_STEPS))

        # in the first block, we remove x0 because we don't include the initial
        # (current) state as an optimization variable
        self.A_mask[: self.nc, : model.ni + model.ns] = A_mask0[:, model.ns :]

        # remaining blocks
        for i in range(1, MPC_STEPS):
            row_idx0 = i * self.nc
            row_slice = slice(row_idx0, row_idx0 + A_mask0.shape[0])

            col_idx0 = model.ni + (i - 1) * (model.ns + model.ni)
            col_slice = slice(col_idx0, col_idx0 + A_mask0.shape[1])

            self.A_mask[row_slice, col_slice] = A_mask0
        self.A_nz_idx = np.nonzero(self.A_mask)

        self.err_jac = jax.jit(jax.jacfwd(self.error_unrolled, argnums=3))
        self.xbar_jac = jax.jit(jax.jacfwd(self.extract_xbar, argnums=0))
        self.ubar_jac = jax.jit(jax.jacfwd(self.extract_ubar, argnums=0))

    @partial(jax.jit, static_argnums=(0,))
    def extract_xbar(self, var):
        var2d = var.reshape((MPC_STEPS, self.model.ni + self.model.ns))
        xbar2d = var2d[:, self.model.ni :]
        return xbar2d.flatten()

    @partial(jax.jit, static_argnums=(0,))
    def extract_ubar(self, var):
        var2d = var.reshape((MPC_STEPS, self.model.ni + self.model.ns))
        ubar2d = var2d[:, : self.model.ni]
        return ubar2d.flatten()

    @partial(jax.jit, static_argnums=(0,))
    def error_unrolled(self, x0, P_we_d, V_ee_d, var):
        """Unroll the pose error over the time horizon."""

        def error_func(u0x1):
            _, x = u0x1[: self.model.ni], u0x1[self.model.ni :]
            P_we, V_ew_w = x[:7], x[7:]

            # TODO rewrite in terms of state
            pose_err = pose_error(P_we_d, P_we)
            V_err = V_ee_d - V_ew_w

            e = jnp.concatenate((pose_err, V_err))
            return e

        var2d = var.reshape((MPC_STEPS, self.model.ni + self.model.ns))
        ebar = jax.lax.map(error_func, var2d)
        return ebar.flatten()

    @partial(jax.jit, static_argnums=(0,))
    def ineq_constraints(self, P_we, V_ew_w, A_ew_w, jnp=jnp):
        """Calculate inequality constraints for a single timestep."""
        _, Q_we = pose_to_pos_quat(P_we)
        ω_ew_w = V_ew_w[3:]
        a_ew_w = A_ew_w[:3]
        α_ew_w = A_ew_w[3:]
        C_we = SO3.from_quaternion_xyzw(Q_we).as_matrix()

        h = jnp.concatenate(
            [
                balancing.object_balance_constraints(obj, C_we, ω_ew_w, a_ew_w, α_ew_w)
                for obj in self.obj_to_constrain
            ]
        )
        return h

    @partial(jax.jit, static_argnums=(0,))
    def constraints_unrolled(self, x0, P_we_d, V_ew_w_d, var):
        """Unroll the inequality constraints over the time horizon."""

        def one_step_con_func(x0, u0x1):
            u0, x1 = u0x1[: self.model.ni], u0x1[self.model.ni :]

            # we actually two sets of constraints for each timestep: one at the
            # start and one at the end
            # at the start of the timestep, we need to ensure the new inputs
            # satisfy constraints
            P_we0, V_ew_w0 = x0[:7], x0[7:]
            ineq_con0 = self.ineq_constraints(P_we0, V_ew_w0, u0)

            # at the end of the timestep, we need to make sure that the robot
            # ends up in a state where constraints are still satisfied given
            # the input
            P_we1, V_ew_w1 = x1[:7], x1[7:]
            ineq_con1 = self.ineq_constraints(P_we1, V_ew_w1, u0)

            # equality constraint on the dynamics
            eq_con = state_error(x1, self.model.simulate(x0, u0))

            return x1, jnp.concatenate((ineq_con0, ineq_con1, eq_con))

        var2d = var.reshape((MPC_STEPS, self.nv))
        _, con2d = jax.lax.scan(one_step_con_func, x0, var2d)
        return con2d.flatten()

    @partial(jax.jit, static_argnums=(0,))
    def balance_constraints_unrolled(self, x0, P_we_d, V_ew_w_d, var):
        """Inequality constraints for object balancing."""

        def one_step_con_func(x0, u0x1):
            u0, x1 = u0x1[: self.model.ni], u0x1[self.model.ni :]

            # we actually two sets of constraints for each timestep: one at the
            # start and one at the end
            # at the start of the timestep, we need to ensure the new inputs
            # satisfy constraints
            P_we0, V_ew_w0 = x0[:7], x0[7:]
            ineq_con0 = self.ineq_constraints(P_we0, V_ew_w0, u0)

            # at the end of the timestep, we need to make sure that the robot
            # ends up in a state where constraints are still satisfied given
            # the input
            P_we1, V_ew_w1 = x1[:7], x1[7:]
            ineq_con1 = self.ineq_constraints(P_we1, V_ew_w1, u0)

            return x1, jnp.concatenate((ineq_con0, ineq_con1))

        var2d = var.reshape((MPC_STEPS, self.nv))
        _, con2d = jax.lax.scan(one_step_con_func, x0, var2d)
        return con2d.flatten()

    @partial(jax.jit, static_argnums=(0,))
    def dynamics_constraints_unrolled(self, x0, P_we_d, V_ew_w_d, var):
        """Equality constraints on the dynamics."""

        def one_step_con_func(x0, u0x1):
            u0, x1 = u0x1[: self.model.ni], u0x1[self.model.ni :]
            eq_con = state_error(x1, self.model.simulate(x0, u0))

            return x1, eq_con

        var2d = var.reshape((MPC_STEPS, self.nv))
        _, con2d = jax.lax.scan(one_step_con_func, x0, var2d)
        return con2d.flatten()

    @partial(jax.jit, static_argnums=(0,))
    def obj_jac_hess(self, x0, P_we_d, V_ew_w_d, var):
        """Calculate objective Hessian and Jacobian."""

        ebar = self.error_unrolled(x0, P_we_d, V_ew_w_d, var)
        dedvar = self.err_jac(x0, P_we_d, V_ew_w_d, var)

        xbar = self.extract_xbar(var)
        ubar = self.extract_ubar(var)

        # Jacobians of state and input w.r.t. optimization variable, which
        # should just be sparse matrices of ones.
        dxdvar = self.xbar_jac(var)
        dudvar = self.ubar_jac(var)

        # Function
        f = (
            ebar.T @ self.Wbar @ ebar
            + (xbar - self.xtbar).T @ self.Qbar @ (xbar - self.xtbar)
            + ubar.T @ self.Rbar @ ubar
        )

        # Jacobian
        g = (
            ebar.T @ self.Wbar @ dedvar
            + xbar.T @ self.Qbar @ dxdvar
            - self.xtbar.T @ self.Qbar @ dxdvar
            + ubar.T @ self.Rbar @ dudvar
        )

        # (Approximate) Hessian
        H = (
            dedvar.T @ self.Wbar @ dedvar
            + dxdvar.T @ self.Qbar @ dxdvar
            + dudvar.T @ self.Rbar @ dudvar
        )

        return f, g, H

    # TODO not ideal, but maybe works for now
    def obj_fun(self, X0, P_we_d, V_ew_w_d, var):
        f, _, _ = self.obj_jac_hess(X0, P_we_d, V_ew_w_d, var)
        return f

    def obj_jac(self, X0, P_we_d, V_ew_w_d, var):
        _, J, _ = self.obj_jac_hess(X0, P_we_d, V_ew_w_d, var)
        return J

    def objective(self):
        return sqp.SparseObjective(self.obj_jac_hess, self.H_nz_idx, self.H_mask.shape)

    def bounds(self):
        """Compute bounds for the problem."""
        ub0 = np.concatenate(
            (
                np.full(self.model.ni, ACC_LIM),  # acceleration
                np.full(7, np.infty),  # no bounds on position (for now)
                np.full(self.model.ni, VEL_LIM),  # velocity
            )
        )
        ub = np.tile(ub0, MPC_STEPS)
        lb = -ub
        return sqp.Bounds(lb, ub)

    def constraints(self):
        """Compute the constraints for the problem."""

        # physics constraints are currently formulated such that they are
        # constrainted to be positive
        # equality constraints just have lb = ub = 0
        lb = np.zeros(MPC_STEPS * self.nc)
        ub0 = np.concatenate((np.full(self.nc_ineq, np.infty), np.zeros(self.nc_eq)))
        ub = np.tile(ub0, MPC_STEPS)

        func = self.constraints_unrolled
        jac_func = jax.jit(jax.jacfwd(self.constraints_unrolled, argnums=3))

        return sqp.SparseConstraints(
            func, jac_func, lb, ub, self.A_nz_idx, self.A_mask.shape
        )

    def scipy_ineq_constraints(self):
        # physics constraints are currently formulated such that they are
        # constrainted to be positive
        # equality constraints just have lb = ub = 0
        # lb = np.zeros(MPC_STEPS * self.nc)
        # ub0 = np.concatenate((np.full(self.nc_ineq, np.infty), np.zeros(self.nc_eq)))
        # ub = np.tile(ub0, MPC_STEPS)

        fun = self.balance_constraints_unrolled
        jac = jax.jit(jax.jacfwd(fun, argnums=3))
        return sqp.Constraints(fun, jac, None, None)

    def scipy_eq_constraints(self):
        fun = self.dynamics_constraints_unrolled
        jac = jax.jit(jax.jacfwd(fun, argnums=3))
        return sqp.Constraints(fun, jac, None, None)


def settle_sim(duration):
    """Briefly simulate to let the simulation settle."""
    t = 0
    while t < 1.0:
        pyb.stepSimulation()
        t += SIM_DT


def setup_sim():
    """Setup pybullet simulation."""
    pyb.connect(pyb.GUI)

    pyb.setGravity(0, 0, -GRAVITY)
    pyb.setTimeStep(SIM_DT)

    pyb.resetDebugVisualizerCamera(
        cameraDistance=4.6,
        cameraYaw=5.2,
        cameraPitch=-27,
        cameraTargetPosition=[1.18, 0.11, 0.05],
    )

    # get rid of extra parts of the GUI
    pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)

    # record video
    # pyb.startStateLogging(pyb.STATE_LOGGING_VIDEO_MP4, "no_robot.mp4")

    # setup ground plane
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pyb.loadURDF("plane.urdf", [0, 0, 0])

    # setup floating end effector
    ee = EndEffector(SIM_DT, side_length=EE_SIDE_LENGTH, position=(0, 0, 1))

    # setup tray
    tray = Cylinder(
        r_tau=EE_INSCRIBED_RADIUS,
        support_area=geometry.CircleSupportArea(EE_INSCRIBED_RADIUS),
        mass=TRAY_MASS,
        radius=TRAY_RADIUS,
        height=2 * TRAY_COM_HEIGHT,
        mu=TRAY_MU,
        bullet_mu=TRAY_MU,
    )
    ee_pos, _ = ee.get_pose()
    tray.bullet.reset_pose(position=ee_pos + [0, 0, TRAY_COM_HEIGHT + 0.05])

    settle_sim(1.0)

    return ee, tray


def calc_r_te_e(r_ew_w, Q_we, r_tw_w):
    """Calculate position of tray relative to the EE."""
    r_te_w = r_tw_w - r_ew_w
    return SO3.from_quaternion_xyzw(Q_we) @ r_te_w


def calc_Q_et(Q_we, Q_wt):
    SO3_we = SO3.from_quaternion_xyzw(Q_we)
    SO3_wt = SO3.from_quaternion_xyzw(Q_wt)
    return SO3_we.inverse().multiply(SO3_wt).as_quaternion_xyzw()


def sample_inputs(problem):
    x0 = np.random.random(problem.model.ns)
    x0[3:7] = x0[3:7] / np.linalg.norm(x0[3:7])  # normalize the quaternion

    P_we_d = np.random.random(7)
    P_we_d[3:] = P_we_d[3:] / np.linalg.norm(P_we_d[3:])

    V_ew_w_d = np.random.random(6)

    # NOTE: didn't normalize quaternions here
    var = np.random.random(problem.nv * MPC_STEPS)

    return x0, P_we_d, V_ew_w_d, var


def sample_hessian(problem):
    """Compute Hessian matrices with random state and input

    Useful for determining the sparsity pattern.
    """
    x0, P_we_d, V_ew_w_d, var = sample_inputs(problem)
    _, _, H = problem.obj_jac_hess(x0, P_we_d, V_ew_w_d, var)
    return H


def sample_constraint_jac(problem):
    jac_func = problem.constraints()
    x0, P_we_d, V_ew_w_d, var = sample_inputs(problem)
    A = jac_func(x0, P_we_d, V_ew_w_d, var)
    return A


def main():
    np.set_printoptions(precision=3, suppress=True)

    if EE_INSCRIBED_RADIUS < TRAY_MU * TRAY_COM_HEIGHT:
        print("warning: w < μh")

    N = int(DURATION / SIM_DT) + 1
    N_record = int(DURATION / (SIM_DT * RECORD_PERIOD))

    # simulation objects and model
    ee, tray = setup_sim()

    model = EndEffectorModel(MPC_DT)

    x = ee.get_state()
    r_ew_w, Q_we = ee.get_pose()
    v_ew_w, ω_ew_w = ee.get_velocity()

    r_tw_w, Q_wt = tray.bullet.get_pose()
    tray.body.com = calc_r_te_e(r_ew_w, Q_we, r_tw_w)

    # desired quaternion: same as the starting orientation
    Qd = Q_we

    # construct the tray balance problem
    problem = TrayBalanceOptimizationEE(model, tray)

    ts = RECORD_PERIOD * SIM_DT * np.arange(N_record)
    us = np.zeros((N_record, model.ni))
    r_ew_ws = np.zeros((N_record, 3))
    r_ew_wds = np.zeros((N_record, 3))
    v_ew_ws = np.zeros((N_record, 3))
    ω_ew_ws = np.zeros((N_record, 3))
    p_tw_ws = np.zeros((N_record, 3))
    r_te_es = np.zeros((N_record, 3))
    Q_ets = np.zeros((N_record, 4))
    ineq_cons = np.zeros((N_record, 3))

    r_te_es[0, :] = tray.body.com
    Q_ets[0, :] = calc_Q_et(Q_we, Q_wt)
    r_ew_ws[0, :] = r_ew_w
    v_ew_ws[0, :] = v_ew_w
    ω_ew_ws[0, :] = ω_ew_w

    # reference trajectory
    # setpoints = np.array([[1, 0, -0.5], [2, 0, -0.5], [3, 0, 0.5]]) + r_ew_w
    setpoints = np.array([[2, 0, 0]]) + r_ew_w
    setpoint_idx = 0

    # initial guess for the solver
    # [u0, x1, u1, x1, ...]
    var00 = np.zeros(problem.nv)
    var00[model.ni + 6] = 1  # w in quaterion = 1
    var0 = np.tile(var00, MPC_STEPS)

    # Construct the SQP controller
    # controller = sqp.SQP(
    #     problem.nv * MPC_STEPS,
    #     problem.nc * MPC_STEPS,
    #     problem.objective(),
    #     problem.constraints(),
    #     problem.bounds(),
    #     # num_wsr=300,
    #     num_iter=SQP_ITER,
    #     verbose=True,
    #     solver="osqp",
    #     var0=var0,
    # )

    controller = sqp.SQP(
        nv=problem.nv * MPC_STEPS,
        nc=problem.nc * MPC_STEPS,
        obj_fun=problem.obj_fun,
        obj_jac=problem.obj_jac,
        ineq_cons=problem.scipy_ineq_constraints(),
        eq_cons=problem.scipy_eq_constraints(),
        bounds=problem.bounds(),
        num_iter=SQP_ITER,
        verbose=False,
        solver="scipy",
        var0=var0,
    )

    for i in range(N - 1):
        if i % CTRL_PERIOD == 0:
            r_ew_w_d = setpoints[setpoint_idx, :]
            P_we_d = pose_from_pos_quat(r_ew_w_d, Qd)
            V_we_d = jnp.zeros(6)

            x = ee.get_state()
            var = controller.solve(x, P_we_d, V_we_d)
            u = var[: model.ni]  # joint acceleration input
            ee.command_acceleration(u)
            print(np.array(problem.obj_fun(x, P_we_d, V_we_d, var)))
            print(u)

        # step simulation forward
        ee.step()
        pyb.stepSimulation()

        if i % RECORD_PERIOD == 0:
            idx = i // RECORD_PERIOD
            r_ew_w, Q_we = ee.get_pose()
            v_ew_w, ω_ew_w = ee.get_velocity()

            x = ee.get_state()
            P_we, V_ew_w = x[:7], x[7:]
            ineq_cons[idx, :] = np.array(problem.ineq_constraints(P_we, V_ew_w, u))

            r_tw_w, Q_wt = tray.bullet.get_pose()
            r_ew_w_d = setpoints[setpoint_idx, :]

            # record
            us[idx, :] = u
            r_ew_wds[idx, :] = r_ew_w_d
            r_ew_ws[idx, :] = r_ew_w
            v_ew_ws[idx, :] = v_ew_w
            ω_ew_ws[idx, :] = ω_ew_w
            p_tw_ws[idx, :] = r_tw_w
            r_te_es[idx, :] = calc_r_te_e(r_ew_w, Q_we, r_tw_w)
            Q_ets[0, :] = calc_Q_et(Q_we, Q_wt)

        if np.linalg.norm(r_ew_w_d - r_ew_w) < 0.01:
            print("Position within 1 cm.")
            setpoint_idx += 1
            if setpoint_idx >= setpoints.shape[0]:
                break

            # update r_ew_w_d to avoid falling back into this block right away
            r_ew_w_d = setpoints[setpoint_idx, :]

        time.sleep(SIM_DT)

    controller.benchmark.print_stats()

    print(f"Min constraint value = {np.min(ineq_cons)}")

    idx = i // RECORD_PERIOD

    plt.figure()
    plt.plot(ts[1:idx], r_ew_wds[1:idx, 0], label="$x_d$", color="r", linestyle="--")
    plt.plot(ts[1:idx], r_ew_wds[1:idx, 1], label="$y_d$", color="g", linestyle="--")
    plt.plot(ts[1:idx], r_ew_wds[1:idx, 2], label="$z_d$", color="b", linestyle="--")
    plt.plot(ts[1:idx], r_ew_ws[1:idx, 0], label="$x$", color="r")
    plt.plot(ts[1:idx], r_ew_ws[1:idx, 1], label="$y$", color="g")
    plt.plot(ts[1:idx], r_ew_ws[1:idx, 2], label="$z$", color="b")
    plt.grid()
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.title("End effector position")

    plt.figure()
    plt.plot(ts[1:idx], v_ew_ws[1:idx, 0], label="$v_x$")
    plt.plot(ts[1:idx], v_ew_ws[1:idx, 1], label="$v_y$")
    plt.plot(ts[1:idx], v_ew_ws[1:idx, 2], label="$v_z$")
    plt.plot(ts[1:idx], ω_ew_ws[1:idx, 0], label="$ω_x$")
    plt.plot(ts[1:idx], ω_ew_ws[1:idx, 1], label="$ω_y$")
    plt.plot(ts[1:idx], ω_ew_ws[1:idx, 2], label="$ω_z$")
    plt.grid()
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity")
    plt.title("End effector velocity")

    plt.figure()
    plt.plot(ts[:idx], tray.body.com[0] - r_te_es[:idx, 0], label="$x$", color="r")
    plt.plot(ts[:idx], tray.body.com[1] - r_te_es[:idx, 1], label="$y$", color="g")
    plt.plot(ts[:idx], tray.body.com[2] - r_te_es[:idx, 2], label="$z$", color="b")
    plt.plot(ts[:idx], Q_ets[:idx, 0], label="$Q_x$")
    plt.plot(ts[:idx], Q_ets[:idx, 1], label="$Q_y$")
    plt.plot(ts[:idx], Q_ets[:idx, 2], label="$Q_z$")
    plt.grid()
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("$p^{te}_e$ error")
    plt.title("$p^{te}_e$ error")

    plt.figure()
    for j in range(ineq_cons.shape[1]):
        plt.plot(ts[:idx], ineq_cons[:idx, j], label=f"$h_{j+1}$")
    plt.plot([0, ts[idx]], [0, 0], color="k")
    plt.grid()
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.title("Inequality constraints")

    plt.figure()
    for j in range(model.ni):
        plt.plot(ts[:idx], us[:idx, j], label=f"$u_{j+1}$")
    plt.grid()
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Commanded joint acceleration")
    plt.title("Acceleration commands")

    plt.show()


if __name__ == "__main__":
    main()
