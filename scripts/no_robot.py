#!/usr/bin/env python
"""Baseline tray balancing formulation."""
from functools import partial
import time

import jax.numpy as jnp
import jax
from jaxlie import SO3
import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb
import pybullet_data

import sqp
from util import (
    skew3,
    pose_error,
    pose_to_pos_quat,
    pose_from_pos_quat,
    equilateral_triangle_inscribed_radius,
    cylinder_inertia_matrix,
    quat_multiply,
    debug_frame
)
from tray import Tray
from end_effector import EndEffector, EndEffectorModel

import IPython


# EE geometry parameters
EE_SIDE_LENGTH = 0.3

# EE motion parameters
VEL_LIM = 4
ACC_LIM = 8

GRAVITY = 9.81

# tray parameters
TRAY_RADIUS = 0.25
TRAY_MASS = 0.5
TRAY_MU = 0.5
TRAY_W = equilateral_triangle_inscribed_radius(EE_SIDE_LENGTH)
TRAY_H = (
    0.5  # height of center of mass from bottom of tray  TODO confusing
)
# TRAY_INERTIA = TRAY_MASS * (3 * TRAY_RADIUS ** 2 + (2 * TRAY_H) ** 2) / 12.0
TRAY_INERTIA = cylinder_inertia_matrix(TRAY_MASS, TRAY_RADIUS, 2 * TRAY_H)

# simulation parameters
SIM_DT = 0.001  # simulation timestep (s)
MPC_DT = 0.1  # lookahead timestep of the controller
MPC_STEPS = 20  # number of timesteps to lookahead
SQP_ITER = 3  # number of iterations for the SQP solved by the controller
PLOT_PERIOD = 100  # update plot every PLOT_PERIOD timesteps
CTRL_PERIOD = 100  # generate new control signal every CTRL_PERIOD timesteps
RECORD_PERIOD = 10
DURATION = 10.0  # duration of trajectory (s)


class TrayBalanceOptimizationEE:
    def __init__(self, model, r_te_e):
        self.model = model
        self.r_te_e = r_te_e

        self.nv = model.ni  # number of optimization variables per MPC step
        self.nc_eq = 0  # number of equality constraints
        self.nc_ineq = 7  # number of inequality constraints
        self.nc = self.nc_eq + self.nc_ineq

        # MPC weights
        Q = np.diag(
            np.concatenate((np.zeros(7), 0.01 * np.ones(model.ni)))
        )  # joint state error
        W = np.diag(np.concatenate((np.ones(3), np.full(3, 1), np.zeros(6))))  # Cartesian state error
        R = 0.1 * np.eye(self.nv)
        V = MPC_DT * np.eye(self.nv)

        # lifted weight matrices
        Ibar = np.eye(MPC_STEPS)
        self.Qbar = np.kron(Ibar, Q)
        self.Wbar = np.kron(Ibar, W)
        self.Rbar = np.kron(Ibar, R)

        # velocity constraint matrix
        self.Vbar = np.kron(np.tril(np.ones((MPC_STEPS, MPC_STEPS))), V)

        self.err_jac = jax.jit(jax.jacfwd(self.error_unrolled, argnums=3))
        self.state_jac = jax.jit(jax.jacfwd(self.state_unrolled, argnums=1))

    @partial(jax.jit, static_argnums=(0,))
    def state_unrolled(self, X0, ubar):
        """Unroll the joint state of the robot over the time horizon."""

        def state_func(X, u):
            X = self.model.simulate(X, u)
            return X, X

        u = ubar.reshape((MPC_STEPS, self.model.ni))
        _, Xbar = jax.lax.scan(state_func, X0, u)
        return Xbar.flatten()

    @partial(jax.jit, static_argnums=(0,))
    def error_unrolled(self, X0, P_we_d, V_ee_d, var):
        """Unroll the pose error over the time horizon."""

        def error_func(X, u):
            X = self.model.simulate(X, u)
            P_we, V_ew_w = X[:7], X[7:]

            pose_err = pose_error(P_we_d, P_we)
            V_err = V_ee_d - V_ew_w

            e = jnp.concatenate((pose_err, V_err))
            return X, e

        u = var.reshape((MPC_STEPS, self.model.ni))
        X, ebar = jax.lax.scan(error_func, X0, u)
        return ebar.flatten()

    @partial(jax.jit, static_argnums=(0,))
    def ineq_constraints(self, P_we, V_ew_w, A_ew_w, jnp=jnp):
        """Calculate inequality constraints for a single timestep."""
        _, Q_we = pose_to_pos_quat(P_we)
        ω_ew_w = V_ew_w[3:]
        a_ew_w = A_ew_w[:3]
        α_ew_w = A_ew_w[3:]

        # TODO: we could probably reformulate all of this in terms of
        # quaternions, if desired
        C_we = SO3.from_quaternion_xyzw(Q_we).as_matrix()
        C_ew = C_we.T
        Sω_ew_w = skew3(ω_ew_w)
        ddC_we = (skew3(α_ew_w) + Sω_ew_w @ Sω_ew_w) @ C_we

        g = jnp.array([0, 0, -GRAVITY])

        α = TRAY_MASS * C_ew @ (a_ew_w + ddC_we @ self.r_te_e - g)

        # rotational
        Iw = C_we @ TRAY_INERTIA @ C_we.T
        β = C_ew @ Sω_ew_w @ Iw @ ω_ew_w + TRAY_INERTIA @ C_ew @ α_ew_w
        S = np.array([[0, 1], [-1, 0]])

        rz = -TRAY_H
        r = TRAY_W

        γ = rz * S.T @ α[:2] - β[:2]

        # NOTE: these constraints are currently written to be >= 0, in
        # constraint to the notes which have everything <= 0.
        # NOTE the addition of a small term in the square root to ensure
        # derivative is well-defined at 0
        ε2 = 0.01

        # h1 = (TRAY_MU * α[2])**2 - (α[0]**2 + α[1]**2)  # friction cone
        # h1 = TRAY_MU * α[2] - jnp.sqrt(α[0] ** 2 + α[1] ** 2 + 0.01)  # friction cone

        # Friction cone with rotational component: this is always a tighter
        # bound than when the rotational component isn't considered (which
        # makes sense).
        # Splitting the absolute value into two constraints appears to be
        # better numerically for the solver
        h1 = TRAY_MU * α[2] - jnp.sqrt(α[0] ** 2 + α[1] ** 2 + ε2)
        h1a = h1 + β[2] / r
        h1b = h1 - β[2] / r

        # h1a = h1
        # h1b = 1

        # this approximation actually works less well than the correct
        # quadratic expression above:
        # TODO probably because of the poor gradient info the absolute values
        # h1 = TRAY_MU * α[2] - jnp.abs(α[0]) - jnp.abs(α[1])

        h2 = α[2]  # α3 >= 0

        h3 = r * α[2] - jnp.sqrt(γ[0] ** 2 + γ[1] ** 2 + ε2)
        # h3 = 1

        return jnp.array([h1a, h1b, h2, h3, 1, 1, 1])

    @partial(jax.jit, static_argnums=(0,))
    def ineq_constraints_unrolled(self, X0, P_we_d, V_ew_w_d, var):
        """Unroll the inequality constraints over the time horizon."""

        def ineq_func(X, u):

            # we actually two sets of constraints for each timestep: one at the
            # start and one at the end
            # at the start of the timestep, we need to ensure the new inputs
            # satisfy constraints
            P_we, V_ew_w = X[:7], X[7:]
            ineq_con1 = self.ineq_constraints(P_we, V_ew_w, u)

            X = self.model.simulate(X, u)

            # at the end of the timestep, we need to make sure that the robot
            # ends up in a state where constraints are still satisfied given
            # the input
            P_we, V_ew_w = X[:7], X[7:]
            ineq_con2 = self.ineq_constraints(P_we, V_ew_w, u)

            return X, jnp.concatenate((ineq_con1, ineq_con2))

        u = var.reshape((MPC_STEPS, self.model.ni))
        X, ineq_con = jax.lax.scan(ineq_func, X0, u)
        return ineq_con.flatten()

    @partial(jax.jit, static_argnums=(0,))
    def obj_hess_jac(self, X0, P_we_d, V_ew_w_d, var):
        """Calculate objective Hessian and Jacobian."""
        u = var

        e = self.error_unrolled(X0, P_we_d, V_ew_w_d, u)
        dedu = self.err_jac(X0, P_we_d, V_ew_w_d, u)

        x = self.state_unrolled(X0, u)
        dxdu = self.state_jac(X0, u)

        # Function
        f = e.T @ self.Wbar @ e + x.T @ self.Qbar @ x + u.T @ self.Rbar @ u

        # Jacobian
        g = e.T @ self.Wbar @ dedu + x.T @ self.Qbar @ dxdu + u.T @ self.Rbar

        # (Approximate) Hessian
        H = dedu.T @ self.Wbar @ dedu + dxdu.T @ self.Qbar @ dxdu + self.Rbar

        return f, g, H

    @partial(jax.jit, static_argnums=(0,))
    def vel_ineq_constraints(self, X0, P_we_d, V_ew_w_d, var):
        """Inequality constraints on EE velocity."""
        V0 = X0[7:]
        return self.Vbar @ var + jnp.tile(V0, MPC_STEPS)

    @partial(jax.jit, static_argnums=(0,))
    def vel_ineq_jacobian(self, X0, P_we_d, V_ew_w_d, var):
        """Jacobian of joint velocity constraints."""
        return self.Vbar

    @partial(jax.jit, static_argnums=(0,))
    def con_fun(self, X0, P_we_d, V_ew_w_d, var):
        """Combined constraint function."""
        con1 = self.vel_ineq_constraints(X0, P_we_d, V_ew_w_d, var)
        con2 = self.ineq_constraints_unrolled(X0, P_we_d, V_ew_w_d, var)
        return jnp.concatenate((con1, con2))

    @partial(jax.jit, static_argnums=(0,))
    def con_jac(self, X0, P_we_d, V_ew_w_d, var):
        """Combined constraint Jacobian."""
        J1 = self.vel_ineq_jacobian(X0, P_we_d, V_ew_w_d, var)
        J2 = jax.jacfwd(self.ineq_constraints_unrolled, argnums=3)(
            X0, P_we_d, V_ew_w_d, var
        )
        return jnp.vstack((J1, J2))

    def bounds(self):
        """Compute bounds for the problem."""
        lb_acc = -ACC_LIM * np.ones(MPC_STEPS * self.nv)
        ub_acc = ACC_LIM * np.ones(MPC_STEPS * self.nv)
        return sqp.Bounds(lb_acc, ub_acc)

    def constraints(self):
        """Compute the constraints for the problem."""
        lb_vel = -VEL_LIM * np.ones(MPC_STEPS * self.nv)
        ub_vel = VEL_LIM * np.ones(MPC_STEPS * self.nv)

        lb_physics = np.zeros(MPC_STEPS * self.nc * 2)
        ub_physics = np.infty * np.ones(MPC_STEPS * self.nc * 2)

        con_lb = np.concatenate((lb_vel, lb_physics))
        con_ub = np.concatenate((ub_vel, ub_physics))

        # constraint mask for velocity constraints is static
        vel_con_mask = self.Vbar

        physics_con_mask = np.kron(
            np.tril(np.ones((MPC_STEPS, MPC_STEPS))), np.ones((2 * self.nc, self.nv))
        )
        con_mask = np.vstack((vel_con_mask, physics_con_mask))
        con_nz_idx = np.nonzero(con_mask)

        return sqp.Constraints(
            self.con_fun, self.con_jac, con_lb, con_ub, nz_idx=con_nz_idx
        )


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

    debug_frame(0.1, ee.uid, -1)

    # setup tray
    tray = Tray(mass=TRAY_MASS, radius=TRAY_RADIUS, height=2 * TRAY_H, mu=TRAY_MU)
    ee_pos, _ = ee.get_pose()
    tray.reset_pose(position=ee_pos + [0, 0, TRAY_H + 0.05])

    settle_sim(1.0)

    return ee, tray


def calc_r_te_e(r_ew_w, Q_we, r_tw_w):
    """Calculate position of tray relative to the EE."""
    # C_{ew} @ (r^{tw}_w - r^{ew}_w
    r_te_w = r_tw_w - r_ew_w
    return SO3.from_quaternion_xyzw(Q_we).inverse() @ r_te_w


def calc_Q_et(Q_we, Q_wt):
    SO3_we = SO3.from_quaternion_xyzw(Q_we)
    SO3_wt = SO3.from_quaternion_xyzw(Q_wt)
    return SO3_we.inverse().multiply(SO3_wt).as_quaternion_xyzw()


def main():
    np.set_printoptions(precision=3, suppress=True)

    if TRAY_W < TRAY_MU * TRAY_H:
        print("warning: w < μh")

    N = int(DURATION / SIM_DT) + 1
    N_record = int(DURATION / (SIM_DT * RECORD_PERIOD))

    # simulation objects and model
    ee, tray = setup_sim()

    model = EndEffectorModel(MPC_DT)

    x = ee.get_state()
    r_ew_w, Q_we = ee.get_pose()
    v_ew_w, ω_ew_w = ee.get_velocity()
    r_tw_w, Q_wt = tray.get_pose()
    r_te_e = calc_r_te_e(r_ew_w, Q_we, r_tw_w)

    # construct the tray balance problem
    problem = TrayBalanceOptimizationEE(model, r_te_e)

    ts = RECORD_PERIOD * SIM_DT * np.arange(N_record)
    us = np.zeros((N_record, model.ni))
    r_ew_ws = np.zeros((N_record, 3))
    r_ew_wds = np.zeros((N_record, 3))
    Q_wes = np.zeros((N_record, 4))
    Q_des = np.zeros((N_record, 4))
    v_ew_ws = np.zeros((N_record, 3))
    ω_ew_ws = np.zeros((N_record, 3))
    r_tw_ws = np.zeros((N_record, 3))
    r_te_es = np.zeros((N_record, 3))
    Q_ets = np.zeros((N_record, 4))
    ineq_cons = np.zeros((N_record, problem.nc_ineq))

    r_te_es[0, :] = r_te_e
    Q_wes[0, :] = Q_we
    Q_ets[0, :] = calc_Q_et(Q_we, Q_wt)
    r_ew_ws[0, :] = r_ew_w
    v_ew_ws[0, :] = v_ew_w
    ω_ew_ws[0, :] = ω_ew_w

    # reference trajectory
    # setpoints = np.array([[1, 0, -0.5], [2, 0, -0.5], [3, 0, 0.5]]) + r_ew_w
    setpoints = np.array([[2, 0, 0]]) + r_ew_w
    setpoint_idx = 0

    # desired quaternion
    # Qd = Q_we
    R_ed = SO3.from_z_radians(0)
    R_we = SO3.from_quaternion_xyzw(Q_we)
    R_wd = R_we.multiply(R_ed)
    Qd = R_wd.as_quaternion_xyzw()
    Qd_inv = R_wd.inverse().as_quaternion_xyzw()

    Q_des[0, :] = quat_multiply(Qd_inv, Q_we)

    # Construct the SQP controller
    controller = sqp.SQP(
        problem.nv * MPC_STEPS,
        2 * problem.nc * MPC_STEPS,
        problem.obj_hess_jac,
        problem.constraints(),
        problem.bounds(),
        num_wsr=300,
        num_iter=SQP_ITER,
        verbose=False,
        solver="qpoases",
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

            r_tw_w, Q_wt = tray.get_pose()
            r_ew_w_d = setpoints[setpoint_idx, :]

            # orientation error
            Q_de = quat_multiply(Qd_inv, Q_we)

            # record
            us[idx, :] = u
            r_ew_wds[idx, :] = r_ew_w_d
            r_ew_ws[idx, :] = r_ew_w
            Q_wes[idx, :] = Q_we
            Q_des[idx, :] = Q_de
            v_ew_ws[idx, :] = v_ew_w
            ω_ew_ws[idx, :] = ω_ew_w
            r_tw_ws[idx, :] = r_tw_w
            r_te_es[idx, :] = calc_r_te_e(r_ew_w, Q_we, r_tw_w)
            Q_ets[0, :] = calc_Q_et(Q_we, Q_wt)

        if np.linalg.norm(r_ew_w_d - r_ew_w) < 0.01 and np.linalg.norm(Q_de[:3]) < 0.01:
            print("Close to desired pose - stopping.")
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
    plt.plot(ts[1:idx], Q_des[1:idx, 0], label="$ΔQ_x$", color="r")
    plt.plot(ts[1:idx], Q_des[1:idx, 1], label="$ΔQ_y$", color="g")
    plt.plot(ts[1:idx], Q_des[1:idx, 2], label="$ΔQ_z$", color="b")
    plt.grid()
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Orientation error")
    plt.title("End effector orientation error")

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
    r_te_e_err = r_te_e - r_te_es[:idx, :]
    plt.plot(ts[:idx], r_te_e_err[:, 0], label="$x$")
    plt.plot(ts[:idx], r_te_e_err[:, 1], label="$y$")
    plt.plot(ts[:idx], r_te_e_err[:, 2], label="$z$")
    plt.plot(ts[:idx], np.linalg.norm(r_te_e_err, axis=1), label="$||r||$")
    plt.plot(ts[:idx], Q_ets[:idx, 0], label="$Q_x$")
    plt.plot(ts[:idx], Q_ets[:idx, 1], label="$Q_y$")
    plt.plot(ts[:idx], Q_ets[:idx, 2], label="$Q_z$")
    plt.grid()
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("$r^{te}_e$ error")
    plt.title("$r^{te}_e$ error")

    plt.figure()
    for j in range(problem.nc_ineq):
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
