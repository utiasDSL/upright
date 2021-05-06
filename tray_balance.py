#!/usr/bin/env python
"""Baseline tray balancing formulation."""
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb
import pybullet_data

from mm2d import trajectory as trajectories

import sqp
from util import rotation_matrix
from tray import Tray
from robot_planar import SimulatedPlanarRobot, PlanarRobotModel

import IPython


BASE_HOME = [0, 0, 0]
UR10_HOME_STANDARD = [
    0.0,
    -0.75 * np.pi,
    -0.5 * np.pi,
    -0.75 * np.pi,
    -0.5 * np.pi,
    0.5 * np.pi,
]
UR10_HOME_TRAY_BALANCE = [
    0.0,
    -0.75 * np.pi,
    -0.5 * np.pi,
    -0.25 * np.pi,
    -0.5 * np.pi,
    0.5 * np.pi,
]
ROBOT_HOME = BASE_HOME + UR10_HOME_TRAY_BALANCE


# robot parameters
L1 = 1
L2 = 1
VEL_LIM = 1
ACC_LIM = 8

GRAVITY = 9.81

# tray parameters
TRAY_RADIUS = 0.25
TRAY_MASS = 0.5
TRAY_MU = 0.5
TRAY_W = 0.1  # TODO this isn't correct now
TRAY_H = 0.01  # 0.5
TRAY_INERTIA = TRAY_MASS * (3 * TRAY_RADIUS ** 2 + (2 * TRAY_H) ** 2) / 12.0

# simulation parameters
SIM_DT = 0.001  # simulation timestep (s)
MPC_DT = 0.1  # lookahead timestep of the controller
MPC_STEPS = 10  # number of timesteps to lookahead
SQP_ITER = 1  # number of iterations for the SQP solved by the controller
PLOT_PERIOD = 100  # update plot every PLOT_PERIOD timesteps
CTRL_PERIOD = 100  # generate new control signal every CTRL_PERIOD timesteps
RECORD_PERIOD = 10
DURATION = 10.0  # duration of trajectory (s)

ns_ee = 6  # num EE states
ns_q = 8  # num joint states
ni = 4  # num inputs
nc_eq = 0
nc_ineq = 7  # num inequality constraints
nv = ni  # num opt vars
nc = nc_eq + nc_ineq  # num constraints

# MPC weights
Q = np.diag([0, 0, 0, 0, 0.01, 0.01, 0.01, 0.01])
W = np.diag([1, 1, 1, 0, 0, 0])
R = 0.01 * np.eye(ni)
V = MPC_DT * np.eye(ni)

# lifted weight matrices
Ibar = np.eye(MPC_STEPS)
Qbar = np.kron(Ibar, Q)
Wbar = np.kron(Ibar, W)
Rbar = np.kron(Ibar, R)

# velocity constraint matrix
Vbar = np.kron(np.tril(np.ones((MPC_STEPS, MPC_STEPS))), V)


def skew1(x):
    return np.array([[0, -x], [x, 0]])


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

    # setup ground plane
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pyb.loadURDF("plane.urdf", [0, 0, 0])

    # setup robot
    robot = SimulatedPlanarRobot(SIM_DT, ROBOT_HOME)
    robot.reset_joint_configuration(ROBOT_HOME)

    # simulate briefly to let the robot settle down after being positioned
    settle_sim(1.0)

    # arm gets bumped by the above settling, so we reset it again
    robot.reset_arm_joints(UR10_HOME_TRAY_BALANCE)

    # setup tray
    # TODO we want to add a constraint to the tray
    tray = Tray(mass=TRAY_MASS, radius=TRAY_RADIUS, height=2 * TRAY_H, mu=TRAY_MU)
    ee_pos, _ = robot.link_pose()
    tray.reset_pose(position=ee_pos + [0, 0, 0.05])

    settle_sim(1.0)

    return robot, tray


def main():
    np.set_printoptions(precision=3, suppress=True)

    if TRAY_W < TRAY_MU * TRAY_H:
        print("warning: w < μh")

    N = int(DURATION / SIM_DT) + 1
    N_record = int(DURATION / (SIM_DT * RECORD_PERIOD))

    p_te_e = np.array([0, 0.05 + TRAY_H])

    ts = RECORD_PERIOD * SIM_DT * np.arange(N_record)
    us = np.zeros((N_record, ni))
    P_ew_ws = np.zeros((N_record, 3))
    P_ew_wds = np.zeros((N_record, 3))
    V_ew_ws = np.zeros((N_record, 3))
    P_tw_ws = np.zeros((N_record, 3))
    p_te_es = np.zeros((N_record, 2))
    X_qs = np.zeros((N_record, ns_q))
    ineq_cons = np.zeros((N_record, nc_ineq))

    p_te_es[0, :] = p_te_e

    # simulation objects and model
    robot, tray = setup_sim()
    model = PlanarRobotModel(MPC_DT, ROBOT_HOME)

    # q, v = robot.joint_states()
    # print(model.tool_pose(q))
    # IPython.embed()
    # return

    # state of joints
    q0, dq0 = robot.joint_states()
    X_q = np.concatenate((q0, np.zeros_like(dq0)))

    P_ew_w = model.tool_pose(X_q)
    V_ew_w = model.tool_velocity(X_q)
    P_ew_ws[0, :] = P_ew_w
    V_ew_ws[0, :] = V_ew_w

    # reference trajectory
    # TODO still need to deal with this
    setpoints = np.array([[1, -0.5], [2, -0.5], [3, 0.5]]) + P_ew_w[:2]
    setpoint_idx = 0
    trajectory = trajectories.Point(setpoints[setpoint_idx, :])

    def joint_state_unrolled(X_q_0, ubar):
        """Unroll the joint state of the robot over the time horizon."""

        def state_func(X_q, u):
            X_q = model.simulate(X_q, u)
            return X_q, X_q

        u = ubar.reshape((MPC_STEPS, ni))
        _, X_q_bar = jax.lax.scan(state_func, X_q_0, u)
        return X_q_bar.flatten()

    def error_unrolled(X_q_0, X_ee_d, var):
        """Unroll the pose error over the time horizon."""
        X_ee_d0 = X_ee_d[:ns_ee]

        def error_func(X_q, u):
            X_q = model.simulate(X_q, u)
            X_ee = model.tool_state(X_q)
            e = X_ee_d0 - X_ee  # TODO this is assuming setpoint
            return X_q, e

        u = var.reshape((MPC_STEPS, ni))
        X_q, ebar = jax.lax.scan(error_func, X_q_0, u)
        return ebar.flatten()

    def ineq_constraints(X_ee, a_ee, jnp=jnp):
        """Calculate inequality constraints for a single timestep."""
        θ_ew, dθ_ew = X_ee[2], X_ee[5]
        a_ew_w, ddθ_ew = a_ee[:2], a_ee[2]
        # R_ew = jnp.array([[ jnp.cos(θ_ew), jnp.sin(θ_ew)],
        #                   [-jnp.sin(θ_ew), jnp.cos(θ_ew)]])
        R_ew = rotation_matrix(-θ_ew, np=jnp)
        S1 = skew1(1)
        g = jnp.array([0, GRAVITY])

        α1, α2 = (
            TRAY_MASS * R_ew @ (a_ew_w + g)
            + TRAY_MASS * (ddθ_ew * S1 - dθ_ew ** 2 * jnp.eye(2)) @ p_te_e
        )
        α3 = TRAY_INERTIA * ddθ_ew

        # NOTE: this is written to be >= 0
        # h1 = TRAY_MU*α2 - jnp.abs(α1)
        h1a = TRAY_MU * α2 + α1
        h1b = TRAY_MU * α2 - α1
        h2 = α2
        # h2 = 1

        w1 = TRAY_W
        w2 = TRAY_W
        h3a = α3 + w1 * α2 + TRAY_H * α1
        h3b = α3 + w1 * α2 - TRAY_H * α1
        # h3a = 1
        # h3b = 1

        h4a = -α3 + w2 * α2 + TRAY_H * α1
        h4b = -α3 + w2 * α2 - TRAY_H * α1
        # h4a = 1
        # h4b = 1

        return jnp.array([h1a, h1b, h2, h3a, h3b, h4a, h4b])

    def ineq_constraints_unrolled(X_q_0, X_ee_d, var):
        """Unroll the inequality constraints over the time horizon."""

        def ineq_func(X_q, u):

            # we actually two sets of constraints for each timestep: one at the
            # start and one at the end
            # at the start of the timestep, we need to ensure the new inputs
            # satisfy constraints
            X_ee = model.tool_state(X_q)
            a_ee = model.tool_acceleration(X_q, u)
            ineq_con1 = ineq_constraints(X_ee, a_ee)

            X_q = model.simulate(X_q, u)

            # at the end of the timestep, we need to make sure that the robot
            # ends up in a state where constraints are still satisfied given
            # the input
            X_ee = model.tool_state(X_q)
            a_ee = model.tool_acceleration(X_q, u)
            ineq_con2 = ineq_constraints(X_ee, a_ee)

            return X_q, jnp.concatenate((ineq_con1, ineq_con2))

        u = var.reshape((MPC_STEPS, ni))
        X_q, ineq_con = jax.lax.scan(ineq_func, X_q_0, u)
        return ineq_con.flatten()

    err_jac = jax.jit(jax.jacfwd(error_unrolled, argnums=2))
    joint_state_jac = jax.jit(jax.jacfwd(joint_state_unrolled, argnums=1))

    @jax.jit
    def obj_hess_jac(X_q_0, X_ee_d, var):
        """Calculate objective Hessian and Jacobian."""
        u = var

        e = error_unrolled(X_q_0, X_ee_d, u)
        dedu = err_jac(X_q_0, X_ee_d, u)

        x = joint_state_unrolled(X_q_0, u)
        dxdu = joint_state_jac(X_q_0, u)

        # Function
        f = e.T @ Wbar @ e + x.T @ Qbar @ x + u.T @ Rbar @ u

        # Jacobian
        g = e.T @ Wbar @ dedu + x.T @ Qbar @ dxdu + u.T @ Rbar

        # (Approximate) Hessian
        H = dedu.T @ Wbar @ dedu + dxdu.T @ Qbar @ dxdu + Rbar

        return f, g, H

    def vel_ineq_constraints(X_q_0, X_ee_d, var):
        """Inequality constraints on joint velocity."""
        dq0 = X_q_0[ni:]
        return Vbar @ var + jnp.tile(dq0, MPC_STEPS)

    def vel_ineq_jacobian(X_q_0, X_ee_d, var):
        """Jacobian of joint velocity constraints."""
        return Vbar

    @jax.jit
    def con_fun(X_q_0, X_ee_d, var):
        """Combined constraint function."""
        con1 = vel_ineq_constraints(X_q_0, X_ee_d, var)
        con2 = ineq_constraints_unrolled(X_q_0, X_ee_d, var)
        return jnp.concatenate((con1, con2))

    @jax.jit
    def con_jac(X_q_0, X_ee_d, var):
        """Combined constraint Jacobian."""
        J1 = vel_ineq_jacobian(X_q_0, X_ee_d, var)
        J2 = jax.jacfwd(ineq_constraints_unrolled, argnums=2)(X_q_0, X_ee_d, var)
        return jnp.vstack((J1, J2))

    # Construct the SQP controller
    lb_vel = -VEL_LIM * np.ones(MPC_STEPS * ni)
    ub_vel = VEL_LIM * np.ones(MPC_STEPS * ni)

    lb_physics = np.zeros(MPC_STEPS * nc * 2)
    ub_physics = np.infty * np.ones(MPC_STEPS * nc * 2)

    con_lb = np.concatenate((lb_vel, lb_physics))
    con_ub = np.concatenate((ub_vel, ub_physics))

    constraints = sqp.Constraints(con_fun, con_jac, con_lb, con_ub)

    lb_acc = -ACC_LIM * np.ones(MPC_STEPS * nv)
    ub_acc = ACC_LIM * np.ones(MPC_STEPS * nv)
    bounds = sqp.Bounds(lb_acc, ub_acc)

    controller = sqp.SQP(
        nv * MPC_STEPS,
        2 * nc * MPC_STEPS,
        obj_hess_jac,
        constraints,
        bounds,
        num_wsr=300,
        num_iter=SQP_ITER,
        verbose=True,
        solver="qpoases",
    )

    for i in range(N - 1):
        t = i * SIM_DT

        if i % CTRL_PERIOD == 0:
            t_sample = np.minimum(t + MPC_DT * np.arange(MPC_STEPS), DURATION)
            pd, vd, _ = trajectory.sample(t_sample, flatten=False)
            z = np.zeros((MPC_STEPS, 1))
            X_ee_d = np.hstack((pd, z, vd, z)).flatten()
            # -0.5*np.pi*np.ones((MPC_STEPS, 1))

            var = controller.solve(X_q, X_ee_d)
            u = var[:ni]  # joint acceleration input
            robot.command_acceleration(u)

        # step simulation forward
        robot.step()
        pyb.stepSimulation()
        X_q = np.concatenate(robot.joint_states())

        if i % RECORD_PERIOD == 0:
            idx = i // RECORD_PERIOD
            P_ew_w = model.tool_pose(X_q)
            V_ew_w = model.tool_velocity(X_q)

            # NOTE: calculating these quantities is fairly expensive
            X_ee = model.tool_state(X_q)
            a_ee = model.tool_acceleration(X_q, u)
            ineq_cons[idx, :] = ineq_constraints(X_ee, a_ee, jnp=np)

            # TODO this needs to be redone
            # tray position is (ideally) a constant offset from EE frame
            # θ_ew = P_ew_w[2]
            # R_we = rotation_matrix(θ_ew)
            # P_tw_w = np.array(
            #     [tray.body.position.x, tray.body.position.y, tray.body.angle]
            # )

            pd, _, _ = trajectory.sample(t, flatten=False)

            # record
            us[idx, :] = u
            X_qs[idx, :] = X_q
            P_ew_wds[idx, :2] = pd
            P_ew_ws[idx, :] = P_ew_w
            V_ew_ws[idx, :] = V_ew_w
            # P_tw_ws[idx, :] = P_tw_w
            # p_te_es[idx, :] = R_we.T @ (P_tw_w[:2] - P_ew_w[:2])

        if np.linalg.norm(pd - P_ew_w[:2]) < 0.01:
            print("Position within 1 cm.")
            setpoint_idx += 1
            if setpoint_idx >= setpoints.shape[0]:
                break

            trajectory = trajectories.Point(setpoints[setpoint_idx, :])

            # update pd to avoid falling back into this block right away
            pd, _, _ = trajectory.sample(t, flatten=False)

    controller.benchmark.print_stats()

    print(np.min(ineq_cons))

    idx = i // RECORD_PERIOD

    plt.figure()
    plt.plot(ts[1:idx], P_ew_wds[1:idx, 0], label="$x_d$", color="b", linestyle="--")
    plt.plot(ts[1:idx], P_ew_wds[1:idx, 1], label="$y_d$", color="r", linestyle="--")
    plt.plot(ts[1:idx], P_ew_ws[1:idx, 0], label="$x$", color="b")
    plt.plot(ts[1:idx], P_ew_ws[1:idx, 1], label="$y$", color="r")
    # plt.plot(ts[:i], P_tw_ws[:i, 0],  label='$t_x$')
    # plt.plot(ts[:i], P_tw_ws[:i, 1],  label='$t_y$')
    plt.grid()
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.title("End effector position")

    plt.figure()
    plt.plot(ts[:idx], p_te_e[0] - p_te_es[:idx, 0], label="$x$", color="b")
    plt.plot(ts[:idx], p_te_e[1] - p_te_es[:idx, 1], label="$y$", color="r")
    plt.grid()
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("$p^{te}_e$ error")
    plt.title("$p^{te}_e$ error")

    plt.figure()
    for j in range(nc_ineq):
        plt.plot(ts[:idx], ineq_cons[:idx, j], label=f"$h_{j+1}$")
    # plt.plot(ts[:N], ineq_cons[:, 1], label='$h_2$')
    # plt.plot(ts[:N], ineq_cons[:, 2], label='$h_3$')
    # plt.plot(ts[:N], ineq_cons[:, 3], label='$h_4$')
    plt.plot([0, ts[idx]], [0, 0], color="k")
    plt.grid()
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.title("Inequality constraints")

    plt.figure()
    plt.plot(ts[:idx], us[:idx, 0], label="$u_1$")
    plt.plot(ts[:idx], us[:idx, 1], label="$u_2$")
    plt.plot(ts[:idx], us[:idx, 2], label="$u_3$")
    plt.plot(ts[:idx], us[:idx, 3], label="$u_4$")
    plt.grid()
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Commanded joint acceleration")
    plt.title("Acceleration commands")

    plt.figure()
    plt.plot(ts[:idx], X_qs[:idx, 4], label=r"$\dot{q}_1$")
    plt.plot(ts[:idx], X_qs[:idx, 5], label=r"$\dot{q}_2$")
    plt.plot(ts[:idx], X_qs[:idx, 6], label=r"$\dot{q}_3$")
    plt.plot(ts[:idx], X_qs[:idx, 7], label=r"$\dot{q}_4$")
    plt.grid()
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocities")
    plt.title("Joint velocities")

    plt.show()


if __name__ == "__main__":
    main()
