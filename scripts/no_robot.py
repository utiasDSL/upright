#!/usr/bin/env python
"""Baseline tray balancing formulation."""
from functools import partial

import jax.numpy as jnp
import jax
from jaxlie import SO3, SE3
import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb
import pybullet_data

import sqp
import util
from util import rot2d, skew1, pose_error, pose_to_pos_quat, pose_from_pos_quat
from tray import Tray
from robot import SimulatedRobot, RobotModel

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

INPUT_MASK = np.array([True, False, False, False, True, True, True, False, False])
BASE_VEL_LIM = [1, 1, 2]
ARM_VEL_LIM = [2.16, 2.16, 3.15, 3.2, 3.2, 3.2]
ROBOT_VEL_LIM = np.array(BASE_VEL_LIM + ARM_VEL_LIM)
PLANAR_VEL_LIM = ROBOT_VEL_LIM[INPUT_MASK]


# robot parameters
ACC_LIM = 8  # TODO

GRAVITY = 9.81

# tray parameters
TRAY_RADIUS = 0.25
TRAY_MASS = 0.5
TRAY_MU = 0.5
TRAY_W = 0.085
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


class TrayBalanceOptimization:
    def __init__(self, model, p_te_e):
        self.model = model
        self.p_te_e = np.array([p_te_e[0], p_te_e[2]])  # NOTE

        self.nv = model.ni  # number of optimization variables per MPC step
        self.nc_eq = 0  # number of equality constraints
        self.nc_ineq = 7  # number of inequality constraints
        self.nc = self.nc_eq + self.nc_ineq

        self.ns_q = 2 * model.ni  # dimension of joint state
        self.ns_ee = 12  # dimension of EE (Cartesian) states

        # MPC weights
        Q = np.diag(
            np.concatenate((np.zeros(model.ni), 0.01 * np.ones(model.ni)))
        )  # joint state error
        W = np.diag(np.concatenate((np.ones(6), np.zeros(6))))  # Cartesian state error
        R = 0.01 * np.eye(self.nv)
        V = MPC_DT * np.eye(self.nv)

        # lifted weight matrices
        Ibar = np.eye(MPC_STEPS)
        self.Qbar = np.kron(Ibar, Q)
        self.Wbar = np.kron(Ibar, W)
        self.Rbar = np.kron(Ibar, R)

        # velocity constraint matrix
        self.Vbar = np.kron(np.tril(np.ones((MPC_STEPS, MPC_STEPS))), V)

        self.err_jac = jax.jit(jax.jacfwd(self.error_unrolled, argnums=3))
        self.joint_state_jac = jax.jit(jax.jacfwd(self.joint_state_unrolled, argnums=1))

    @partial(jax.jit, static_argnums=(0,))
    def joint_state_unrolled(self, X_q_0, ubar):
        """Unroll the joint state of the robot over the time horizon."""

        def state_func(X_q, u):
            X_q = self.model.simulate(X_q, u)
            return X_q, X_q

        u = ubar.reshape((MPC_STEPS, self.model.ni))
        _, X_q_bar = jax.lax.scan(state_func, X_q_0, u)
        return X_q_bar.flatten()

    @partial(jax.jit, static_argnums=(0,))
    def error_unrolled(self, X_q_0, P_we_d, V_ee_d, var):
        """Unroll the pose error over the time horizon."""

        def error_func(X_q, u):
            X_q = self.model.simulate(X_q, u)
            P_we = self.model.tool_pose(X_q)
            V_ee = self.model.tool_velocity(X_q)

            pose_err = pose_error(P_we_d, P_we)
            V_err = V_ee_d - V_ee

            e = jnp.concatenate((pose_err, V_err))
            return X_q, e

        u = var.reshape((MPC_STEPS, self.model.ni))
        X_q, ebar = jax.lax.scan(error_func, X_q_0, u)
        return ebar.flatten()

    @partial(jax.jit, static_argnums=(0,))
    def ineq_constraints(self, P_we, V_ew_w, A_ew_w, jnp=jnp):
        """Calculate inequality constraints for a single timestep."""
        _, Q_we = pose_to_pos_quat(P_we)
        θ_ew = util.pitch_from_quat(Q_we)
        dθ_ew = V_ew_w[5]  # pitch
        a_ew_w = jnp.array([A_ew_w[0], A_ew_w[2]])
        ddθ_ew = A_ew_w[5]

        R_ew = rot2d(-θ_ew, np=jnp)
        S1 = skew1(1)
        g = jnp.array([0, GRAVITY])

        α1, α2 = (
            TRAY_MASS * R_ew @ (a_ew_w + g)
            + TRAY_MASS * (ddθ_ew * S1 - dθ_ew ** 2 * jnp.eye(2)) @ self.p_te_e
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

    @partial(jax.jit, static_argnums=(0,))
    def ineq_constraints_unrolled(self, X_q_0, P_we_d, V_ew_w_d, var):
        """Unroll the inequality constraints over the time horizon."""

        def ineq_func(X_q, u):

            # we actually two sets of constraints for each timestep: one at the
            # start and one at the end
            # at the start of the timestep, we need to ensure the new inputs
            # satisfy constraints
            P_we = self.model.tool_pose(X_q)
            V_ee = self.model.tool_velocity(X_q)
            a_ee = self.model.tool_acceleration(X_q, u)
            ineq_con1 = self.ineq_constraints(P_we, V_ee, a_ee)

            X_q = self.model.simulate(X_q, u)

            # at the end of the timestep, we need to make sure that the robot
            # ends up in a state where constraints are still satisfied given
            # the input
            P_we = self.model.tool_pose(X_q)
            V_ee = self.model.tool_velocity(X_q)
            a_ee = self.model.tool_acceleration(X_q, u)
            ineq_con2 = self.ineq_constraints(P_we, V_ee, a_ee)

            return X_q, jnp.concatenate((ineq_con1, ineq_con2))

        u = var.reshape((MPC_STEPS, self.model.ni))
        X_q, ineq_con = jax.lax.scan(ineq_func, X_q_0, u)
        return ineq_con.flatten()

    @partial(jax.jit, static_argnums=(0,))
    def obj_hess_jac(self, X_q_0, P_we_d, V_ew_w_d, var):
        """Calculate objective Hessian and Jacobian."""
        u = var

        e = self.error_unrolled(X_q_0, P_we_d, V_ew_w_d, u)
        dedu = self.err_jac(X_q_0, P_we_d, V_ew_w_d, u)

        x = self.joint_state_unrolled(X_q_0, u)
        dxdu = self.joint_state_jac(X_q_0, u)

        # Function
        f = e.T @ self.Wbar @ e + x.T @ self.Qbar @ x + u.T @ self.Rbar @ u

        # Jacobian
        g = e.T @ self.Wbar @ dedu + x.T @ self.Qbar @ dxdu + u.T @ self.Rbar

        # (Approximate) Hessian
        H = dedu.T @ self.Wbar @ dedu + dxdu.T @ self.Qbar @ dxdu + self.Rbar

        return f, g, H

    @partial(jax.jit, static_argnums=(0,))
    def vel_ineq_constraints(self, X_q_0, P_we_d, V_ew_w_d, var):
        """Inequality constraints on joint velocity."""
        dq0 = X_q_0[self.model.ni :]
        return self.Vbar @ var + jnp.tile(dq0, MPC_STEPS)

    @partial(jax.jit, static_argnums=(0,))
    def vel_ineq_jacobian(self, X_q_0, P_we_d, V_ew_w_d, var):
        """Jacobian of joint velocity constraints."""
        return self.Vbar

    @partial(jax.jit, static_argnums=(0,))
    def con_fun(self, X_q_0, P_we_d, V_ew_w_d, var):
        """Combined constraint function."""
        con1 = self.vel_ineq_constraints(X_q_0, P_we_d, V_ew_w_d, var)
        con2 = self.ineq_constraints_unrolled(X_q_0, P_we_d, V_ew_w_d, var)
        return jnp.concatenate((con1, con2))

    @partial(jax.jit, static_argnums=(0,))
    def con_jac(self, X_q_0, P_we_d, V_ew_w_d, var):
        """Combined constraint Jacobian."""
        J1 = self.vel_ineq_jacobian(X_q_0, P_we_d, V_ew_w_d, var)
        J2 = jax.jacfwd(self.ineq_constraints_unrolled, argnums=3)(
            X_q_0, P_we_d, V_ew_w_d, var
        )
        return jnp.vstack((J1, J2))

    def bounds(self):
        """Compute bounds for the problem."""
        lb_acc = -ACC_LIM * np.ones(MPC_STEPS * self.nv)
        ub_acc = ACC_LIM * np.ones(MPC_STEPS * self.nv)
        return sqp.Bounds(lb_acc, ub_acc)

    def constraints(self):
        """Compute the constraints for the problem."""
        lb_vel = np.tile(-ROBOT_VEL_LIM, MPC_STEPS)
        ub_vel = np.tile(ROBOT_VEL_LIM, MPC_STEPS)

        lb_physics = np.zeros(MPC_STEPS * self.nc * 2)
        ub_physics = np.infty * np.ones(MPC_STEPS * self.nc * 2)

        con_lb = np.concatenate((lb_vel, lb_physics))
        con_ub = np.concatenate((ub_vel, ub_physics))

        return sqp.Constraints(self.con_fun, self.con_jac, con_lb, con_ub)


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
    pyb.startStateLogging(pyb.STATE_LOGGING_VIDEO_MP4, "tray3d.mp4")

    # setup ground plane
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pyb.loadURDF("plane.urdf", [0, 0, 0])

    # setup robot
    # robot = SimulatedRobot(SIM_DT)
    # robot.reset_joint_configuration(ROBOT_HOME)
    # TODO want just an EE

    # simulate briefly to let the robot settle down after being positioned
    settle_sim(1.0)

    # arm gets bumped by the above settling, so we reset it again
    # robot.reset_arm_joints(UR10_HOME_TRAY_BALANCE)

    # setup tray
    tray = Tray(mass=TRAY_MASS, radius=TRAY_RADIUS, height=2 * TRAY_H, mu=TRAY_MU)
    ee_pos, _ = robot.link_pose()
    tray.reset_pose(position=ee_pos + [0, 0, TRAY_H + 0.05])

    settle_sim(1.0)

    return robot, tray


def calc_r_te_e(P_we, r_tw_w):
    r_ew_w, Q_we = pose_to_pos_quat(P_we)
    r_te_w = r_tw_w - r_ew_w
    return SO3.from_quaternion_xyzw(Q_we) @ r_te_w


def main():
    np.set_printoptions(precision=3, suppress=True)

    if TRAY_W < TRAY_MU * TRAY_H:
        print("warning: w < μh")

    N = int(DURATION / SIM_DT) + 1
    N_record = int(DURATION / (SIM_DT * RECORD_PERIOD))

    # simulation objects and model
    robot, tray = setup_sim()
    model = RobotModel(MPC_DT, ROBOT_HOME)

    # state of joints
    q0, dq0 = robot.joint_states()
    X_q = np.concatenate((q0, np.zeros_like(dq0)))

    P_we = model.tool_pose(X_q)
    r_ew_w, Q_we = pose_to_pos_quat(P_we)
    V_ew_w = model.tool_velocity(X_q)
    r_tw_w, _ = tray.get_pose()
    r_te_e = calc_r_te_e(P_we, r_tw_w)

    # desired quaternion: same as the starting orientation
    _, Qd = pose_to_pos_quat(P_we)

    # construct the tray balance problem
    problem = TrayBalanceOptimization(model, r_te_e)

    ts = RECORD_PERIOD * SIM_DT * np.arange(N_record)
    us = np.zeros((N_record, model.ni))
    p_ew_ws = np.zeros((N_record, 3))
    p_ew_wds = np.zeros((N_record, 3))
    V_ew_ws = np.zeros((N_record, 6))
    p_tw_ws = np.zeros((N_record, 3))
    p_te_es = np.zeros((N_record, 3))
    X_qs = np.zeros((N_record, problem.ns_q))
    ineq_cons = np.zeros((N_record, problem.nc_ineq))

    p_te_es[0, :] = r_te_e
    p_ew_ws[0, :] = r_ew_w
    V_ew_ws[0, :] = V_ew_w

    # reference trajectory
    setpoints = np.array([[1, 0, -0.5], [2, 0, -0.5], [3, 0, 0.5]]) + r_ew_w
    setpoint_idx = 0
    # trajectory = trajectories.Point(setpoints[setpoint_idx, :])

    # Construct the SQP controller
    controller = sqp.SQP(
        problem.nv * MPC_STEPS,
        2 * problem.nc * MPC_STEPS,
        problem.obj_hess_jac,
        problem.constraints(),
        problem.bounds(),
        num_wsr=300,
        num_iter=SQP_ITER,
        verbose=True,
        solver="qpoases",
    )

    for i in range(N - 1):
        # t = i * SIM_DT

        if i % CTRL_PERIOD == 0:
            r_ew_w_d = setpoints[setpoint_idx, :]
            P_we_d = pose_from_pos_quat(r_ew_w_d, Qd)
            V_we_d = jnp.zeros(6)

            var = controller.solve(X_q, P_we_d, V_we_d)
            u = var[: model.ni]  # joint acceleration input
            robot.command_acceleration(u)

        # step simulation forward
        robot.step()
        pyb.stepSimulation()
        X_q = np.concatenate(robot.joint_states())

        if i % RECORD_PERIOD == 0:
            idx = i // RECORD_PERIOD
            P_we = model.tool_pose(X_q)
            V_ew_w = model.tool_velocity(X_q)

            # NOTE: calculating these quantities is fairly expensive
            P_we = model.tool_pose(X_q)
            V_ew_w = model.tool_velocity(X_q)
            A_ew_w = model.tool_acceleration(X_q, u)
            ineq_cons[idx, :] = np.array(problem.ineq_constraints(P_we, V_ew_w, A_ew_w))

            r_tw_w, Q_wt = tray.get_pose()
            r_ew_w_d = setpoints[setpoint_idx, :]

            # record
            us[idx, :] = u
            X_qs[idx, :] = X_q
            p_ew_wds[idx, :] = r_ew_w_d
            p_ew_ws[idx, :] = pose_to_pos_quat(P_we)[0]
            V_ew_ws[idx, :] = V_ew_w
            p_tw_ws[idx, :] = r_tw_w
            p_te_es[idx, :] = calc_r_te_e(P_we, r_tw_w)

        if np.linalg.norm(r_ew_w_d - P_we[:3]) < 0.01:
            print("Position within 1 cm.")
            setpoint_idx += 1
            if setpoint_idx >= setpoints.shape[0]:
                break

            # update r_ew_w_d to avoid falling back into this block right away
            r_ew_w_d = setpoints[setpoint_idx, :]

    controller.benchmark.print_stats()

    print(np.min(ineq_cons))

    idx = i // RECORD_PERIOD

    plt.figure()
    plt.plot(ts[1:idx], p_ew_wds[1:idx, 0], label="$x_d$", color="b", linestyle="--")
    plt.plot(ts[1:idx], p_ew_wds[1:idx, 2], label="$z_d$", color="r", linestyle="--")
    plt.plot(ts[1:idx], p_ew_ws[1:idx, 0], label="$x$", color="b")
    plt.plot(ts[1:idx], p_ew_ws[1:idx, 2], label="$z$", color="r")
    plt.grid()
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.title("End effector position")

    plt.figure()
    plt.plot(ts[:idx], r_te_e[0] - p_te_es[:idx, 0], label="$x$", color="b")
    plt.plot(ts[:idx], r_te_e[2] - p_te_es[:idx, 2], label="$z$", color="r")
    plt.grid()
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("$p^{te}_e$ error")
    plt.title("$p^{te}_e$ error")

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
        # TODO: size or indexing of us is wrong:
        # >    plt.plot(ts[:idx], us[:idx, i], label=f"$u_{j+1}$")
        # > IndexError: index 8100 is out of bounds for axis 1 with size 9
        plt.plot(ts[:idx], us[:idx, j], label=f"$u_{j+1}$")
    plt.grid()
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Commanded joint acceleration")
    plt.title("Acceleration commands")

    # plt.figure()
    # plt.plot(ts[:idx], X_qs[:idx, 4], label=r"$\dot{q}_1$")
    # plt.plot(ts[:idx], X_qs[:idx, 5], label=r"$\dot{q}_2$")
    # plt.plot(ts[:idx], X_qs[:idx, 6], label=r"$\dot{q}_3$")
    # plt.plot(ts[:idx], X_qs[:idx, 7], label=r"$\dot{q}_4$")
    # plt.grid()
    # plt.legend()
    # plt.xlabel("Time (s)")
    # plt.ylabel("Joint velocities")
    # plt.title("Joint velocities")

    plt.show()


if __name__ == "__main__":
    main()
