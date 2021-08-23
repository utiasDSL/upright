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
import util
from bodies import Cylinder, Cuboid, compose_bodies
from end_effector import EndEffector, EndEffectorModel
import geometry
import balancing

import IPython


# EE geometry parameters
EE_SIDE_LENGTH = 0.3
EE_INSCRIBED_RADIUS = geometry.equilateral_triangle_inscribed_radius(EE_SIDE_LENGTH)

# EE motion parameters
VEL_LIM = 4
ACC_LIM = 8

GRAVITY_MAG = 9.81
GRAVITY_VECTOR = np.array([0, 0, -GRAVITY_MAG])

# tray parameters
TRAY_RADIUS = 0.25
TRAY_MASS = 0.5
TRAY_MU = 0.5
TRAY_COM_HEIGHT = 0.01
TRAY_NUM_ZMP_CONSTRAINTS = 1
TRAY_NUM_CONSTRAINTS = 2 + TRAY_NUM_ZMP_CONSTRAINTS

OBJ_MASS = 1
OBJ_TRAY_MU = 0.5
OBJ_TRAY_MU_BULLET = OBJ_TRAY_MU / TRAY_MU
OBJ_RADIUS = 0.1
OBJ_SIDE_LENGTHS = (0.2, 0.2, 0.4)
OBJ_COM_HEIGHT = 0.2
OBJ_ZMP_MARGIN = 0.01
OBJ_NUM_ZMP_CONSTRAINTS = 4
OBJ_NUM_CONSTRAINTS = 2 + OBJ_NUM_ZMP_CONSTRAINTS

NUM_OBJECTS = 2

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
    def __init__(self, model, tray, obj):
        self.model = model

        # create the composite
        obj_tray_composite = tray.copy()
        obj_tray_composite.body = compose_bodies([tray.body, obj.body])
        delta = tray.body.com - obj_tray_composite.body.com
        obj_tray_composite.support_area.offset = delta[:2]
        obj_tray_composite.com_height = tray.com_height - delta[2]

        self.obj_to_constrain = [obj_tray_composite, obj]
        assert len(self.obj_to_constrain) == NUM_OBJECTS

        self.nv = model.ni  # number of optimization variables per MPC step
        self.nc_eq = 0  # number of equality constraints

        # NOTE: n_balance_con multiplied by 2 because we enforce two
        # constraints per timestep
        self.n_balance_con = TRAY_NUM_CONSTRAINTS + OBJ_NUM_CONSTRAINTS
        self.nc = self.nc_eq + 2 * self.n_balance_con

        # MPC weights
        # Joint state weight
        Q_POSE = np.zeros(7)
        Q_VELOCITY = 0.01 * np.ones(model.ni)
        Q = np.diag(np.concatenate((Q_POSE, Q_VELOCITY)))

        # Cartesian state error weight
        W_EE_POSE = np.ones(6)
        W_EE_VELOCITY = np.zeros(6)
        W = np.diag(np.concatenate((W_EE_POSE, W_EE_VELOCITY)))

        # Input weight
        R = 0.01 * np.eye(self.nv)

        # Velocity constraint matrix
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

            pose_err = util.pose_error(P_we_d, P_we)
            V_err = V_ee_d - V_ew_w

            e = jnp.concatenate((pose_err, V_err))
            return X, e

        u = var.reshape((MPC_STEPS, self.model.ni))
        X, ebar = jax.lax.scan(error_func, X0, u)
        return ebar.flatten()

    @partial(jax.jit, static_argnums=(0,))
    def ineq_constraints(self, P_we, V_ew_w, A_ew_w, jnp=jnp):
        """Calculate inequality constraints for a single timestep."""
        _, Q_we = util.pose_to_pos_quat(P_we)
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
    def obj_fun(self, X0, P_we_d, V_ew_w_d, var):
        ubar = var
        e = self.error_unrolled(X0, P_we_d, V_ew_w_d, ubar)
        x = self.state_unrolled(X0, ubar)
        f = e.T @ self.Wbar @ e + x.T @ self.Qbar @ x + ubar.T @ self.Rbar @ ubar
        return f

    @partial(jax.jit, static_argnums=(0,))
    def obj_jac(self, X0, P_we_d, V_ew_w_d, var):
        ubar = var

        e = self.error_unrolled(X0, P_we_d, V_ew_w_d, ubar)
        dedu = self.err_jac(X0, P_we_d, V_ew_w_d, ubar)

        x = self.state_unrolled(X0, ubar)
        dxdu = self.state_jac(X0, ubar)

        g = e.T @ self.Wbar @ dedu + x.T @ self.Qbar @ dxdu + ubar.T @ self.Rbar
        return g

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

        lb_physics = np.zeros(MPC_STEPS * self.nc)
        ub_physics = np.infty * np.ones(MPC_STEPS * self.nc)

        con_lb = np.concatenate((lb_vel, lb_physics))
        con_ub = np.concatenate((ub_vel, ub_physics))

        # constraint mask for velocity constraints is static
        vel_con_mask = self.Vbar

        physics_con_mask = np.kron(
            np.tril(np.ones((MPC_STEPS, MPC_STEPS))), np.ones((self.nc, self.nv))
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

    pyb.setGravity(0, 0, -GRAVITY_MAG)
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

    util.debug_frame(0.1, ee.uid, -1)

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

    # object on tray
    # obj = Cylinder(
    #     r_tau=geometry.circle_r_tau(OBJ_RADIUS),
    #     support_area=geometry.CircleSupportArea(OBJ_RADIUS, margin=OBJ_ZMP_MARGIN),
    #     mass=OBJ_MASS,
    #     radius=OBJ_RADIUS,
    #     height=2 * OBJ_COM_HEIGHT,
    #     mu=OBJ_TRAY_MU,
    #     bullet_mu=OBJ_TRAY_MU_BULLET,
    #     color=(0, 1, 0, 1),
    # )
    # obj.bullet.reset_pose(position=ee_pos + [0, 0, 2 * TRAY_COM_HEIGHT + OBJ_COM_HEIGHT + 0.05])

    obj_support = geometry.PolygonSupportArea(
        geometry.cuboid_support_vertices(OBJ_SIDE_LENGTHS), margin=OBJ_ZMP_MARGIN
    )
    obj = Cuboid(
        r_tau=geometry.circle_r_tau(OBJ_RADIUS),
        support_area=obj_support,
        mass=OBJ_MASS,
        side_lengths=OBJ_SIDE_LENGTHS,
        mu=OBJ_TRAY_MU,
        bullet_mu=OBJ_TRAY_MU_BULLET,
        color=(0, 1, 0, 1),
    )
    obj.bullet.reset_pose(
        position=ee_pos + [0.05, 0, 2 * TRAY_COM_HEIGHT + 0.5 * OBJ_SIDE_LENGTHS[2] + 0.05]
    )

    settle_sim(1.0)

    return ee, tray, obj


def main():
    np.set_printoptions(precision=3, suppress=True)

    if EE_INSCRIBED_RADIUS < TRAY_MU * TRAY_COM_HEIGHT:
        print("warning: w < μh")

    N = int(DURATION / SIM_DT) + 1
    N_record = int(DURATION / (SIM_DT * RECORD_PERIOD))

    # simulation objects and model
    ee, tray, obj = setup_sim()

    model = EndEffectorModel(MPC_DT)

    x = ee.get_state()
    r_ew_w, Q_we = ee.get_pose()
    v_ew_w, ω_ew_w = ee.get_velocity()

    # set CoMs relative to EE
    r_tw_w, Q_wt = tray.bullet.get_pose()
    tray.body.com = util.calc_r_te_e(r_ew_w, Q_we, r_tw_w)

    r_ow_w, Q_wo = obj.bullet.get_pose()
    obj.body.com = util.calc_r_te_e(r_ew_w, Q_we, r_ow_w)

    # construct the tray balance problem
    problem = TrayBalanceOptimizationEE(model, tray, obj)

    ts = RECORD_PERIOD * SIM_DT * np.arange(N_record)
    us = np.zeros((N_record, model.ni))
    r_ew_ws = np.zeros((N_record, 3))
    r_ew_wds = np.zeros((N_record, 3))
    Q_wes = np.zeros((N_record, 4))
    Q_des = np.zeros((N_record, 4))
    v_ew_ws = np.zeros((N_record, 3))
    ω_ew_ws = np.zeros((N_record, 3))
    r_tw_ws = np.zeros((N_record, 3))
    ineq_cons = np.zeros((N_record, problem.n_balance_con))

    r_te_es = np.zeros((N_record, 3))
    Q_ets = np.zeros((N_record, 4))

    r_oe_es = np.zeros((N_record, 3))
    Q_eos = np.zeros((N_record, 4))

    r_ot_ts = np.zeros((N_record, 3))
    Q_tos = np.zeros((N_record, 4))

    r_te_es[0, :] = tray.body.com
    r_oe_es[0, :] = obj.body.com
    r_ot_ts[0, :] = util.calc_r_te_e(r_tw_w, Q_wt, r_ow_w)
    Q_wes[0, :] = Q_we
    Q_ets[0, :] = util.calc_Q_et(Q_we, Q_wt)
    Q_eos[0, :] = util.calc_Q_et(Q_we, Q_wo)
    Q_tos[0, :] = util.calc_Q_et(Q_wt, Q_wo)
    r_ew_ws[0, :] = r_ew_w
    v_ew_ws[0, :] = v_ew_w
    ω_ew_ws[0, :] = ω_ew_w

    # reference trajectory
    # setpoints = np.array([[1, 0, -0.5], [2, 0, -0.5], [3, 0, 0.5]]) + r_ew_w
    setpoints = np.array([[2, 0, 0]]) + r_ew_w
    setpoint_idx = 0

    # desired quaternion
    R_ed = SO3.from_z_radians(np.pi)
    # R_ed = SO3.identity()
    R_we = SO3.from_quaternion_xyzw(Q_we)
    R_wd = R_we.multiply(R_ed)
    Qd = R_wd.as_quaternion_xyzw()
    Qd_inv = R_wd.inverse().as_quaternion_xyzw()

    Q_des[0, :] = util.quat_multiply(Qd_inv, Q_we)

    # Construct the SQP controller
    # controller = sqp.SQP(
    #     problem.nv * MPC_STEPS,
    #     problem.nc * MPC_STEPS,
    #     problem.obj_hess_jac,
    #     problem.constraints(),
    #     problem.bounds(),
    #     num_wsr=300,
    #     num_iter=SQP_ITER,
    #     verbose=False,
    #     solver="qpoases",
    # )

    controller = sqp.SQP(
        nv=problem.nv * MPC_STEPS,
        nc=problem.nc * MPC_STEPS,
        obj_fun=problem.obj_fun,
        obj_jac=problem.obj_jac,
        ineq_cons=problem.constraints(),
        eq_cons=None,
        bounds=problem.bounds(),
        num_iter=SQP_ITER,
        verbose=False,
        solver="scipy",
    )

    x_pred = np.zeros(7 + 6)

    for i in range(N - 1):
        if i % CTRL_PERIOD == 0:
            r_ew_w_d = setpoints[setpoint_idx, :]
            P_we_d = util.pose_from_pos_quat(r_ew_w_d, Qd)
            V_we_d = jnp.zeros(6)

            x_ctrl = ee.get_state()
            var = controller.solve(x_ctrl, P_we_d, V_we_d)
            u = var[: model.ni]  # joint acceleration input
            ee.command_acceleration(u)

            # evaluate model error
            if i > 0:
                model_error = np.linalg.norm(util.state_error(x_ctrl, x_pred))
                print(model_error)
                if model_error > 0.1:
                    IPython.embed()

            x_pred = model.simulate(x_ctrl, u)

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
            r_ow_w, Q_wo = obj.bullet.get_pose()
            r_ew_w_d = setpoints[setpoint_idx, :]

            # orientation error
            Q_de = util.quat_multiply(Qd_inv, Q_we)

            # record
            us[idx, :] = u
            r_ew_wds[idx, :] = r_ew_w_d
            r_ew_ws[idx, :] = r_ew_w
            Q_wes[idx, :] = Q_we
            Q_des[idx, :] = Q_de
            v_ew_ws[idx, :] = v_ew_w
            ω_ew_ws[idx, :] = ω_ew_w
            r_tw_ws[idx, :] = r_tw_w
            r_te_es[idx, :] = util.calc_r_te_e(r_ew_w, Q_we, r_tw_w)
            r_oe_es[idx, :] = util.calc_r_te_e(r_ew_w, Q_we, r_ow_w)
            r_ot_ts[idx, :] = util.calc_r_te_e(r_tw_w, Q_wt, r_ow_w)
            Q_ets[idx, :] = util.calc_Q_et(Q_we, Q_wt)
            Q_eos[idx, :] = util.calc_Q_et(Q_we, Q_wo)
            Q_tos[idx, :] = util.calc_Q_et(Q_wt, Q_wo)

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
    r_te_e_err = tray.body.com - r_te_es[:idx, :]
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
    r_oe_e_err = obj.body.com - r_oe_es[:idx, :]
    plt.plot(ts[:idx], r_oe_e_err[:, 0], label="$x$")
    plt.plot(ts[:idx], r_oe_e_err[:, 1], label="$y$")
    plt.plot(ts[:idx], r_oe_e_err[:, 2], label="$z$")
    plt.plot(ts[:idx], Q_eos[:idx, 0], label="$Q_x$")
    plt.plot(ts[:idx], Q_eos[:idx, 1], label="$Q_y$")
    plt.plot(ts[:idx], Q_eos[:idx, 2], label="$Q_z$")
    plt.grid()
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("$r^{oe}_e$ error")
    plt.title("$r^{oe}_e$ error")

    plt.figure()
    r_ot_t_err = r_ot_ts[0, :] - r_ot_ts[:idx, :]
    plt.plot(ts[:idx], r_ot_t_err[:, 0], label="$x$")
    plt.plot(ts[:idx], r_ot_t_err[:, 1], label="$y$")
    plt.plot(ts[:idx], r_ot_t_err[:, 2], label="$z$")
    plt.plot(ts[:idx], Q_tos[:idx, 0], label="$Q_x$")
    plt.plot(ts[:idx], Q_tos[:idx, 1], label="$Q_y$")
    plt.plot(ts[:idx], Q_tos[:idx, 2], label="$Q_z$")
    plt.grid()
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("$r^{ot}_t$ error")
    plt.title("$r^{ot}_t$ error")

    plt.figure()
    for j in range(problem.n_balance_con):
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
