#!/usr/bin/env python
"""Baseline tray balancing formulation."""
from functools import partial

import jax.numpy as jnp
import jax
from jaxlie import SO3
import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb

import sqp
import util
from end_effector import EndEffectorModel
import balancing
from simulation import Simulation
from recording import Recorder

import IPython


# EE motion parameters
VEL_LIM = 4
ACC_LIM = 8

# simulation parameters
MPC_DT = 0.1  # lookahead timestep of the controller
MPC_STEPS = 20  # number of timesteps to lookahead
SQP_ITER = 3  # number of iterations for the SQP solved by the controller
CTRL_PERIOD = 100  # generate new control signal every CTRL_PERIOD timesteps
RECORD_PERIOD = 10
DURATION = 10.0  # duration of trajectory (s)


class TrayBalanceOptimizationEE:
    def __init__(self, model, obj_to_constrain):
        self.model = model
        self.obj_to_constrain = obj_to_constrain

        self.nv = model.ni  # number of optimization variables per MPC step
        self.nc_eq = 0  # number of equality constraints

        # each object has 1 normal constraint, 1 friction constraint, and >= 1
        # ZMP constraint depending on the geometry of the support area
        num_obj = len(self.obj_to_constrain)
        self.n_balance_con = 2 * num_obj + sum(
            [obj.support_area.num_constraints for obj in self.obj_to_constrain]
        )

        # n_balance_con multiplied by 2 because we enforce two
        # constraints per timestep
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

    def qpoases_controller(self):
        # TODO not sure if this currently works
        return sqp.SQP(
            self.nv * MPC_STEPS,
            self.nc * MPC_STEPS,
            self.obj_hess_jac,
            self.constraints(),
            self.bounds(),
            num_wsr=300,
            num_iter=SQP_ITER,
            verbose=False,
            solver="qpoases",
        )

    def scipy_controller(self):
        return sqp.SQP(
            nv=self.nv * MPC_STEPS,
            obj_fun=self.obj_fun,
            obj_jac=self.obj_jac,
            ineq_cons=self.constraints(),
            eq_cons=None,
            bounds=self.bounds(),
            num_iter=SQP_ITER,
            verbose=False,
            solver="scipy",
        )


def main():
    np.set_printoptions(precision=3, suppress=True)

    sim = Simulation(dt=0.001)

    N = int(DURATION / sim.dt) + 1

    # simulation objects and model
    ee, objects, composites = sim.setup(obj_names=["tray", "cuboid1"])
    tray = objects["tray"]
    obj = objects["cuboid1"]

    robot_model = EndEffectorModel(MPC_DT)

    x = ee.get_state()
    r_ew_w, Q_we = ee.get_pose()
    v_ew_w, ω_ew_w = ee.get_velocity()
    r_tw_w, Q_wt = tray.bullet.get_pose()
    r_ow_w, Q_wo = obj.bullet.get_pose()

    # construct the tray balance problem
    problem = TrayBalanceOptimizationEE(robot_model, composites)

    # data recorder and plotter
    recorder = Recorder(
        sim.dt, DURATION, RECORD_PERIOD, model=robot_model, problem=problem
    )

    recorder.r_te_es[0, :] = tray.body.com
    recorder.r_oe_es[0, :] = obj.body.com
    recorder.r_ot_ts[0, :] = util.calc_r_te_e(r_tw_w, Q_wt, r_ow_w)
    recorder.Q_wes[0, :] = Q_we
    recorder.Q_ets[0, :] = util.calc_Q_et(Q_we, Q_wt)
    recorder.Q_eos[0, :] = util.calc_Q_et(Q_we, Q_wo)
    recorder.Q_tos[0, :] = util.calc_Q_et(Q_wt, Q_wo)
    recorder.r_ew_ws[0, :] = r_ew_w
    recorder.v_ew_ws[0, :] = v_ew_w
    recorder.ω_ew_ws[0, :] = ω_ew_w

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

    recorder.Q_des[0, :] = util.quat_multiply(Qd_inv, Q_we)

    # Construct the SQP controller
    controller = problem.scipy_controller()

    for i in range(N - 1):
        if i % CTRL_PERIOD == 0:
            r_ew_w_d = setpoints[setpoint_idx, :]
            P_we_d = util.pose_from_pos_quat(r_ew_w_d, Qd)
            V_we_d = jnp.zeros(6)

            x_ctrl = ee.get_state()
            var = controller.solve(x_ctrl, P_we_d, V_we_d)
            u = var[: robot_model.ni]  # joint acceleration input
            ee.command_acceleration(u)

            print(f"u = {u}")

        # step simulation forward
        ee.step()
        pyb.stepSimulation()

        if recorder.now_is_the_time(i):
            idx = recorder.record_index(i)
            r_ew_w, Q_we = ee.get_pose()
            v_ew_w, ω_ew_w = ee.get_velocity()

            x = ee.get_state()
            P_we, V_ew_w = x[:7], x[7:]
            recorder.ineq_cons[idx, :] = np.array(
                problem.ineq_constraints(P_we, V_ew_w, u)
            )

            r_tw_w, Q_wt = tray.bullet.get_pose()
            r_ow_w, Q_wo = obj.bullet.get_pose()
            r_ew_w_d = setpoints[setpoint_idx, :]

            # orientation error
            Q_de = util.quat_multiply(Qd_inv, Q_we)

            # record
            recorder.us[idx, :] = u
            recorder.r_ew_wds[idx, :] = r_ew_w_d
            recorder.r_ew_ws[idx, :] = r_ew_w
            recorder.Q_wes[idx, :] = Q_we
            recorder.Q_des[idx, :] = Q_de
            recorder.v_ew_ws[idx, :] = v_ew_w
            recorder.ω_ew_ws[idx, :] = ω_ew_w
            recorder.r_tw_ws[idx, :] = r_tw_w
            recorder.r_te_es[idx, :] = util.calc_r_te_e(r_ew_w, Q_we, r_tw_w)
            recorder.r_oe_es[idx, :] = util.calc_r_te_e(r_ew_w, Q_we, r_ow_w)
            recorder.r_ot_ts[idx, :] = util.calc_r_te_e(r_tw_w, Q_wt, r_ow_w)
            recorder.Q_ets[idx, :] = util.calc_Q_et(Q_we, Q_wt)
            recorder.Q_eos[idx, :] = util.calc_Q_et(Q_we, Q_wo)
            recorder.Q_tos[idx, :] = util.calc_Q_et(Q_wt, Q_wo)

            if (
                np.linalg.norm(r_ew_w_d - r_ew_w) < 0.01
                and np.linalg.norm(Q_de[:3]) < 0.01
            ):
                print("Close to desired pose - stopping.")
                setpoint_idx += 1
                if setpoint_idx >= setpoints.shape[0]:
                    break

                # update r_ew_w_d to avoid falling back into this block right away
                r_ew_w_d = setpoints[setpoint_idx, :]

    controller.benchmark.print_stats()

    print(f"Min constraint value = {np.min(recorder.ineq_cons)}")

    last_sim_index = i
    recorder.plot_ee_position(last_sim_index)
    recorder.plot_ee_orientation(last_sim_index)
    recorder.plot_ee_velocity(last_sim_index)
    recorder.plot_r_te_e_error(last_sim_index)
    recorder.plot_r_oe_e_error(last_sim_index)
    recorder.plot_r_ot_t_error(last_sim_index)
    recorder.plot_balancing_constraints(last_sim_index)
    recorder.plot_commands(last_sim_index)

    plt.show()


if __name__ == "__main__":
    main()
