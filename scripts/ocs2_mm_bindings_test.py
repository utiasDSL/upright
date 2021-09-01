#!/usr/bin/env python
"""Baseline tray balancing formulation."""
import rospy

import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb

import util
from robot import RobotModel
import balancing
from simulation import MobileManipulatorSimulation, ROBOT_HOME
from recording import Recorder

# from ros_interface import ROSInterface

from ocs2_mobile_manipulator_modified import (
    mpc_interface,
    scalar_array,
    vector_array,
    matrix_array,
    TargetTrajectories,
)

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


class Obstacle:
    def __init__(self, initial_position, collision_uid, visual_uid, velocity=None):
        self.uid = pyb.createMultiBody(
            baseMass=0,  # non-dynamic body
            baseCollisionShapeIndex=collision_uid,
            baseVisualShapeIndex=visual_uid,
            basePosition=list(initial_position),
            baseOrientation=(0, 0, 0, 1),
        )
        self.initial_position = initial_position

        self.velocity = velocity
        if self.velocity is None:
            self.velocity = np.zeros(3)
        pyb.resetBaseVelocity(self.uid, linearVelocity=list(self.velocity))

    # def update_velocity(self, t):
    #     v = self.velocity_func(self.initial_position, t)
    #     pyb.resetBaseVelocity(self.uid, linearVelocity=v)

    def sample_position(self, t):
        """Sample the position of the object at a given time."""
        # assume constant velocity
        return self.initial_position + t * self.velocity
        # positions = np.array([self.initial_position + t * self.velocity for t in ts])
        # return positions


def main():
    np.set_printoptions(precision=3, suppress=True)

    sim = MobileManipulatorSimulation(dt=0.001)

    N = int(DURATION / sim.dt) + 1

    # simulation objects and model
    robot, objects, composites = sim.setup(obj_names=["tray", "cuboid1"])
    tray = objects["tray"]
    obj = objects["cuboid1"]

    robot_model = RobotModel(MPC_DT, ROBOT_HOME)

    q, v = robot.joint_states()
    r_ew_w, Q_we = robot_model.tool_pose(q)
    v_ew_w, ω_ew_w = robot_model.tool_velocity(q, v)
    r_tw_w, Q_wt = tray.bullet.get_pose()
    r_ow_w, Q_wo = obj.bullet.get_pose()

    # data recorder and plotter
    recorder = Recorder(sim.dt, DURATION, RECORD_PERIOD, model=robot_model)

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
    # setpoints = np.array([[2, 0, 0]]) + r_ew_w
    # setpoint_idx = 0

    # desired quaternion
    # R_ed = SO3.from_z_radians(np.pi)
    # # R_ed = SO3.identity()
    # R_we = SO3.from_quaternion_xyzw(Q_we)
    # R_wd = R_we.multiply(R_ed)
    # Qd = R_wd.as_quaternion_xyzw()
    # Qd_inv = R_wd.inverse().as_quaternion_xyzw()
    #
    # recorder.Q_des[0, :] = util.quat_multiply(Qd_inv, Q_we)

    r_obs0 = np.array(r_ew_w) + [0, -10, 0]
    v_obs = np.array([0, 0.4, 0])
    obs_radius = 0.1
    obs_collision_uid = pyb.createCollisionShape(
        shapeType=pyb.GEOM_SPHERE,
        radius=obs_radius,
    )
    obs_visual_uid = pyb.createVisualShape(
        shapeType=pyb.GEOM_SPHERE,
        radius=obs_radius,
        rgbaColor=(1, 0, 0, 1),
    )
    obstacle = Obstacle(r_obs0, -1, obs_visual_uid, velocity=v_obs)

    # r_obs5 = obstacle.sample_position(5)

    # initial time, state, and input
    t = 0
    x = np.concatenate((q, v))
    u = np.zeros(robot_model.ni)

    target_times = [0, 3, 6, 9]

    # setup MPC and initial EE target pose
    mpc = mpc_interface("mpc")
    t_target = scalar_array()
    for time in target_times:
        t_target.push_back(time)

    input_target = vector_array()
    for _ in target_times:
        input_target.push_back(u)

    state_target = vector_array()
    r_ew_w_d = np.array(r_ew_w) + [0, 0, 0]
    Qd = np.array(Q_we)
    state_target.push_back(np.concatenate((r_ew_w_d, Qd, r_obs0)))
    state_target.push_back(np.concatenate((r_ew_w_d + [1, 0, 0], Qd, r_obs0)))
    state_target.push_back(np.concatenate((r_ew_w_d + [2, 0, 0], Qd, r_obs0)))
    state_target.push_back(np.concatenate((r_ew_w_d + [3, 0, 0], Qd, r_obs0)))

    target_trajectories = TargetTrajectories(t_target, state_target, input_target)
    mpc.reset(target_trajectories)

    assert len(state_target) == len(target_times)

    for i in range(N - 1):
        # t = i * sim.dt

        q, v = robot.joint_states()
        x = np.concatenate((q, v))
        mpc.setObservation(t, x, u)

        # TODO this should be set to reflect the MPC time step
        if i % CTRL_PERIOD == 0:
            # t0 = time.time()
            robot.cmd_vel = v  # NOTE
            mpc.advanceMpc()
            # t1 = time.time()
            # print(f"dt = {t1 - t0}")

        # evaluate the current MPC policy (returns an entire trajectory of
        # waypoints, starting from the current time)
        t_result = scalar_array()
        x_result = vector_array()
        u_result = vector_array()
        mpc.getMpcSolution(t_result, x_result, u_result)

        u = u_result[0]
        # x_ref = x + sim.dt * mpc.flowMap(t, x, u)
        # v_cmd = K.dot(x_ref[:9] - x[:9]) + x_ref[9:]
        # robot.command_velocity(v_cmd)

        robot.command_acceleration(u)
        sim.step(step_robot=True)

        # dx = mpc.flowMap(t, x, u)
        # x += sim.dt * dx

        t += sim.dt

        # if i % 300 == 0:
        #     IPython.embed()

        # t_result_prev = t_result
        # x_result_prev = x_result
        # u_result_prev = u_result

        continue

        # if recorder.now_is_the_time(i):
        #     idx = recorder.record_index(i)
        #     r_ew_w, Q_we = robot_model.tool_pose(q)
        #     v_ew_w, ω_ew_w = robot_model.tool_velocity(q, v)

        # P_we, V_ew_w = robot.link_pose()
        # recorder.ineq_cons[idx, :] = np.array(
        #     problem.ineq_constraints(P_we, V_ew_w, u)
        # )

        # r_tw_w, Q_wt = tray.bullet.get_pose()
        # r_ow_w, Q_wo = obj.bullet.get_pose()
        # r_ew_w_d = setpoints[setpoint_idx, :]

        # orientation error
        # Q_de = util.quat_multiply(Qd_inv, Q_we)

        # record
        # recorder.us[idx, :] = u
        # recorder.r_ew_wds[idx, :] = r_ew_w_d
        # recorder.r_ew_ws[idx, :] = r_ew_w
        # recorder.Q_wes[idx, :] = Q_we
        # recorder.Q_des[idx, :] = Q_de
        # recorder.v_ew_ws[idx, :] = v_ew_w
        # recorder.ω_ew_ws[idx, :] = ω_ew_w
        # recorder.r_tw_ws[idx, :] = r_tw_w
        # recorder.r_te_es[idx, :] = util.calc_r_te_e(r_ew_w, Q_we, r_tw_w)
        # recorder.r_oe_es[idx, :] = util.calc_r_te_e(r_ew_w, Q_we, r_ow_w)
        # recorder.r_ot_ts[idx, :] = util.calc_r_te_e(r_tw_w, Q_wt, r_ow_w)
        # recorder.Q_ets[idx, :] = util.calc_Q_et(Q_we, Q_wt)
        # recorder.Q_eos[idx, :] = util.calc_Q_et(Q_we, Q_wo)
        # recorder.Q_tos[idx, :] = util.calc_Q_et(Q_wt, Q_wo)

        # if (
        #     np.linalg.norm(r_ew_w_d - r_ew_w) < 0.01
        #     and np.linalg.norm(Q_de[:3]) < 0.01
        # ):
        #     print("Close to desired pose - stopping.")
        #     setpoint_idx += 1
        #     if setpoint_idx >= setpoints.shape[0]:
        #         break
        #
        #     # update r_ew_w_d to avoid falling back into this block right away
        #     r_ew_w_d = setpoints[setpoint_idx, :]
        # i += 1
    print("done: time expired")

    # controller.benchmark.print_stats()

    # print(f"Min constraint value = {np.min(recorder.ineq_cons)}")

    # last_sim_index = i
    # recorder.plot_ee_position(last_sim_index)
    # recorder.plot_ee_orientation(last_sim_index)
    # recorder.plot_ee_velocity(last_sim_index)
    # recorder.plot_r_te_e_error(last_sim_index)
    # recorder.plot_r_oe_e_error(last_sim_index)
    # recorder.plot_r_ot_t_error(last_sim_index)
    # # recorder.plot_balancing_constraints(last_sim_index)
    # recorder.plot_commands(last_sim_index)
    #
    # plt.show()


if __name__ == "__main__":
    main()
