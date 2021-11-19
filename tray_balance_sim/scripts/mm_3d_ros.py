#!/usr/bin/env python
"""Baseline tray balancing formulation."""
import rospy

import numpy as np
import matplotlib.pyplot as plt

import util
from robot import RobotModel
import balancing
from simulation import MobileManipulatorSimulation, ROBOT_HOME
from recording import Recorder
from ros_interface import ROSInterface

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
    recorder = Recorder(
        sim.dt, DURATION, RECORD_PERIOD, model=robot_model
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
    interface = ROSInterface("mm_pybullet_sim")
    rate = rospy.Rate(1. / sim.dt)

    i = 0
    # import time
    # t1 = time.time()

    # wait until we receive something from the controller
    while not rospy.is_shutdown() and not interface.initialized():
        rate.sleep()

    # t = interface.time
    t = 0

    while not rospy.is_shutdown():
        # if interface.initialized():
        u = interface.command
        # TODO we could actually just command velocity directly
        robot.command_acceleration(u)
        # v_cmd = interface.v
        # robot.command_velocity(v_cmd)
        # print(f"u = {u}")

        # step simulation forward
        sim.step(step_robot=True)
        t += sim.dt
        q, v = robot.joint_states()
        interface.publish_state(t, q, v)
        rate.sleep()

        # t2 = time.time()
        # print(f"dt = {t2 - t1}")
        # t1 = t2

        # if i % 1000 == 0:

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
        i += 1

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
