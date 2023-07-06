#!/usr/bin/env python3
import os
import resource
import time

import numpy as np
import rospy
import matplotlib.pyplot as plt

import mobile_manipulation_central as mm
import upright_core as core
import upright_cmd as cmd
import upright_robust as rob
from upright_core.logging import DataLogger


VERBOSE = False
DURATION = 5.0

RATE = 125  # Hz
TIMESTEP = 1.0 / RATE

# TODO adjust this
MAX_JOINT_VELOCITY = np.array([0.5, 0.5, 0.5, 1, 1, 1, 1, 1, 1])

PROCESS_COV = 1000
MEASUREMENT_COV = 1


def main():
    # os.nice(-10)
    # resource.setrlimit(resource.RLIMIT_RTPRIO, (98, 98))

    np.set_printoptions(precision=6, suppress=True)

    # load configuration
    cli_args = cmd.cli.sim_arg_parser().parse_args()
    config = core.parsing.load_config(cli_args.config)
    ctrl_config = config["controller"]

    # parse controller model
    model = rob.RobustControllerModel(ctrl_config, TIMESTEP)
    kp, kv = model.kp, model.kv
    controller = model.controller
    robot_model = model.robot

    # data logging
    logger = DataLogger(config)

    # start ROS
    rospy.init_node("upright_robust_controller", disable_signals=True)
    robot_interface = mm.MobileManipulatorROSInterface()
    signal_handler = mm.RobotSignalHandler(robot_interface)

    # wait until robot feedback has been received
    rate = rospy.Rate(RATE)
    while not rospy.is_shutdown() and not robot_interface.ready():
        rate.sleep()

    # initial condition
    q, v = robot_interface.q, np.zeros(robot_interface.nv)
    u = np.zeros(robot_interface.nv)

    # desired waypoint
    robot_model.forward(q, v)
    r_ew_w_0, _ = robot_model.link_pose()
    rd = r_ew_w_0 + ctrl_config["waypoints"][0]["position"]
    vd = np.zeros(3)
    ad = np.zeros(3)

    # state estimate
    nq = robot_interface.nq
    x0 = np.concatenate((q, v))
    P0 = 0.01 * np.eye(x0.shape[0])
    Iq = np.eye(nq)
    Q0 = PROCESS_COV * Iq
    R0 = MEASUREMENT_COV * Iq
    C = np.hstack((Iq, 0 * Iq))
    estimate = mm.GaussianEstimate(x=x0, P=P0)

    # TODO compare raw and filtered state data

    avg_solve_time = 0
    num_solves = 0

    # control loop
    t0 = rospy.Time.now().to_sec()
    t = t0
    while not rospy.is_shutdown() and t - t0 < DURATION:
        last_t = t
        t = rospy.Time.now().to_sec()
        dt = t - last_t

        # robot feedback
        q_meas, v_meas = robot_interface.q, robot_interface.v
        y = robot_interface.q
        A = np.block([[Iq, dt * Iq], [0 * Iq, Iq]])
        B = np.vstack((0.5 * dt**2 * Iq, dt * Iq))
        Q = B @ Q0 @ B.T
        estimate = mm.KalmanFilter.predict(estimate, A, Q, B @ u)
        estimate = mm.KalmanFilter.correct(estimate, C, R0, y)
        q, v = estimate.x[:nq], estimate.x[nq:]

        # estimated EE state
        robot_model.forward(q, v)
        r_ew_w, C_we = robot_model.link_pose(rotation_matrix=True)
        v_ew_w, _ = robot_model.link_velocity()

        # commanded EE acceleration
        a_ew_w_cmd = kp * (rd - r_ew_w) + kv * (vd - v_ew_w) + ad

        # compute command taking balancing into account
        # ta = rospy.Time.now().to_sec()
        ta = time.time()
        u, A_e = controller.solve(q, v, a_ew_w_cmd)
        # tb = rospy.Time.now().to_sec()
        tb = time.time()

        avg_solve_time = (num_solves * avg_solve_time + tb - ta) / (1 + num_solves)
        num_solves += 1
        rospy.loginfo(f"curr = {1000 * (tb - ta)} ms")
        rospy.loginfo(f"avg  = {1000 * avg_solve_time} ms")

        # integrate acceleration command to get new commanded velocity from the
        # current velocity
        cmd_vel = v + dt * u
        cmd_vel = mm.bound_array(cmd_vel, lb=-MAX_JOINT_VELOCITY, ub=MAX_JOINT_VELOCITY)
        robot_interface.publish_cmd_vel(cmd_vel, bodyframe=False)

        if logger.ready(t):
            logger.append("ts", t - t0)
            logger.append("qs_meas", q_meas)
            logger.append("vs_meas", v_meas)
            logger.append("qs_est", q)
            logger.append("vs_est", v)

        # if VERBOSE:
        #     A_w = block_diag(C_we, C_we) @ A_e

        rate.sleep()

    robot_interface.brake()

    # plotting
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    ts = np.array(logger.data["ts"])
    qs_meas = np.array(logger.data["qs_meas"])
    vs_meas = np.array(logger.data["vs_meas"])
    qs = np.array(logger.data["qs_est"])
    vs = np.array(logger.data["vs_est"])

    plt.figure()
    for i in range(robot_interface.nq):
        plt.plot(ts, qs[:, i], label=f"q_{i}", color=colors[i])
    for i in range(robot_interface.nq):
        plt.plot(ts, qs_meas[:, i], label=f"q_meas_{i}", linestyle="--", color=colors[i])
    plt.grid()
    plt.legend()
    plt.xlabel("Time [s]")
    plt.xlabel("Joint position")
    plt.title("Joint positions vs. time")

    plt.figure()
    for i in range(robot_interface.nv):
        plt.plot(ts, vs[:, i], label=f"v_{i}", color=colors[i])
    for i in range(robot_interface.nv):
        plt.plot(ts, vs_meas[:, i], label=f"v_meas_{i}", linestyle="--", color=colors[i])
    plt.grid()
    plt.legend()
    plt.xlabel("Time [s]")
    plt.xlabel("Joint velocity")
    plt.title("Joint velocities vs. time")

    plt.show()


if __name__ == "__main__":
    main()
