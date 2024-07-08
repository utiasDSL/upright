#!/usr/bin/env python3
import argparse
import os
import resource
import signal
import time

import numpy as np
import rospy
import matplotlib.pyplot as plt

import mobile_manipulation_central as mm
import upright_core as core
import upright_cmd as cmd
import upright_robust as rob
from upright_core.logging import DataLogger


# TODO make some of these into command line arguments?
DRY_RUN = False
VERBOSE = True
USE_DATA_LOGGER = False
CHECK_VICON_RATE = False
EXPECTED_VICON_RATE = 100  # Hz
VICON_OBJECT_NAME = "ThingBase_3"
DURATION = 30.0

# TODO should something be done to account for the Vicon rate being different
# from this? Note also the Ridgeback command rate is different...
RATE = 125  # Hz
TIMESTEP = 1.0 / RATE

MAX_JOINT_VELOCITY = np.array([1.0, 1, 1, 1, 1, 1, 1, 1, 1])
MAX_JOINT_ACCELERATION = np.array([2.5, 2.5, 1, 5, 5, 5, 5, 5, 5])

MAX_EE_LIN_VEL = 2.0
MAX_EE_ANG_VEL = 1.0
MAX_EE_LIN_ACC = 1.0
MAX_EE_ANG_ACC = 1.0
TILT_ANGLE_MAX = np.deg2rad(15)

EE_LIN_ACC_WEIGHT = 1

# this needs to at least ~5, otherwise the tilting action won't be strong
# enough against the linear acceleration
EE_ANG_ACC_WEIGHT = 10
JOINT_ACC_WEIGHT = 0.01 * np.array([1, 1, 1, 1, 1, 1, 1, 100, 1])

# needs to be high enough to actually converge
JOINT_VEL_WEIGHT = 3
JOINT_JERK_WEIGHT = 0

USE_KALMAN_FILTER = True
PROCESS_COV = 1000
MEASUREMENT_COV = 1


def main():
    # os.nice(-10)
    # resource.setrlimit(resource.RLIMIT_RTPRIO, (98, 98))

    np.set_printoptions(precision=6, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configuration file.")
    parser.add_argument("--save", help="Save data to this file.")
    parser.add_argument("--notes", help="Additional information written to notes.txt.")

    # load configuration
    args = parser.parse_args()
    config = core.parsing.load_config(args.config)
    ctrl_config = config["controller"]

    # parse controller model
    model = rob.RobustControllerModel(
        ctrl_config,
        TIMESTEP,
        v_joint_max=MAX_JOINT_VELOCITY,
        a_joint_max=MAX_JOINT_ACCELERATION,
        a_cart_max=MAX_EE_LIN_ACC,
        α_cart_max=MAX_EE_ANG_ACC,
        v_cart_max=MAX_EE_LIN_VEL,
        ω_cart_max=MAX_EE_ANG_VEL,
        a_cart_weight=EE_LIN_ACC_WEIGHT,
        α_cart_weight=EE_ANG_ACC_WEIGHT,
        a_joint_weight=JOINT_ACC_WEIGHT,
        v_joint_weight=JOINT_VEL_WEIGHT,
        j_joint_weight=JOINT_JERK_WEIGHT,
        tilt_angle_max=TILT_ANGLE_MAX,
    )
    kp, kv = model.kp, model.kv
    controller = model.controller
    robot_model = model.robot

    # data logging
    if USE_DATA_LOGGER:
        logger = DataLogger(config)

    # start ROS
    rospy.init_node("upright_robust_controller", disable_signals=True)
    robot_interface = mm.MobileManipulatorROSInterface()
    signal_handler = mm.SimpleSignalHandler()

    # check that Vicon rate is as we expect
    if CHECK_VICON_RATE:
        print("Checking Vicon rate...")
        checker = mm.ViconRateChecker(vicon_object_name=VICON_OBJECT_NAME)
        if not checker.check_rate(expected_rate=EXPECTED_VICON_RATE):
            print("Vicon rate is wrong!")
            return

    if args.save is not None:
        recorder = rob.DataRecorder(name=args.save, notes=args.notes)
        recorder.record()
        print(f"Recording data to {recorder.log_dir}")

        # wait for bag to be set up properly
        time.sleep(5.0)

    # wait until robot feedback has been received
    rate = rospy.Rate(RATE)
    while not rospy.is_shutdown() and not robot_interface.ready():
        rate.sleep()
    print("Robot interface is ready.")

    # initial condition
    q, v = robot_interface.q, np.zeros(robot_interface.nv)
    u = np.zeros(robot_interface.nv)
    cmd_vel = np.zeros(robot_interface.nv)

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
    estimate = mm.kf.GaussianEstimate(x0, P0)

    ctrl_prof = rob.RunningAverage()

    # control loop
    t0 = rospy.Time.now().to_sec()
    t = t0
    while not rospy.is_shutdown() and t - t0 < DURATION and not signal_handler.received:
        last_t = t
        t = rospy.Time.now().to_sec()
        dt = t - last_t

        # robot feedback
        q_meas, v_meas = robot_interface.q, robot_interface.v

        # Kalman filtering
        if USE_KALMAN_FILTER:
            y = q_meas
            A = np.block([[Iq, dt * Iq], [0 * Iq, Iq]])
            B = np.vstack((0.5 * dt**2 * Iq, dt * Iq))
            Q = B @ Q0 @ B.T
            estimate = mm.kf.predict_and_correct(estimate, A, Q, B @ u, C, R0, y)
            q, v = estimate.x[:nq], estimate.x[nq:]
        else:
            q = q_meas
            v = v_meas

        # estimated EE state
        robot_model.forward(q, v, u)
        r_ew_w, C_we = robot_model.link_pose(rotation_matrix=True)
        v_ew_w, _ = robot_model.link_velocity()
        a_ew_w, _ = robot_model.link_classical_acceleration()

        # commanded EE acceleration
        a_ew_w_cmd = kp * (rd - r_ew_w) + kv * (vd - v_ew_w) + ad

        # compute command taking balancing into account
        ta = time.perf_counter_ns()
        u, A_e = controller.solve(q, v, a_ew_w_cmd, u)
        tb = time.perf_counter_ns()

        if VERBOSE:
            ctrl_prof.update(tb - ta)
            rospy.loginfo(f"curr = {(tb - ta) / 1e6} ms")
            rospy.loginfo(f"avg  = {ctrl_prof.average / 1e6} ms")
            rospy.loginfo(f"max  = {ctrl_prof.max / 1e6} ms")

        # integrate acceleration command to get new commanded velocity from the
        # current velocity
        cmd_vel = v + dt * u
        # cmd_vel = cmd_vel + dt * u
        cmd_vel = mm.bound_array(cmd_vel, lb=-MAX_JOINT_VELOCITY, ub=MAX_JOINT_VELOCITY)

        if DRY_RUN:
            print(f"cmd_vel = {cmd_vel}")
        else:
            robot_interface.publish_cmd_vel(cmd_vel, bodyframe=False)

        if USE_DATA_LOGGER and logger.ready(t):
            logger.append("ts", t - t0)
            logger.append("qs_meas", q_meas)
            logger.append("vs_meas", v_meas)
            logger.append("qs_est", q)
            logger.append("vs_est", v)
            logger.append("us", u)
            logger.append("r_ew_ws", r_ew_w)
            logger.append("v_ew_ws", v_ew_w)
            logger.append("a_ew_ws", a_ew_w)
            logger.append("a_ew_ws_cmd", a_ew_w_cmd)
            logger.append(
                "a_ew_ws_feas", A_e[:3]
            )  # feasible acceleration under balancing

        rate.sleep()

    print("Braking robot")
    robot_interface.brake()
    if args.save is not None:
        recorder.close()
    time.sleep(0.5)  # wait a bit to make sure brake is published

    if not USE_DATA_LOGGER:
        return

    # plotting
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    ts = np.array(logger.data["ts"])
    qs_meas = np.array(logger.data["qs_meas"])
    vs_meas = np.array(logger.data["vs_meas"])
    qs = np.array(logger.data["qs_est"])
    vs = np.array(logger.data["vs_est"])
    us = np.array(logger.data["us"])
    r_ew_ws = np.array(logger.data["r_ew_ws"])
    v_ew_ws = np.array(logger.data["v_ew_ws"])
    a_ew_ws = np.array(logger.data["a_ew_ws"])
    a_ew_ws_cmd = np.array(logger.data["a_ew_ws_cmd"])
    a_ew_ws_feas = np.array(logger.data["a_ew_ws_feas"])

    plt.figure()
    for i in range(robot_interface.nq):
        plt.plot(ts, qs[:, i], label=f"q_{i}", color=colors[i])
    for i in range(robot_interface.nq):
        plt.plot(
            ts, qs_meas[:, i], label=f"q_meas_{i}", linestyle="--", color=colors[i]
        )
    plt.grid()
    plt.legend(ncols=2)
    plt.xlabel("Time [s]")
    plt.xlabel("Joint position")
    plt.title("Joint positions vs. time")

    plt.figure()
    for i in range(robot_interface.nv):
        plt.plot(ts, vs[:, i], label=f"v_{i}", color=colors[i])
    for i in range(robot_interface.nv):
        plt.plot(
            ts, vs_meas[:, i], label=f"v_meas_{i}", linestyle="--", color=colors[i]
        )
    plt.grid()
    plt.legend(ncols=2)
    plt.xlabel("Time [s]")
    plt.xlabel("Joint velocity")
    plt.title("Joint velocities vs. time")

    plt.figure()
    for i in range(robot_interface.nv):
        plt.plot(ts, us[:, i], label=f"u_{i}", color=colors[i])
    plt.grid()
    plt.legend()
    plt.xlabel("Time [s]")
    plt.xlabel("Joint acceleration commands")
    plt.title("Joint acceleration commands vs. time")

    plt.figure()
    plt.plot(ts, r_ew_ws[:, 0], label="x", color=colors[0])
    plt.plot(ts, r_ew_ws[:, 1], label="y", color=colors[1])
    plt.plot(ts, r_ew_ws[:, 2], label="z", color=colors[2])
    plt.grid()
    plt.legend()
    plt.xlabel("Time [s]")
    plt.xlabel("Position [m]")
    plt.title("End effector position vs. time")

    plt.figure()
    plt.plot(ts, v_ew_ws[:, 0], label="x", color=colors[0])
    plt.plot(ts, v_ew_ws[:, 1], label="y", color=colors[1])
    plt.plot(ts, v_ew_ws[:, 2], label="z", color=colors[2])
    plt.grid()
    plt.legend()
    plt.xlabel("Time [s]")
    plt.xlabel("Velocity [m/s]")
    plt.title("End effector velocity vs. time")

    plt.figure()
    plt.plot(ts, a_ew_ws[:, 0], label="x", color=colors[0])
    plt.plot(ts, a_ew_ws[:, 1], label="y", color=colors[1])
    plt.plot(ts, a_ew_ws[:, 2], label="z", color=colors[2])
    plt.plot(ts, a_ew_ws_cmd[:, 0], label="x_cmd", linestyle="--", color=colors[0])
    plt.plot(ts, a_ew_ws_cmd[:, 1], label="y_cmd", linestyle="--", color=colors[1])
    plt.plot(ts, a_ew_ws_cmd[:, 2], label="z_cmd", linestyle="--", color=colors[2])
    plt.plot(
        ts, a_ew_ws_feas[:, 0], label="x_feas", linestyle="dotted", color=colors[0]
    )
    plt.plot(
        ts, a_ew_ws_feas[:, 1], label="y_feas", linestyle="dotted", color=colors[1]
    )
    plt.plot(
        ts, a_ew_ws_feas[:, 2], label="z_feas", linestyle="dotted", color=colors[2]
    )
    plt.grid()
    plt.legend()
    plt.xlabel("Time [s]")
    plt.xlabel("Acceleration [m/s^2]")
    plt.title("End effector acceleration vs. time")

    plt.show()


if __name__ == "__main__":
    main()
