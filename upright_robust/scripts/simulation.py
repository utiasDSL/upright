#!/usr/bin/env python3
"""Closed-loop upright reactive simulation using Pybullet."""
import datetime
import time
import signal
import sys

import numpy as np
import pybullet as pyb
import pyb_utils
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

import mobile_manipulation_central as mm
import upright_sim as sim
import upright_core as core
import upright_cmd as cmd
import upright_robust as rob
from upright_core.logging import DataLogger, DataPlotter

import IPython


USE_KALMAN_FILTER = True
PROCESS_COV = 1
MEASUREMENT_COV = 1

# TODO specify all of these properly via config
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
# JOINT_ACC_WEIGHT = 0.01
JOINT_ACC_WEIGHT = 0.01 * np.array([1, 1, 1, 1, 1, 1, 1, 100, 1])

# needs to be high enough to actually converge
# JOINT_VEL_WEIGHT = 3
JOINT_VEL_WEIGHT = 3.0 * np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
JOINT_JERK_WEIGHT = 0


def sigint_handler(sig, frame):
    print("Ctrl-C pressed: exiting.")
    pyb.disconnect()
    sys.exit(0)


def main():
    np.set_printoptions(precision=6, suppress=True)
    signal.signal(signal.SIGINT, sigint_handler)

    cli_args = cmd.cli.sim_arg_parser().parse_args()

    # load configuration
    config = core.parsing.load_config(cli_args.config)
    sim_config = config["simulation"]
    ctrl_config = config["controller"]

    timestamp = datetime.datetime.now()
    env = sim.simulation.UprightSimulation(
        config=sim_config,
        timestamp=timestamp,
        video_name=cli_args.video,
        extra_gui=sim_config.get("extra_gui", False),
    )
    env.settle(5.0)
    print("Sim object info")
    for name, obj in env.objects.items():
        info = pyb_utils.getDynamicsInfo(obj.uid, -1)
        print(f"{name} inertia diag = {info.localInertiaDiagonal}")

    # data logging
    logger = DataLogger(config)
    logger.add("object_names", [str(name) for name in env.objects.keys()])

    # parse controller model
    model = rob.RobustControllerModel(
        ctrl_config,
        env.timestep,
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
    robot = model.robot

    t = 0.0
    q, v = env.robot.joint_states()
    u = np.zeros(env.robot.nv)
    v_cmd = np.zeros(env.robot.nv)

    robot.forward(q, v)
    r_ew_w_0, Q_we_0 = robot.link_pose()
    r_ew_w_d = r_ew_w_0 + ctrl_config["waypoints"][0]["position"]

    # desired trajectory
    # trajectory = mm.PointToPointTrajectory.quintic(
    #     r_ew_w_0, r_ew_w_d, max_vel=2, max_acc=4
    # )

    # ensure that PyBullet sim matches the Pinocchio model
    r_ew_w_sim, Q_we_sim = env.robot.link_pose()
    assert np.allclose(r_ew_w_0, r_ew_w_sim)
    assert np.allclose(Q_we_0, Q_we_sim)

    # state estimate
    # obviously not needed here, but used to test the effect of the KF on the
    # overall system dynamics
    x0 = np.concatenate((q, v))
    P0 = 0.01 * np.eye(x0.shape[0])
    Iq = np.eye(env.robot.nq)
    Q0 = PROCESS_COV * Iq
    R0 = MEASUREMENT_COV * Iq
    C = np.hstack((Iq, 0 * Iq))
    estimate = mm.kf.GaussianEstimate(x0, P0)

    # goal position
    pyb_utils.debug_frame_world(0.2, list(r_ew_w_d), orientation=Q_we_0, line_width=3)

    # profiling for controller solve time
    ctrl_prof = rob.RunningAverage()
    kf_prof = rob.RunningAverage()

    # simulation loop
    while t <= env.duration:
        # current joint state
        q_meas, v_meas = env.robot.joint_states(add_noise=False)

        # Kalman filtering
        if USE_KALMAN_FILTER:
            y = q_meas
            A = np.block([[Iq, env.timestep * Iq], [0 * Iq, Iq]])
            B = np.vstack((0.5 * env.timestep**2 * Iq, env.timestep * Iq))
            Q = B @ Q0 @ B.T

            t0 = time.perf_counter_ns()
            estimate = mm.kf.predict_and_correct(estimate, A, Q, B @ u, C, R0, y)
            t1 = time.perf_counter_ns()
            kf_prof.update(t1 - t0)
            # print(f"kf avg = {kf_prof.average / 1e6} ms")

            q, v = estimate.x[: env.robot.nq], estimate.x[env.robot.nq :]
        else:
            q = q_meas
            v = v_meas

        # current EE state
        robot.forward(q, v, u)
        r_ew_w, C_we = robot.link_pose(rotation_matrix=True)
        v_ew_w, ω_ew_w = robot.link_velocity()
        a_ew_w, α_ew_w = robot.link_classical_acceleration()

        # desired EE state
        rd = r_ew_w_d
        vd = np.zeros(3)
        ad = np.zeros(3)

        # commanded EE acceleration
        a_ew_w_cmd = kp * (rd - r_ew_w) + kv * (vd - v_ew_w) + ad

        # compute command
        t0 = time.perf_counter_ns()
        u, A_e = controller.solve(q, v, a_ew_w_cmd, u)
        # u[:3] = C_we @ u[:3]  # TODO: rotate into world frame?
        t1 = time.perf_counter_ns()

        A_w = block_diag(C_we, C_we) @ A_e

        ctrl_prof.update(t1 - t0)
        print(f"curr = {(t1 - t0) / 1e6} ms")
        print(f"avg  = {ctrl_prof.average / 1e6} ms")
        print(f"max  = {ctrl_prof.max / 1e6} ms")
        # print(f"a_cmd = {a_ew_w_cmd}")
        # print(f"A_w = {A_w}")

        # NOTE: we use v_cmd rather than v here because PyBullet doesn't
        # respond to small velocities well, and it screws up angular velocity
        # tracking
        v_cmd = v_cmd + env.timestep * u
        env.robot.command_velocity(v_cmd, bodyframe=False)
        t = env.step(t, step_robot=False)[0]

        if logger.ready(t):
            logger.append("ts", t)
            logger.append("qs_meas", q_meas)
            logger.append("vs_meas", v_meas)
            logger.append("qs_est", q)
            logger.append("vs_est", v)
            logger.append("us", u)
            # logger.append("r_ew_ws", r_ew_w)
            logger.append("v_ew_ws", v_ew_w)
            logger.append("a_ew_ws", a_ew_w)
            logger.append("a_ew_ws_cmd", a_ew_w_cmd)
            logger.append(
                "a_ew_ws_feas", A_e[:3]
            )  # feasible acceleration under balancing

            logger.append("α_ew_ws", α_ew_w)
            logger.append("ω_ew_ws", ω_ew_w)

            # tilt angle
            z = np.array([0, 0, 1])
            logger.append("tilt_angles", np.arccos(z @ C_we @ z))

            # from the simulation
            # TODO: check this is the same as the model
            r_ew_w, Q_we = env.robot.link_pose()
            r_ow_ws, Q_wos = env.object_poses()
            logger.append("r_ew_ws", r_ew_w)
            logger.append("Q_wes", Q_we)
            logger.append("r_ow_ws", r_ow_ws)
            logger.append("Q_wos", Q_wos)

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
    tilt_angles = np.array(logger.data["tilt_angles"])

    ω_ew_ws = np.array(logger.data["ω_ew_ws"])
    α_ew_ws = np.array(logger.data["α_ew_ws"])

    DataPlotter.from_logger(logger).plot_object_error(obj_index=0)

    plt.figure()
    for i in range(env.robot.nq):
        plt.plot(ts, qs[:, i], label=f"q_{i}", color=colors[i])
    for i in range(env.robot.nq):
        plt.plot(
            ts, qs_meas[:, i], label=f"q_meas_{i}", linestyle="--", color=colors[i]
        )
    plt.grid()
    plt.legend(ncols=2)
    plt.xlabel("Time [s]")
    plt.xlabel("Joint position")
    plt.title("Joint positions vs. time")

    plt.figure()
    for i in range(env.robot.nv):
        plt.plot(ts, vs[:, i], label=f"v_{i}", color=colors[i])
    for i in range(env.robot.nv):
        plt.plot(
            ts, vs_meas[:, i], label=f"v_meas_{i}", linestyle="--", color=colors[i]
        )
    plt.grid()
    plt.legend(ncols=2)
    plt.xlabel("Time [s]")
    plt.xlabel("Joint velocity")
    plt.title("Joint velocities vs. time")

    plt.figure()
    for i in range(env.robot.nv):
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
    plt.title("EE linear velocity vs. time")

    plt.figure()
    plt.plot(ts, ω_ew_ws[:, 0], label="x", color=colors[0])
    plt.plot(ts, ω_ew_ws[:, 1], label="y", color=colors[1])
    plt.plot(ts, ω_ew_ws[:, 2], label="z", color=colors[2])
    plt.grid()
    plt.legend()
    plt.xlabel("Time [s]")
    plt.xlabel("Velocity [rad/s]")
    plt.title("EE angular velocity vs. time")

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
    plt.title("EE linear acceleration vs. time")

    plt.figure()
    plt.plot(ts, α_ew_ws[:, 0], label="x", color=colors[0])
    plt.plot(ts, α_ew_ws[:, 1], label="y", color=colors[1])
    plt.plot(ts, α_ew_ws[:, 2], label="z", color=colors[2])
    plt.grid()
    plt.legend()
    plt.xlabel("Time [s]")
    plt.xlabel("Acceleration [rad/s^2]")
    plt.title("EE angular acceleration vs. time")

    plt.figure()
    plt.plot(ts, tilt_angles)
    plt.grid()
    plt.xlabel("Time [s]")
    plt.xlabel("Tilt angle [rad]")
    plt.title("Tilt angle vs. time")

    plt.show()


if __name__ == "__main__":
    main()
