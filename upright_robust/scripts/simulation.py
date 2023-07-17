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
from upright_core.logging import DataLogger

import IPython


USE_KALMAN_FILTER = True
PROCESS_COV = 1
MEASUREMENT_COV = 1

# TODO specify these properly via config
MAX_JOINT_VELOCITY = np.array([0.5, 0.5, 0.5, 1, 1, 1, 1, 1, 1])
MAX_JOINT_ACCELERATION = np.array([2.5, 2.5, 1, 5, 5, 5, 5, 5, 5])


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

    # parse controller model
    model = rob.RobustControllerModel(
        ctrl_config,
        env.timestep,
        v_joint_max=MAX_JOINT_VELOCITY,
        a_joint_max=MAX_JOINT_ACCELERATION,
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
            print(f"kf avg = {kf_prof.average / 1e6} ms")

            q, v = estimate.x[: env.robot.nq], estimate.x[env.robot.nq :]
        else:
            q = q_meas
            v = v_meas

        # current EE state
        robot.forward(q, v)
        r_ew_w, C_we = robot.link_pose(rotation_matrix=True)
        v_ew_w, _ = robot.link_velocity()

        # desired EE state
        # rd, vd, ad = trajectory.sample(t)

        rd = r_ew_w_d
        vd = np.zeros(3)
        ad = np.zeros(3)

        # commanded EE acceleration
        a_ew_w_cmd = kp * (rd - r_ew_w) + kv * (vd - v_ew_w) + ad

        # compute command
        t0 = time.perf_counter_ns()
        u, A_e = controller.solve(q, v, a_ew_w_cmd)
        t1 = time.perf_counter_ns()

        A_w = block_diag(C_we, C_we) @ A_e

        ctrl_prof.update(t1 - t0)
        print(f"curr = {(t1 - t0) / 1e6} ms")
        print(f"avg  = {ctrl_prof.average / 1e6} ms")
        # print(f"a_cmd = {a_ew_w_cmd}")
        # print(f"A_w = {A_w}")

        # NOTE: we want to use v_cmd rather than v here because PyBullet
        # doesn't respond to small velocities well, and it screws up angular
        # velocity tracking
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
            logger.append("r_ew_ws", r_ew_w)
            logger.append("v_ew_ws", v_ew_w)

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
    plt.title("End effector velocity vs. time")

    plt.show()


if __name__ == "__main__":
    main()
