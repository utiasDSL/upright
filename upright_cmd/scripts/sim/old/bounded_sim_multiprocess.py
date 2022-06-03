#!/usr/bin/env python3
"""PyBullet simulation using the bounded balancing constraints"""
import argparse
import time
import datetime
import sys
import os
from pathlib import Path
from multiprocessing import Process, Manager, Pipe, Lock
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb
from pyb_utils.ghost import GhostSphere
from pyb_utils.frame import debug_frame_world

from upright_sim import util, camera, simulation

from tray_balance_constraints.logging import DataLogger, DataPlotter
import tray_balance_constraints as core
import tray_balance_ocs2 as ctrl
import upright_cmd as cmd

import IPython


def outer_control_loop(
    ctrl_wrapper, r_ew_w, Q_we, sync_lock, outer_ctrl_con, outer_ctrl_txu
):
    """Outer control loop containing MPC optimization.

    Advance MPC at a fixed control rate."""
    try:
        # initialize and run first iteration
        mpc = ctrl_wrapper.controller(r_ew_w, Q_we)
        t, x, u = outer_ctrl_txu.recv()
        mpc.setObservation(t, x, u)
        mpc.advanceMpc()
        outer_ctrl_con.send(mpc.getLinearController())
    finally:
        sync_lock.release()

    # run MPC as fast as possible
    # rate = core.util.Rate.from_hz(40)
    while True:
        # get the latest state
        while outer_ctrl_txu.poll(timeout=0):
            t, x, u = outer_ctrl_txu.recv()
        mpc.setObservation(t, x, u)
        mpc.advanceMpc()
        outer_ctrl_con.send(mpc.getLinearController())
        # rate.sleep()


def main():
    np.set_printoptions(precision=3, suppress=True)

    cli_args = cmd.cli.sim_arg_parser().parse_args()

    # load configuration
    config = core.parsing.load_config(cli_args.config)
    sim_config = config["simulation"]
    ctrl_config = config["controller"]
    log_config = config["logging"]

    # timing
    duration_millis = sim_config["duration"]
    timestep_millis = sim_config["timestep"]
    timestep_secs = core.parsing.millis_to_secs(timestep_millis)
    duration_secs = core.parsing.millis_to_secs(duration_millis)
    num_timesteps = int(duration_millis / timestep_millis)
    ctrl_period = ctrl_config["control_period"]

    # start the simulation
    sim = simulation.MobileManipulatorSimulation(sim_config)
    robot = sim.robot

    # setup sim objects
    r_ew_w, Q_we = robot.link_pose()
    sim_objects = simulation.sim_object_setup(r_ew_w, sim_config)
    num_objects = len(sim_objects)

    # mark frame at the initial position
    debug_frame_world(0.2, list(r_ew_w), orientation=Q_we, line_width=3)

    # initial time, state, input
    t = 0.0
    q, v = robot.joint_states()
    a = np.zeros(robot.nv)
    x = np.concatenate((q, v, a))
    u = np.zeros(robot.nu)

    # video recording
    now = datetime.datetime.now()
    video_manager = camera.VideoManager.from_config_dict(
        video_name=cli_args.video, config=sim_config, timestamp=now, r_ew_w=r_ew_w
    )

    ctrl_wrapper = ctrl.parsing.ControllerConfigWrapper(ctrl_config, x0=x)
    ctrl_objects = ctrl_wrapper.objects()

    # data logging
    log_dir = Path(log_config["log_dir"])
    log_dt = log_config["timestep"]
    logger = DataLogger(config)

    logger.add("sim_timestep", timestep_secs)
    logger.add("duration", duration_secs)
    logger.add("control_period", ctrl_period)
    logger.add("object_names", [str(name) for name in sim_objects.keys()])

    logger.add("nq", ctrl_config["robot"]["dims"]["q"])
    logger.add("nv", ctrl_config["robot"]["dims"]["v"])
    logger.add("nx", ctrl_config["robot"]["dims"]["x"])
    logger.add("nu", ctrl_config["robot"]["dims"]["u"])

    # create reference trajectory and controller
    ref = ctrl_wrapper.reference_trajectory(r_ew_w, Q_we)

    # TODO this is only used to easily compute constraint values
    # would be better to separate out that functionality to avoid having to
    # create a whole other object (which requires autodiff compilation, etc.)
    mpc_inner = ctrl_wrapper.controller(r_ew_w, Q_we)

    # frames and ghost (i.e., pure visual) objects
    ghosts = []
    for state in ref.states:
        r_ew_w_d, Q_we_d = ref.pose(state)
        # ghosts.append(GhostSphere(radius=0.05, position=r_ew_w_d, color=(0, 1, 0, 1)))
        debug_frame_world(0.2, list(r_ew_w_d), orientation=Q_we_d, line_width=3)

    target_idx = 0
    static_stable = True

    # namespace to manage shared controller resources
    outer_ctrl_con, inner_ctrl_con = Pipe()
    outer_ctrl_txu, inner_ctrl_txu = Pipe()

    # acquire a lock which is only released after MPC is initialized in the
    # outer-loop process
    sync_lock = Lock()
    sync_lock.acquire()

    # send the initial state
    inner_ctrl_txu.send((t, x, u))

    # start separate control process
    outer_control_proc = Process(
        target=outer_control_loop,
        args=(
            ctrl_wrapper,
            r_ew_w,
            Q_we,
            sync_lock,
            outer_ctrl_con,
            outer_ctrl_txu,
        ),
    )
    outer_control_proc.start()

    # wait until MPC is initialized
    sync_lock.acquire()

    # simulation loop
    # this loop sets the MPC observation and computes the control input at a
    # faster rate than the outer loop MPC optimization problem
    # TODO: ideally we'd seperate this cleanly into its own function
    for i in range(num_timesteps):
        q, v = robot.joint_states()
        x = np.concatenate((q, v, a))

        # add noise to state variables
        q_noisy, v_noisy = robot.joint_states(add_noise=True)
        x_noisy = np.concatenate((q_noisy, v_noisy, a))

        # send updated states
        if ctrl_config["use_noisy_state_to_plan"]:
            inner_ctrl_txu.send((t, x_noisy, u))
        else:
            inner_ctrl_txu.send((t, x, u))

        # get a new controller if available
        if inner_ctrl_con.poll(timeout=0):
            lin_ctrl = inner_ctrl_con.recv()

        # compute the input using the current controller
        u = lin_ctrl.computeInput(t, x)
        a = np.copy(robot.cmd_acc)
        robot.command_jerk(u)

        if i % log_dt == 0:
            if ctrl_wrapper.settings.tray_balance_settings.enabled:
                if (
                    ctrl_wrapper.settings.tray_balance_settings.constraint_type
                    == ctrl.bindings.ConstraintType.Hard
                ):
                    balance_cons = mpc_inner.stateInputInequalityConstraint(
                        "trayBalance", t, x, u
                    )
                else:
                    balance_cons = mpc_inner.softStateInputInequalityConstraint(
                        "trayBalance", t, x, u
                    )
                logger.append("ineq_cons", balance_cons)
            # if ctrl_wrapper.settings.dynamic_obstacle_settings.enabled:
            #     recorder.dynamic_obs_distance[idx, :] = mpc.stateInequalityConstraint(
            #         "dynamicObstacleAvoidance", t, x
            #     )
            # if ctrl_wrapper.settings.collision_avoidance_settings.enabled:
            #     recorder.collision_pair_distance[
            #         idx, :
            #     ] = mpc.stateInequalityConstraint("collisionAvoidance", t, x)

            r_ew_w, Q_we = robot.link_pose()
            v_ew_w, ω_ew_w = robot.link_velocity()
            r_ew_w_d, Q_we_d = ref.pose(ref.states[target_idx])

            logger.append("ts", t)
            logger.append("us", u)
            logger.append("xs", x)
            logger.append("xs_noisy", x_noisy)
            logger.append("r_ew_w_ds", r_ew_w_d)
            logger.append("r_ew_ws", r_ew_w)
            logger.append("Q_wes", Q_we)
            logger.append("Q_we_ds", Q_we_d)
            logger.append("v_ew_ws", v_ew_w)
            logger.append("ω_ew_ws", ω_ew_w)
            logger.append("cmd_vels", robot.cmd_vel.copy())

            # compute distance outside of support area
            d = core.util.support_area_distance(ctrl_objects[1], Q_we)
            logger.append("ds", d)

            if d > 0:
                if static_stable:
                    sim_objects["box"].change_color((1, 0, 0, 1))
                static_stable = False
            else:
                if not static_stable:
                    sim_objects["box"].change_color((0, 1, 0, 1))
                static_stable = True

            r_ow_ws = np.zeros((num_objects, 3))
            Q_wos = np.zeros((num_objects, 4))
            for j, obj in enumerate(sim_objects.values()):
                r_ow_ws[j, :], Q_wos[j, :] = obj.get_pose()
            logger.append("r_ow_ws", r_ow_ws)
            logger.append("Q_wos", Q_wos)

        sim.step(step_robot=True)
        if ctrl_wrapper.settings.dynamic_obstacle_settings.enabled:
            obstacle.step()
        for ghost in ghosts:
            ghost.update()
        t += timestep_secs

        # if we have multiple targets, step through them
        if t >= ref.times[target_idx] and target_idx < len(ref.times) - 1:
            target_idx += 1

        video_manager.record(i)

    # rejoin the control process
    # TODO probably put this in the finally part of a try-finally
    outer_control_proc.terminate()
    print("killed outer control loop process")

    try:
        print(f"Min constraint value = {np.min(logger.data['ineq_cons'])}")
    except:
        pass

    # save logged data
    if cli_args.log is not None:
        logger.save(log_dir, now, name=cli_args.log)

    if video_manager.save:
        print(f"Saved video to {video_manager.path}")

    # visualize data
    plotter = DataPlotter(logger)
    plotter.plot_ee_position()
    plotter.plot_ee_orientation()
    plotter.plot_ee_velocity()
    for j in range(num_objects):
        plotter.plot_object_error(j)

    plotter.plot_value_vs_time(
        "ineq_cons",
        legend_prefix="g",
        ylabel="Constraint Value",
        title="Balancing Inequality Constraints vs. Time",
    )
    plotter.plot_value_vs_time(
        "us",
        indices=range(robot.nu),
        legend_prefix="u",
        ylabel="Commanded Input",
        title="Commanded Inputs vs. Time",
    )

    # plotter.plot_control_durations()
    plotter.plot_cmd_vs_real_vel()

    plotter.plot_value_vs_time(
        "xs",
        indices=range(robot.nq),
        legend_prefix="q",
        ylabel="Joint Position",
        title="Joint Positions vs. Time",
    )
    plotter.plot_value_vs_time(
        "xs",
        indices=range(robot.nq + robot.nv, robot.nq + 2 * robot.nv),
        legend_prefix="a",
        ylabel="Joint Acceleration",
        title="Joint Accelerations vs. Time",
    )
    ax = plotter.plot_value_vs_time(
        "ds",
        ylabel="Distance (m)",
        title="Distance Outside of SA vs. Time",
    )

    plt.show()


if __name__ == "__main__":
    main()