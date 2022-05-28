#!/usr/bin/env python3
"""PyBullet simulation using the bounded balancing constraints"""
import argparse
import time
import datetime
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb
from pyb_utils.ghost import GhostSphere
from pyb_utils.frame import debug_frame_world

from tray_balance_sim import util, camera, simulation

from tray_balance_constraints.logging import DataLogger, DataPlotter
import tray_balance_constraints as core
import tray_balance_ocs2 as ctrl
import upright_cmd as cmd

import IPython

# TODO add operating points to config as a trajectory to load from a file
# def use_operating_points(ctrl_manager):
#     with np.load("short_box_trajectory.npz") as data:
#         ts_op = data["ts"]
#         xs_op = data["xs"]
#         us_op = data["us"]
#
#     step = 1
#     for i in range(0, ts_op.shape[0], step):
#         ctrl_manager.settings.operating_times.push_back(ts_op[i])
#         ctrl_manager.settings.operating_states.push_back(xs_op[i, :])
#         ctrl_manager.settings.operating_inputs.push_back(us_op[i, :])
#
#     ctrl_manager.settings.use_operating_points = True


def main():
    np.set_printoptions(precision=3, suppress=True)

    cli_args = cmd.cli.sim_arg_parser().parse_args()

    # load configuration
    config = core.parsing.load_config(cli_args.config)
    sim_config = config["simulation"]
    ctrl_config = config["controller"]
    log_config = config["logging"]

    # start the simulation
    timestamp = datetime.datetime.now()
    sim = simulation.MobileManipulatorSimulation(
        config=sim_config, timestamp=timestamp, cli_args=cli_args
    )

    # setup sim objects
    r_ew_w, Q_we = sim.robot.link_pose()
    num_objects = len(sim.objects)

    # initial time, state, input
    t = 0.0
    q, v = sim.robot.joint_states()
    a = np.zeros(sim.robot.nv)
    x = np.concatenate((q, v, a))
    u = np.zeros(sim.robot.nu)

    # controller
    ctrl_manager = ctrl.manager.ControllerManager(ctrl_config, x0=x)

    # use_operating_points(ctrl_manager)

    # data logging
    logger = DataLogger(config)

    logger.add("sim_timestep", sim.timestep)
    logger.add("duration", sim.duration)
    logger.add("ctrl_timestep", ctrl_manager.period)
    logger.add("object_names", [str(name) for name in sim.objects.keys()])

    logger.add("nq", ctrl_config["robot"]["dims"]["q"])
    logger.add("nv", ctrl_config["robot"]["dims"]["v"])
    logger.add("nx", ctrl_config["robot"]["dims"]["x"])
    logger.add("nu", ctrl_config["robot"]["dims"]["u"])

    # frames and ghost (i.e., pure visual) objects
    for r_ew_w_d, Q_we_d in ctrl_manager.ref.poses():
        # sim.ghosts.append(GhostSphere(radius=0.05, position=r_ew_w_d, color=(0, 1, 0, 1)))
        debug_frame_world(0.2, list(r_ew_w_d), orientation=Q_we_d, line_width=3)

    static_stable = True

    ctrl_manager.warmstart()

    # simulation loop
    while t <= sim.duration:
        q, v = sim.robot.joint_states(add_noise=True)
        x = np.concatenate((q, v, a))

        try:
            x_opt, u = ctrl_manager.step(t, x)
        except RuntimeError as e:
            print(e)
            print("exit the interpreter to proceed to plots")
            IPython.embed()
            break

        a = np.copy(x_opt[-sim.robot.nv :])
        sim.robot.command_jerk(u)

        if logger.ready(t):
            # if ctrl_manager.settings.dynamic_obstacle_settings.enabled:
            #     recorder.dynamic_obs_distance[idx, :] = mpc.stateInequalityConstraint(
            #         "dynamicObstacleAvoidance", t, x
            #     )
            # if ctrl_manager.settings.collision_avoidance_settings.enabled:
            #     recorder.collision_pair_distance[
            #         idx, :
            #     ] = mpc.stateInequalityConstraint("collisionAvoidance", t, x)

            # TODO
            ctrl_manager.log(logger)

            # TODO this could be a ref.log_desired_pose()
            # log controller stuff
            r_ew_w_d, Q_we_d = ctrl_manager.ref.get_desired_pose(t)
            logger.append("r_ew_w_ds", r_ew_w_d)
            logger.append("Q_we_ds", Q_we_d)

            # log sim stuff
            r_ew_w, Q_we = sim.robot.link_pose()
            v_ew_w, ω_ew_w = sim.robot.link_velocity()
            r_ow_ws, Q_wos = sim.object_poses()
            logger.append("ts", t)
            logger.append("us", u)
            logger.append("xs", x)
            logger.append("r_ew_ws", r_ew_w)
            logger.append("Q_wes", Q_we)
            logger.append("v_ew_ws", v_ew_w)
            logger.append("ω_ew_ws", ω_ew_w)
            logger.append("cmd_vels", sim.robot.cmd_vel.copy())
            logger.append("r_ow_ws", r_ow_ws)
            logger.append("Q_wos", Q_wos)

            # TODO put this computation in a function
            ddC_we = (
                core.math.skew3(α_ew_w)
                + core.math.skew3(ω_ew_w) @ core.math.skew3(ω_ew_w)
            ) @ C_we
            logger.append("ddC_we_norm", np.linalg.norm(ddC_we, ord=2))


        t = sim.step(t, step_robot=True)
        # if ctrl_manager.settings.dynamic_obstacle_settings.enabled:
        #     obstacle.step()

    try:
        print(f"Min constraint value = {np.min(logger.data['ineq_cons'])}")
    except:
        pass

    logger.add("control_durations", ctrl_manager.replanning_durations)

    # save logged data
    if cli_args.log is not None:
        logger.save(timestamp, name=cli_args.log)

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
        "balancing_constraints",
        legend_prefix="g",
        ylabel="Constraint Value",
        title="Balancing Inequality Constraints vs. Time",
    )
    plotter.plot_value_vs_time(
        "us",
        indices=range(sim.robot.nu),
        legend_prefix="u",
        ylabel="Commanded Input",
        title="Commanded Inputs vs. Time",
    )

    plotter.plot_control_durations()
    plotter.plot_cmd_vs_real_vel()

    plotter.plot_value_vs_time(
        "xs",
        indices=range(sim.robot.nq),
        legend_prefix="q",
        ylabel="Joint Position",
        title="Joint Positions vs. Time",
    )
    plotter.plot_value_vs_time(
        "xs",
        indices=range(sim.robot.nq + sim.robot.nv, sim.robot.nq + 2 * sim.robot.nv),
        legend_prefix="a",
        ylabel="Joint Acceleration",
        title="Joint Accelerations vs. Time",
    )
    ax = plotter.plot_value_vs_time(
        "ds",
        ylabel="Distance (m)",
        title="Distance Outside of SA vs. Time",
    )
    plotter.plot_value_vs_time(
        "orn_err",
        ylabel="Angle error (rad)",
        title="Angle between tray normal and total acceleration",
    )
    plotter.plot_value_vs_time(
        "ddC_we_norm",
        ylabel="ddC_we norm",
        title="ddC_we norm",
    )

    plt.show()


if __name__ == "__main__":
    main()
