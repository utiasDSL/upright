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
from tray_balance_sim.recording import DataLogger, DataPlotter

import tray_balance_constraints as core
import tray_balance_ocs2 as ctrl

import IPython


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configuration file.")
    parser.add_argument(
        "--log",
        nargs="?",
        default=None,
        const="",
        help="Log data. Optionally specify prefix for log directoy.",
    )
    parser.add_argument(
        "--video",
        nargs="?",
        default=None,
        const="",
        help="Record video. Optionally specify prefix for video directoy.",
    )
    return parser.parse_args()


def main():
    np.set_printoptions(precision=3, suppress=True)

    cli_args = parse_cli_args()

    # load configuration
    config = util.load_config(cli_args.config)
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
    mpc = ctrl_wrapper.controller(r_ew_w, Q_we)

    # visual indicator for target - note debug frame doesn't show up in video
    r_ew_w_d, Q_we_d = ref.pose(ref.states[0])
    debug_frame_world(0.2, list(r_ew_w_d), orientation=Q_we_d, line_width=3)

    # ghost (i.e., pure visual) objects
    ghosts = [GhostSphere(radius=0.05, position=r_ew_w_d, color=(0, 1, 0, 1))]

    target_idx = 0
    x_opt = np.copy(x)

    # simulation loop
    for i in range(num_timesteps):
        q, v = robot.joint_states()
        x = np.concatenate((q, v, a))

        # add noise to state variables
        q_noisy, v_noisy = robot.joint_states(add_noise=True)
        x_noisy = np.concatenate((q_noisy, v_noisy, a))

        # by using x_opt, we're basically just doing pure open-loop planning,
        # since the state never deviates from the optimal trajectory (at least
        # due to noise)
        # this does have the benefit of smoothing out the state used for
        # computation, which is important for constraint handling
        if ctrl_config["use_noisy_state_to_plan"]:
            mpc.setObservation(t, x_noisy, u)
        else:
            mpc.setObservation(t, x_opt, u)

        # this should be set to reflect the MPC time step
        # we can increase it if the MPC rate is faster
        if i % ctrl_period == 0:
            try:
                t0 = time.time()
                mpc.advanceMpc()
                t1 = time.time()
            except RuntimeError as e:
                print(e)
                print("exit the interpreter to proceed to plots")
                IPython.embed()
                # i -= 1  # for the recorder
                break
            # recorder.control_durations[i // ctrl_period] = t1 - t0
            logger.append("control_durations", t1 - t0)

        # As far as I can tell, evaluateMpcSolution actually computes the input
        # for the particular time and state (the input is often at least
        # state-varying in DDP, with linear feedback on state error). OTOH,
        # getMpcSolution just gives the current MPC policy trajectory over the
        # entire time horizon, without accounting for the given state. So it is
        # like doing feedforward input only, which is bad.
        mpc.evaluateMpcSolution(t, x_noisy, x_opt, u)
        a = np.copy(x_opt[-robot.nv:])
        # robot.command_acceleration(u)
        robot.command_jerk(u)

        if i % log_dt == 0:
            if ctrl_wrapper.settings.tray_balance_settings.enabled:
                if (
                    ctrl_wrapper.settings.tray_balance_settings.constraint_type
                    == ctrl.bindings.ConstraintType.Hard
                ):
                    balance_cons = mpc.stateInputInequalityConstraint(
                        "trayBalance", t, x, u
                    )
                else:
                    balance_cons = mpc.softStateInputInequalityConstraint(
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

    if len(logger.data["ineq_cons"]) > 0:
        print(f"Min constraint value = {np.min(logger.data['ineq_cons'])}")

    # save logged data
    if cli_args.log is not None:
        logger.save(log_dir, now, name=cli_args.log)

    # visualize data
    plotter = DataPlotter(logger)
    plotter.plot_ee_position()
    plotter.plot_ee_orientation()
    plotter.plot_ee_velocity()
    for j in range(num_objects):
        plotter.plot_object_error(j)
    plotter.plot_balancing_constraints()
    plotter.plot_commands()
    plotter.plot_control_durations()
    plotter.plot_cmd_vs_real_vel()
    plotter.plot_joint_config()
    plotter.plot_joint_acceleration()

    # last_sim_index = i
    # recorder.plot_ee_position(last_sim_index)
    # recorder.plot_ee_orientation(last_sim_index)
    # recorder.plot_ee_velocity(last_sim_index)
    # for j in range(len(sim_objects)):
    #     recorder.plot_object_error(last_sim_index, j)
    # recorder.plot_balancing_constraints(last_sim_index)
    # recorder.plot_commands(last_sim_index)
    # recorder.plot_control_durations(last_sim_index)
    # recorder.plot_cmd_vs_real_vel(last_sim_index)
    # recorder.plot_joint_config(last_sim_index)

    # if ctrl_wrapper.settings.dynamic_obstacle_settings.enabled:
    #     print(
    #         f"Min dynamic obstacle distance = {np.min(recorder.dynamic_obs_distance, axis=0)}"
    #     )
    #     recorder.plot_dynamic_obs_dist(last_sim_index)

    plt.show()


if __name__ == "__main__":
    main()
