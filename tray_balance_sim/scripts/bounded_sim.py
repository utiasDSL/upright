#!/usr/bin/env python3
"""PyBullet simulation using the bounded balancing constraints"""
import datetime

import numpy as np
import pybullet as pyb
from pyb_utils.ghost import GhostSphere
from pyb_utils.frame import debug_frame_world

from tray_balance_sim import simulation
from tray_balance_constraints.logging import DataLogger, DataPlotter
import tray_balance_constraints as core
import tray_balance_ocs2 as ctrl
import upright_cmd as cmd

import IPython


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
    sim = simulation.BulletSimulation(
        config=sim_config, timestamp=timestamp, cli_args=cli_args
    )

    # initial time, state, input
    t = 0.0
    q, v = sim.robot.joint_states()
    a = np.zeros(sim.robot.nv)
    x = np.concatenate((q, v, a))
    u = np.zeros(sim.robot.nu)

    # controller
    ctrl_manager = ctrl.manager.ControllerManager.from_config(ctrl_config, x0=x)

    # data logging
    logger = DataLogger(config)

    logger.add("sim_timestep", sim.timestep)
    logger.add("duration", sim.duration)
    logger.add("ctrl_timestep", ctrl_manager.timestep)
    logger.add("object_names", [str(name) for name in sim.objects.keys()])

    logger.add("nq", ctrl_config["robot"]["dims"]["q"])
    logger.add("nv", ctrl_config["robot"]["dims"]["v"])
    logger.add("nx", ctrl_config["robot"]["dims"]["x"])
    logger.add("nu", ctrl_config["robot"]["dims"]["u"])

    # frames and ghost (i.e., pure visual) objects
    for r_ew_w_d, Q_we_d in ctrl_manager.ref.poses():
        # sim.ghosts.append(GhostSphere(radius=0.05, position=r_ew_w_d, color=(0, 1, 0, 1)))
        debug_frame_world(0.2, list(r_ew_w_d), orientation=Q_we_d, line_width=3)

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

        # TODO more logger reforms to come
        if logger.ready(t):
            # if ctrl_manager.settings.dynamic_obstacle_settings.enabled:
            #     recorder.dynamic_obs_distance[idx, :] = mpc.stateInequalityConstraint(
            #         "dynamicObstacleAvoidance", t, x
            #     )
            # if ctrl_manager.settings.collision_avoidance_settings.enabled:
            #     recorder.collision_pair_distance[
            #         idx, :
            #     ] = mpc.stateInequalityConstraint("collisionAvoidance", t, x)

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

            # log controller stuff
            r_ew_w_d, Q_we_d = ctrl_manager.ref.get_desired_pose(t)
            logger.append("r_ew_w_ds", r_ew_w_d)
            logger.append("Q_we_ds", Q_we_d)

            ctrl_manager.model.update(x, u)
            logger.append("ddC_we_norm", ctrl_manager.model.ddC_we_norm())
            logger.append("balancing_constraints", ctrl_manager.model.balancing_constraints())
            logger.append("sa_dists", ctrl_manager.model.support_area_distances())
            logger.append("orn_err", ctrl_manager.model.angle_between_acc_and_normal())

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

    if sim.video_manager.save:
        print(f"Saved video to {sim.video_manager.path}")

    # visualize data
    DataPlotter(logger).plot_all(show=True)


if __name__ == "__main__":
    main()
