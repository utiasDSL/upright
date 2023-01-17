#!/usr/bin/env python3
"""PyBullet simulation using the bounded balancing constraints"""
import datetime

import numpy as np
import pybullet as pyb
from pyb_utils.ghost import GhostSphere
from pyb_utils.frame import debug_frame_world

from upright_sim import simulation
from upright_core.logging import DataLogger, DataPlotter
import upright_core as core
import upright_control as ctrl
import upright_cmd as cmd

import IPython


def main():
    np.set_printoptions(precision=3, suppress=True)

    argparser = cmd.cli.sim_arg_parser()
    argparser.add_argument("trajectory_file", help="NPZ file to load the trajectory from.")
    cli_args = argparser.parse_args()

    # load configuration
    config = core.parsing.load_config(cli_args.config)
    sim_config = config["simulation"]
    ctrl_config = config["controller"]
    log_config = config["logging"]

    # start the simulation
    timestamp = datetime.datetime.now()
    sim = simulation.UprightSimulation(
        config=sim_config, timestamp=timestamp, cli_args=cli_args
    )

    # initial time, state, input
    t = 0.0
    q, v = sim.robot.joint_states()
    a = np.zeros(sim.robot.nv)
    x = np.concatenate((q, v, a))

    # controller
    ref = ctrl.trajectory.StateInputTrajectory.load(cli_args.trajectory_file)
    model = ctrl.manager.ControllerModel.from_config(ctrl_config)
    mapping = ctrl.trajectory.StateInputMapping(model.settings.dims)
    Kp = np.eye(model.settings.dims.q)
    interpolator = ctrl.trajectory.TrajectoryInterpolator(mapping, ref)

    # data logging
    logger = DataLogger(config)

    logger.add("sim_timestep", sim.timestep)
    logger.add("duration", sim.duration)
    logger.add("object_names", [str(name) for name in sim.objects.keys()])

    logger.add("nq", ctrl_config["robot"]["dims"]["q"])
    logger.add("nv", ctrl_config["robot"]["dims"]["v"])
    logger.add("nx", ctrl_config["robot"]["dims"]["x"])
    logger.add("nu", ctrl_config["robot"]["dims"]["u"])

    # simulation loop
    while t <= sim.duration:
        q, v = sim.robot.joint_states(add_noise=True)
        x, _ = mapping.qva2xu(q, v, a)

        xd = interpolator.interpolate(t)
        qd, vd, a = mapping.xu2qva(xd)

        v_cmd = Kp @ (qd - q) + vd
        sim.robot.command_velocity(v_cmd)

        if logger.ready(t):
            # log sim stuff
            r_ew_w, Q_we = sim.robot.link_pose()
            v_ew_w, ω_ew_w = sim.robot.link_velocity()
            r_ow_ws, Q_wos = sim.object_poses()
            logger.append("ts", t)
            logger.append("xs", x)
            logger.append("xds", xd)
            logger.append("r_ew_ws", r_ew_w)
            logger.append("Q_wes", Q_we)
            logger.append("v_ew_ws", v_ew_w)
            logger.append("ω_ew_ws", ω_ew_w)
            logger.append("cmd_vels", sim.robot.cmd_vel.copy())
            logger.append("r_ow_ws", r_ow_ws)
            logger.append("Q_wos", Q_wos)

            # log controller stuff
            model.update(xd)
            r_ew_w_d, Q_we_d = model.robot.link_pose()
            logger.append("r_ew_w_ds", r_ew_w_d)
            logger.append("Q_we_ds", Q_we_d)

            model.update(x)
            logger.append("ddC_we_norm", model.ddC_we_norm())
            logger.append("balancing_constraints", model.balancing_constraints())
            logger.append("sa_dists", model.support_area_distances())
            logger.append("orn_err", model.angle_between_acc_and_normal())

        t = sim.step(t, step_robot=False)

    try:
        print(f"Min constraint value = {np.min(logger.data['ineq_cons'])}")
    except:
        pass

    # save logged data
    if cli_args.log is not None:
        logger.save(timestamp, name=cli_args.log)

    if sim.video_manager.save:
        print(f"Saved video to {sim.video_manager.path}")

    # visualize data
    DataPlotter(logger).plot_all(show=True)


if __name__ == "__main__":
    main()
