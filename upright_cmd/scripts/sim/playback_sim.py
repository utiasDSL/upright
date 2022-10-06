#!/usr/bin/env python3
"""Closed-loop upright simulation using Pybullet over ROS."""
import argparse
import datetime
import glob
import time
from pathlib import Path

import rospy
import rosbag
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


def parse_args():
    """Parse CLI args into the required config and bag file paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory", help="Directory in which to find config and bag files."
    )
    parser.add_argument(
        "--config_name",
        help="Name of config file within the given directory.",
        required=False,
    )
    parser.add_argument(
        "--bag_name",
        help="Name of bag file within the given directory.",
        required=False,
    )
    cli_args = parser.parse_args()

    dir_path = Path(cli_args.directory)

    if cli_args.config_name is not None:
        config_path = dir_path / cli_args.config_name
    else:
        config_files = glob.glob(dir_path.as_posix() + "/*.yaml")
        if len(config_files) == 0:
            raise FileNotFoundError("Error: could not find a config file in the specified directory.")
        if len(config_files) > 1:
            raise FileNotFoundError("Error: multiple possible config files in the specified directory. Please specify the name using the `--config_name` option.")
        config_path = config_files[0]

    if cli_args.bag_name is not None:
        bag_path = dir_path / cli_args.bag_name
    else:
        bag_files = glob.glob(dir_path.as_posix() + "/*.bag")
        if len(bag_files) == 0:
            raise FileNotFoundError("Error: could not find a bag file in the specified directory.")
        if len(config_files) > 1:
            print(
                "Error: multiple bag files in the specified directory. Please specify the name using the `--bag_name` option."
            )
        bag_path = bag_files[0]
    return config_path, bag_path


def main():
    np.set_printoptions(precision=3, suppress=True)

    config_path, bag_path = parse_args()

    # load configuration
    config = core.parsing.load_config(config_path)
    sim_config = config["simulation"]
    ctrl_config = config["controller"]
    log_config = config["logging"]

    # start the simulation
    timestamp = datetime.datetime.now()
    sim = simulation.BulletSimulation(config=sim_config, timestamp=timestamp)

    # settle sim to make sure everything is touching comfortably
    sim.settle(5.0)

    # initial time, state, input
    t = 0.0
    q, v = sim.robot.joint_states()
    a = np.zeros(sim.robot.nv)
    x_obs = sim.dynamic_obstacle_state()
    x = np.concatenate((q, v, a, x_obs))
    u = np.zeros(sim.robot.nu)

    Q_obs = np.array([0, 0, 0, 1])

    # data logging
    logger = DataLogger(config)

    logger.add("sim_timestep", sim.timestep)
    logger.add("duration", sim.duration)
    logger.add("object_names", [str(name) for name in sim.objects.keys()])

    logger.add("nq", ctrl_config["robot"]["dims"]["q"])
    logger.add("nv", ctrl_config["robot"]["dims"]["v"])
    logger.add("nx", ctrl_config["robot"]["dims"]["x"])
    logger.add("nu", ctrl_config["robot"]["dims"]["u"])

    # TODO the problem is that the simulation does not currently have dynamic
    # obstacles enabled, I think
    model = ctrl.manager.ControllerModel.from_config(ctrl_config, x0=x)

    bag = rosbag.Bag(bag_path)
    base_cmd_msgs = [msg for _, msg, _ in bag.read_messages("/ridgeback/cmd_vel")]
    base_cmd_vels = np.array(
        [[msg.linear.x, msg.linear.y, msg.angular.z] for msg in base_cmd_msgs]
    )
    base_cmd_ts = np.array(
        [t.to_sec() for _, _, t in bag.read_messages("/ridgeback/cmd_vel")]
    )
    t0 = base_cmd_ts[0]
    base_cmd_ts -= t0
    base_cmd_index = 0

    arm_cmd_msgs = [msg for _, msg, _ in bag.read_messages("/ur10/cmd_vel")]
    arm_cmd_vels = np.array([msg.data for msg in arm_cmd_msgs])
    arm_cmd_ts = np.array(
        [t.to_sec() for _, _, t in bag.read_messages("/ur10/cmd_vel")]
    )
    arm_cmd_ts -= t0
    arm_cmd_index = 0

    # projectile_msgs = [
    #     msg for _, msg, _ in bag.read_messages("/Projectile/joint_states")
    # ]
    # projectile_positions = np.array([msg.position for msg in projectile_msgs])
    # projectile_velocities = np.array([msg.velocity for msg in projectile_msgs])
    projectile_msgs = [
        msg for _, msg, _ in bag.read_messages("/vicon/Projectile/Projectile")
    ]
    projectile_positions = np.array(
        [
            [
                msg.transform.translation.x,
                msg.transform.translation.y,
                msg.transform.translation.z,
            ]
            for msg in projectile_msgs
        ]
    )
    projectile_velocities = np.zeros_like(projectile_positions)

    projectile_ts = np.array([msg.header.stamp.to_sec() for msg in projectile_msgs])
    projectile_ts -= t0
    projectile = simulation.BulletDynamicObstacle(
        projectile_positions[0, :], projectile_velocities[0, :]
    )
    projectile.start(0)
    projectile_index = 0
    K_proj = 100

    # reference pose trajectory
    model.update(x)
    r_ew_w, Q_we = model.robot.link_pose()
    ref = ctrl.wrappers.TargetTrajectories.from_config(ctrl_config, r_ew_w, Q_we, u)

    # frames and ghost (i.e., pure visual) objects
    for r_ew_w_d, Q_we_d in ref.poses():
        debug_frame_world(0.2, list(r_ew_w_d), orientation=Q_we_d, line_width=3)

    IPython.embed()

    t = 0
    duration = arm_cmd_ts[-1]

    # simulation loop
    while not rospy.is_shutdown() and t <= duration:
        # feedback is in the world frame
        q, v = sim.robot.joint_states(add_noise=True, bodyframe=False)

        # commands are always in the body frame (to match the real robot)
        while (
            base_cmd_index + 1 < base_cmd_ts.shape[0]
            and base_cmd_ts[base_cmd_index + 1] <= t
        ):
            base_cmd_index += 1
        while (
            arm_cmd_index + 1 < arm_cmd_ts.shape[0]
            and arm_cmd_ts[arm_cmd_index + 1] <= t
        ):
            arm_cmd_index += 1
        while (
            projectile_index + 1 < projectile_ts.shape[0]
            and projectile_ts[projectile_index + 1] <= t
        ):
            projectile_index += 1
        cmd_vel = np.concatenate(
            (base_cmd_vels[base_cmd_index, :], arm_cmd_vels[arm_cmd_index, :])
        )
        sim.robot.command_velocity(cmd_vel, bodyframe=True)

        # manually steer the projectile
        projectile_cmd_vel = (
            K_proj
            * (projectile_positions[projectile_index, :] - projectile.joint_state()[0])
            + projectile_velocities[projectile_index, :]
        )
        # pyb.resetBaseVelocity(
        #     projectile.body.uid, linearVelocity=list(projectile_cmd_vel)
        # )
        pyb.resetBasePositionAndOrientation(
            projectile.body.uid,
            list(projectile_positions[projectile_index, :]),
            [0, 0, 0, 1],
        )

        if logger.ready(t):
            x = np.concatenate((q, v, a, x_obs))

            # log sim stuff
            r_ew_w, Q_we = sim.robot.link_pose()
            v_ew_w, ω_ew_w = sim.robot.link_velocity()
            r_ow_ws, Q_wos = sim.object_poses()
            logger.append("ts", t)
            logger.append("xs", x)
            logger.append("r_ew_ws", r_ew_w)
            logger.append("Q_wes", Q_we)
            logger.append("v_ew_ws", v_ew_w)
            logger.append("ω_ew_ws", ω_ew_w)
            logger.append("cmd_vels", cmd_vel)
            logger.append("r_ow_ws", r_ow_ws)
            logger.append("Q_wos", Q_wos)

            # log controller stuff
            r_ew_w_d, Q_we_d = ref.get_desired_pose(t)
            logger.append("r_ew_w_ds", r_ew_w_d)
            logger.append("Q_we_ds", Q_we_d)

            # NOTE: not accurate due to lack of acceleration info
            model.update(x)
            logger.append("ddC_we_norm", model.ddC_we_norm())
            logger.append("balancing_constraints", model.balancing_constraints())
            logger.append("sa_dists", model.support_area_distances())
            logger.append("orn_err", model.angle_between_acc_and_normal())

        t = sim.step(t, step_robot=False)

    try:
        print(f"Min constraint value = {np.min(logger.data['balancing_constraints'])}")
    except:
        pass

    # visualize data
    DataPlotter.from_logger(logger).plot_all(show=True)


if __name__ == "__main__":
    main()
