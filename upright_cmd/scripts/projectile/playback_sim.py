#!/usr/bin/env python3
"""Simulation to play back the real projectile Vicon data and simulated robot response."""
import argparse
import datetime
import time

import rospy
import rosbag
import numpy as np
import pybullet as pyb
import matplotlib.pyplot as plt
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

    # parse CLI args (directory containing bag and config file)
    parser = argparse.ArgumentParser()
    cmd.cli.add_bag_dir_arguments(parser)
    config_path, bag_path = cmd.cli.parse_bag_dir_args(parser.parse_args())

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
    # sim.launch_dynamic_obstacles()

    # initial time, state, input
    t = 0.0
    q, v = sim.robot.joint_states()
    a = np.zeros(sim.robot.nv)

    # we don't care about the simulated obstacle, so just give it zero values
    x_obs = np.zeros(9)
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
        [0], [projectile_positions[0, :]], [projectile_velocities[0, :]], [np.zeros(3)],
    )
    projectile.start(0)
    projectile_index = 0
    K_proj = 100

    projectile_msgs_est = [msg for _, msg, _ in bag.read_messages("/Projectile/joint_states")]
    projectile_ts_est = np.array([msg.header.stamp.to_sec() for msg in projectile_msgs_est])
    projectile_ts_est -= t0
    projectile_rs_est = np.array([msg.position for msg in projectile_msgs_est])
    projectile_vs_est = np.array([msg.velocity for msg in projectile_msgs_est])

    # reference pose trajectory
    model.update(x)
    r_ew_w, Q_we = model.robot.link_pose()
    ref = ctrl.wrappers.TargetTrajectories.from_config(ctrl_config, r_ew_w, Q_we, u)

    # frames and ghost (i.e., pure visual) objects
    for r_ew_w_d, Q_we_d in ref.poses():
        debug_frame_world(0.2, list(r_ew_w_d), orientation=Q_we_d, line_width=3)

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
        cmd_vel_world = sim.robot.command_velocity(cmd_vel, bodyframe=True)

        pyb.resetBasePositionAndOrientation(
            projectile.body.uid,
            list(projectile_positions[projectile_index, :]),
            [0, 0, 0, 1],
        )

        # for recording the obstacle position
        x_obs[:3] = projectile_positions[projectile_index, :]

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
            logger.append("cmd_vels", cmd_vel_world)
            logger.append("r_ow_ws", r_ow_ws)
            logger.append("Q_wos", Q_wos)

            # log controller stuff
            r_ew_w_d, Q_we_d = ref.get_desired_pose(t)
            logger.append("r_ew_w_ds", r_ew_w_d)
            logger.append("Q_we_ds", Q_we_d)

            d_obs = np.linalg.norm(r_ew_w - x_obs[:3])
            logger.append("collision_pair_distances", np.array([d_obs]))

            # NOTE: not accurate due to lack of acceleration info
            model.update(x)
            logger.append("ddC_we_norm", model.ddC_we_norm())
            logger.append("balancing_constraints", model.balancing_constraints())
            logger.append("sa_dists", model.support_area_distances())
            logger.append("orn_err", model.angle_between_acc_and_normal())

        t = sim.step(t, step_robot=False)[0]
        time.sleep(sim.timestep)

    try:
        print(f"Min constraint value = {np.min(logger.data['balancing_constraints'])}")
    except:
        pass

    # visualize data
    DataPlotter.from_logger(logger).plot_all(show=False)

    plt.figure()
    plt.plot(projectile_ts_est, projectile_rs_est[:, 0], label="x")
    plt.plot(projectile_ts_est, projectile_rs_est[:, 1], label="y")
    plt.plot(projectile_ts_est, projectile_rs_est[:, 2], label="z")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Estimated obstacle position")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(projectile_ts_est, projectile_vs_est[:, 0], label="x")
    plt.plot(projectile_ts_est, projectile_vs_est[:, 1], label="y")
    plt.plot(projectile_ts_est, projectile_vs_est[:, 2], label="z")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Estimated obstacle velocity")
    plt.legend()
    plt.grid()

    plt.show()

if __name__ == "__main__":
    main()
