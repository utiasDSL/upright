#!/usr/bin/env python3
"""Plot end effector position, velocity, and acceleration from a bag file.

Also computes the maximum velocity and acceleration.
"""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt

from mobile_manipulation_central import ros_utils
import upright_core as core
import upright_control as ctrl
import upright_cmd as cmd
from upright_ros_interface.parsing import parse_mpc_observation_msgs


def parse_control_model(config_path):
    config = core.parsing.load_config(config_path)
    ctrl_config = config["controller"]
    return ctrl.manager.ControllerModel.from_config(ctrl_config), config


def main():
    parser = argparse.ArgumentParser()
    cmd.cli.add_bag_dir_arguments(parser)
    config_path, bag_path = cmd.cli.parse_bag_dir_args(parser.parse_args())

    model, config = parse_control_model(config_path)
    robot = model.robot

    bag = rosbag.Bag(bag_path)

    mpc_obs_msgs = [
        msg for _, msg, _ in bag.read_messages("/mobile_manipulator_mpc_observation")
    ]
    ts, xs, us = parse_mpc_observation_msgs(mpc_obs_msgs, normalize_time=True)

    n = len(ts)

    ee_poses = np.zeros((n, 7))
    ee_velocities = np.zeros((n, 6))
    ee_accelerations = np.zeros((n, 6))

    for i in range(n):
        robot.forward_xu(xs[i, :])
        ee_poses[i, :] = np.concatenate(robot.link_pose())
        ee_velocities[i, :] = np.concatenate(robot.link_velocity())
        ee_accelerations[i, :] = np.concatenate(robot.link_classical_acceleration())

    # reference position is relative to the initial position
    ref = config["controller"]["waypoints"][0]["position"] + ee_poses[0, :3]

    velocity_magnitudes = np.linalg.norm(ee_velocities[:, :3], axis=1)
    max_vel_idx = np.argmax(velocity_magnitudes)
    max_vel = velocity_magnitudes[max_vel_idx]
    print(f"Max velocity = {max_vel:.3f} m/s at time = {ts[max_vel_idx]} seconds.")

    acceleration_magnitudes = np.linalg.norm(ee_accelerations[:, :3], axis=1)
    max_acc_idx = np.argmax(acceleration_magnitudes)
    max_acc = acceleration_magnitudes[max_acc_idx]
    print(
        f"Max acceleration = {max_acc:.3f} m/s^2 at time = {ts[max_acc_idx]} seconds."
    )

    plt.figure()
    lx, = plt.plot(ts, ee_poses[:, 0], label="x")
    ly, = plt.plot(ts, ee_poses[:, 1], label="y")
    lz, = plt.plot(ts, ee_poses[:, 2], label="z")
    plt.axhline(ref[0], label="xd", linestyle="--", color=lx.get_color())
    plt.axhline(ref[1], label="yd", linestyle="--", color=ly.get_color())
    plt.axhline(ref[2], label="zd", linestyle="--", color=lz.get_color())
    plt.title("EE position")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(ts, ee_velocities[:, 0], label="x")
    plt.plot(ts, ee_velocities[:, 1], label="y")
    plt.plot(ts, ee_velocities[:, 2], label="z")
    plt.title("EE velocity")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(ts, ee_accelerations[:, 0], label="x")
    plt.plot(ts, ee_accelerations[:, 1], label="y")
    plt.plot(ts, ee_accelerations[:, 2], label="z")
    plt.title("EE acceleration")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s^2]")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
