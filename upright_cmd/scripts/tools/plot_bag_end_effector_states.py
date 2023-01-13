#!/usr/bin/env python3
"""Plot end effector position, velocity, and acceleration from a bag file.

Also computes the maximum velocity and acceleration.
"""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt
from scipy import fft, signal

from mobile_manipulation_central import ros_utils
import upright_core as core
import upright_control as ctrl
import upright_cmd as cmd
from upright_ros_interface.parsing import parse_mpc_observation_msgs

import IPython


def parse_control_model(config_path):
    config = core.parsing.load_config(config_path)
    ctrl_config = config["controller"]
    return ctrl.manager.ControllerModel.from_config(ctrl_config)


def parse_states_from_vicon_data(bag):
    # TODO we would also like to compute the velocity and accelerations from
    # Vicon data directly
    tray_msgs = [
        msg for _, msg, _ in bag.read_messages("/vicon/ThingWoodTray/ThingWoodTray")
    ]
    ts, tray_poses = ros_utils.parse_transform_stamped_msgs(
        tray_msgs, normalize_time=True
    )
    n = len(ts)
    dts = ts[1:] - ts[:-1]

    positions = tray_poses[:, :3]

    # NOTE: we can align the measurements if desired
    dt = np.mean(dts)
    # new_ts = dt * np.arange(n)
    # positions = np.array(ros_utils.interpolate_list(new_ts, ts, positions))
    # ts = new_ts

    velocities = signal.savgol_filter(positions, 10, 2, deriv=1, delta=dt, axis=0)
    accelerations = signal.savgol_filter(positions, 30, 2, deriv=2, delta=dt, axis=0)

    # velocities = np.zeros_like(positions)
    # accelerations = np.zeros_like(positions)
    # for i in range(1, n - 1):
    #     velocities[i, :] = (positions[i + 1, :] - positions[i - 1, :]) / (
    #         ts[i + 1] - ts[i - 1]
    #     )
    #
    # for i in range(2, n - 2):
    #     accelerations[i, :] = (velocities[i + 1, :] - velocities[i - 1, :]) / (
    #         ts[i + 1] - ts[i - 1]
    #     )
    #
    # smoothed_velocities = np.zeros_like(velocities)
    # for i in range(2, n - 2):
    #     smoothed_velocities[i, :] = np.mean(velocities[i-1:i+2, :], axis=0)
    # velocities = smoothed_velocities
    #
    # w = 12
    # smoothed_accelerations = np.zeros_like(accelerations)
    # for i in range(w + 2, n - w - 2):
    #     smoothed_accelerations[i, :] = np.mean(accelerations[i-w:i+w+1, :], axis=0)
    # accelerations = smoothed_accelerations

    plt.figure()
    plt.plot(ts, positions[:, 0], label="x")
    plt.plot(ts, positions[:, 1], label="y")
    plt.plot(ts, positions[:, 2], label="z")
    plt.title("EE position")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(ts, velocities[:, 0], label="x")
    plt.plot(ts, velocities[:, 1], label="y")
    plt.plot(ts, velocities[:, 2], label="z")
    plt.title("EE velocity")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(ts, accelerations[:, 0], label="x")
    plt.plot(ts, accelerations[:, 1], label="y")
    plt.plot(ts, accelerations[:, 2], label="z")
    plt.title("EE acceleration")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s]")
    plt.legend()
    plt.grid()

    # F = fft.fft(accelerations, axis=0)
    # ω = fft.fftfreq(n, dts[0])[:n//2]
    # plt.figure()
    # plt.plot(ω, 2.0/n * np.abs(F[0:n//2]))
    # plt.grid()
    #
    # for i in range(len(ω)):
    #     if ω[i] >= 6:
    #         idx = i
    #         break
    # idx = 100
    #
    # F[idx:n//2+1, :] = 0
    # F[n//2+idx:, :] = 0
    # y = fft.ifft(F, axis=0)
    # y = np.real(y)
    #
    # plt.figure()
    # plt.plot(ts, y[:, 0], label="x")
    # plt.plot(ts, y[:, 1], label="y")
    # plt.plot(ts, y[:, 2], label="z")
    # plt.title("EE acceleration")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Acceleration [m/s]")
    # plt.legend()
    # plt.grid()
    # plt.show()


    # IPython.embed()


def main():
    parser = argparse.ArgumentParser()
    cmd.cli.add_bag_dir_arguments(parser)
    config_path, bag_path = cmd.cli.parse_bag_dir_args(parser.parse_args())

    model = parse_control_model(config_path)
    robot = model.robot

    bag = rosbag.Bag(bag_path)

    parse_states_from_vicon_data(bag)
    # plt.show()
    # return

    mpc_obs_msgs = [
        msg for _, msg, _ in bag.read_messages("/mobile_manipulator_mpc_observation")
    ]
    ts, xs, us = parse_mpc_observation_msgs(mpc_obs_msgs, normalize_time=True)

    n = len(ts)

    ee_poses = np.zeros((n, 7))
    ee_velocities = np.zeros((n, 6))
    ee_accelerations = np.zeros((n, 6))

    for i in range(n):
        robot.forward(xs[i, :])
        ee_poses[i, :] = np.concatenate(robot.link_pose())
        ee_velocities[i, :] = np.concatenate(robot.link_velocity())
        ee_accelerations[i, :] = np.concatenate(robot.link_acceleration())

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
    plt.plot(ts, ee_poses[:, 0], label="x")
    plt.plot(ts, ee_poses[:, 1], label="y")
    plt.plot(ts, ee_poses[:, 2], label="z")
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
