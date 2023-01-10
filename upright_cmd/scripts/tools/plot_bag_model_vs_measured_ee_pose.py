#!/usr/bin/env python3
"""Plot pose of wood tray as measured by Vicon compared to the pose from the model."""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt
import IPython

from mobile_manipulation_central import ros_utils
import upright_core as core
import upright_control as ctrl
import upright_cmd as cmd


def parse_control_model(config_path):
    config = core.parsing.load_config(config_path)
    ctrl_config = config["controller"]
    return ctrl.manager.ControllerModel.from_config(ctrl_config)


def main():
    np.set_printoptions(precision=3, suppress=True)

    # parse CLI args (directory containing bag and config file)
    parser = argparse.ArgumentParser()
    cmd.cli.add_bag_dir_arguments(parser)
    config_path, bag_path = cmd.cli.parse_bag_dir_args(parser.parse_args())

    model = parse_control_model(config_path)
    robot = model.robot

    bag = rosbag.Bag(bag_path)

    ur10_msgs = [msg for _, msg, _ in bag.read_messages("/ur10/joint_states")]
    ridgeback_msgs = [msg for _, msg, _ in bag.read_messages("/ridgeback/joint_states")]
    tray_msgs = [
        msg for _, msg, _ in bag.read_messages("/vicon/ThingWoodTray/ThingWoodTray")
    ]

    ts = ros_utils.parse_time(tray_msgs, normalize_time=False)
    ur10_ts, ur10_qs, _ = ros_utils.parse_ur10_joint_state_msgs(
        ur10_msgs, normalize_time=False
    )
    rb_ts, rb_qs, _ = ros_utils.parse_ridgeback_joint_state_msgs(
        ridgeback_msgs, normalize_time=False
    )

    ur10_qs_aligned = ros_utils.interpolate_list(ts, ur10_ts, ur10_qs)
    rb_qs_aligned = ros_utils.interpolate_list(ts, rb_ts, rb_qs)
    qs = np.hstack((rb_qs_aligned, ur10_qs_aligned))
    n = qs.shape[0]

    # prepend default obstacle positions, which we don't care about
    qs = np.hstack((np.zeros((n, 3)), qs))
    ts -= ts[0]

    # just for comparison
    q_home = model.settings.initial_state[:9]
    q_home = np.concatenate((np.zeros(3), q_home))
    robot.forward_qva(q_home)
    r_home, Q_home = robot.link_pose()

    # compute modelled EE poses
    z = np.array([0, 0, 1])
    ee_positions = np.zeros((n, 3))
    ee_orientations = np.zeros((n, 4))
    ee_angles = np.zeros(n)
    for i in range(n):
        robot.forward_qva(qs[i, :])
        ee_positions[i, :], ee_orientations[i, :] = robot.link_pose()
        R = core.math.quat_to_rot(ee_orientations[i, :])

        # angle from the upright direction
        ee_angles[i] = np.arccos(z @ R @ z)

    # compute measured tray poses
    tray_positions = np.zeros((n, 3))
    tray_orientations = np.zeros((n, 4))
    tray_angles = np.zeros(n)
    for i in range(n):
        p = tray_msgs[i].transform.translation
        tray_positions[i, :] = [p.x, p.y, p.z]
        Q = tray_msgs[i].transform.rotation
        orientation = np.array([Q.x, Q.y, Q.z, Q.w])
        tray_orientations[i, :] = orientation
        R = core.math.quat_to_rot(orientation)
        tray_angles[i] = np.arccos(z @ R @ z)

    # error between measured and modelled orientation
    orientation_errors = np.zeros((n, 4))
    angle_errors = np.zeros(n)
    for i in range(n):
        R1 = core.math.quat_to_rot(ee_orientations[i, :])
        R2 = core.math.quat_to_rot(tray_orientations[i, :])
        ΔQ = core.math.rot_to_quat(R1 @ R2.T)
        orientation_errors[i, :] = ΔQ
        angle_errors[i] = core.math.quat_angle(ΔQ)

    # EE (model) position vs. time
    plt.figure()
    plt.plot(ts, ee_positions[:, 0], label="x")
    plt.plot(ts, ee_positions[:, 1], label="y")
    plt.plot(ts, ee_positions[:, 2], label="z")
    plt.xlabel("Time (s)")
    plt.ylabel("EE position (m)")
    plt.title(f"EE position vs. time")
    plt.legend()
    plt.grid()

    # EE (model) quaternion orientation vs. time
    plt.figure()
    plt.plot(ts, ee_orientations[:, 0], label="x")
    plt.plot(ts, ee_orientations[:, 1], label="y")
    plt.plot(ts, ee_orientations[:, 2], label="z")
    plt.plot(ts, ee_orientations[:, 3], label="w")
    plt.plot(ts, ee_angles, label="angle")
    plt.xlabel("Time (s)")
    plt.ylabel("EE orientation")
    plt.title(f"EE orientation vs. time")
    plt.legend()
    plt.grid()

    # Tray (measured) position vs. time
    plt.figure()
    plt.plot(ts, tray_positions[:, 0], label="x")
    plt.plot(ts, tray_positions[:, 1], label="y")
    plt.plot(ts, tray_positions[:, 2], label="z")
    plt.xlabel("Time (s)")
    plt.ylabel("Tray position (m)")
    plt.title(f"Tray position vs. time")
    plt.legend()
    plt.grid()

    # Tray (measured) quaternion orientation vs. time
    plt.figure()
    plt.plot(ts, tray_orientations[:, 0], label="x")
    plt.plot(ts, tray_orientations[:, 1], label="y")
    plt.plot(ts, tray_orientations[:, 2], label="z")
    plt.plot(ts, tray_orientations[:, 3], label="w")
    plt.plot(ts, tray_angles, label="angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Tray orientation")
    plt.title(f"Tray orientation vs. time")
    plt.legend()
    plt.grid()

    # Orientation error between model and measured values
    plt.figure()
    plt.plot(ts, orientation_errors[:, 0], label="x")
    plt.plot(ts, orientation_errors[:, 1], label="y")
    plt.plot(ts, orientation_errors[:, 2], label="z")
    plt.plot(ts, orientation_errors[:, 3], label="w")
    plt.plot(ts, angle_errors, label="angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Orientation error")
    plt.title(f"Orientation error vs. time")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
