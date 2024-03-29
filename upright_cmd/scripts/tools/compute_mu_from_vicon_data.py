#!/usr/bin/env python3
"""Compute coefficient of friction from Vicon data.

This assumes that the z-axis of the tray Vicon model is actually aligned with
the tray normal, and that the world frame z-axis is gravity-aligned.
"""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt
from mobile_manipulation_central import ros_utils
import upright_core as core


TRAY_VICON_NAME = "ThingWoodTray"
SLIP_MARGIN = 0.005  # 5 mm slip is considered "slipping"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("object_name", help="Name of object in Vicon system.")
    parser.add_argument("bagfile", help="Bag file to plot.")
    args = parser.parse_args()
    bag = rosbag.Bag(args.bagfile)

    tray_topic = ros_utils.vicon_topic_name(TRAY_VICON_NAME)
    tray_msgs = [msg for _, msg, _ in bag.read_messages(tray_topic)]
    ts, tray_poses = ros_utils.parse_transform_stamped_msgs(
        tray_msgs, normalize_time=False
    )

    obj_topic = ros_utils.vicon_topic_name(args.object_name)
    obj_msgs = [msg for _, msg, _ in bag.read_messages(obj_topic)]
    obj_ts, obj_poses = ros_utils.parse_transform_stamped_msgs(
        obj_msgs, normalize_time=False
    )
    obj_poses = np.array(ros_utils.interpolate_list(ts, obj_ts, obj_poses))
    ts -= ts[0]

    # compute angle of tray from horizontal (which is equal to angle from the
    # z-axis)
    z = np.array([0, 0, 1])
    n = ts.shape[0]
    angles = np.zeros(n)
    for i in range(n):
        q_we = tray_poses[i, 3:]
        normal = core.math.quat_rotate(q_we, z)
        angles[i] = np.arccos(z @ normal)

    # corresponding coefficients of friction
    mus = np.tan(angles)

    # compute offset of object from tray in tray's local frame
    r_locals = np.zeros((n, 3))
    for i in range(n):
        C_we = core.math.quat_to_rot(tray_poses[i, 3:])
        r_world = obj_poses[i, :3] - tray_poses[i, :3]
        r_locals[i, :] = C_we.T @ r_world

    # normalize by initial offset
    r_locals -= r_locals[0, :]

    # slip is the distance in the contact (x-y) plane
    slips = np.linalg.norm(r_locals[:, :2], axis=1)

    # find time and mu when object has slipped by 5mm
    slip_time = None
    for i in range(n):
        if np.abs(slips[i]) >= SLIP_MARGIN:
            break
    slip_time = ts[i]
    mu = mus[i]

    print(f"mu = {mu}")

    plt.plot(ts, angles, label="angle")
    plt.plot(ts, mus, label="mu")
    plt.plot(ts, slips, label="slip")
    plt.axvline(slip_time, color="k")
    plt.axhline(mu, color="k")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
