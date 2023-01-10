#!/usr/bin/env python3
"""Compute coefficient of friction from Vicon data."""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt
from mobile_manipulation_central import ros_utils
import upright_core as core

import IPython

TRAY_VICON_NAME = "ThingWoodTray"
OBJECT_VICON_NAME = "ThingWoodBlock"


def vicon_topic_name(name):
    return "/".join(["/vicon", name, name])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file to plot.")
    args = parser.parse_args()
    bag = rosbag.Bag(args.bagfile)

    tray_msgs = [
        msg for _, msg, _ in bag.read_messages(vicon_topic_name(TRAY_VICON_NAME))
    ]
    ts, tray_poses = ros_utils.parse_transform_stamped_msgs(
        tray_msgs, normalize_time=False
    )

    obj_msgs = [
        msg for _, msg, _ in bag.read_messages(vicon_topic_name(OBJECT_VICON_NAME))
    ]
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

    # compute slip in the contact plane
    slips = np.zeros(n)
    for i in range(n):
        C_we = core.math.quat_to_rot(tray_poses[i, 3:])
        r_world = obj_poses[i, :3] - tray_poses[i, :3]
        r_local = C_we.T @ r_world
        slips[i] = np.linalg.norm(r_local[:2])

    mus = np.tan(angles)
    slip0 = np.mean(slips[:10])

    # find time and mu when object has slipped by 5mm
    slip_time = None
    for i in range(n):
        if np.abs(slips[i] - slip0) >= 0.005:
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
