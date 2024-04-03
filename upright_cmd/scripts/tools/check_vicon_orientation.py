#!/usr/bin/env python3
import argparse
import glob
from pathlib import Path

import numpy as np
import rospy
import rosbag
from spatialmath.base import q2r

from mobile_manipulation_central import ros_utils

import IPython


VICON_OBJECT_NAME = "ThingWoodTray"
# VICON_OBJECT_NAME = "ThingBase_3"
# VICON_OBJECT_NAME = "ThingPinkBottle"


def main():
    np.set_printoptions(suppress=True, precision=6)

    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Bag file to plot.")
    parser.add_argument(
        "--glob", action="store_true", help="Glob for bag files under directory."
    )
    args = parser.parse_args()

    if args.glob:
        paths = glob.glob(args.path + "/**/*.bag", recursive=True)
    else:
        paths = [args.path]

    print(f"Parsing {len(paths)} bag files...")

    topic = ros_utils.vicon_topic_name(VICON_OBJECT_NAME)
    a = np.array([0, 0, 1])  # axis of interest
    warning = 0  # number of bags to warn about

    for path in paths:
        print(Path(path).parent.name)
        bag = rosbag.Bag(path)
        msgs = [msg for _, msg, _ in bag.read_messages(topic)]
        ts, poses = ros_utils.parse_transform_stamped_msgs(msgs, normalize_time=True)

        min_val = np.infty
        max_val = -np.infty

        C0 = q2r(poses[0][3:], order="xyzs")  # initial orientation
        a0 = C0 @ a  # initial axis direction

        for pose in poses:
            q = pose[3:]
            C = q2r(q, order="xyzs")
            min_val = min(min_val, a0 @ C @ a)
            max_val = max(max_val, a0 @ C @ a)
        print(f"min = {min_val}, max = {max_val}")

        # NOTE tilt angle measurement is not really valid, since the
        # orientation no longer matches the calibrated one
        # print(f"max tilt angle = {np.rad2deg(np.arccos(min_val))} deg")
        if min_val < 0:
            print("WARNING: axis is down!")
            print(path)
            warning += 1
        print("-")

    if warning > 0:
        print(f"{warning} bags have an orientation flip.")
    else:
        print("All bag files appear good.")


main()
