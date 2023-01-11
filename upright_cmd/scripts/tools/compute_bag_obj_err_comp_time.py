#!/usr/bin/env python3
"""Plot robot true and estimated joint state from a bag file."""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt

import upright_core as core
from mobile_manipulation_central import ros_utils

import IPython


TRAY_VICON_NAME = "ThingWoodTray"
OBJECT_VICON_NAME = "ThingPinkBottle"


def parse_mpc_solve_times(bag, plot=False):
    policy_msgs = [
        msg for _, msg, _ in bag.read_messages("/mobile_manipulator_mpc_policy")
    ]
    policy_times = np.array(
        [t.to_sec() for _, _, t in bag.read_messages("/mobile_manipulator_mpc_policy")]
    )

    policy_solve_times = np.array([msg.solveTime for msg in policy_msgs])

    print("SOLVE TIME")
    print(f"max  = {np.max(policy_solve_times):.2f} ms")
    print(f"min  = {np.min(policy_solve_times):.2f} ms")
    print(f"mean = {np.mean(policy_solve_times):.2f} ms")

    if plot:
        plt.figure()
        plt.plot(policy_times, policy_solve_times)
        plt.grid()


def get_bag_topics(bag):
    return list(bag.get_type_and_topic_info()[1].keys())


def vicon_object_topics(bag):
    topics = get_bag_topics(bag)

    def func(topic):
        if not topic.startswith("/vicon"):
            return False
        if (
            topic.endswith("markers")
            or topic.endswith("ThingBase")
            or topic.endswith("ThingWoodTray")
        ):
            return False
        return True

    topics = list(filter(func, topics))
    if len(topics) == 0:
        print("No object topic found!")
    elif len(topics) > 1:
        print("Multiple object topics found!")
    return topics[0]


def parse_object_error(bag):
    tray_topic = ros_utils.vicon_topic_name(TRAY_VICON_NAME)
    tray_msgs = [msg for _, msg, _ in bag.read_messages(tray_topic)]

    obj_topic = ros_utils.vicon_topic_name(OBJECT_VICON_NAME)
    obj_msgs = [msg for _, msg, _ in bag.read_messages(obj_topic)]

    # parse and align messages
    ts, tray_poses = ros_utils.parse_transform_stamped_msgs(
        tray_msgs, normalize_time=False
    )
    ts_obj, obj_poses = ros_utils.parse_transform_stamped_msgs(
        obj_msgs, normalize_time=False
    )
    r_ow_ws = np.array(ros_utils.interpolate_list(ts, ts_obj, obj_poses[:, :3]))
    ts -= ts[0]

    n = len(ts)
    r_ot_ts = []
    for i in range(n):
        r_tw_w, Q_wt = tray_poses[i, :3], tray_poses[i, 3:]
        r_ow_w = r_ow_ws[i, :]

        # tray rotation w.r.t. world
        C_wt = core.math.quat_to_rot(Q_wt)
        C_tw = C_wt.T

        # compute offset of object in tray's frame
        r_ot_w = r_ow_w - r_tw_w
        r_ot_t = C_tw @ r_ot_w
        r_ot_ts.append(r_ot_t)

    r_ot_ts = np.array(r_ot_ts)

    # compute distance w.r.t. the initial position
    # TODO this does not really seem right: error gets large and then goes down
    # over time?
    r_ot_t_err = r_ot_ts - r_ot_ts[0, :]
    distances = np.linalg.norm(r_ot_t_err, axis=1)

    plt.figure()
    plt.plot(ts, 1000 * distances)
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Distance error [mm]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file to plot.")
    args = parser.parse_args()

    bag = rosbag.Bag(args.bagfile)
    parse_mpc_solve_times(bag, plot=True)
    parse_object_error(bag)
    plt.show()



if __name__ == "__main__":
    main()
