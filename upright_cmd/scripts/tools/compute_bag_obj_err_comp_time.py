#!/usr/bin/env python3
"""Plot robot true and estimated joint state from a bag file."""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt

import upright_core as core
from mobile_manipulation_central import ros_utils
from upright_ros_interface.parsing import parse_object_error, parse_mpc_solve_times

import IPython


TRAY_VICON_NAME = "ThingWoodTray"


def get_bag_topics(bag):
    return list(bag.get_type_and_topic_info()[1].keys())


def vicon_object_topics(bag):
    topics = get_bag_topics(bag)

    def func(topic):
        if not topic.startswith("/vicon"):
            return False
        if (
            topic.endswith("markers")
            or "ThingBase" in topic
            or TRAY_VICON_NAME in topic
        ):
            return False
        return True

    topics = list(filter(func, topics))
    if len(topics) == 0:
        print("No object topic found!")
    elif len(topics) > 1:
        raise ValueError("Multiple object topics found!")
    return topics[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file to plot.")
    args = parser.parse_args()

    bag = rosbag.Bag(args.bagfile)
    solve_times, ts1 = parse_mpc_solve_times(bag, max_time=5, return_times=True)
    object_vicon_name = vicon_object_topics(bag).split("/")[-1]
    print(f"Object is {object_vicon_name}")
    errors, ts2 = parse_object_error(
        bag, TRAY_VICON_NAME, object_vicon_name, return_times=True
    )

    print("SOLVE TIME")
    print(f"max  = {np.max(solve_times):.2f} ms")
    print(f"min  = {np.min(solve_times):.2f} ms")
    print(f"mean = {np.mean(solve_times):.2f} ms")

    plt.figure()
    plt.plot(ts1, solve_times)
    plt.xlabel("Time [s]")
    plt.ylabel("Solve time [ms]")
    plt.grid()

    plt.figure()
    plt.plot(ts2, 1000 * errors)  # convert to mm
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Distance error [mm]")

    plt.show()


if __name__ == "__main__":
    main()
