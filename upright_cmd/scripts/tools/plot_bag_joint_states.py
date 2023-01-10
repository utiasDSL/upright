#!/usr/bin/env python3
"""Plot robot true and estimated joint state from a bag file."""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt

from mobile_manipulation_central import ros_utils
from upright_ros_interface.parsing import parse_mpc_observation_msgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file to plot.")
    args = parser.parse_args()

    bag = rosbag.Bag(args.bagfile)

    ur10_msgs = [msg for _, msg, _ in bag.read_messages("/ur10/joint_states")]
    ridgeback_msgs = [msg for _, msg, _ in bag.read_messages("/ridgeback/joint_states")]
    mpc_obs_msgs = [
        msg for _, msg, _ in bag.read_messages("/mobile_manipulator_mpc_observation")
    ]

    tas, qas, vas = ros_utils.parse_ur10_joint_state_msgs(
        ur10_msgs, normalize_time=False
    )
    tbs, qbs, vbs = ros_utils.parse_ridgeback_joint_state_msgs(
        ridgeback_msgs, normalize_time=False
    )
    tms, xms, ums = parse_mpc_observation_msgs(mpc_obs_msgs, normalize_time=False)

    # use arm messages for timing
    ts = tas

    # align base messages with the arm messages
    qbs_aligned = ros_utils.interpolate_list(ts, tbs, qbs)
    vbs_aligned = ros_utils.interpolate_list(ts, tbs, vbs)

    qs_real = np.hstack((qbs_aligned, qas))
    vs_real = np.hstack((vbs_aligned, vas))

    # align the estimates and input
    n = 9
    xms_aligned = np.array(ros_utils.interpolate_list(ts, tms, xms))
    ums_aligned = np.array(ros_utils.interpolate_list(ts, tms, ums))
    qs_obs = xms_aligned[:, :n]
    vs_obs = xms_aligned[:, n:2*n]
    as_obs = xms_aligned[:, 2*n:3*n]
    us_obs = ums_aligned[:, :n]

    ts -= ts[0]

    plt.figure()
    for i in range(n):
        plt.plot(ts, qs_real[:, i], label=f"q_{i+1}")
    plt.title("Real Joint Positions")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, qs_obs[:, i], label=f"q_{i+1}")
    plt.title("Estimated Joint Positions")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, qs_real[:, i] - qs_obs[:, i], label=f"q_{i+1}")
    plt.title("Joint Position Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, vs_real[:, i], label=f"v_{i+1}")
    plt.title("Real Joint Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, vs_obs[:, i], label=f"v_{i+1}")
    plt.title("Estimated Joint Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, vs_real[:, i] - vs_obs[:, i], label=f"v_{i+1}")
    plt.title("Joint Velocity Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, as_obs[:, i], label=f"a_{i+1}")
    plt.title("Estimated Joint Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint acceleration")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, us_obs[:, i], label=f"j_{i+1}")
    plt.title("Joint (Jerk) Input")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint jerk")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
