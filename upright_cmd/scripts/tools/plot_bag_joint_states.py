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
    mpc_plan_msgs = [
        msg for _, msg, _ in bag.read_messages("/mobile_manipulator_mpc_plan")
    ]

    tas, qas, vas = ros_utils.parse_ur10_joint_state_msgs(
        ur10_msgs, normalize_time=False
    )
    tbs, qbs, vbs = ros_utils.parse_ridgeback_joint_state_msgs(
        ridgeback_msgs, normalize_time=False
    )

    # alternatively we can use finite differences, since the vbs are already
    # low-pass filtered otherwise
    # vbs = np.zeros_like(qbs)
    # for i in range(1, vbs.shape[0]):
    #     vbs[i, :] = (qbs[i, :] - qbs[i - 1, :]) / (tbs[i] - tbs[i - 1])

    tms, xms, ums = parse_mpc_observation_msgs(mpc_obs_msgs, normalize_time=False)
    tps, xps, ups = parse_mpc_observation_msgs(mpc_plan_msgs, normalize_time=False)

    # use arm messages for timing
    # TODO no need to include anything before the first policy message is
    # received
    ts = tas

    # only start just before the first observation is published
    start_idx = np.argmax(ts >= tms[0]) - 1
    assert start_idx >= 0

    ts = ts[start_idx:]
    qas = qas[start_idx:, :]
    vas = vas[start_idx:, :]

    # import IPython
    # IPython.embed()
    # return

    # align base messages with the arm messages
    qbs_aligned = ros_utils.interpolate_list(ts, tbs, qbs)
    vbs_aligned = ros_utils.interpolate_list(ts, tbs, vbs)

    qs_real = np.hstack((qbs_aligned, qas))
    vs_real = np.hstack((vbs_aligned, vas))

    # qs_real = qas
    # vs_real = vas

    # align the estimates and input
    n = 9
    xms_aligned = np.array(ros_utils.interpolate_list(ts, tms, xms))
    ums_aligned = np.array(ros_utils.interpolate_list(ts, tms, ums))
    qs_obs = xms_aligned[:, :n]
    vs_obs = xms_aligned[:, n:2*n]
    as_obs = xms_aligned[:, 2*n:3*n]
    us_obs = ums_aligned[:, :n]

    # align MPC optimal trajectory
    xps_aligned = np.array(ros_utils.interpolate_list(ts, tps, xps))
    qs_plan = xps_aligned[:, :n]
    vs_plan = xps_aligned[:, n:2*n]
    as_plan = xps_aligned[:, 2*n:3*n]

    ts -= ts[0]

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    plt.figure()
    for i in range(n):
        plt.plot(ts, qs_real[:, i], label=f"$q_{i+1}$")
    plt.title("Real Joint Positions")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, qs_obs[:, i], label=f"$\hat{{q}}_{i+1}$")
    for i in range(n):
        plt.plot(ts, qs_plan[:, i], label=f"$q^{{plan}}_{i+1}$", linestyle="--", color=colors[i])
    plt.title("Estimated Joint Positions")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position")
    plt.legend(ncols=2)
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, qs_real[:, i] - qs_obs[:, i], label=f"$q_{i+1}$")
    plt.title("Joint Position Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, vs_real[:, i], label=f"$v_{i+1}$")
    plt.title("Real Joint Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, vs_obs[:, i], label=f"$\hat{{v}}_{i+1}$")
    for i in range(n):
        plt.plot(ts, vs_plan[:, i], label=f"$v^{{plan}}_{i+1}$", linestyle="--", color=colors[i])
    plt.title("Estimated and Planned Joint Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend(ncol=2)
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, vs_real[:, i] - vs_obs[:, i], label=f"$v_{i+1}$")
    plt.title("Joint Velocity Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, as_obs[:, i], label=f"$\hat{{a}}_{i+1}$")
    for i in range(n):
        plt.plot(ts, as_plan[:, i], label=f"$a^{{plan}}_{i+1}$", linestyle="--", color=colors[i])
    plt.title("Estimated Joint Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint acceleration")
    plt.legend(ncol=2)
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, us_obs[:, i], label=f"$j_{i+1}$")
    plt.title("Joint (Jerk) Input")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint jerk")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
