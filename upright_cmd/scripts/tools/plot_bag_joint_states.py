"""Plot UR10 and Ridgeback joint position and velocity from a ROS bag."""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt
from mobile_manipulation_central import ros_utils

import IPython


def parse_mpc_observation_msgs(msgs, normalize_time=True):
    ts = []
    xs = []
    us = []

    for msg in msgs:
        ts.append(msg.time)
        xs.append(msg.state.value)
        us.append(msg.input.value)

    ts = np.array(ts)
    if normalize_time:
        ts -= ts[0]

    return ts, np.array(xs), np.array(us)


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
    mpc_est_msgs = [
        msg for _, msg, _ in bag.read_messages("/mobile_manipulator_state_estimate")
    ]

    tas, qas, vas = ros_utils.parse_ur10_joint_state_msgs(
        ur10_msgs, normalize_time=False
    )
    tbs, qbs, vbs = ros_utils.parse_ridgeback_joint_state_msgs(
        ridgeback_msgs, normalize_time=False
    )
    tms, xms, ums = parse_mpc_observation_msgs(mpc_obs_msgs, normalize_time=False)
    tes, xes, _ = parse_mpc_observation_msgs(mpc_est_msgs, normalize_time=False)

    ts = tas
    qbs_aligned = ros_utils.interpolate_list(ts, tbs, qbs)
    vbs_aligned = ros_utils.interpolate_list(ts, tbs, vbs)

    qs_real = np.hstack((qbs_aligned, qas))
    vs_real = np.hstack((vbs_aligned, vas))

    xms_aligned = np.array(ros_utils.interpolate_list(ts, tms, xms))
    qs_obs = xms_aligned[:, :9]
    vs_obs = xms_aligned[:, 9:18]

    xes_aligned = np.array(ros_utils.interpolate_list(ts, tes, xes))
    qs_est = xes_aligned[:, :9]
    vs_est = xes_aligned[:, 9:18]

    ts -= ts[0]

    plt.figure()
    for i in range(9):
        plt.plot(ts, qs_real[:, i], label=f"q_{i+1}")
    plt.title("Real Joint Positions")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(9):
        plt.plot(ts, qs_obs[:, i], label=f"q_{i+1}")
    plt.title("Observed Joint Positions")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(9):
        plt.plot(ts, qs_est[:, i], label=f"q_{i+1}")
    plt.title("Estimated Joint Position")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(9):
        plt.plot(ts, qs_real[:, i] - qs_obs[:, i], label=f"q_{i+1}")
    plt.title("Joint Position Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(9):
        plt.plot(ts, qs_real[:, i] - qs_est[:, i], label=f"q_{i+1}")
    plt.title("Joint Position Estimate Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(9):
        plt.plot(ts, vs_real[:, i], label=f"v_{i+1}")
    plt.title("Real Joint Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(9):
        plt.plot(ts, vs_obs[:, i], label=f"v_{i+1}")
    plt.title("Observed Joint Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(9):
        plt.plot(ts, vs_est[:, i], label=f"v_{i+1}")
    plt.title("Estimated Joint Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(9):
        plt.plot(ts, vs_real[:, i] - vs_obs[:, i], label=f"v_{i+1}")
    plt.title("Joint Velocity Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(9):
        plt.plot(ts, vs_real[:, i] - vs_est[:, i], label=f"v_{i+1}")
    plt.title("Joint Velocity Estimate Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
