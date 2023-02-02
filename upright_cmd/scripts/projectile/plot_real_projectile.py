#!/usr/bin/env python3
"""Plot position of Vicon objects from a ROS bag."""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt
from mobile_manipulation_central import ros_utils

import IPython


VICON_PROJECTILE_NAME = "ThingVolleyBall"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file to plot.")
    args = parser.parse_args()

    bag = rosbag.Bag(args.bagfile)

    est_msgs = [msg for _, msg, _ in bag.read_messages("/projectile/joint_states")]
    est_positions = np.array([msg.position for msg in est_msgs])
    est_velocities = np.array([msg.velocity for msg in est_msgs])

    vicon_topic = ros_utils.vicon_topic_name(VICON_PROJECTILE_NAME)
    vicon_msgs = [msg for _, msg, _ in bag.read_messages(vicon_topic)]
    _, vicon_poses = ros_utils.parse_transform_stamped_msgs(
        vicon_msgs, normalize_time=False
    )
    vicon_positions = vicon_poses[:, :3]

    t0 = ros_utils.msg_time(vicon_msgs[0])
    vicon_times = ros_utils.parse_time(vicon_msgs, t0=t0)
    est_times = ros_utils.parse_time(est_msgs, t0=t0)

    # find time at which Kalman filtering starts
    nv = vicon_positions.shape[0]
    for i in range(nv):
        if vicon_positions[i, 2] >= 0.8:
            active_time = vicon_times[i]
            break

    # compare to measured (numerically-differentiated) velocities
    num_diff_velocities = np.zeros((nv - 1, 3))
    Δs = np.zeros(nv - 1)
    for i in range(nv - 1):
        dt = vicon_times[i + 1] - vicon_times[i]
        dp = vicon_positions[i + 1, :] - vicon_positions[i, :]
        Δs[i] = np.linalg.norm(dp)
        num_diff_velocities[i, :] = dp / dt

    # x, y, z position vs. time
    plt.figure()
    plt.plot(vicon_times, vicon_positions[:, 0], label="x", color="r")
    plt.plot(vicon_times, vicon_positions[:, 1], label="y", color="g")
    plt.plot(vicon_times, vicon_positions[:, 2], label="z", color="b")
    plt.plot(
        est_times, est_positions[:, 0], label="$\hat{x}$", linestyle="--", color="r"
    )
    plt.plot(
        est_times, est_positions[:, 1], label="$\hat{y}$", linestyle="--", color="g"
    )
    plt.plot(
        est_times, est_positions[:, 2], label="$\hat{z}$", linestyle="--", color="b"
    )
    plt.axvline(active_time, color="k")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title(f"Projectile position vs. time")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(vicon_times[1:], num_diff_velocities[:, 0], label="$v_x$", color="r")
    plt.plot(vicon_times[1:], num_diff_velocities[:, 1], label="$v_y$", color="g")
    plt.plot(vicon_times[1:], num_diff_velocities[:, 2], label="$v_z$", color="b")
    plt.plot(
        est_times, est_velocities[:, 0], label="$\hat{v}_x$", linestyle="--", color="r"
    )
    plt.plot(
        est_times, est_velocities[:, 1], label="$\hat{v}_y$", linestyle="--", color="g"
    )
    plt.plot(
        est_times, est_velocities[:, 2], label="$\hat{v}_z$", linestyle="--", color="b"
    )
    plt.axvline(active_time, color="k")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title(f"Projectile velocity vs. time")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(vicon_times[1:], Δs)
    plt.axvline(active_time, color="k")
    plt.axhline(0.2, color=(0.5, 0.5, 0.5, 1))
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.title("Distance between position measurements")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
