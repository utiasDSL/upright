"""Plot position of Vicon objects from a ROS bag."""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt
from mobile_manipulation_central import ros_utils


def parse_vicon_msgs(msgs, t0=None):
    ts = ros_utils.parse_time(msgs, t0=t0)
    poses = np.zeros((len(msgs), 7))
    for i, msg in enumerate(msgs):
        r = msg.transform.translation
        q = msg.transform.rotation
        poses[i, :] = [r.x, r.y, r.z, q.x, q.y, q.z, q.w]
    return ts, poses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file to plot.")
    args = parser.parse_args()

    bag = rosbag.Bag(args.bagfile)

    est_msgs = [msg for _, msg, _ in bag.read_messages("/Projectile/joint_states")]
    est_positions = np.array([msg.position for msg in est_msgs])
    est_velocities = np.array([msg.velocity for msg in est_msgs])

    vicon_msgs = [msg for _, msg, _ in bag.read_messages("/vicon/Projectile/Projectile")]
    _, vicon_poses = parse_vicon_msgs(vicon_msgs)
    vicon_positions = vicon_poses[:, :3]

    t0 = ros_utils.msg_time(vicon_msgs[0])
    vicon_times = ros_utils.parse_time(vicon_msgs, t0=t0)
    est_times = ros_utils.parse_time(est_msgs, t0=t0)

    # x, y, z position vs. time
    plt.figure()
    plt.plot(vicon_times, vicon_positions[:, 0], label="x", color="r")
    plt.plot(vicon_times, vicon_positions[:, 1], label="y", color="g")
    plt.plot(vicon_times, vicon_positions[:, 2], label="z", color="b")
    plt.plot(est_times, est_positions[:, 0], label="x_est", linestyle="--", color="r")
    plt.plot(est_times, est_positions[:, 1], label="y_est", linestyle="--", color="g")
    plt.plot(est_times, est_positions[:, 2], label="z_est", linestyle="--", color="b")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title(f"Projectile position vs. time")
    plt.legend()
    plt.grid()

    # plt.figure()
    # plt.plot(gt_times, gt_velocities[:, 0], label="x", color="r")
    # plt.plot(gt_times, gt_velocities[:, 1], label="y", color="g")
    # plt.plot(gt_times, gt_velocities[:, 2], label="z", color="b")
    # plt.plot(est_times, est_velocities[:, 0], label="x_est", linestyle="--", color="r")
    # plt.plot(est_times, est_velocities[:, 1], label="y_est", linestyle="--", color="g")
    # plt.plot(est_times, est_velocities[:, 2], label="z_est", linestyle="--", color="b")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Velocity (m/s)")
    # plt.title(f"Projectile velocity vs. time")
    # plt.legend()
    # plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
