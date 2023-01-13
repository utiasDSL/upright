#!/usr/bin/env python3
"""Plot distance of projectile to:
1. initial position of the EE (to verify that it would have hit the EE)
2. actual position of the EE (to verify that the EE actually got out of the way)

Also computes the maximum velocity and acceleration.
"""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt

from mobile_manipulation_central import ros_utils
import upright_core as core

import IPython


# TODO we would prefer to measure offset to the actual collision sphere
VICON_PROJECTILE_NAME = "ThingProjectile"
VICON_EE_NAME = "ThingWoodTray"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file to plot.")
    args = parser.parse_args()

    bag = rosbag.Bag(args.bagfile)

    ee_topic = ros_utils.vicon_topic_name(VICON_EE_NAME)
    ee_msgs = [msg for _, msg, _ in bag.read_messages(ee_topic)]
    ee_ts, ee_poses = ros_utils.parse_transform_stamped_msgs(
        ee_msgs, normalize_time=False
    )

    projectile_topic = ros_utils.vicon_topic_name(VICON_PROJECTILE_NAME)
    projectile_msgs = [msg for _, msg, _ in bag.read_messages(projectile_topic)]
    projectile_ts, projectile_poses = ros_utils.parse_transform_stamped_msgs(
        projectile_msgs, normalize_time=False
    )

    proj_est_msgs = [
        msg for _, msg, _ in bag.read_messages("/ThingProjectile/joint_states")
    ]
    proj_est_ts = ros_utils.parse_time(proj_est_msgs, normalize_time=False)
    proj_pos_est = np.array([msg.position for msg in proj_est_msgs])

    ts = ee_ts
    n = len(ts)
    projectile_positions = np.array(
        ros_utils.interpolate_list(ts, projectile_ts, projectile_poses[:, :3])
    )
    ee_positions = ee_poses[:, :3]
    proj_pos_est = np.array(ros_utils.interpolate_list(ts, proj_est_ts, proj_pos_est))
    ts -= ts[0]

    # TODO may be easier to compute the EE position using the joint states
    # (possibly the estimate)---then we can directly get the collision sphere
    # position using the model

    # only start when robot sees the ball when z >= 0
    start_idx = np.argmax(proj_pos_est[:, 2] >= 1.0)

    r0 = ee_positions[0, :]
    distance_to_origin = np.linalg.norm(projectile_positions - r0, axis=1)
    distance_to_ee = np.linalg.norm(projectile_positions - ee_positions, axis=1)
    distance_to_origin_est = np.linalg.norm(proj_pos_est - r0, axis=1)
    distance_to_ee_est = np.linalg.norm(proj_pos_est - ee_positions, axis=1)

    plt.figure()
    plt.plot(ts[start_idx:], distance_to_origin[start_idx:], label="Dist to Origin")
    plt.plot(ts[start_idx:], distance_to_ee[start_idx:], label="Dist to EE")
    plt.plot(
        ts[start_idx:],
        distance_to_origin_est[start_idx:],
        label="Est. Dist to Origin",
        linestyle="--",
    )
    plt.plot(
        ts[start_idx:],
        distance_to_ee_est[start_idx:],
        label="Est. Dist to EE",
        linestyle="--",
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.title("Projectile Distance")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
