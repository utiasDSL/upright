#!/usr/bin/env python3
"""Compute center point of tray from Vicon data.

A marker is placed at the center of the tray; we just want its position
relative to the Vicon tray object origin.
"""
import numpy as np
import rosbag
import matplotlib.pyplot as plt
from mobile_manipulation_central import ros_utils

import IPython


BAGFILE = "/media/adam/Data/PhD/Data/upright/experiments/calibration/tray_2022-11-22-17-43-41.bag"


def remove_named_markers(msgs):
    for msg in msgs:
        msg.markers = [marker for marker in msg.markers if marker.marker_name == ""]
    return msgs


def marker_xyz(marker_msg):
    return [marker_msg.translation.x, marker_msg.translation.y, marker_msg.translation.z]


def main():
    bag = rosbag.Bag(BAGFILE)

    tray_msgs = [msg for _, msg, _ in bag.read_messages("/vicon/ThingTray2/ThingTray2")]
    _, tray_poses = ros_utils.parse_transform_stamped_msgs(tray_msgs)
    tray_position = np.mean(tray_poses[:, :3], axis=0)

    marker_msgs = [msg for _, msg, _ in bag.read_messages("/vicon/markers")]
    remove_named_markers(marker_msgs)

    center_marker_positions = []
    for msg in marker_msgs:
        positions = np.array([marker_xyz(marker) for marker in msg.markers]) / 1000
        dists = np.linalg.norm(positions - tray_position, axis=1)
        idx = np.argmin(dists)
        center_marker_positions.append(positions[idx, :])
    center_marker_position = np.mean(center_marker_positions, axis=0)

    center_marker_relative_position = center_marker_position - tray_position
    print(center_marker_relative_position)


if __name__ == "__main__":
    main()
