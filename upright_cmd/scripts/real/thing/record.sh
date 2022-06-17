#!/bin/sh
BAG_DIR=/media/adam/Data/PhD/Data/upright/real-thing/bags
rosbag record -o "$BAG_DIR/$1" --regex "/scaled_vel_joint_traj_controller/(.*)" --regex "/mobile_manipulator_(.*)"
