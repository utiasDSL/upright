#!/bin/sh
BAG_DIR=/media/adam/Data/PhD/Data/upright/real-thing/bags
rosbag record -o "$BAG_DIR/$1" /ur10_joint_states /ur10_cmd_vel --regex "/scaled_vel_joint_traj_controller/(.*)" --regex "/mobile_manipulator_(.*)"
