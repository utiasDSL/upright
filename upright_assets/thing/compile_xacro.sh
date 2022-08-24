#!/bin/sh
# Compile xacro versions of URDFs to regular versions for consumption by
# libraries. This is mainly used to resolve ROS package paths without
# hardcoding.
mkdir -p urdf

xacro xacro/ocs2/thing_ocs2_tray.urdf.xacro -o urdf/thing_ocs2_tray.urdf
xacro xacro/ocs2/thing_ocs2_fingers.urdf.xacro -o urdf/thing_ocs2_fingers.urdf

xacro xacro/pyb/thing_pyb_tray.urdf.xacro -o urdf/thing_pyb_tray.urdf
xacro xacro/pyb/thing_pyb_fingers.urdf.xacro -o urdf/thing_pyb_fingers.urdf
xacro xacro/obstacles.urdf.xacro -o urdf/obstacles.urdf

xacro xacro/ros/thing_ros_tray.urdf.xacro -o urdf/thing_ros_tray.urdf

# second run to resolve mesh paths
xacro urdf/thing_pyb_tray.urdf -o urdf/thing_pyb_tray.urdf
xacro urdf/thing_ocs2_tray.urdf -o urdf/thing_ocs2_tray.urdf
xacro urdf/thing_pyb_fingers.urdf -o urdf/thing_pyb_fingers.urdf
xacro urdf/thing_ocs2_fingers.urdf -o urdf/thing_ocs2_fingers.urdf
xacro urdf/thing_ros_tray.urdf -o urdf/thing_ros_tray.urdf

# xacro thing_pyb_static_obs.urdf.xacro -o thing_pyb_static_obs.urdf
# xacro thing_static_tray_pyb.urdf.xacro -o thing_static_tray_pyb.urdf
