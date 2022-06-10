#!/bin/sh
# Compile xacro versions of URDFs to regular versions for consumption by
# libraries. This is mainly used to resolve ROS package paths without
# hardcoding.
mkdir -p urdf
# xacro mm_ocs2.urdf.xacro -o mm_ocs2.urdf
xacro xacro/ocs2/mm_ocs2_tray.urdf.xacro -o urdf/mm_ocs2_tray.urdf
xacro xacro/ocs2/mm_ocs2_fingers.urdf.xacro -o urdf/mm_ocs2_fingers.urdf

xacro xacro/pyb/mm_pyb_tray.urdf.xacro -o urdf/mm_pyb_tray.urdf
xacro xacro/pyb/mm_pyb_fingers.urdf.xacro -o urdf/mm_pyb_fingers.urdf
xacro xacro/obstacles.urdf.xacro -o urdf/obstacles.urdf
# xacro mm_pyb_static_obs.urdf.xacro -o mm_pyb_static_obs.urdf
# xacro mm_static_tray_pyb.urdf.xacro -o mm_static_tray_pyb.urdf