#!/bin/sh
# Compile xacro versions of URDFs to regular versions for consumption by
# libraries. This is mainly used to resolve ROS package paths without
# hardcoding.
xacro mm_ocs2.urdf.xacro -o mm_ocs2.urdf
xacro mm_pyb.urdf.xacro -o mm_pyb.urdf
