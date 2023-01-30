#!/bin/sh
rosbag play $1 --clock --topics /vicon/ThingBase/ThingBase /vicon/ThingVolleyBall/ThingVolleyBall
