#!/usr/bin/env python3
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray


if __name__ == "__main__":
    rospy.init_node("velocity_test")

    pub = rospy.Publisher("mobile_manipulator_cmd_vel", Float64MultiArray, queue_size=1)

    rospy.sleep(1.0)

    v = -0.5
    duration = 1.0
    cmd = [0, 0, 0, 0, 0, v]
    rate = rospy.Rate(125)

    now = rospy.Time.now().to_sec()
    time = now
    while time < now + duration and not rospy.is_shutdown():
        msg = Float64MultiArray()
        msg.data = cmd
        pub.publish(msg)

        rate.sleep()
        time = rospy.Time.now().to_sec()

    # stop the robot
    msg = Float64MultiArray()
    msg.data = [0, 0, 0, 0, 0, 0]
    pub.publish(msg)
