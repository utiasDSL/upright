#!/usr/bin/env python3
import numpy as np
import rospy
from geometry_msgs.msg import Twist


if __name__ == "__main__":
    rospy.init_node("velocity_test")

    pub = rospy.Publisher("/ridgeback_velocity_controller/cmd_vel", Twist, queue_size=1)

    rospy.sleep(1.0)

    v = 0.5
    duration = 1.0
    msg = Twist()
    msg.linear.x = v
    rate = rospy.Rate(125)

    now = rospy.Time.now().to_sec()
    time = now
    while time < now + duration and not rospy.is_shutdown():
        pub.publish(msg)
        rate.sleep()
        time = rospy.Time.now().to_sec()

    # stop the robot
    msg = Twist()
    pub.publish(msg)
