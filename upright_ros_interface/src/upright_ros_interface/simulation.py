from threading import Lock

import rospy
import numpy as np

from std_msgs.msg import Float64MultiArray
from rosgraph_msgs.msg import Clock
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState

from upright_ros_interface.trajectory_client import UR10_JOINT_NAMES


class ROSSimulationInterface:
    """Interface between the MPC node and the simulation."""
    def __init__(self, topic_prefix):
        rospy.init_node("pyb_interface")

        self.cmd_vel = None

        self.clock_pub = rospy.Publisher("/clock", Clock, queue_size=1)
        self.feedback_pub = rospy.Publisher("/ur10_joint_states", JointState, queue_size=1)
        self.cmd_sub = rospy.Subscriber("/ur10_cmd_vel", Float64MultiArray, self._cmd_cb)

        # wait for everything to be setup
        rospy.sleep(1.0)

    def ready(self):
        return self.cmd_vel is not None

    def publish_feedback(self, t, q, v):
        msg = JointState()
        msg.header.stamp = rospy.Time(t)
        msg.name = UR10_JOINT_NAMES
        msg.position = q
        msg.velocity = v
        self.feedback_pub.publish(msg)

    def _cmd_cb(self, msg):
        self.cmd_vel = np.array(msg.data)

    def publish_time(self, t):
        msg = Clock()
        msg.clock = rospy.Time(t)
        self.clock_pub.publish(msg)
