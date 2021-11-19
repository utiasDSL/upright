import numpy as np
import rospy

from tray_balance_msgs.msg import TrayBalanceControllerInfo
from sensor_msgs.msg import JointState


class ROSInterface:
    def __init__(self, name):
        rospy.init_node(name)

        self.received_msg = False

        self.control_info_sub = rospy.Subscriber(
            "/mm/control_info", TrayBalanceControllerInfo, self.control_info_cb
        )
        # self.reference_pub =
        self.state_pub = rospy.Publisher("/mm/current_state", JointState, queue_size=1)

        rospy.sleep(1.0)

    def initialized(self):
        return self.received_msg

    def control_info_cb(self, msg):
        self.q = np.array(msg.joints.position)
        self.v = np.array(msg.joints.velocity)
        self.command = np.array(msg.command)
        self.time = msg.header.stamp.to_sec()
        print(f"msg time = {self.time}")
        self.received_msg = True

    def publish_state(self, t, q, v):
        msg = JointState()
        msg.header.stamp = rospy.Time(t)
        msg.position = list(q)
        msg.velocity = list(v)
        self.state_pub.publish(msg)


def main():
    interface = ROSInterface("tray_balance_sim")

    rate = rospy.Rate(100)

    while not rospy.is_shutdown():
        if interface.command is not None:
            interface.publish_state(interface.q, interface.v)
        rate.sleep()


if __name__ == "__main__":
    main()
