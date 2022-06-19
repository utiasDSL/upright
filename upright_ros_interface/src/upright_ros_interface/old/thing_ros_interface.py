import numpy as np
import rospy

from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

import upright_core as core


# TODO not immediately useful
class ThingRobot:
    def __init__(self, config):
        # dimensions
        self.nq = config["dims"]["q"]  # num positions
        self.nv = config["dims"]["v"]  # num velocities
        self.nx = config["dims"]["x"]  # num states
        self.nu = config["dims"]["u"]  # num inputs

        # home position
        self.home = core.parsing.parse_array(config["x0"])[:self.nq]

        # commands
        self.cmd_vel = np.zeros(self.nv)
        self.cmd_acc = np.zeros_like(self.cmd_vel)
        self.cmd_jerk = np.zeros_like(self.cmd_vel)

        # feedback
        self.q_base = np.zeros(3)
        self.v_base = np.zeros(3)
        self.q_arm = np.zeros(6)
        self.v_arm = np.zeros(6)

        # set to true once feedback has been received
        self.base_msg_received = False
        self.arm_msg_received = False

        # publishers for velocity commands
        self.base_cmd_pub = rospy.Publisher("/ridgeback_velocity_controller/cmd_vel", Twist, queue_size=1)
        self.arm_cmd_pub = rospy.Publisher("/ur_driver/joint_speed", JointTrajectory, queue_size=1)

        # subscribers for joint states
        self.base_joint_sub = rospy.Subscriber("/rb_joint_states", JointState, self._base_joint_cb)
        self.arm_joint_sub = rospy.Subscriber("/ur10_joint_states", JointState, self._arm_joint_cb)

    def ready(self):
        return self.base_msg_received and self.arm_msg_received

    def _base_rotation_matrix(self):
        """Get rotation matrix for the base.

        This is just the rotation about the z-axis by the yaw angle.
        """
        yaw = self.q_base[2]
        C_wb = core.math.rotz(yaw)
        return C_wb

    def _pub_base_vel_cmd(self, cmd_vel_base):
        """Publish base velocity commands."""
        msg = Twist()
        msg.linear.x = cmd_vel_base[0]
        msg.linear.y = cmd_vel_base[1]
        msg.angular.z = cmd_vel_base[2]
        self.base_cmd_pub.publish(msg)

    def _pub_arm_vel_cmd(self, cmd_vel_arm):
        """Publish arm velocity commands."""
        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now()

        point = JointTrajectoryPoint()
        point.velocities = list(cmd_vel_arm)
        msg.points.append(point)

        self.arm_cmd_pub.publish(msg)

    def command_velocity(self, cmd_vel):
        """Command the velocity of the robot's joints."""
        self._pub_base_vel_cmd(cmd_vel[:3])
        self._pub_arm_vel_cmd(cmd_vel[3:])

    def command_acceleration(self, cmd_acc):
        """Command acceleration of the robot's joints."""
        self.cmd_acc = cmd_acc

    def command_jerk(self, cmd_jerk):
        """Command jerk of the robot's joints."""
        self.cmd_jerk = cmd_jerk

    def step(self, secs):
        """Step the robot kinematics forward by `secs` seconds."""
        # input (acceleration) and velocity are both in the body frame
        self.cmd_acc += secs * self.cmd_jerk
        self.cmd_vel += secs * self.cmd_acc
        self.command_velocity(self.cmd_vel)

    def _base_joint_cb(self, msg):
        """Callback for base joint feedback."""
        self.q_base = np.array(msg.position)
        self.v_base = np.array(msg.velocity)
        self.base_msg_received = True

    def _arm_joint_cb(self, msg):
        """Callback for arm joint feedback."""
        self.q_arm = np.array(msg.position)
        self.v_arm = np.array(msg.velocity)
        self.arm_msg_received = True

    def joint_states(self):
        """Get the current state of the joints.

        Return a tuple (q, v), where q is the n-dim array of positions and v is
        the n-dim array of velocities.
        """
        # Velocity representation is in the body frame. Vicon gives feedback in
        # the world frame, so we have to rotate the linear base velocity.
        C_wb = self._base_rotation_matrix()
        v_base_body = np.append((C_wb.T @ self.v_base[:2], self.v_base[2]))

        q = np.concatenate((self.q_base, self.q_arm))
        v = np.concatenate((v_base_body, self.v_arm))
        return q, v
