# Action client for sending joint trajectories.
import sys

import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from controller_manager_msgs.srv import SwitchControllerRequest, SwitchController
from controller_manager_msgs.srv import LoadControllerRequest, LoadController
import geometry_msgs.msg as geometry_msgs


# If your robot description is created with a tf_prefix, those would have to be adapted
UR10_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

JOINT_TRAJECTORY_CONTROLLERS = [
    "scaled_pos_joint_traj_controller",
    "scaled_vel_joint_traj_controller",
    "pos_joint_traj_controller",
    "vel_joint_traj_controller",
    "forward_joint_traj_controller",
]


# We'll have to make sure that none of these controllers are running, as they will
# be conflicting with the joint trajectory controllers
CONFLICTING_CONTROLLERS = ["joint_group_vel_controller", "twist_controller"]


class TrajectoryClient:
    """Small trajectory client to test a joint trajectory"""

    def __init__(self, joint_trajectory_controller=None):
        rospy.init_node("test_move")

        timeout = rospy.Duration(5)
        self.switch_srv = rospy.ServiceProxy(
            "controller_manager/switch_controller", SwitchController
        )
        self.load_srv = rospy.ServiceProxy(
            "controller_manager/load_controller", LoadController
        )
        try:
            self.switch_srv.wait_for_service(timeout.to_sec())
        except rospy.exceptions.ROSException as err:
            rospy.logerr(
                "Could not reach controller switch service. Msg: {}".format(err)
            )
            sys.exit(-1)

        if joint_trajectory_controller is not None:
            self.switch_controller(joint_trajectory_controller)

    def send_joint_trajectory(self, trajectory):
        """Creates a trajectory and sends it using the selected action server"""
        trajectory_client = actionlib.SimpleActionClient(
            "{}/follow_joint_trajectory".format(self.joint_trajectory_controller),
            FollowJointTrajectoryAction,
        )

        timeout = rospy.Duration(5)
        if not trajectory_client.wait_for_server(timeout):
            rospy.logerr("Could not reach controller action server.")
            sys.exit(-1)

        # Create and fill trajectory goal
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = trajectory

        rospy.loginfo("Executing trajectory...")

        trajectory_client.send_goal(goal, feedback_cb=None)  # TODO
        trajectory_client.wait_for_result()

        result = trajectory_client.get_result()
        rospy.loginfo(
            "Trajectory execution finished in state {}".format(result.error_code)
        )

    def switch_controller(self, target_controller):
        """Activates the desired controller and stops all others from the predefined list above"""
        other_controllers = JOINT_TRAJECTORY_CONTROLLERS + CONFLICTING_CONTROLLERS

        other_controllers.remove(target_controller)

        # load the desired controller
        srv = LoadControllerRequest()
        srv.name = target_controller
        self.load_srv(srv)

        # switch to the desired controller from whatever is currently active
        srv = SwitchControllerRequest()
        srv.stop_controllers = other_controllers
        srv.start_controllers = [target_controller]
        srv.strictness = SwitchControllerRequest.BEST_EFFORT
        self.switch_srv(srv)