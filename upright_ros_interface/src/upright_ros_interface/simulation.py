from threading import Lock

import rospy
import numpy as np
import upright_control as ctrl

from trajectory_msgs.msg import JointTrajectory
from ocs2_msgs.msg import (
    mpc_flattened_controller,
    mpc_observation,
    mpc_state,
    mpc_input,
)
from ocs2_msgs.srv import reset as mpc_reset
from ocs2_msgs.srv import resetRequest as mpc_reset_request


class ROSSimulationInterface:
    """Interface between the MPC node and the simulation."""
    def __init__(self, topic_prefix, interpolator):
        rospy.init_node("pyb_interface")

        # optimal trajectory
        self.trajectory = None
        self.trajectory_lock = Lock()
        self.interpolator = interpolator

        # optimal policy
        self.policy = None
        self.policy_lock = Lock()

        self.policy_sub = rospy.Subscriber(
            topic_prefix + "_mpc_policy", mpc_flattened_controller, self._policy_cb
        )
        self.trajectory_sub = rospy.Subscriber(
            topic_prefix + "_joint_trajectory", JointTrajectory, self._trajectory_cb
        )
        self.observation_pub = rospy.Publisher(
            topic_prefix + "_mpc_observation", mpc_observation, queue_size=1
        )

        # wait for everything to be setup
        # TODO try latching
        rospy.sleep(1.0)

    def reset_mpc(self, ref):
        # call service to reset, repeating until done
        srv_name = "mobile_manipulator_mpc_reset"

        print("Waiting for MPC reset service...")

        rospy.wait_for_service(srv_name)
        mpc_reset_service = rospy.ServiceProxy(srv_name, mpc_reset)

        req = mpc_reset_request()
        req.reset = True
        req.targetTrajectories.timeTrajectory = ref.ts
        for x in ref.xs:
            msg = mpc_state()
            msg.value = x
            req.targetTrajectories.stateTrajectory.append(msg)
        for u in ref.us:
            msg = mpc_input()
            msg.value = u
            req.targetTrajectories.inputTrajectory.append(msg)

        try:
            resp = mpc_reset_service(req)
        except rospy.ServiceException as e:
            print("MPC reset failed.")
            print(e)
            return 1

        print("MPC reset done.")

    def publish_observation(self, t, x, u):
        msg = mpc_observation()
        msg.time = t
        msg.state.value = x
        msg.input.value = u
        self.observation_pub.publish(msg)

    def _trajectory_cb(self, msg):
        t_opt = []
        x_opt = []
        u_opt = []
        for i in range(len(msg.points)):
            t_opt.append(msg.points[i].time_from_start.to_sec())

            q = msg.points[i].positions
            v = msg.points[i].velocities
            a = msg.points[i].accelerations
            x_opt.append(q + v + a)

            u_opt.append(msg.points[i].effort)

        with self.trajectory_lock:
            self.trajectory = ctrl.trajectory.StateInputTrajectory(
                np.array(t_opt), np.array(x_opt), np.array(u_opt)
            )
            self.interpolator.update(self.trajectory)

    def _policy_cb(self, msg):
        # info to reconstruct the linear controller
        time_array = ctrl.bindings.scalar_array()
        state_dims = []
        input_dims = []
        data = []
        for i in range(len(msg.timeTrajectory)):
            time_array.push_back(msg.timeTrajectory[i])
            state_dims.append(len(msg.stateTrajectory[i].value))
            input_dims.append(len(msg.inputTrajectory[i].value))
            data.append(msg.data[i].data)

        # TODO better is to keep another controller for use if the current one
        # is locked?
        with self.policy_lock:
            if msg.controllerType == mpc_flattened_controller.CONTROLLER_FEEDFORWARD:
                self.policy = ctrl.bindings.FeedforwardController.unflatten(
                    time_array, data
                )
            elif msg.controllerType == mpc_flattened_controller.CONTROLLER_LINEAR:
                self.policy = ctrl.bindings.LinearController.unflatten(
                    state_dims, input_dims, time_array, data
                )
            else:
                rospy.logwarn("Unknown controller type received!")


