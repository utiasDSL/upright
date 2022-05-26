#!/usr/bin/env python3
"""PyBullet simulation using the bounded balancing constraints"""
import argparse
import time
import datetime
import sys
import os
from pathlib import Path
from threading import Lock

import rospy

import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb
from pyb_utils.ghost import GhostSphere
from pyb_utils.frame import debug_frame_world
from trajectory_msgs.msg import JointTrajectory

from tray_balance_sim import util, camera, simulation
from tray_balance_constraints.logging import DataLogger, DataPlotter
import tray_balance_constraints as core
import tray_balance_ocs2 as ctrl
import upright_cmd as cmd

from ocs2_msgs.msg import (
    mpc_flattened_controller,
    mpc_observation,
    mpc_state,
    mpc_input,
)
from ocs2_msgs.srv import reset as mpc_reset
from ocs2_msgs.srv import resetRequest as mpc_reset_request

import IPython


class QuinticInterpolator:
    def __init__(self, duration, q1, q2, v1, v2, a1, a2):
        T = duration
        A = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [1, T, T ** 2, T ** 3, T ** 4, T ** 5],
                [0, 1, 0, 0, 0, 0],
                [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                [0, 0, 2, 0, 0, 0],
                [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3],
            ]
        )
        b = np.array([q1, q2, v1, v2, a1, a2])
        self.coeffs = np.linalg.solve(A, b)

    def eval(self, t):
        q = self.coeffs.dot([1, t, t ** 2, t ** 3, t ** 4, t ** 5])
        v = self.coeffs[1:].dot(
            [1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4]
        )
        a = self.coeffs[2:].dot(
            [2, 6 * t, 12 * t ** 2, 20 * t ** 3]
        )
        return q, v, a


def decompose_state(x, nq, nv):
    q = x[: nq]
    v = x[nq : nq + nv]
    a = x[nq + nv : nq + 2 * nv]
    return q, v, a


def interpolate(t_opt, x_opt, t, nq, nv):
    if t <= t_opt[0]:
        return decompose_state(x_opt[0], nq, nv)
    elif t >= t_opt[-1]:
        return decompose_state(x_opt[-1], nq, nv)
    else:
        for i in range(len(t_opt) - 1):
            if t_opt[i] <= t <= t_opt[i + 1]:
                break

    t1 = t_opt[i]
    t2 = t_opt[i + 1]
    Δt = t2 - t1

    # if the timestep between the states is really small, then we don't need to
    # bother interpolating
    if Δt <= 1e-3:
        return decompose_state(x_opt[i], nq, nv)

    q1, v1, a1 = decompose_state(x_opt[i], nq, nv)
    q2, v2, a2 = decompose_state(x_opt[i + 1], nq, nv)
    q = np.zeros(nq)
    v = np.zeros(nv)
    a = np.zeros(nv)
    for j in range(nq):
        q[j], v[j], a[j] = QuinticInterpolator(Δt, q1[j], q2[j], v1[j], v2[j], a1[j], a2[j]).eval(t - t1)
    return q, v, a


class ROSInterface:
    def __init__(self, topic_prefix):
        rospy.init_node("pyb_interface")

        self.policy = None

        self.t_opt = None
        self.x_opt = None
        self.u_opt = None

        self.policy_lock = Lock()
        self.trajectory_lock = Lock()

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
        req.targetTrajectories.timeTrajectory = ref.times
        for state in ref.states:
            msg = mpc_state()
            msg.value = state
            req.targetTrajectories.stateTrajectory.append(msg)
        for inp in ref.inputs:
            msg = mpc_input()
            msg.value = inp
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
            self.t_opt = np.array(t_opt)
            self.x_opt = np.array(x_opt)
            self.u_opt = np.array(u_opt)

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

        # optimal trajectory
        # t_opt = []
        # x_opt = []
        # u_opt = []
        # for i in range(len(msg.timeTrajectory)):
        #     t_opt.append(msg.timeTrajectory[i])
        #     x_opt.append(msg.stateTrajectory[i].value)
        #     u_opt.append(msg.inputTrajectory[i].value)
        #
        # with self.trajectory_lock:
        #     self.t_opt = np.array(t_opt)
        #     self.x_opt = np.array(x_opt)
        #     self.u_opt = np.array(u_opt)

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


def main():
    np.set_printoptions(precision=3, suppress=True)

    cli_args = cmd.cli.sim_arg_parser().parse_args()

    # load configuration
    config = core.parsing.load_config(cli_args.config)
    sim_config = config["simulation"]
    ctrl_config = config["controller"]
    log_config = config["logging"]

    # timing
    duration_millis = sim_config["duration"]
    timestep_millis = sim_config["timestep"]
    timestep_secs = core.parsing.millis_to_secs(timestep_millis)
    duration_secs = core.parsing.millis_to_secs(duration_millis)
    num_timesteps = int(duration_millis / timestep_millis)
    ctrl_period = ctrl_config["control_period"]

    # start the simulation
    sim = simulation.MobileManipulatorSimulation(sim_config)
    robot = sim.robot

    # setup sim objects
    r_ew_w, Q_we = robot.link_pose()
    sim_objects = simulation.sim_object_setup(r_ew_w, sim_config)
    num_objects = len(sim_objects)

    # mark frame at the initial position
    debug_frame_world(0.2, list(r_ew_w), orientation=Q_we, line_width=3)

    # initial time, state, input
    t = 0.0
    q, v = robot.joint_states()
    a = np.zeros(robot.nv)
    x = np.concatenate((q, v, a))
    u = np.zeros(robot.nu)

    # video recording
    now = datetime.datetime.now()
    video_manager = camera.VideoManager.from_config_dict(
        video_name=cli_args.video, config=sim_config, timestamp=now, r_ew_w=r_ew_w
    )

    ctrl_wrapper = ctrl.parsing.ControllerConfigWrapper(ctrl_config, x0=x)
    ctrl_objects = ctrl_wrapper.objects()

    # data logging
    log_dir = Path(log_config["log_dir"])
    log_dt = log_config["timestep"]
    logger = DataLogger(config)

    logger.add("sim_timestep", timestep_secs)
    logger.add("duration", duration_secs)
    logger.add("control_period", ctrl_period)
    logger.add("object_names", [str(name) for name in sim_objects.keys()])

    logger.add("nq", ctrl_config["robot"]["dims"]["q"])
    logger.add("nv", ctrl_config["robot"]["dims"]["v"])
    logger.add("nx", ctrl_config["robot"]["dims"]["x"])
    logger.add("nu", ctrl_config["robot"]["dims"]["u"])

    # create reference trajectory and controller
    ref = ctrl_wrapper.reference_trajectory(r_ew_w, Q_we)

    # TODO this is only used to easily compute constraint values
    # would be better to separate out that functionality to avoid having to
    # create a whole other object (which requires autodiff compilation, etc.)
    mpc_inner = ctrl_wrapper.controller(ref)

    # frames and ghost (i.e., pure visual) objects
    ghosts = []
    for state in ref.states:
        r_ew_w_d, Q_we_d = ref.pose(state)
        # ghosts.append(GhostSphere(radius=0.05, position=r_ew_w_d, color=(0, 1, 0, 1)))
        debug_frame_world(0.2, list(r_ew_w_d), orientation=Q_we_d, line_width=3)

    target_idx = 0
    static_stable = True

    # setup the ROS interface
    ros_interface = ROSInterface("mobile_manipulator")
    ros_interface.reset_mpc(ref)
    ros_interface.publish_observation(t, x, u)

    # custom PD controller
    K = 1 * np.eye(robot.nq)

    # wait for an initial policy to be computed
    rate = rospy.Rate(1.0 / timestep_secs)
    # rate = rospy.Rate(10)
    while (ros_interface.policy is None or ros_interface.x_opt is None) and not rospy.is_shutdown():
        rate.sleep()

    # simulation loop
    # this loop sets the MPC observation and computes the control input at a
    # faster rate than the outer loop MPC optimization problem
    # TODO: ideally we'd seperate this cleanly into its own function
    for i in range(num_timesteps):
        q, v = robot.joint_states()
        x = np.concatenate((q, v, a))

        # add noise to state variables
        q_noisy, v_noisy = robot.joint_states(add_noise=True)
        x_noisy = np.concatenate((q_noisy, v_noisy, a))

        # publish latest observation
        if ctrl_config["use_noisy_state_to_plan"]:
            ros_interface.publish_observation(t, x_noisy, u)
        else:
            ros_interface.publish_observation(t, x, u)

        # compute the input using the current controller
        # with ros_interface.policy_lock:
        #     u = ros_interface.policy.computeInput(t, x)
        # a = np.copy(robot.cmd_acc)
        # robot.command_jerk(u)

        with ros_interface.trajectory_lock:
            t_opt = np.copy(ros_interface.t_opt)
            x_opt = np.copy(ros_interface.x_opt)
        qd, vd, a = interpolate(t_opt, x_opt, t, robot.nq, robot.nv)
        v_cmd = K @ (qd - q) + vd
        robot.command_velocity(v_cmd)

        sim.step(step_robot=False)
        t += timestep_secs
        rate.sleep()

        # if we have multiple targets, step through them
        if t >= ref.times[target_idx] and target_idx < len(ref.times) - 1:
            target_idx += 1

    IPython.embed()


if __name__ == "__main__":
    main()
