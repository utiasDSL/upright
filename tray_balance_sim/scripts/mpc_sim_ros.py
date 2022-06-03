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
        t0 = time.time()
        self.coeffs = np.linalg.solve(A, b)
        t1 = time.time()
        print(f"first time = {t1 - t0}")

    def eval(self, t):
        q = self.coeffs.dot([1, t, t ** 2, t ** 3, t ** 4, t ** 5])
        v = self.coeffs[1:].dot([1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4])
        a = self.coeffs[2:].dot([2, 6 * t, 12 * t ** 2, 20 * t ** 3])
        return q, v, a


def decompose_state(x, nq, nv):
    q = x[:nq]
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
        q[j], v[j], a[j] = QuinticInterpolator(
            Δt, q1[j], q2[j], v1[j], v2[j], a1[j], a2[j]
        ).eval(t - t1)
    return q, v, a


class ROSInterface:
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


def main():
    np.set_printoptions(precision=3, suppress=True)

    cli_args = cmd.cli.sim_arg_parser().parse_args()

    # load configuration
    config = core.parsing.load_config(cli_args.config)
    sim_config = config["simulation"]
    ctrl_config = config["controller"]
    log_config = config["logging"]

    # start the simulation
    timestamp = datetime.datetime.now()
    sim = simulation.BulletSimulation(
        config=sim_config, timestamp=timestamp, cli_args=cli_args
    )

    # initial time, state, input
    t = 0.0
    q, v = sim.robot.joint_states()
    a = np.zeros(sim.robot.nv)
    x = np.concatenate((q, v, a))
    u = np.zeros(sim.robot.nu)

    # data logging
    logger = DataLogger(config)

    logger.add("sim_timestep", sim.timestep)
    logger.add("duration", sim.duration)
    logger.add("object_names", [str(name) for name in sim.objects.keys()])

    logger.add("nq", ctrl_config["robot"]["dims"]["q"])
    logger.add("nv", ctrl_config["robot"]["dims"]["v"])
    logger.add("nx", ctrl_config["robot"]["dims"]["x"])
    logger.add("nu", ctrl_config["robot"]["dims"]["u"])

    model = ctrl.manager.ControllerModel.from_config(ctrl_config)
    mapping = ctrl.manager.StateInputMapping(model.settings.dims)
    Kp = np.eye(model.settings.dims.q)
    interpolator = ctrl.manager.TrajectoryInterpolator(mapping, None)

    # reference pose trajectory
    model.update(x=model.settings.initial_state)
    r_ew_w, Q_we = model.robot.link_pose()
    ref = ctrl.wrappers.TargetTrajectories.from_config(ctrl_config, r_ew_w, Q_we, u)

    # frames and ghost (i.e., pure visual) objects
    for r_ew_w_d, Q_we_d in ref.poses():
        debug_frame_world(0.2, list(r_ew_w_d), orientation=Q_we_d, line_width=3)

    # setup the ROS interface
    ros_interface = ROSInterface("mobile_manipulator", interpolator)
    ros_interface.reset_mpc(ref)
    ros_interface.publish_observation(t, x, u)

    # wait for an initial policy to be computed
    rate = rospy.Rate(1.0 / sim.timestep)
    # rate = rospy.Rate(10)
    while (
        ros_interface.policy is None or ros_interface.trajectory is None
    ) and not rospy.is_shutdown():
        rate.sleep()

    # simulation loop
    while t <= sim.duration:
        q, v = sim.robot.joint_states(add_noise=True)
        x = np.concatenate((q, v, a))
        ros_interface.publish_observation(t, x, u)

        with ros_interface.trajectory_lock:
            xd = interpolator.interpolate(t)
            qd, vd, a = mapping.xu2qva(xd)

        v_cmd = Kp @ (qd - q) + vd
        sim.robot.command_velocity(v_cmd)

        t = sim.step(t, step_robot=False)

    IPython.embed()


if __name__ == "__main__":
    main()
