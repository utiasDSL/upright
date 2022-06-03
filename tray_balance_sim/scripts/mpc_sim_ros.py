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

        if logger.ready(t):
            # log sim stuff
            r_ew_w, Q_we = sim.robot.link_pose()
            v_ew_w, ω_ew_w = sim.robot.link_velocity()
            r_ow_ws, Q_wos = sim.object_poses()
            logger.append("ts", t)
            logger.append("us", u)
            logger.append("xs", x)
            logger.append("r_ew_ws", r_ew_w)
            logger.append("Q_wes", Q_we)
            logger.append("v_ew_ws", v_ew_w)
            logger.append("ω_ew_ws", ω_ew_w)
            logger.append("cmd_vels", sim.robot.cmd_vel.copy())
            logger.append("r_ow_ws", r_ow_ws)
            logger.append("Q_wos", Q_wos)

            # log controller stuff
            r_ew_w_d, Q_we_d = ref.get_desired_pose(t)
            logger.append("r_ew_w_ds", r_ew_w_d)
            logger.append("Q_we_ds", Q_we_d)

            model.update(x, u)
            logger.append("ddC_we_norm", model.ddC_we_norm())
            logger.append("balancing_constraints", model.balancing_constraints())
            logger.append("sa_dists", model.support_area_distances())
            logger.append("orn_err", model.angle_between_acc_and_normal())

        t = sim.step(t, step_robot=False)

    # TODO can I get control durations somehow?
    # - could log frequency of message updates, though this would incur some
    #   overhead
    # - could somehow log and publish it separately

    if cli_args.log is not None:
        logger.save(timestamp, name=cli_args.log)

    if sim.video_manager.save:
        print(f"Saved video to {sim.video_manager.path}")

    # visualize data
    DataPlotter(logger).plot_all(show=True)


if __name__ == "__main__":
    main()
