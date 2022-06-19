#!/usr/bin/env python3
"""PyBullet simulation using the bounded balancing constraints"""
import argparse
import time
import datetime
import sys
import os
from pathlib import Path
from multiprocessing import Process, Manager, Pipe, Lock
import pickle

import numpy as np
import matplotlib.pyplot as plt

from upright_sim import util, camera, simulation

from upright_core.logging import DataLogger, DataPlotter
import upright_core as core
import tray_balance_ocs2 as ctrl

import IPython


class DoubleIntegrator:
    def __init__(self, v0, a0):
        self.dt = dt
        self.v = np.copy(v0)
        self.a = np.copy(a0)

    def integrate(self, dt, cmd):
        self.v += dt * self.a
        self.a += dt * cmd


def outer_control_loop(
    ctrl_wrapper, r_ew_w, Q_we, sync_lock, outer_ctrl_con, outer_ctrl_txu
):
    """Outer control loop containing MPC optimization.

    Advance MPC at a fixed control rate."""
    try:
        # initialize and run first iteration
        mpc = ctrl_wrapper.controller(r_ew_w, Q_we)
        t, x, u = outer_ctrl_txu.recv()
        mpc.setObservation(t, x, u)
        mpc.advanceMpc()
        outer_ctrl_con.send(mpc.getLinearController())
    finally:
        sync_lock.release()

    # run MPC as fast as possible
    # rate = core.util.Rate.from_hz(40)
    while True:
        # get the latest state
        while outer_ctrl_txu.poll(timeout=0):
            t, x, u = outer_ctrl_txu.recv()
        mpc.setObservation(t, x, u)
        mpc.advanceMpc()
        outer_ctrl_con.send(mpc.getLinearController())
        # rate.sleep()


def main():
    np.set_printoptions(precision=3, suppress=True)

    cli_args = util.parse_cli_args()

    # load configuration
    config = core.parsing.load_config(cli_args.config)
    ctrl_config = config["controller"]
    perls2_config = config["perls2"]
    log_config = config["logging"]

    # timing
    # TODO this needs to be revised
    duration_millis = sim_config["duration"]
    timestep_millis = sim_config["timestep"]
    timestep_secs = core.parsing.millis_to_secs(timestep_millis)
    duration_secs = core.parsing.millis_to_secs(duration_millis)
    num_timesteps = int(duration_millis / timestep_millis)
    ctrl_period = ctrl_config["control_period"]

    nq = nv = nu = 7

    # initial time, state, input
    t = 0.0
    q = np.array(perls2_config["neutral_joint_angles"])
    v = np.zeros(nv)
    a = np.zeros(nv)
    x = np.concatenate((q, v, a))
    u = np.zeros(nu)

    ctrl_wrapper = ctrl.parsing.ControllerConfigWrapper(ctrl_config, x0=x)

    # namespace to manage shared controller resources
    outer_ctrl_con, inner_ctrl_con = Pipe()
    outer_ctrl_txu, inner_ctrl_txu = Pipe()

    # acquire a lock which is only released after MPC is initialized in the
    # outer-loop process
    sync_lock = Lock()
    sync_lock.acquire()

    # send the initial state
    inner_ctrl_txu.send((t, x, u))

    # start separate control process
    outer_control_proc = Process(
        target=outer_control_loop,
        args=(
            ctrl_wrapper,
            r_ew_w,  # TODO need to get these
            Q_we,
            sync_lock,
            outer_ctrl_con,
            outer_ctrl_txu,
        ),
    )
    outer_control_proc.start()

    # wait until MPC is initialized
    sync_lock.acquire()

    # TODO need control based on real time now
    rate = core.util.Rate.from_timestep_secs(dt)
    integrator = DoubleIntegrator(v, a)

    robot = RealPandaInterface(perls2_config, controlType="JointVelocity")
    robot.reset()

    # simulation loop
    # this loop sets the MPC observation and computes the control input at a
    # faster rate than the outer loop MPC optimization problem
    # TODO: ideally we'd seperate this cleanly into its own function
    try:
        for i in range(num_timesteps):
            q, v = robot.q, robot.dq
            a = integrator.a
            x = np.concatenate((q, v, a))

            inner_ctrl_txu.send((t, x, u))

            # get a new controller if available
            if inner_ctrl_con.poll(timeout=0):
                lin_ctrl = inner_ctrl_con.recv()

            # compute the input using the current controller
            u = lin_ctrl.computeInput(t, x)

            # TODO need to make sure loop rate is correct
            integrator.step(dt, u)
            robot.set_joint_velocities(integator.v)

            rate.sleep()
            t += timestep_secs
    finally:
        # rejoin the control process
        outer_control_proc.terminate()
        print("killed outer control loop process")

        robot.reset()
        robot.disconnect()


if __name__ == "__main__":
    main()
