#!/usr/bin/env python3
import time
import datetime

import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb

from upright_sim import util, simulation

import upright_core as core
import upright_cmd as cmd

import IPython


def main():
    np.set_printoptions(precision=3, suppress=True)

    cli_args = cmd.cli.sim_arg_parser().parse_args()

    # load configuration
    config = core.parsing.load_config(cli_args.config)
    sim_config = config["simulation"]

    # timing
    duration_millis = sim_config["duration"]
    timestep_millis = sim_config["timestep"]
    timestep_secs = core.parsing.millis_to_secs(timestep_millis)
    num_timesteps = int(duration_millis / timestep_millis)

    # start the simulation
    timestamp = datetime.datetime.now()
    sim = simulation.BulletSimulation(
        config=sim_config, timestamp=timestamp, cli_args=cli_args
    )
    robot = sim.robot

    # control params
    K = 0.5 * np.eye(7)
    amp = 0.1
    freq = 2

    # initial time, state, input
    t = 0.0
    q, _ = robot.joint_states()
    q0 = np.copy(q)

    # simulation loop
    for i in range(num_timesteps):
        # qd = q0 + [0, 0, 0, 0, 0, amp * (1 - np.cos(freq * t)), 0]
        # vd = np.array([0, 0, 0, 0, 0, amp * freq * np.sin(freq * t), 0])

        # q, _ = robot.joint_states()
        # u = K @ (qd - q) + vd
        u = np.array([0, 0, 0, 0, 0, 0.1])

        robot.command_velocity(u)

        t += sim.step(t, step_robot=False)
        time.sleep(timestep_secs)


if __name__ == "__main__":
    main()
