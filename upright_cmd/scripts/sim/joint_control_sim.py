#!/usr/bin/env python3
import time

import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb

from upright_sim import util, simulation

import tray_balance_constraints as core

import IPython


def main():
    np.set_printoptions(precision=3, suppress=True)

    cli_args = util.parse_cli_args()

    # load configuration
    config = util.load_config(cli_args.config)
    sim_config = config["simulation"]

    # timing
    duration_millis = sim_config["duration"]
    timestep_millis = sim_config["timestep"]
    timestep_secs = core.parsing.millis_to_secs(timestep_millis)
    num_timesteps = int(duration_millis / timestep_millis)

    # start the simulation
    sim = simulation.MobileManipulatorSimulation(sim_config)
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
        qd = q0 + [0, 0, 0, 0, 0, amp * (1 - np.cos(freq * t)), 0]
        vd = np.array([0, 0, 0, 0, 0, amp * freq * np.sin(freq * t), 0])

        q, _ = robot.joint_states()
        u = K @ (qd - q) + vd

        robot.command_velocity(u)

        sim.step(step_robot=False)
        time.sleep(timestep_secs)
        t += timestep_secs


if __name__ == "__main__":
    main()
