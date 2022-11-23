#!/usr/bin/env python3
import time
import datetime

import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb

from upright_sim import util, simulation
from mobile_manipulation_central.trajectory_generation import (
    PointToPointTrajectory,
    QuinticTimeScaling,
    CompositeTrajectory,
)

import upright_core as core
import upright_cmd as cmd

import IPython


DESIRED_BASE_CONFIGURATION = np.array([-1, 0, 0])

# fmt: off
# last arm configuration is same as the first, so the robot goes back when done
# calibrating
DESIRED_ARM_CONFIGURATIONS = np.array([
  [1.5708, -0.7854,  1.5708, -0.7854,  1.5708, 1.3100],
  [0,      -0.7854,  1.5708, -0.7854,  1.5708, 1.3100],
  [2.3562, -0.7854,  1.5708, -0.7854,  1.5708, 1.3100],
  [2.3562, -1.5708,  1.5708, -0.7854,  1.5708, 1.3100],
  [2.3562, -1.5708,  0.7854, -0.7854,  1.5708, 1.3100],
  [2.3562, -1.5708,  0.7854,       0,  1.5708, 1.3100],
  [2.3562, -1.5708,  0.7854,       0,       0, 1.3100],
  [1.5708, -1.5708,  0.7854,       0,       0, 0.5236],
  [0,      -1.5708,  0.7854,       0,       0, 0.5236],
  [1.5708, -0.7854,  1.5708, -0.7854,  1.5708, 1.3100]])
# fmt: on


def main():
    np.set_printoptions(precision=3, suppress=True)

    cli_args = cmd.cli.sim_arg_parser().parse_args()

    # load configuration
    config = core.parsing.load_config(cli_args.config)
    sim_config = config["simulation"]

    # start the simulation
    timestamp = datetime.datetime.now()
    sim = simulation.BulletSimulation(
        config=sim_config, timestamp=timestamp, cli_args=cli_args
    )
    robot = sim.robot

    # control params
    K = 10 * np.eye(robot.nq)

    # initial time, state, input
    t = 0.0
    q, _ = robot.joint_states()
    q0 = np.copy(q)

    num_configs = DESIRED_ARM_CONFIGURATIONS.shape[0]
    goals = np.hstack(
        (
            np.tile(DESIRED_BASE_CONFIGURATION, (num_configs, 1)),
            DESIRED_ARM_CONFIGURATIONS,
        )
    )

    max_vel = 0.5
    max_acc = 1.0
    goal = goals[0, :]
    trajectory = PointToPointTrajectory.quintic(q0, goal, max_vel, max_acc)
    print(f"Moving to position 0 with duration {trajectory.duration} seconds.")

    # simulation loop
    idx = 0
    while True:
        q, _ = robot.joint_states()

        # if we've reached the current goal, move to the next one
        dist = np.linalg.norm(goal - q)
        if dist < 1e-3:
            print(f"Converged to position {idx} with error {dist}.")
            idx += 1
            if idx == num_configs:
                print("Sequence complete.")
                break
            goal = goals[idx, :]
            trajectory = PointToPointTrajectory.quintic(q, goal, max_vel, max_acc)
            print(f"Moving to position {idx} with duration {trajectory.duration} seconds.")

        qd, vd, _ = trajectory.sample(t)
        u = K @ (qd - q) + vd

        robot.command_velocity(u)

        t = sim.step(t, step_robot=False)
        # time.sleep(sim.timestep)


if __name__ == "__main__":
    main()
