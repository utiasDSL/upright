#!/usr/bin/env python3
"""Generate and save a trajectory by rolling out the MPC."""
import numpy as np

import tray_balance_constraints as core
import tray_balance_ocs2 as ctrl
import upright_cmd as cmd

import IPython


def main():
    np.set_printoptions(precision=3, suppress=True)

    argparser = cmd.cli.basic_arg_parser()
    argparser.add_argument("trajectory_file", help="NPZ file to save the trajectory to.")
    cli_args = argparser.parse_args()

    # load configuration
    config = core.parsing.load_config(cli_args.config)
    sim_config = config["simulation"]
    ctrl_config = config["controller"]

    timestep = sim_config["timestep"]
    duration = sim_config["duration"]

    # rollout the controller to generate a trajectory
    ctrl_manager = ctrl.manager.ControllerManager.from_config(ctrl_config)
    trajectory = ctrl_manager.plan(timestep, duration)
    trajectory.save(cli_args.trajectory_file)

    print(f"Saved trajectory to {cli_args.outfile}.")


if __name__ == "__main__":
    main()
