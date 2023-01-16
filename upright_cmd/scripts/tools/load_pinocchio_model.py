#!/usr/bin/env python3
import argparse
import numpy as np
import pinocchio
import hppfcl as fcl

import upright_core as core
import upright_control as ctrl

import IPython


def main():
    np.random.seed(0)
    np.set_printoptions(precision=3, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file.")
    args = parser.parse_args()

    # get config path from command line argument
    config = core.parsing.load_config(args.config)["controller"]
    ctrl_model = ctrl.manager.ControllerModel.from_config(config)
    settings = ctrl_model.settings
    robot = ctrl_model.robot
    geom = ctrl_model.geom

    x = settings.initial_state
    u = np.zeros(settings.dims.u())
    q = robot.mapping.get_pinocchio_joint_position(x)

    # add offset to the dynamic obstacles
    robot.forward(x)
    r = robot.link_pose()[0]
    for i in range(settings.dims.o):
        q[i*3:(i+1)*3] += r

    robot.forward_qva(q)

    dists = geom.compute_distances()
    geom.visualize(q)

    print(f"Collision distances = {dists}")

    # forward kinematics (position, velocity, acceleration)
    r, Q = robot.link_pose()
    dr, ω = robot.link_velocity()
    J = robot.jacobian(q)

    print(f"Tool pose: r = {r}, Q = {Q}")

    # note that this is the classical acceleration, not the spatial acceleration
    ddr, α = robot.link_classical_acceleration()

    # forward kinematics derivatives
    robot.forward_derivatives(x, u)
    dVdq, dVdv = robot.link_velocity_derivatives()
    dAdq, dAdv, dAda = robot.link_classical_acceleration_derivatives()

    # J == dVdv == dAda
    np.testing.assert_allclose(J, dVdv, atol=1e-5)
    np.testing.assert_allclose(J, dAda, atol=1e-5)

    IPython.embed()


if __name__ == "__main__":
    main()
