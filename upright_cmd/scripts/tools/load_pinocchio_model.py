#!/usr/bin/env python3
import argparse
import numpy as np

import upright_core as core
import upright_control as ctrl

import IPython


def main():
    np.random.seed(0)
    np.set_printoptions(precision=3, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file.")
    # parser.add_argument(
    #     "--visualize",
    #     help="Visualize the scene in meshcat",
    #     action="store_true",
    # )
    args = parser.parse_args()

    # get config path from command line argument
    config = core.parsing.load_config(args.config)["controller"]
    model = ctrl.manager.ControllerModel.from_config(config)
    robot = model.robot

    x = model.settings.initial_state
    u = np.zeros(robot.dims.u)

    # create geometry interface
    geom = ctrl.robot.PinocchioGeometry(robot)
    geom.add_geometry_objects_from_config(config["static_obstacles"])
    geom.add_collision_pairs(config["static_obstacles"]["collision_pairs"])

    robot.forward(x, u)

    # compute distances between collision pairs
    dists = geom.compute_distances()
    print(f"Collision distances = {dists}")

    # visualize the robot
    q = x[:robot.dims.q]
    viz = geom.visualize(q)

    # forward kinematics (position, velocity, acceleration)
    r, Q = robot.link_pose()
    dr, ω = robot.link_velocity()
    J = robot.jacobian(q)

    # note that this is the classical acceleration, not the spatial acceleration
    ddr, α = robot.link_acceleration()

    # forward kinematics derivatives
    robot.forward_derivatives(x, u)
    dVdq, dVdv = robot.link_velocity_derivatives()
    dAdq, dAdv, dAda = robot.link_acceleration_derivatives()

    # J == dVdv == dAda
    np.testing.assert_allclose(J, dVdv, atol=1e-5)
    np.testing.assert_allclose(J, dAda, atol=1e-5)

    IPython.embed()


if __name__ == "__main__":
    main()
