#!/usr/bin/env python3
import argparse
import numpy as np
import pinocchio
import hppfcl as fcl

import upright_core as core
import upright_control as ctrl

import IPython


def obstacle_model():
    model = pinocchio.Model()
    model.name = "obstacle"
    geom_model = pinocchio.GeometryModel()

    # free-floating joint
    joint_name = "obstacle_joint"
    joint_placement = pinocchio.SE3.Identity()
    joint_id = model.addJoint(0, pinocchio.JointModelTranslation(), joint_placement, joint_name)

    # body
    mass = 1.0
    radius = 0.1
    inertia = pinocchio.Inertia.FromSphere(mass, radius)
    body_placement = pinocchio.SE3.Identity()
    model.appendBodyToJoint(joint_id, inertia, body_placement)

    # visual model
    geom_name = "obstacle"
    shape = fcl.Sphere(radius)
    geom_obj = pinocchio.GeometryObject(geom_name, joint_id, shape, body_placement)
    geom_obj.meshColor = np.ones((4))
    geom_model.addGeometryObject(geom_obj)

    return model, geom_model


def main():
    np.random.seed(0)
    np.set_printoptions(precision=3, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file.")
    args = parser.parse_args()

    # get config path from command line argument
    config = core.parsing.load_config(args.config)["controller"]
    model = ctrl.manager.ControllerModel.from_config(config)
    robot = model.robot

    obs_model, obs_geom_model = obstacle_model()
    new_model = pinocchio.appendModel(
        robot.model,
        obs_model,
        # geom.visual_model,
        # obs_geom_model,
        0,
        pinocchio.SE3.Identity(),
    )
    IPython.embed()
    # robot.model = new_model
    # robot.dims.x += 9
    # robot.dims.q += 3
    # robot.dims.v += 3

    x = np.concatenate((model.settings.initial_state, np.zeros(9)))
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
    q = x[: robot.dims.q]
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
