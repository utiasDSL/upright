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
    joint_id = model.addJoint(
        0, pinocchio.JointModelTranslation(), joint_placement, joint_name
    )

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

    geom = ctrl.robot.PinocchioGeometry(robot)

    obs_model, obs_geom_model = obstacle_model()
    new_model, new_geom_model = pinocchio.appendModel(
        robot.model,
        obs_model,
        geom.collision_model,
        obs_geom_model,
        0,
        pinocchio.SE3.Identity(),
    )

    for i in range(obs_geom_model.ngeoms):
        geom.collision_model.addGeometryObject(obs_geom_model.geometryObjects[i])
    new_geom_model = geom.collision_model

    data = new_model.createData()

    x = model.settings.initial_state
    q = np.concatenate((x[robot.dims.x : robot.dims.x + 3], x[: robot.dims.q]))

    pinocchio.forwardKinematics(new_model, data, q)
    pinocchio.updateFramePlacements(new_model, data)

    r = data.oMf[robot.tool_idx].translation.copy()
    # q[-3:] += r
    q[:3] = r

    q = np.array(
        [
            1.394,
            -0.043,
            0.756,
            -0.0,
            0.0,
            0.0,
            1.571,
            -0.785,
            1.571,
            -0.785,
            1.571,
            1.312,
        ]
    )

    pinocchio.forwardKinematics(new_model, data, q)
    pinocchio.updateFramePlacements(new_model, data)

    geom_data = pinocchio.GeometryData(new_geom_model)
    pinocchio.updateGeometryPlacements(new_model, data, new_geom_model, geom_data)
    pinocchio.computeDistances(new_geom_model, geom_data)
    d1s = np.array([result.min_distance for result in geom_data.distanceResults])

    # q[:3] += [0, -1, 0]
    # pinocchio.forwardKinematics(new_model, data, q)
    # pinocchio.updateFramePlacements(new_model, data)
    # pinocchio.updateGeometryPlacements(new_model, data, new_geom_model, geom_data)
    # pinocchio.computeDistances(new_geom_model, geom_data)
    # d2s = np.array([result.min_distance for result in geom_data.distanceResults])

    viz = pinocchio.visualize.MeshcatVisualizer(
        new_model, new_geom_model, new_geom_model
    )
    viz.initViewer()
    viz.loadViewerModel()
    viz.display(q)

    IPython.embed()
    return

    # robot.model = new_model
    # robot.dims.x += 9
    # robot.dims.q += 3
    # robot.dims.v += 3

    x = model.settings.initial_state
    u = np.zeros(model.settings.dims.u())

    x_robot = x[: robot.dims.x]
    u_robot = u[: robot.dims.u]

    # create geometry interface
    geom = ctrl.robot.PinocchioGeometry(robot)
    geom.add_geometry_objects_from_config(config["static_obstacles"])
    geom.add_collision_pairs(config["static_obstacles"]["collision_pairs"])

    robot.forward(x_robot, u_robot)

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
