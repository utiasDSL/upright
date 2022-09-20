#!/usr/bin/env python3
import argparse
import numpy as np
import pinocchio
import hppfcl as fcl

import upright_core as core
import upright_control as ctrl

import IPython


def build_obstacle_model(obstacles):
    model = pinocchio.Model()
    model.name = "dynamic_obstacles"
    geom_model = pinocchio.GeometryModel()

    for obstacle in obstacles:
        # free-floating joint
        joint_name = obstacle.name + "_joint"
        joint_placement = pinocchio.SE3.Identity()
        joint_id = model.addJoint(
            0, pinocchio.JointModelTranslation(), joint_placement, joint_name
        )

        # body
        mass = 1.0
        inertia = pinocchio.Inertia.FromSphere(mass, obstacle.radius)
        body_placement = pinocchio.SE3.Identity()
        model.appendBodyToJoint(joint_id, inertia, body_placement)

        # visual model
        shape = fcl.Sphere(obstacle.radius)
        geom_obj = pinocchio.GeometryObject(obstacle.name, joint_id, shape, body_placement)
        geom_obj.meshColor = np.ones((4))
        geom_model.addGeometryObject(geom_obj)

    return model, geom_model


def append_model(
    robot, geom, model, geom_model, frame_index=0, placement=pinocchio.SE3.Identity()
):
    new_model, new_collision_model = pinocchio.appendModel(
        robot.model, model, geom.collision_model, geom_model, 0, placement
    )
    _, new_visual_model = pinocchio.appendModel(
        robot.model, model, geom.visual_model, geom_model, 0, placement
    )

    new_robot = ctrl.robot.PinocchioRobot(
        new_model, robot.mapping, robot.tool_link_name
    )
    new_geom = ctrl.robot.PinocchioGeometry(
        new_robot, new_collision_model, new_visual_model
    )
    return new_robot, new_geom


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
    geom = ctrl.robot.PinocchioGeometry.from_robot_and_urdf(
        robot, settings.robot_urdf_path
    )

    obs_model, obs_geom_model = build_obstacle_model(settings.obstacle_settings.dynamic_obstacles)
    robot, geom = append_model(robot, geom, obs_model, obs_geom_model)

    x = settings.initial_state
    u = np.zeros(settings.dims.u())
    q = robot.mapping.get_pinocchio_joint_position(x)

    # add offset to the dynamic obstacles
    robot.forward(x)
    r = robot.link_pose()[0]
    for i in range(settings.dims.o):
        q[i*3:(i+1)*3] += r

    robot.forward_qva(q)

    geom.add_geometry_objects_from_config(config["obstacles"])
    if config["obstacles"]["collision_pairs"] is not None:
        geom.add_collision_pairs(config["obstacles"]["collision_pairs"])

    dists = geom.compute_distances()
    geom.visualize(q)

    print(f"Collision distances = {dists}")

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
